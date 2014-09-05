//
//  SaltoConsoleController.m
//  OpenSalto GUI
//
//  Copyright 2014 Marko Havu. Released under the terms of
//  GNU General Public License version 3 or later.
//

#import "SaltoConsoleController.h"
#import "SaltoGuiAppDelegate.h"
#include "salto.h"

@implementation SaltoConsoleController

@synthesize console;
@synthesize textView;
@synthesize insertionPoint;

- (instancetype)init {
    self = [super initWithWindowNibName: @"SaltoConsole"];
    if (self) {
        console = [[SaltoConsole alloc] init];
    }

    return self;
}

- (void)dealloc {
    [console release];
    [super dealloc];
}

- (void)windowDidLoad {
    [super windowDidLoad];
    for (NSString *string in [console outputStrings]) {
        [textView setString:[textView.string stringByAppendingString:string]];
    }
    insertionPoint = [textView.string length];
    [textView scrollRangeToVisible:NSMakeRange(insertionPoint, 0)];
}

- (BOOL)textView:(NSTextView *)view shouldChangeTextInRange:(NSRange)range replacementString:(NSString *)string {
    BOOL shouldChange;
    
    // TODO: Use NSParagraphStyle
    if (range.location < insertionPoint) {
        [view.textStorage appendAttributedString:[[[NSAttributedString alloc] initWithString:string] autorelease]];
        [view setSelectedRange:NSMakeRange(view.string.length, 0)];
        shouldChange = NO;
    } else {
        shouldChange = YES;
    }
    // TODO: if interactive, [inputHandle writeData:[string dataUsingEncoding:NSUTF8StringEncoding]];
    
    return shouldChange;
}

- (BOOL)textView:(NSTextView *)view doCommandBySelector:(SEL)aSelector {
    BOOL result = NO;
    
    if (aSelector == @selector(insertNewline:)) {
        if (![console isExecuting]) {
            if (NSApplication.sharedApplication.currentEvent.modifierFlags & NSShiftKeyMask) {
                // Add a newline at the end of the edit buffer.
                [view setSelectedRange:NSMakeRange(view.string.length, 0)];
                [view insertNewlineIgnoringFieldEditor:self];

                // Send the contents of the edit buffer to the Python interpreter.
                [console execute:[view.string substringWithRange:NSMakeRange(insertionPoint, view.string.length -insertionPoint)]];

                insertionPoint = textView.string.length;
                
                result = YES;
            }
        } else {
            // TODO: Make readline return
        }
    }

    return result;
}

- (void)insertOutput:(NSString *)string {
    NSRange range = NSMakeRange(insertionPoint, 0);
    [textView.textStorage replaceCharactersInRange:range withString:string];
    insertionPoint += string.length;
    [console appendOutput:string];
    [textView scrollRangeToVisible:NSMakeRange(insertionPoint, 0)];
}

@end


@implementation SaltoConsole

@synthesize inputArray;
@synthesize outputArray;
@synthesize executing;

- (instancetype)init {
    self = [super init];
    if (self) {
        inputArray = [[NSMutableArray alloc] init];
        outputArray = [[NSMutableArray alloc] init];
        [outputArray addObject:[NSMutableString string]];
    }

    return self;
}

- (void)dealloc {
    [inputArray release];
    [outputArray release];
    [super dealloc];
}

- (NSArray *)inputStrings {
    return [NSArray arrayWithArray:inputArray];
}

- (NSArray *)outputStrings {
    return [NSArray arrayWithArray:outputArray];
}

- (void)appendOutput:(NSString *)string {
    NSMutableString *currentOutput = [outputArray objectAtIndex:outputArray.count - 1];
    [currentOutput appendString:string];
}

- (void)execute:(NSString *)statement {
    SaltoGuiAppDelegate *appDelegate = NSApplication.sharedApplication.delegate;
    dispatch_async(appDelegate.queue,
                   ^{
                       executing = YES;
                       PyObject *output = saltoEval([statement UTF8String]);
                       executing = NO;
                       if (output) {
                           [outputArray addObject:[NSMutableString string]];
                           NSString *string = [NSString stringWithUTF8String:PyUnicode_AsUTF8(output)];
                           dispatch_async(dispatch_get_main_queue(),
                                          ^{ [appDelegate.consoleController insertOutput:string]; });
                       }
                   });
    [inputArray addObject:statement];
    // TODO: handle output
}

@end


static PyObject* readFromConsole(PyObject* self, PyObject* args) {
    Py_ssize_t size;

    if (PyArg_ParseTuple(args, "|n:read", &size)) {
        NSLog(@"read is not implemented yet\n");
        // TODO: implement
        // Read and return up to size bytes.
        // If the argument is omitted, None, or negative, data is read and returned until EOF is reached.
        // Maybe capture keypresses directly?
    } else {
        PyErr_SetString(PyExc_TypeError, "read() takes an optional integer argument");
    }

    return PyUnicode_FromString("");
}

static PyObject* readLineFromConsole(PyObject* self, PyObject* args) {
    Py_ssize_t size;

    if (PyArg_ParseTuple(args, "|n:readline", &size)) {
        NSLog(@"readline is not implemented yet\n");
        // TODO: implement
        // Read and return one line from the stream. If size is specified, at most size bytes will be read.
        // Return when newline is entered.
        // We should probably block in another thread.
    } else {
        PyErr_SetString(PyExc_TypeError, "readline() takes an optional integer argument");
    }

    return PyUnicode_FromString("");
}

static PyObject* readLinesFromConsole(PyObject* self, PyObject* args) {
    PyObject *list = NULL;
    Py_ssize_t hint;

    if (PyArg_ParseTuple(args, "|n:readlines", &hint)) {
        NSLog(@"readlines is not implemented yet\n");
        // TODO: implement
        // Read and return a list of lines from the stream.
        // A hint can be specified to control the number of lines read:
        // no more lines will be read if the total size (in bytes/characters) of all lines so far exceeds hint.
    } else {
        PyErr_SetString(PyExc_TypeError, "readlines() takes an optional integer argument");
    }

    return list;
}

static PyObject* writeToConsole(PyObject* self, PyObject* args) {
    const char *str;
 
    if (PyArg_ParseTuple(args, "s:write", &str)) {
        NSString *string = [NSString stringWithUTF8String:str];
        SaltoGuiAppDelegate *appDelegate = NSApplication.sharedApplication.delegate;
        SaltoConsoleController *consoleController = appDelegate.consoleController;
        // TODO: Use NSParagraphStyle
        dispatch_async(dispatch_get_main_queue(), ^{ [consoleController insertOutput:string]; });
    } else {
        PyErr_SetString(PyExc_TypeError, "write() takes a string argument");
    }
 
    return Py_BuildValue("i", strlen(str));
}

static PyObject* writeLinesToConsole(PyObject* self, PyObject* args) {
    PyObject *iterable, *iterator, *o, *s;
    NSString *string;
    NSRange range;
    
    if (PyArg_ParseTuple(args, "O:writelines", &iterable)) {
        SaltoGuiAppDelegate *appDelegate = NSApplication.sharedApplication.delegate;
        SaltoConsoleController *consoleController = appDelegate.consoleController;
        iterator = PyObject_GetIter(iterable);  // new
        while ((o = PyIter_Next(iterator))) {  // new
            if (PyUnicode_Check(o)) {
                s = PyUnicode_AsUTF8String(o);  // new
            } else {
                s = o;
                Py_INCREF(s);
            }
            if (PyBytes_Check(s)) {
                string = [NSString stringWithUTF8String:PyBytes_AsString(s)];
                range = NSMakeRange(consoleController.insertionPoint, 0);
                // TODO: Use NSParagraphStyle
                dispatch_async(dispatch_get_main_queue(), ^{ [consoleController insertOutput:string]; });
                Py_DECREF(s);
            } else {
                PyErr_SetString(PyExc_TypeError, "writelines() takes an iterable of strings as an argument");
                continue;
            }
            Py_DECREF(o);
        }
        Py_XDECREF(iterator);
    } else {
        PyErr_SetString(PyExc_TypeError, "writelines() takes an iterable of strings as an argument");
    }

    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject* consoleEncoding(PyObject* self, PyObject* args) {
    return PyUnicode_FromString("utf-8");
}

static PyObject* consoleName(PyObject* self, PyObject* args) {
    return PyUnicode_FromString("OpenSALTO console window");
}

static PyObject* returnNone(PyObject* self, PyObject* args) {
    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject* returnTrue(PyObject* self, PyObject* args) {
    Py_INCREF(Py_True);
    return Py_True;
}

static PyObject* returnFalse(PyObject* self, PyObject* args) {
    Py_INCREF(Py_False);
    return Py_False;
}

// salto.gui.addChannel()
// salto.gui.removeChannel()
// -> inform SaltoChannelViewController

static PyMethodDef saltoGuiMethods[] = {
    {"closed", returnFalse, METH_VARARGS, "returns False (the stream is not closed)"},
    {"encoding", consoleEncoding, METH_VARARGS, "returns 'utf-8'"},
    {"flush", returnNone, METH_VARARGS, "does nothing"},
    {"isatty", returnTrue, METH_VARARGS, "returns True (the stream is interactive)"},
    {"name", consoleName, METH_VARARGS, "returns \"OpenSALTO console window\""},
    {"read", readFromConsole, METH_VARARGS, "read input from console"},
    {"readable", returnTrue, METH_VARARGS, "returns True (the stream is readable)"},
    {"readline", readLineFromConsole, METH_VARARGS, "read a line from the console"},
    {"readlines", readLinesFromConsole, METH_VARARGS, "read a list of lines from the console"},
    {"seekable", returnFalse, METH_VARARGS, "returns False (the stream is not seekable: seek(), tell() and truncate() will raise OSError)"},
    {"writable", returnTrue, METH_VARARGS, "returns True (the stream is writable)"},
    {"write", writeToConsole, METH_VARARGS, "write a string to console"},
    {"writelines", writeLinesToConsole, METH_VARARGS, "write an iterable of strings to console"},
    {0, 0, 0, 0} // sentinel
};

static PyModuleDef saltoGuiModuleDef = {
    PyModuleDef_HEAD_INIT,
    "salto.gui",
    "OpenSALTO GUI extension module",
    -1,
    saltoGuiMethods,
    NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC PyInit_saltoGui(void) {
    PyObject* module = PyModule_Create(&saltoGuiModuleDef);
    if (module) {
        PySys_SetObject("stdout", module);
        PySys_SetObject("stderr", module);
        PySys_SetObject("stdin", module);
    }
 
    return module;
}
