//
//  salto_gui.m
//  OpenSALTO GUI
//
//  Copyright 2014 Marko Havu. Released under the terms of
//  GNU General Public License version 3 or later.
//

#import <Cocoa/Cocoa.h>
#import "SaltoGuiDelegate.h"
#import "SaltoChannelWrapper.h"
#import "saltoGui.h"

static PyObject *readFromConsole(PyObject *pyself, PyObject *args) {
    Py_ssize_t size;

    if (PyArg_ParseTuple(args, "|n:read", &size)) {
        NSLog(@"read is not implemented yet");
        // TODO: implement
        // Read and return up to size bytes.
        // If the argument is omitted, None, or negative, data is read and returned until EOF is reached.
        // Maybe capture keypresses directly?
    } else {
        PyErr_SetString(PyExc_TypeError, "read() takes an optional integer argument");
    }

    return PyUnicode_FromString("");
}

static PyObject *readLineFromConsole(PyObject *pyself, PyObject *args) {
    Py_ssize_t size;

    if (PyArg_ParseTuple(args, "|n:readline", &size)) {
        NSLog(@"readline is not implemented yet");
        // TODO: implement
        // Read and return one line from the stream. If size is specified, at most size bytes will be read.
        // Return when newline is entered.
        // We should probably block in another thread.
    } else {
        PyErr_SetString(PyExc_TypeError, "readline() takes an optional integer argument");
    }

    return PyUnicode_FromString("");
}

static PyObject *readLinesFromConsole(PyObject *pyself, PyObject *args) {
    PyObject *list = NULL;
    Py_ssize_t hint;

    if (PyArg_ParseTuple(args, "|n:readlines", &hint)) {
        NSLog(@"readlines is not implemented yet");
        // TODO: implement
        // Read and return a list of lines from the stream.
        // A hint can be specified to control the number of lines read:
        // no more lines will be read if the total size (in bytes/characters) of all lines so far exceeds hint.
    } else {
        PyErr_SetString(PyExc_TypeError, "readlines() takes an optional integer argument");
    }

    return list;
}

static PyObject *writeToConsole(PyObject *pyself, PyObject *args) {
    const char *str;
    PyObject *result = NULL;

    if (PyArg_ParseTuple(args, "s:write", &str)) {
        NSString *string = [NSString stringWithUTF8String:str];
        SaltoConsoleController *consoleController = [[NSApp delegate] consoleController];
        // TODO: Use NSParagraphStyle
        dispatch_async(dispatch_get_main_queue(), ^{ [consoleController insertOutput:string]; });
        result = Py_BuildValue("i", strlen(str));  // new
    } else {
        PyErr_SetString(PyExc_TypeError, "write() takes a string argument");
    }

    return result;
}

static PyObject *writeLinesToConsole(PyObject *pyself, PyObject *args) {
    PyObject *iterable, *iterator, *o, *s, *result = NULL;
    NSString *string;
    NSRange range;

    if (PyArg_ParseTuple(args, "O:writelines", &iterable)) {
        SaltoConsoleController *consoleController = [[NSApp delegate] consoleController];
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
        Py_INCREF(Py_None);
        result = Py_None;
    } else {
        PyErr_SetString(PyExc_TypeError, "writelines() takes an iterable of strings as an argument");
    }

    return result;
}

static PyObject *consoleEncoding(PyObject *pyself, PyObject *args) {
    return PyUnicode_FromString("utf-8");
}

static PyObject *consoleName(PyObject *pyself, PyObject *args) {
    return PyUnicode_FromString("OpenSALTO console window");
}

static PyObject *returnNone(PyObject *pyself, PyObject *args) {
    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject *returnTrue(PyObject *pyself, PyObject *args) {
    Py_INCREF(Py_True);
    return Py_True;
}

static PyObject *returnFalse(PyObject *pyself, PyObject *args) {
    Py_INCREF(Py_False);
    return Py_False;
}

PyObject *saltoGuiAddChannel(PyObject *pyself, PyObject *args) {
    Channel *ch;
    PyObject *result = NULL;
    const char *name;
    
    if (PyArg_ParseTuple(args, "O!s:addChannel", &ChannelType, &ch, &name)) {
        SaltoChannelWrapper *channel = [SaltoChannelWrapper wrapperForChannel:ch];
        [channel setLabel:[NSString stringWithUTF8String:name]];
        [[NSApp delegate] performSelectorOnMainThread:@selector(addChannel:)
                                           withObject:channel waitUntilDone:NO];
        Py_INCREF(Py_None);
        result = Py_None;
    } else {
        PyErr_SetString(PyExc_TypeError, "addChannel() takes a Channel argument");
    }
    
    return result;
}

PyObject *saltoGuiRemoveChannel(PyObject *pyself, PyObject *args) {
    Channel *ch;
    PyObject *result = NULL;
    
    if (PyArg_ParseTuple(args, "O!:removeChannel", &ChannelType, &ch)) {
        SaltoChannelWrapper *channel = [SaltoChannelWrapper wrapperForChannel:ch];
        [[NSApp delegate] performSelectorOnMainThread:@selector(removeChannel:)
                                           withObject:channel waitUntilDone:NO];
        Py_INCREF(Py_None);
        result = Py_None;
    } else {
        PyErr_SetString(PyExc_TypeError, "removeChannel() takes a Channel argument");
    }

    return result;
}

PyObject *saltoGuiTerminate(PyObject *pyself) {
    [NSApp terminate:nil];
    Py_INCREF(Py_None);
    return Py_None;
}

static int interrupt(void *arg) {
    PyErr_SetInterrupt();
    return -1;
}

void saltoGuiInterrupt(void) {
    Py_AddPendingCall(&interrupt, NULL);
}


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
    {"addChannel", saltoGuiAddChannel, METH_VARARGS, "called from salto.ChannelTable.add()"},
    {"removeChannel", saltoGuiRemoveChannel, METH_VARARGS, "called from salto.ChannelTable.remove()"},
    {"quit", (PyCFunction)saltoGuiTerminate, METH_NOARGS, "close OpenSALTO GUI"},
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
    PyObject *module = PyModule_Create(&saltoGuiModuleDef);
    if (module) {
        PySys_SetObject("stdout", module);
        PySys_SetObject("stderr", module);
        PySys_SetObject("stdin", module);
    }
    
    return module;
}
