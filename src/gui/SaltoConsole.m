//
//  SaltoConsole.m
//  OpenSalto GUI
//
//  Copyright 2014 Marko Havu. Released under the terms of
//  GNU General Public License version 3 or later.
//

#import "SaltoConsole.h"
#import "SaltoGuiDelegate.h"
#include "salto.h"

@implementation SaltoConsole

@synthesize inputArray = _inputArray;
@synthesize outputArray = _outputArray;

- (instancetype)init {
    self = [super init];
    if (self) {
        _inputArray = [[NSMutableArray alloc] init];
        _outputArray = [[NSMutableArray alloc] init];
        [_outputArray addObject:[NSMutableString string]];
    }

    return self;
}

- (void)dealloc {
    [_inputArray release];
    [_outputArray release];
    [super dealloc];
}

- (NSArray *)inputStrings {
    return [NSArray arrayWithArray:_inputArray];
}

- (NSArray *)outputStrings {
    return [NSArray arrayWithArray:_outputArray];
}

- (void)addInput:(NSString *)string {
    [_inputArray addObject:string];
}

- (void)appendOutput:(NSString *)string {
    NSMutableString *currentOutput = [_outputArray lastObject];
    [currentOutput appendString:string];
}

- (void)execute:(NSString *)statement {
    SaltoGuiDelegate *appDelegate = [NSApp delegate];
    dispatch_async(appDelegate.queue,
                   ^{
                       _executing = YES;
                       PyObject *output = saltoEval([statement UTF8String]);
                       _executing = NO;
                       if (output) {
                           [_outputArray addObject:[NSMutableString string]];
                           PyGILState_STATE state = PyGILState_Ensure();
                           NSString *string = [NSString stringWithUTF8String:PyUnicode_AsUTF8(output)];
                           Py_DECREF(output);
                           PyGILState_Release(state);
                           dispatch_async(dispatch_get_main_queue(),
                                          ^{ [appDelegate.consoleController insertOutput:string]; });
                       }                    
                   });
    [_inputArray addObject:statement];
    // TODO: handle output
}

@end
