//
//  SaltoConsoleController.h
//  OpenSalto GUI
//
//  Copyright 2014 Marko Havu. Released under the terms of
//  GNU General Public License version 3 or later.
//

#import <Cocoa/Cocoa.h>
#include <Python.h>


@interface SaltoConsole : NSObject

@property (readonly, getter = inputStrings) NSMutableArray *inputArray;
@property (readonly, getter = outputStrings) NSMutableArray *outputArray;
@property (readonly, getter = isExecuting) BOOL executing;

- (void)execute:(NSString *)string;
- (void)appendOutput:(NSString *)string;

@end


@interface SaltoConsoleController : NSWindowController <NSTextViewDelegate>

@property (readonly) IBOutlet NSTextView *textView;
@property (retain) SaltoConsole *console;
@property (assign) NSUInteger insertionPoint;

- (void)insertOutput:(NSString *)string;

@end


PyMODINIT_FUNC PyInit_saltoGui(void);