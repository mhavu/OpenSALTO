//
//  SaltoConsoleController.h
//  OpenSalto GUI
//
//  Copyright 2014 Marko Havu. Released under the terms of
//  GNU General Public License version 3 or later.
//

#import <Cocoa/Cocoa.h>


@interface SaltoConsole : NSObject

@property (readonly) NSFileHandle *inputHandle;
@property (readonly) NSFileHandle *outputHandle;
@property (readonly) NSFileHandle *errorOutputHandle;
@property (readonly, getter = inputStrings) NSMutableArray *inputArray;
@property (readonly, getter = outputStrings) NSMutableArray *outputArray;
@property (retain) NSMutableString *editBuffer;
@property (readonly, getter = isRunning) BOOL running;

- (void)run;
- (void)addInputString:(NSString *)string;
- (void)addOutputString:(NSString *)string;

@end


@interface SaltoConsoleController : NSWindowController <NSTextViewDelegate>

@property (readonly) IBOutlet NSTextView *textView;
@property (retain) SaltoConsole *console;
@property (assign) NSUInteger insertionPoint;

- (void)handleNewOutput:(id)notification;

@end