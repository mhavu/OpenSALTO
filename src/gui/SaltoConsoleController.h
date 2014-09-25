//
//  SaltoConsoleController.h
//  OpenSALTO GUI
//
//  Copyright 2014 Marko Havu. Released under the terms of
//  GNU General Public License version 3 or later.
//

#import <Cocoa/Cocoa.h>
#import "SaltoConsole.h"

@interface SaltoConsoleController : NSWindowController <NSTextViewDelegate>

@property (readonly) IBOutlet NSTextView *textView;
@property (retain) SaltoConsole *console;
@property (assign) NSUInteger insertionPoint;

- (void)insertInput:(NSString *)string;
- (void)insertOutput:(NSString *)string;

@end

@interface SaltoConsoleWindow : NSWindow

@end