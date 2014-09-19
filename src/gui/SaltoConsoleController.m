//
//  SaltoConsoleController.m
//  OpenSalto GUI
//
//  Copyright 2014 Marko Havu. Released under the terms of
//  GNU General Public License version 3 or later.
//

#import "SaltoConsoleController.h"

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
            if ([[NSApp currentEvent] modifierFlags] & NSShiftKeyMask) {
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