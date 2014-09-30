//
//  SaltoConsoleController.m
//  OpenSalto GUI
//
//  Copyright 2014 Marko Havu. Released under the terms of
//  GNU General Public License version 3 or later.
//

#import "SaltoConsoleController.h"

@implementation SaltoConsoleController

- (instancetype)init {
    self = [super initWithWindowNibName:@"SaltoConsole"];
    if (self) {
        _console = [[SaltoConsole alloc] init];
    }

    return self;
}

- (void)dealloc {
    [_console release];
    [_textView release];
    [super dealloc];
}

- (void)windowDidLoad {
    [super windowDidLoad];
    for (NSString *string in [_console outputStrings]) {
        [_textView setString:[_textView.string stringByAppendingString:string]];
    }
    _insertionPoint = [_textView.string length];
    [_textView scrollRangeToVisible:NSMakeRange(_insertionPoint, 0)];
}

- (BOOL)textView:(NSTextView *)view shouldChangeTextInRange:(NSRange)range replacementString:(NSString *)string {
    BOOL shouldChange;
    
    // TODO: Use NSParagraphStyle
    if (range.location < _insertionPoint) {
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
        if (![_console isExecuting]) {
            if ([[NSApp currentEvent] modifierFlags] & NSShiftKeyMask) {
                // Add a newline at the end of the edit buffer.
                [view setSelectedRange:NSMakeRange(view.string.length, 0)];
                [view insertNewlineIgnoringFieldEditor:self];

                // Send the contents of the edit buffer to the Python interpreter.
                [_console execute:[view.string substringWithRange:NSMakeRange(_insertionPoint, view.string.length -_insertionPoint)]];

                _insertionPoint = _textView.string.length;
                
                result = YES;
            }
        } else {
            // TODO: Make readline return
        }
    }

    return result;
}

- (void)insertInput:(NSString *)string {
    NSRange range = NSMakeRange(_insertionPoint, 0);
    [_textView selectedRange];
    [_textView setSelectedRange:range];
    [_textView.textStorage replaceCharactersInRange:range withString:string];
    [_textView insertNewlineIgnoringFieldEditor:self];
    _insertionPoint += string.length + 1;
    [_console addInput:string];
    [_textView scrollRangeToVisible:NSMakeRange(_insertionPoint, 0)];
    [_textView setSelectedRange:NSMakeRange(_textView.string.length, 0)];
}

- (void)insertOutput:(NSString *)string {
    NSRange range = NSMakeRange(_insertionPoint, 0);
    [_textView.textStorage replaceCharactersInRange:range withString:string];
    _insertionPoint += string.length;
    [_console appendOutput:string];
    [_textView scrollRangeToVisible:NSMakeRange(_insertionPoint, 0)];
}

@end

@implementation SaltoConsoleWindow

- (BOOL)canBecomeMainWindow {
    return NO;
}

@end