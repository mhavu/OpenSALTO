//
//  SaltoConsoleController.m
//  OpenSalto GUI
//
//  Copyright 2014 Marko Havu. Released under the terms of
//  GNU General Public License version 3 or later.
//

#import "SaltoConsoleController.h"
#include "salto.h"

@implementation SaltoConsoleController

@synthesize console;
@synthesize textView;
@synthesize insertionPoint;

- (instancetype)init {
    self = [super initWithWindowNibName: @"SaltoConsole"];
    if (self) {
        console = [[SaltoConsole alloc] init];
        [[NSNotificationCenter defaultCenter] addObserver:self
                                                 selector:@selector(handleNewErrorOutput:)
                                                     name:NSFileHandleReadCompletionNotification
                                                   object:console.errorOutputHandle];
        [[NSNotificationCenter defaultCenter] addObserver:self
                                                 selector:@selector(handleNewOutput:)
                                                     name:NSFileHandleReadCompletionNotification
                                                   object:console.outputHandle];
        [console.errorOutputHandle readInBackgroundAndNotify];
        [console.outputHandle readInBackgroundAndNotify];
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

- (void)handleNewOutput:(id)notification {
    [console.outputHandle readInBackgroundAndNotify];
    NSString *string = [[NSString alloc] initWithData:[[notification userInfo] objectForKey:NSFileHandleNotificationDataItem] encoding:NSUTF8StringEncoding];
    [console addOutputString:string];
    [textView.textStorage appendAttributedString:[[[NSAttributedString alloc] initWithString:string] autorelease]];
    insertionPoint = [textView.string length];
    [textView scrollRangeToVisible:NSMakeRange(insertionPoint, 0)];
}

- (void)handleNewErrorOutput:(id)notification {
    [console.errorOutputHandle readInBackgroundAndNotify];
    NSString *string = [[NSString alloc] initWithData:[[notification userInfo] objectForKey:NSFileHandleNotificationDataItem] encoding:NSUTF8StringEncoding];
    [textView.textStorage appendAttributedString:[[[NSAttributedString alloc] initWithString:string attributes:[NSDictionary dictionaryWithObject:[NSColor redColor] forKey:NSForegroundColorAttributeName]] autorelease]];
    insertionPoint = [textView.string length];
    [textView scrollRangeToVisible:NSMakeRange(insertionPoint, 0)];
}

- (BOOL)textView:(NSTextView *)view shouldChangeTextInRange:(NSRange)range replacementString:(NSString *)string {
    BOOL shouldChange;
    
    shouldChange = (range.location < insertionPoint) ? NO : YES;
    [[console inputHandle] writeData:[string dataUsingEncoding:NSUTF8StringEncoding]];
    
    return shouldChange;
}

@end


@implementation SaltoConsole

@synthesize inputHandle;
@synthesize outputHandle;
@synthesize errorOutputHandle;
@synthesize inputArray;
@synthesize outputArray;
@synthesize editBuffer;
@synthesize running;

- (instancetype)init {
    self = [super init];
    if (self) {
        NSPipe *stdoutPipe = [NSPipe pipe];
        outputHandle = [stdoutPipe.fileHandleForReading retain];
        if (dup2([stdoutPipe.fileHandleForWriting fileDescriptor], fileno(stdout)) < 0)
            self = nil;
    }
    if (self) {
        NSPipe *stderrPipe = [NSPipe pipe];
        errorOutputHandle = [stderrPipe.fileHandleForReading retain];
        if (dup2([stderrPipe.fileHandleForWriting fileDescriptor], fileno(stderr)) < 0)
            self = nil;
    }
    if (self) {
        NSPipe *stdinPipe = [NSPipe pipe];
        inputHandle = [stdinPipe.fileHandleForWriting retain];
        if (dup2([stdinPipe.fileHandleForReading fileDescriptor], fileno(stdin)) < 0)
            self = nil;
    }
    if (self) {
        inputArray = [[NSMutableArray alloc] init];
        outputArray = [[NSMutableArray alloc] init];
    }

    return self;
}

- (void)dealloc {
    [inputHandle release];
    [outputHandle release];
    [inputArray release];
    [outputArray release];
    [editBuffer release];
    [super dealloc];
}

- (NSArray *)inputStrings {
    return [NSArray arrayWithArray:inputArray];
}

- (NSArray *)outputStrings {
    return [NSArray arrayWithArray:outputArray];
}

- (void)addInputString:(NSString *)string {
    [inputArray addObject:string];
}

- (void)addOutputString:(NSString *)string {
    [outputArray addObject:string];
}

- (void)run {
    if (!running) {
        saltoRun();
        running = YES;
    }
}

@end
