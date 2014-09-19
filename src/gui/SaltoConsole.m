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
    SaltoGuiDelegate *appDelegate = [NSApp delegate];
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
