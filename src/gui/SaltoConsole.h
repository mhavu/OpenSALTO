//
//  SaltoConsole.h
//  OpenSALTO GUI
//
//  Copyright 2014 Marko Havu. Released under the terms of
//  GNU General Public License version 3 or later.
//

#import <Cocoa/Cocoa.h>

@interface SaltoConsole : NSObject

@property (readonly, getter = inputStrings) NSMutableArray *inputArray;
@property (readonly, getter = outputStrings) NSMutableArray *outputArray;
@property (readonly, getter = isExecuting) BOOL executing;

- (void)execute:(NSString *)string;
- (void)appendInput:(NSString *)string;
- (void)appendOutput:(NSString *)string;

@end