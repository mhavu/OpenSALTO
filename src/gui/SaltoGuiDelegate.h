//
//  SaltoGuiDelegate.h
//  OpenSALTO GUI
//
//  Copyright 2014 Marko Havu. Released under the terms of
//  GNU General Public License version 3 or later.
//

#import <Cocoa/Cocoa.h>
#import "SaltoConsoleController.h"
#import "SaltoChannelWrapper.h"
#include "Channel.h"

@interface SaltoGuiDelegate : NSObject

@property (readonly) SaltoConsoleController *consoleController;
@property (readonly) NSMutableArray *channelArray;
@property (nonatomic, assign) dispatch_queue_t queue;

- (IBAction)showConsoleWindow:(id)sender;
- (IBAction)toggleAlignment:(id)sender;
- (void)addChannel:(SaltoChannelWrapper *)channel;
- (void)removeChannel:(SaltoChannelWrapper *)channel;

@end
