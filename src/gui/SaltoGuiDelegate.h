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

#ifndef NS_OPTIONS
#define NS_OPTIONS(_type, _name) _type _name; enum _name
#endif

typedef NS_OPTIONS(NSInteger, SaltoChannelAlignment) {
    SaltoAlignStartTimes = 1 << 0,
    SaltoAlignTimeOfDay = 1 << 1,
    SaltoAlignArbitrary = 1 << 2
};

@interface SaltoGuiDelegate : NSObject

@property (nonatomic, readonly) SaltoConsoleController *consoleController;
@property (nonatomic, readonly) NSMutableArray *channelArray;
@property (nonatomic, assign) dispatch_queue_t queue;
@property (nonatomic, assign) NSTimeInterval xRange;
@property (nonatomic, assign) NSTimeInterval xVisibleRangeStart;
@property (nonatomic, assign) NSTimeInterval xVisibleRangeEnd;
@property (nonatomic, assign) SaltoChannelAlignment alignment;

- (IBAction)showConsoleWindow:(id)sender;
- (IBAction)toggleAlignment:(id)sender;
- (void)addChannel:(SaltoChannelWrapper *)channel;
- (void)removeChannel:(SaltoChannelWrapper *)channel;

@end
