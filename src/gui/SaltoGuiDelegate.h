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
    SaltoAlignCalendarDate = 1 << 1,
    SaltoAlignTimeOfDay = 1 << 2,
    SaltoAlignArbitrary = 1 << 3
};

@interface SaltoGuiDelegate : NSObject

@property (nonatomic, readonly) SaltoConsoleController *consoleController;
@property (nonatomic, retain) IBOutlet NSScrollView *scrollView;
@property (nonatomic, readonly) NSMutableArray *channelArray;
@property (nonatomic, assign) dispatch_queue_t queue;
@property (readonly) NSOperationQueue *opQueue;
@property (nonatomic, assign) NSTimeInterval rangeStart;
@property (nonatomic, assign) NSTimeInterval rangeEnd;
@property (nonatomic, assign) NSTimeInterval visibleRangeStart;
@property (nonatomic, assign) NSTimeInterval visibleRangeEnd;
@property (nonatomic, assign) double pixelsPerSecond;
@property (nonatomic, assign) SaltoChannelAlignment alignment;

- (NSTimeInterval)range;
- (void)setRangeStart:(NSTimeInterval)start end:(NSTimeInterval)end;
- (NSTimeInterval)visibleRange;
- (void)setVisibleRangeStart:(NSTimeInterval)start end:(NSTimeInterval)end;

- (IBAction)showConsoleWindow:(id)sender;
- (IBAction)toggleAlignment:(id)sender;
- (void)addChannel:(SaltoChannelWrapper *)channel;
- (void)removeChannel:(SaltoChannelWrapper *)channel;
- (IBAction)zoomIn:(id)sender;
- (IBAction)zoomOut:(id)sender;
- (IBAction)showAll:(id)sender;
- (IBAction)interruptExecution:(id)sender;

@end
