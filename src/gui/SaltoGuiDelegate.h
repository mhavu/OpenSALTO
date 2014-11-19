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
    SaltoAlignStartTime = 1 << 0,
    SaltoAlignMarkers = SaltoAlignStartTime,
    SaltoAlignCalendarDate = 1 << 1,
    SaltoAlignTimeOfDay = 1 << 2,
};

@interface SaltoGuiDelegate : NSObject

@property (nonatomic, readonly) SaltoConsoleController *consoleController;
@property (nonatomic, retain) IBOutlet NSScrollView *scrollView;
@property (nonatomic, retain) IBOutlet NSMenu *alignmentMenu;
@property (nonatomic, readonly) NSMutableArray *channelArray;
@property (nonatomic, assign) dispatch_queue_t queue;
@property (nonatomic, readonly) NSTimeInterval rangeStart;
@property (nonatomic, readonly) NSTimeInterval rangeEnd;
@property (nonatomic, assign) NSTimeInterval visibleRange;
@property (nonatomic, assign) SaltoChannelAlignment alignment;
@property (nonatomic, assign, getter = isZoomedIn) BOOL zoomedIn;
@property (nonatomic, retain) IBOutlet NSScroller *scroller;
@property (nonatomic, unsafe_unretained) IBOutlet NSLayoutConstraint *scrollerHeightConstraint;

- (NSTimeInterval)range;
- (void)addChannel:(SaltoChannelWrapper *)channel;
- (void)removeChannel:(SaltoChannelWrapper *)channel;
- (IBAction)showConsoleWindow:(id)sender;
- (IBAction)setAlignmentMode:(id)sender;
- (IBAction)zoomIn:(id)sender;
- (IBAction)zoomOut:(id)sender;
- (IBAction)showAll:(id)sender;
- (IBAction)interruptExecution:(id)sender;
- (IBAction)scrollAction:(id)sender;

@end