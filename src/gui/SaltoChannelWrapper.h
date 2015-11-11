//
//  SaltoChannelWrapper.h
//  OpenSALTO GUI
//
//  Copyright 2014 Marko Havu. Released under the terms of
//  GNU General Public License version 3 or later.
//

#import <Cocoa/Cocoa.h>
#import <CorePlot/CorePlot.h>
#include "Channel.h"

@class SaltoChannelView;

typedef struct {
    NSUInteger location;
    NSUInteger length;
    NSUInteger expandedLocation;
    NSUInteger expandedLength;
    NSUInteger pointCount;
} SaltoChannelSegment;

@interface NSValue (SaltoChannelSegment)
+ (id)valueWithChannelSegment:(SaltoChannelSegment)pair;
- (SaltoChannelSegment)channelSegmentValue;
@end

@interface SaltoChannelWrapper : NSObject<CPTPlotDataSource, CPTPlotSpaceDelegate>

@property (retain) CPTXYGraph *graph;
@property (readonly) Channel *channel;
@property (unsafe_unretained) SaltoChannelView *view;
@property (readonly) NSRect frame;
@property (retain) NSString *label;
@property (readonly) NSString *signalType;
@property (readonly) double samplerate;
@property (assign) NSTimeInterval visibleRangeStart;
@property (readonly) NSTimeInterval startTime;
@property (readonly) NSTimeInterval endTime;
@property (readonly) NSTimeInterval duration;
@property (readonly) double yMin;
@property (readonly) double yMax;
@property (readonly) double yVisibleRangeMin;
@property (readonly) double yVisibleRangeMax;
@property (assign, getter = isVisible) BOOL visible;

+ (instancetype)wrapperForChannel:(Channel *)ch;
- (instancetype)initWithChannel:(Channel *)ch;
- (void)updateEventViews;
- (void)setupPlot;
- (NSTimeInterval)timeForPoint:(CGPoint)point;
- (CGFloat)xForTimeInterval:(NSTimeInterval)time;
- (CGFloat)xForTimespec:(struct timespec)t;
- (double)pixelsPerSecond;

@end
