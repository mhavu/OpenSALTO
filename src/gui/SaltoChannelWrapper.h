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

@interface SaltoChannelWrapper : NSObject<CPTPlotDataSource, CPTPlotSpaceDelegate>

@property (retain) CPTXYGraph *graph;
@property (readonly) Channel *channel;
@property (unsafe_unretained) SaltoChannelView *view;
@property (retain) NSMutableArray *eventViewArray;
@property (readonly) NSRect frame;
@property (readonly) NSTimeInterval visibleRange;
@property (readonly) double pixelsPerSecond;
@property (retain) NSString *label;
@property (readonly) NSString *signalType;
@property (readonly) NSString *samplerate;
@property (assign) NSTimeInterval alignment;
@property (readonly) double yVisibleRangeMin;
@property (readonly) double yVisibleRangeMax;
@property (assign, getter = isVisible) BOOL visible;

+ (instancetype)wrapperForChannel:(Channel *)ch;
- (instancetype)initWithChannel:(Channel *)ch;
- (void)updateEventViews;
- (void)setupPlot;

@end
