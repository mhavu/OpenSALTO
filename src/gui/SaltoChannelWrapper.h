//
//  SaltoChannelWrapper.h
//  OpenSALTO GUI
//
//  Copyright 2014 Marko Havu. Released under the terms of
//  GNU General Public License version 3 or later.
//

#import <Cocoa/Cocoa.h>
#include "Channel.h"

@class SaltoChannelView;
@class SaltoYScaleView;

@interface SaltoChannelWrapper : NSObject

@property (readonly) Channel *channel;
@property (unsafe_unretained) SaltoChannelView *view;
@property (unsafe_unretained) SaltoYScaleView *scaleView;
@property (retain) NSMutableArray *eventViewArray;
@property (readonly) NSRect frame;
@property (readonly) NSTimeInterval visibleRange;
@property (readonly) double pixelsPerSecond;
@property (readonly) CGLayerRef layer;
@property (retain) NSOperation *operation;
@property (retain) NSString *label;
@property (readonly) NSString *signalType;
@property (readonly) NSString *samplerate;
@property (assign) NSTimeInterval alignment;
@property (readonly) double yVisibleRangeMin;
@property (readonly) double yVisibleRangeMax;
@property (assign, getter = isVisible) BOOL visible;

+ (instancetype)wrapperForChannel:(Channel *)ch;
- (instancetype)initWithChannel:(Channel *)ch;
- (void)drawLayerForContext:(CGContextRef)context frame:(NSRect)frame;
- (void)drawInContext:(CGContextRef)context;
- (void)drawScaleInContext:(CGContextRef)context;
- (void)updateEventViews;

@end
