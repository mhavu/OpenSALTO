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

@interface SaltoChannelWrapper : NSObject

@property (readonly) Channel *channel;
@property (unsafe_unretained) SaltoChannelView *view;
@property (assign) CGPoint *strokeSegments;
@property (assign) size_t strokeCount;
@property (retain) NSString *label;
@property (retain) NSString *signalType;
@property (retain) NSString *samplerate;
@property (assign) NSTimeInterval alignment;
@property (assign) double yVisibleRangeMin;
@property (assign) double yVisibleRangeMax;
@property (assign, getter = isVisible) BOOL visible;

+ (instancetype)wrapperForChannel:(Channel *)ch;
- (instancetype)initWithChannel:(Channel *)ch;
- (void)drawInContext:(CGContextRef)context;
- (void)drawScaleInContext:(CGContextRef)context;

@end
