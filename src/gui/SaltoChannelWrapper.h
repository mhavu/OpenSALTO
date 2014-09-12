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
@property (retain) SaltoChannelView *view;
@property (retain) NSString *label;
@property (retain) NSString *signalType;
@property (retain) NSString *samplerate;
@property (retain) NSDate *alignment;
@property (retain) NSDate *visibleRangeStart;
@property (retain) NSDate *visibleRangeEnd;
@property (assign, getter = isVisible) BOOL visible;

+ (instancetype)wrapperForChannel:(Channel *)ch;
- (instancetype)initWithChannel:(Channel *)ch;
- (void)drawInContext:(CGContextRef)context;

@end
