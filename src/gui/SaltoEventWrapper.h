//
//  SaltoEventWrapper.h
//  OpenSALTO GUI
//
//  Copyright 2014 Marko Havu. Released under the terms of
//  GNU General Public License version 3 or later.
//

#import <Cocoa/Cocoa.h>
#include "Event.h"

@class SaltoChannelWrapper;

@interface SaltoEventWrapper : NSObject

@property (readonly) Event *event;
@property (unsafe_unretained) SaltoChannelWrapper *channel;
@property (assign) CGColorRef color;

+ (instancetype)eventWithType:(EventVariety)type
                      subtype:(NSString *)subtype
                        start:(NSTimeInterval)start
                          end:(NSTimeInterval)end
                  description:(NSString *)description;
+ (instancetype)wrapperForEvent:(Event *)event;
- (instancetype)initWithType:(EventVariety)type
                     subtype:(NSString *)subtype
                       start:(NSTimeInterval)start
                         end:(NSTimeInterval)end
                 description:(NSString *)description;
- (instancetype)initWithEvent:(Event *)event;
- (EventVariety)type;
- (void)setType:(EventVariety)type;
- (NSTimeInterval)startTime;
- (void)setStartTime:(NSTimeInterval)time;
- (NSTimeInterval)endTime;
- (void)setEndTime:(NSTimeInterval)time;
- (NSString *)subtype;
- (void)setSubtype:(NSString *)subtype;
- (NSString *)descriptionText;
- (void)setDescriptionText:(NSString *)description;

@end
