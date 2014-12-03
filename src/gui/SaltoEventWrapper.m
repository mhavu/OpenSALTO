//
//  SaltoEventWrapper.m
//  OpenSALTO GUI
//
//  Copyright 2014 Marko Havu. Released under the terms of
//  GNU General Public License version 3 or later.
//

#import "SaltoEventWrapper.h"

static NSMutableSet *allEvents = nil;

@implementation SaltoEventWrapper

+ (void)initialize {
    if (!allEvents) {
        allEvents = [[NSMutableSet set] retain];
    }
}

+ (instancetype)eventWithType:(EventVariety)type
                      subtype:(NSString *)subtype
                        start:(NSTimeInterval)start
                          end:(NSTimeInterval)end
                  description:(NSString *)description
{
    struct timespec start_t = endTimeFromDuration(0, 0, start);
    struct timespec end_t = endTimeFromDuration(0, 0, end);
    Event *event = newEvent(type, [subtype UTF8String], start_t, end_t, [description UTF8String]);

    return [SaltoEventWrapper wrapperForEvent:event];
}

+ (instancetype)wrapperForEvent:(Event *)event {
    return [[[SaltoEventWrapper alloc] initWithEvent:event] autorelease];
}

- (instancetype)initWithType:(EventVariety)type
                     subtype:(NSString *)subtype
                       start:(NSTimeInterval)start
                         end:(NSTimeInterval)end
                 description:(NSString *)description
{
    struct timespec start_t = endTimeFromDuration(0, 0, start);
    struct timespec end_t = endTimeFromDuration(0, 0, end);
    Event *event = newEvent(type, [subtype UTF8String], start_t, end_t, [description UTF8String]);

    return [self initWithEvent:event];
}

// Designated initializer
- (instancetype)initWithEvent:(Event *)event {
    // If a wrapper already exists for this event, return it.
    BOOL existsAlready = NO;
    for (NSValue *value in allEvents) {
        SaltoEventWrapper *obj = value.nonretainedObjectValue;
        if (obj.event == event) {
            [self release];
            self = obj;
            existsAlready = YES;
            break;
        }
    }
    // Otherwise create a new one.
    if (!existsAlready) {
        self = [super init];
        if (self) {
            PyGILState_STATE state = PyGILState_Ensure();
            Py_INCREF(event);
            PyGILState_Release(state);
            _event = event;
            [allEvents addObject:[NSValue valueWithNonretainedObject:self]];
        }
    }
    
    return self;
}

- (void)dealloc {
    [allEvents removeObject:[NSValue valueWithNonretainedObject:self]];
    PyGILState_STATE state = PyGILState_Ensure();
    Py_XDECREF(_event);
    PyGILState_Release(state);
    [super dealloc];
}

#pragma mark - Getters and setters

- (EventVariety)type {
    return _event->type;
}

- (void)setType:(EventVariety)type {
    _event->type = type;
}

- (NSTimeInterval)startTime {
    return (_event->start_sec + _event->start_nsec / 1e9);
}

- (void)setStartTime:(NSTimeInterval)time {
    struct timespec t = endTimeFromDuration(0, 0, time);
    _event->start_sec = t.tv_sec;
    _event->start_nsec = t.tv_nsec;
}

- (NSTimeInterval)endTime {
    return (_event->end_sec + _event->end_nsec / 1e9);
}

- (void)setEndTime:(NSTimeInterval)time {
    struct timespec t = endTimeFromDuration(0, 0, time);
    _event->end_sec = t.tv_sec;
    _event->end_nsec = t.tv_nsec;
}

- (NSString *)subtype {
    return [NSString stringWithUTF8String:_event->subtype];
}

- (void)setSubtype:(NSString *)subtype {
    const char *buffer = [subtype UTF8String];
    size_t len = strlen(buffer) + 1;
    if (strcmp(_event->subtype, buffer) != 0) {
        free(_event->subtype);
        _event->subtype = malloc(len);
        strlcpy(_event->subtype, buffer, len);
    }
}

- (NSString *)descriptionText {
    return [NSString stringWithUTF8String:_event->description];
}

- (void)setDescriptionText:(NSString *)description {
    const char *buffer = [description UTF8String];
    size_t len = strlen(buffer) + 1;
    if (strcmp(_event->description, buffer) != 0) {
        free(_event->description);
        _event->description = malloc(len);
        strlcpy(_event->description, buffer, len);
    }
}

@end
