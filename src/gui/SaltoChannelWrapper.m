//
//  SaltoChannelWrapper.m
//  OpenSALTO GUI
//
//  Copyright 2014 Marko Havu. Released under the terms of
//  GNU General Public License version 3 or later.
//

#import "SaltoChannelWrapper.h"
#import "SaltoChannelView.h"
#include "salto.h"

static int typenum;

@implementation SaltoChannelWrapper

@synthesize channel;
@synthesize view;
@synthesize label;
@synthesize signalType;
@synthesize samplerate;
@synthesize alignment;
@synthesize visibleRangeStart;
@synthesize visibleRangeEnd;
@synthesize visible;

+ (void)initialize {
	if (self == SaltoChannelWrapper.class) {
        switch (sizeof(CGFloat)) {
            case 8:
                typenum = NPY_FLOAT64;
                break;
            case 4:
                typenum = NPY_FLOAT32;
                break;
            default:
                NSLog(@"Fatal error: size of CGFloat (%li bytes) is not 4 or 8 bytes.\n", sizeof(CGFloat));
                [NSApp terminate:self];
        }
    }
}

+ (instancetype)wrapperForChannel:(Channel *)ch {
    return [[[SaltoChannelWrapper alloc] initWithChannel:ch] autorelease];
}

- (instancetype)initWithChannel:(Channel *)ch {
    self = [super init];
    if (self) {
        PyGILState_STATE state = PyGILState_Ensure();
        Py_INCREF(ch);
        PyGILState_Release(state);
        channel = ch;
        signalType = [[NSString stringWithUTF8String:channel->type] retain];
        samplerate = [[NSString stringWithFormat:@"%lf Hz", channel->samplerate] retain];
        // TODO: PySetObject *events;
    }

    return self;
}

- (void)dealloc {
    [view release];
    [label release];
    [signalType release];
    [samplerate release];
    [alignment release];
    [visibleRangeStart release];
    [visibleRangeEnd release];
    PyGILState_STATE state = PyGILState_Ensure();
	Py_XDECREF(channel);
	PyGILState_Release(state);
    [super dealloc];
}

- (void)drawInContext:(CGContextRef)context {
    // Use black stroke with width of 1.0, round caps, and bevel joints.
    CGContextSetRGBStrokeColor(context, 0.0, 0.0, 0.0, 1.0);
    CGContextSetLineWidth(context, 1.0);
    CGContextSetLineCap(context, kCGLineCapRound);
    CGContextSetLineJoin(context, kCGLineJoinBevel);

    // Draw the Y coordinate axis.
    CGContextBeginPath(context);
    CGContextMoveToPoint(context, 0.0, -100.0);
    CGContextAddLineToPoint(context, 0.0, 100.0);
    CGContextDrawPath(context, kCGPathStroke);

    // TODO: Determine start and end times and samplerate based on the shown area.
    double pixelRatio = 10.0; // new samplerate
    long long start_sec = channel->start_sec;
    long long end_sec = channel->start_sec + 600;
    long start_nsec = channel->start_nsec;
    long end_nsec = start_nsec;
    
    PyGILState_STATE state = PyGILState_Ensure();
    PyArrayObject *data = (channel->collection || channel->samplerate > 3.0 * pixelRatio) ?
        (PyArrayObject *)PyObject_CallMethod((PyObject *)self.channel, "resampledData",
                                             "dLlLlis", pixelRatio, start_sec, start_nsec,
                                             end_sec, end_nsec, typenum, "VRA") :
        channel->data;
    if (data) {
        size_t count = PyArray_DIM(data, 0);
        CGPoint *strokeSegments = calloc(2 * count - 2, sizeof(CGPoint));
        if (strokeSegments) {
            CGFloat *buffer = (CGFloat *)PyArray_DATA(data);
            strokeSegments[0] = CGPointMake(0, buffer[0]);
            for (NSUInteger i = 1; i < count - 1; i++) {
                strokeSegments[2 * i - 1] = CGPointMake(i, buffer[i]);
                strokeSegments[2 * i] = CGPointMake(i, buffer[i]);
            }
            strokeSegments[2 * count - 3] = CGPointMake(count - 1, buffer[count - 1]);
        } else {
            NSLog(@"calloc failed in %s\n", __func__);
        }
        // TODO: Move up so that negative values are shown.
        CGContextStrokeLineSegments(context, strokeSegments, count);
    }
    PyGILState_Release(state);
}

@end