//
//  SaltoChannelWrapper.m
//  OpenSALTO GUI
//
//  Copyright 2014 Marko Havu. Released under the terms of
//  GNU General Public License version 3 or later.
//

#import "SaltoChannelWrapper.h"
#import "SaltoChannelView.h"
#import "SaltoGuiDelegate.h"
#include "salto.h"

static int typenum;

@implementation SaltoChannelWrapper

@synthesize channel;
@synthesize view;
@synthesize strokeSegments;
@synthesize strokeCount;
@synthesize label;
@synthesize signalType;
@synthesize samplerate;
@synthesize alignment;
@synthesize yVisibleRangeMin;
@synthesize yVisibleRangeMax;
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
                NSLog(@"Fatal error: size of CGFloat (%li bytes) is not 4 or 8 bytes.", sizeof(CGFloat));
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
        BOOL isSigned = (!ch->collection) ?
                        PyArray_ISSIGNED((PyArrayObject *)ch->data) :
                        PyArray_ISSIGNED((PyArrayObject *)ch->fill_values);
        PyGILState_Release(state);
        channel = ch;
        signalType = [[NSString stringWithUTF8String:channel->type] retain];
        samplerate = [[NSString stringWithFormat:@"%lf Hz", channel->samplerate] retain];
        SaltoGuiDelegate *appDelegate = [NSApp delegate];
        NSTimeInterval duration = channelDuration(channel);
        if (duration > appDelegate.xRange) {
            appDelegate.xRange = duration;
            appDelegate.xVisibleRangeStart = 0.0;
            appDelegate.xVisibleRangeEnd = duration;
        }
        if (appDelegate.alignment == SaltoAlignTimeOfDay) {
            // TODO: alignment = ?
        }
        if (isSigned) {
            yVisibleRangeMax = (1 << channel->resolution) * channel->scale + channel->offset;
            yVisibleRangeMin = -(1 << channel->resolution) * channel->scale + channel->offset;
        } else {
            yVisibleRangeMax = (1 << channel->resolution) * channel->scale + channel->offset;
            yVisibleRangeMin = channel->offset;
        }
        // TODO: PySetObject *events;
    }

    return self;
}

- (void)dealloc {
    [label release];
    [signalType release];
    [samplerate release];
    free(strokeSegments);
    PyGILState_STATE state = PyGILState_Ensure();
	Py_XDECREF(channel);
	PyGILState_Release(state);
    [super dealloc];
}

- (void)drawInContext:(CGContextRef)context {
    // Determine start time, end time, and samplerate based on the shown area.
    double yScale = view.frame.size.height / (yVisibleRangeMax - yVisibleRangeMin);
    SaltoGuiDelegate *appDelegate = [NSApp delegate];
    NSTimeInterval visibleInterval = appDelegate.xVisibleRangeEnd - appDelegate.xVisibleRangeStart;
    double pixelsPerSecond = (view.frame.size.width - 5.0) / visibleInterval;
    struct timespec start_t = endTimeFromDuration(0, 0, appDelegate.xVisibleRangeStart);
    struct timespec end_t = endTimeFromDuration(channel->start_sec, channel->start_nsec, visibleInterval);
    NSTimeInterval xMin = MAX(-alignment, 0);
    NSTimeInterval xMax = MIN(channelDuration(channel) - alignment, visibleInterval);
    
    [self drawScaleInContext:context];
    
    CGContextSaveGState(context);
    CGContextTranslateCTM(context, 5.0, yScale * -yVisibleRangeMin);

    // Use blue stroke with width of 1.0, round caps, and miter joints.
    CGContextSetRGBStrokeColor(context, 0.0, 0.0, 0.7, 1.0);
    CGContextSetLineWidth(context, 1.0);
    CGContextSetLineCap(context, kCGLineCapRound);
    CGContextSetLineJoin(context, kCGLineJoinMiter);

    if (!strokeSegments) {
        PyGILState_STATE state = PyGILState_Ensure();
        size_t length = 0;
        channelData(channel, &length);
        PyArrayObject *data = (PyArrayObject *)PyObject_CallMethod((PyObject *)self.channel, "resampledData",
                                                                   "dLlLlis", pixelsPerSecond,
                                                                   start_t.tv_sec, start_t.tv_nsec,
                                                                   end_t.tv_sec, end_t.tv_nsec,
                                                                   typenum, "VRA");
        // TODO: If necessary, optimize.
        // One in every three drawing calls is unnecessary when using VRA.
        if (data) {
            strokeCount  = PyArray_DIM(data, 0);
            strokeSegments = calloc(2 * strokeCount - 2, sizeof(CGPoint));
            if (strokeSegments) {
                CGFloat *buffer = (CGFloat *)PyArray_DATA(data);
                // TODO: Calculate correct x coordinates based on
                // true start time, end time, and number of returned pixels.
                CGFloat x = 6 + xMin * pixelsPerSecond;
                CGFloat xStep = (xMax - xMin) * pixelsPerSecond / (strokeCount - 1);
                strokeSegments[0] = CGPointMake(x, buffer[0] * yScale);
                for (NSUInteger i = 1; i < strokeCount - 1; i++) {
                    x += xStep;
                    strokeSegments[2 * i - 1] = CGPointMake(x, buffer[i] * yScale);
                    strokeSegments[2 * i] = CGPointMake(x, buffer[i] * yScale);
                }
                strokeSegments[2 * strokeCount - 3] = CGPointMake(x + xStep, buffer[strokeCount - 1] * yScale);
            } else {
                NSLog(@"calloc failed in %s", __func__);
            }
        }
        PyGILState_Release(state);
    }
    CGContextStrokeLineSegments(context, strokeSegments, 2 * strokeCount - 2);

    CGContextRestoreGState(context);
}

- (void)drawScaleInContext:(CGContextRef)context {
    double yScale = view.frame.size.height / (yVisibleRangeMax - yVisibleRangeMin);
    CGContextSaveGState(context);
    CGContextTranslateCTM(context, 5.0, yScale * -yVisibleRangeMin);
    CGContextSetRGBStrokeColor(context, 0.0, 0.0, 0.1, 1.0);
    CGContextSetLineCap(context, kCGLineCapSquare);
    // Draw y coordinate axis.
    CGContextSetLineWidth(context, 1.0);
    CGContextBeginPath(context);
    CGContextMoveToPoint(context, 5.0, yVisibleRangeMin * yScale);
    CGContextAddLineToPoint(context, 5.0, yVisibleRangeMax * yScale);
    CGContextMoveToPoint(context, 3.0, yVisibleRangeMin * yScale);
    CGContextAddLineToPoint(context, 5.0, yVisibleRangeMin * yScale);
    CGContextMoveToPoint(context, 3.0, yVisibleRangeMax * yScale);
    CGContextAddLineToPoint(context, 5.0, yVisibleRangeMax * yScale);
    CGContextDrawPath(context, kCGPathStroke);
    // Draw x coordinate axis.
    CGContextSetLineWidth(context, 0.1);
    CGContextBeginPath(context);
    CGContextMoveToPoint(context, 3.0, 0.0);
    CGContextAddLineToPoint(context, view.frame.size.width, 0.0);
    CGContextDrawPath(context, kCGPathStroke);
    CGContextRestoreGState(context);
}

@end