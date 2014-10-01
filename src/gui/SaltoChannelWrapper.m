//
//  SaltoChannelWrapper.m
//  OpenSALTO GUI
//
//  Copyright 2014 Marko Havu. Released under the terms of
//  GNU General Public License version 3 or later.
//

#import "SaltoChannelWrapper.h"
#import "SaltoChannelView.h"
#import "SaltoEventView.h"
#import "SaltoGuiDelegate.h"
#include "salto.h"

static int typenum;

@implementation SaltoChannelWrapper

@synthesize operation = _operation;

- (NSOperation *)operation {
    @synchronized(self) {
        return _operation;
    }
}

- (void)setOperation:(NSOperation *)operation {
    @synchronized(self) {
        if (operation != _operation) {
            [_operation cancel];
            [[[NSApp delegate] opQueue] addOperation:operation];
            [operation retain];
            [_operation release];
            _operation = operation;
        }
    }
}


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
        // TODO: PySetObject *events;
        PyGILState_Release(state);
        _channel = ch;
        _signalType = [[NSString stringWithUTF8String:_channel->type] retain];
        _samplerate = [[NSString stringWithFormat:@"%lf Hz", _channel->samplerate] retain];
        SaltoGuiDelegate *appDelegate = [NSApp delegate];
        NSTimeInterval duration = channelDuration(_channel);
        if (duration > appDelegate.range) {
            appDelegate.rangeStart = 0.0;
            appDelegate.rangeEnd = duration;
            [appDelegate setVisibleRangeStart:0.0 end:duration];
        }
        if (appDelegate.alignment == SaltoAlignTimeOfDay) {
            // TODO: alignment = ?
        }
        if (isSigned) {
            _yVisibleRangeMax = (1 << _channel->resolution) * _channel->scale + _channel->offset;
            _yVisibleRangeMin = -(1 << _channel->resolution) * _channel->scale + _channel->offset;
        } else {
            _yVisibleRangeMax = (1 << _channel->resolution) * _channel->scale + _channel->offset;
            _yVisibleRangeMin = _channel->offset;
        }
    }

    return self;
}

- (void)dealloc {
    [_eventViewArray release];
    [_label release];
    [_signalType release];
    [_samplerate release];
    [_operation release];
    CGLayerRelease(_layer);
    PyGILState_STATE state = PyGILState_Ensure();
	Py_XDECREF(_channel);
	PyGILState_Release(state);
    [super dealloc];
}

- (void)drawLayerForContext:(CGContextRef)context frame:(NSRect)frame {
    SaltoGuiDelegate *appDelegate = [NSApp delegate];
    if (!NSEqualSizes(_frame.size, frame.size) ||
        _visibleRange != appDelegate.visibleRange ||
        _pixelsPerSecond != appDelegate.pixelsPerSecond)
    {
        _frame = frame;
        _visibleRange = appDelegate.visibleRange;
        _pixelsPerSecond = appDelegate.pixelsPerSecond;
        CGLayerRef layer = CGLayerCreateWithContext(context, _frame.size, NULL);
        CGContextRef layerContext = CGLayerGetContext(layer);

        // Determine start time, end time, and samplerate based on the shown area.
        struct timespec start_t = endTimeFromDuration(0, 0, appDelegate.visibleRangeStart);
        struct timespec end_t = endTimeFromDuration(_channel->start_sec, _channel->start_nsec, _visibleRange);
        NSTimeInterval xMin = MAX(-_alignment, 0);
        NSTimeInterval xMax = MIN(channelDuration(_channel) - _alignment, _visibleRange);
        double yScale = NSHeight(_frame) / (_yVisibleRangeMax - _yVisibleRangeMin);
        CGContextTranslateCTM(layerContext, 0.0, yScale * -_yVisibleRangeMin);

        PyGILState_STATE state = PyGILState_Ensure();
        PyArrayObject *data = (PyArrayObject *)PyObject_CallMethod((PyObject *)_channel, "resampledData",
                                                                   "dLlLlis", appDelegate.pixelsPerSecond,
                                                                   start_t.tv_sec, start_t.tv_nsec,
                                                                   end_t.tv_sec, end_t.tv_nsec,
                                                                   typenum, "VRA");
        // TODO: If necessary, optimize.
        // One in every three drawing calls is unnecessary when using VRA.
        if (data) {
            size_t count = PyArray_DIM(data, 0);
            CGPoint *segments = calloc(2 * count - 2, sizeof(CGPoint));
            if (segments) {
                CGFloat *buffer = (CGFloat *)PyArray_DATA(data);
                // TODO: Calculate correct x coordinates based on
                // true start time, end time, and number of returned pixels.
                CGFloat x = xMin * _pixelsPerSecond;
                CGFloat xStep = (xMax - xMin) * _pixelsPerSecond / (count - 1);
                segments[0] = CGPointMake(x, yScale * buffer[0]);
                for (NSUInteger i = 1; i < count - 1; i++) {
                    x += xStep;
                    segments[2 * i - 1] = CGPointMake(x, yScale * buffer[i]);
                    segments[2 * i] = CGPointMake(x, yScale * buffer[i]);
                }
                segments[2 * count - 3] = CGPointMake(x + xStep, yScale * buffer[count - 1]);

                // Use blue stroke with width of 1.0, round caps, and miter joints.
                CGContextSetRGBStrokeColor(layerContext, 0.0, 0.0, 0.7, 1.0);
                CGContextSetLineWidth(layerContext, 1.0);
                CGContextSetLineCap(layerContext, kCGLineCapRound);
                CGContextSetLineJoin(layerContext, kCGLineJoinMiter);
                CGContextStrokeLineSegments(layerContext, segments, 2 * count - 2);
                @synchronized(self) {
                    CGLayerRelease(_layer);
                    _layer = layer;
                }
                free(segments);
            } else {
                NSLog(@"calloc failed in %s", __func__);
            }
        }
        PyGILState_Release(state);
    }

    [_view setNeedsDisplay:YES];
}

- (void)drawInContext:(CGContextRef)context {
    @synchronized(self) {
        if (_layer)
            CGContextDrawLayerAtPoint(context, CGPointMake(0.0, 0.0), _layer);
    }
}

- (void)drawScaleInContext:(CGContextRef)context {
    double yScale = NSHeight(_scaleView.frame) / (_yVisibleRangeMax - _yVisibleRangeMin);
    CGContextSaveGState(context);
    CGContextTranslateCTM(context, 11.0, yScale * -_yVisibleRangeMin);
    CGContextSetRGBStrokeColor(context, 0.0, 0.0, 0.0, 1.0);
    CGContextSetLineCap(context, kCGLineCapSquare);
    CGContextSetLineWidth(context, 1.0);
    CGContextBeginPath(context);
    CGContextMoveToPoint(context, 0.0, yScale * _yVisibleRangeMin);
    CGContextAddLineToPoint(context, 0.0, yScale * _yVisibleRangeMax);
    CGContextMoveToPoint(context, -5.0, 0.0);
    CGContextAddLineToPoint(context, 0.0, 0.0);
    CGContextMoveToPoint(context, -5.0, yScale * _yVisibleRangeMin);
    CGContextAddLineToPoint(context, 0.0, yScale * _yVisibleRangeMin);
    CGContextMoveToPoint(context, -5.0, yScale * _yVisibleRangeMax);
    CGContextAddLineToPoint(context, 0.0, yScale * _yVisibleRangeMax);
    CGContextSetAllowsAntialiasing(context, NO);
    CGContextDrawPath(context, kCGPathStroke);
    CGContextSetAllowsAntialiasing(context, YES);
    CGContextRestoreGState(context);
}

- (void)updateEventViews {
    SaltoGuiDelegate *appDelegate = [NSApp delegate];
    NSTimeInterval visibleInterval = appDelegate.visibleRangeEnd - appDelegate.visibleRangeStart;
    struct timespec start_t = endTimeFromDuration(0, 0, appDelegate.visibleRangeStart);
    struct timespec end_t = endTimeFromDuration(_channel->start_sec, _channel->start_nsec, visibleInterval);
    PyGILState_STATE state = PyGILState_Ensure();
    PyObject *eventSet = PyObject_CallMethod((PyObject *)_channel, "getEvents", "LlLl",
                                             start_t.tv_sec, start_t.tv_nsec,
                                             end_t.tv_sec, end_t.tv_nsec);  // new
    if (eventSet) {
        PyObject *iterator = PyObject_GetIter(eventSet);  // new
        if (iterator) {
            Event *e = (Event *)PyIter_Next(iterator);
            while (e) {
                // TODO: Create a SaltoEventView for each event.
                NSRect eventRect = NSMakeRect(0, 0, 100, _view.frame.size.height);
                SaltoEventView *eventView = [[SaltoEventView alloc] initWithFrame:eventRect];
                [_view addSubview:(NSView *)eventView];
                [eventView release];
                Py_DECREF(e);
                e = (Event *)PyIter_Next(iterator);
            }
            Py_DECREF(iterator);
        }
        Py_DECREF(eventSet);
    }
    PyGILState_Release(state);
}

@end