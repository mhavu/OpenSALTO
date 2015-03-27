//
//  SaltoChannelWrapper.m
//  OpenSALTO GUI
//
//  Copyright 2014 Marko Havu. Released under the terms of
//  GNU General Public License version 3 or later.
//

#define PY_ARRAY_UNIQUE_SYMBOL Py_Array_API_OpenSALTO
#define NO_IMPORT_ARRAY
#import "SaltoChannelWrapper.h"
#import "SaltoChannelView.h"
#import "SaltoGuiDelegate.h"
#import "SaltoEventWrapper.h"
#include "salto.h"

static int typenum;

@implementation NSValue (SaltoChannelSegment)

+ (id)valueWithChannelSegment:(SaltoChannelSegment)channelSegment {
    return [NSValue value:&channelSegment withObjCType:@encode(SaltoChannelSegment)];
}

- (SaltoChannelSegment)channelSegmentValue {
    SaltoChannelSegment channelSegment;
    [self getValue:&channelSegment];
    return channelSegment;
}

@end


@interface SaltoChannelWrapper ()

@property (readonly) NSMutableArray *segments;

@end


@implementation SaltoChannelWrapper

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
    if (self && ch) {
        npy_intp fillCount = 0;
        npy_intp *fill_positions = NULL;
        npy_intp *fill_lengths = NULL;
        double *fill_values = NULL;
 
        // Get Python object data.
        PyGILState_STATE state = PyGILState_Ensure();
        Py_INCREF(ch);
        BOOL isSigned = PyArray_ISSIGNED((PyArrayObject *)ch->data);
        npy_intp length = PyArray_DIM(ch->data, 0);
        if (ch->fill_values) {
            fillCount = PyArray_DIM(ch->fill_values, 0);
            fill_positions = PyArray_DATA(ch->fill_positions);
            fill_lengths = PyArray_DATA(ch->fill_lengths);
            fill_values = PyArray_DATA(ch->fill_values);
        }
        PyGILState_Release(state);
 
        // Form SaltoChannelSegment array.
        NSUInteger segmentCount = 2 * fillCount + 1;
        _segments = [[NSMutableArray arrayWithCapacity:segmentCount] retain];
        NSUInteger sampleIndex = 0;
        NSUInteger expandedSampleIndex = 0;
        for (npy_intp i = 0; i < fillCount; i++) {
            SaltoChannelSegment dataSegment = {
                .location = sampleIndex,
                .length = fill_positions[i] - sampleIndex,
                .expandedLocation = expandedSampleIndex,
                .expandedLength = fill_positions[i] - sampleIndex,
                .value = 0.0};
            sampleIndex = fill_positions[i];
            expandedSampleIndex += dataSegment.length;
            SaltoChannelSegment fillSegment = {
                .location = sampleIndex,
                .length = 0,
                .expandedLocation = expandedSampleIndex,
                .expandedLength = fill_lengths[i],
                .value = fill_values[i] * ch->scale + ch->offset};
            expandedSampleIndex += fill_lengths[i];
            [_segments insertObject:[NSValue valueWithChannelSegment:dataSegment] atIndex:2 * i];
            [_segments insertObject:[NSValue valueWithChannelSegment:fillSegment] atIndex:2 * i + 1];
        }
        SaltoChannelSegment dataSegment = {
            .location = sampleIndex,
            .length = length - sampleIndex,
            .expandedLocation = expandedSampleIndex,
            .expandedLength = length - sampleIndex,
            .value = 0.0};
        [_segments insertObject:[NSValue valueWithChannelSegment:dataSegment] atIndex:segmentCount - 1];

        // Initialize other object properties.
        _channel = ch;
        _signalType = [[NSString stringWithUTF8String:_channel->type] retain];
        _samplerate = _channel->samplerate;
        _startTime = _channel->start_sec + _channel->start_nsec / 1e9;
        _duration = channelDuration(_channel);
        _endTime = _startTime + _duration;
        SaltoGuiDelegate *appDelegate = [NSApp delegate];
        if (appDelegate.alignment == SaltoAlignTimeOfDay) {
            NSCalendar *calendar = [NSCalendar currentCalendar];
            NSCalendarUnit fullDays = (NSDayCalendarUnit | NSMonthCalendarUnit | NSYearCalendarUnit);
            
            NSDate *date = [NSDate dateWithTimeIntervalSince1970:_endTime];
            NSDateComponents *dateComponents = [calendar components:fullDays fromDate:date];
            dateComponents.day++;
            NSDate *midnightAfter = [calendar dateFromComponents:dateComponents];
            NSTimeInterval toMidnight = [midnightAfter timeIntervalSinceDate:date];

            date = [NSDate dateWithTimeIntervalSince1970:_startTime];
            dateComponents = [calendar components:fullDays fromDate:date];
            NSDate *midnightBefore = [calendar dateFromComponents:dateComponents];
            NSTimeInterval sinceMidnight = [date timeIntervalSinceDate:midnightBefore];
            
            NSInteger days = [[calendar components:NSDayCalendarUnit
                                          fromDate:midnightBefore
                                            toDate:midnightAfter
                                           options:0] day];
            if (days <= 2) {
                if (sinceMidnight < toMidnight) {
                    _visibleRangeStart = [midnightBefore timeIntervalSince1970];
                } else {
                    dateComponents.day++;
                    midnightAfter = [calendar dateFromComponents:dateComponents];
                    _visibleRangeStart = [midnightAfter timeIntervalSince1970];
                }
            } else {
                // TODO: Notify of channels longer than one day.
                dateComponents.day++;
                midnightAfter = [calendar dateFromComponents:dateComponents];
                _visibleRangeStart = [midnightAfter timeIntervalSince1970];
            }
        } else if (appDelegate.alignment == SaltoAlignCalendarDate) {
            _visibleRangeStart = ([appDelegate.channelArray count]) ? appDelegate.rangeStart : _startTime;
        } else {
            _visibleRangeStart = _startTime;
        }
        if (isSigned) {
            _yVisibleRangeMax = (1 << _channel->resolution) * _channel->scale + _channel->offset;
            _yVisibleRangeMin = -(1 << _channel->resolution) * _channel->scale + _channel->offset;
        } else {
            _yVisibleRangeMax = (1 << _channel->resolution) * _channel->scale + _channel->offset;
            _yVisibleRangeMin = _channel->offset;
        }
    } else {
        self = nil;
    }

    return self;
}

- (void)dealloc {
    [_graph release];
    [_label release];
    [_signalType release];
    [_segments release];
    PyGILState_STATE state = PyGILState_Ensure();
	Py_XDECREF(_channel);
	PyGILState_Release(state);
    [super dealloc];
}

#pragma mark - Drawing the plot

- (void)updateEventViews {
    SaltoGuiDelegate *appDelegate = [NSApp delegate];
    struct timespec start_t = endTimeFromDuration(0, 0, self.visibleRangeStart);
    struct timespec end_t = endTimeFromDuration(0, 0, self.visibleRangeStart + appDelegate.visibleRange);
    // Remove existing layers and tracking areas.
    [self.view clearEventLayers];
    [self.view clearTrackingAreas];

    PyGILState_STATE state = PyGILState_Ensure();
    PyObject *eventSet = PyObject_CallMethod((PyObject *)_channel, "getEvents", "LlLl",
                                             start_t.tv_sec, start_t.tv_nsec,
                                             end_t.tv_sec, end_t.tv_nsec);  // new
    if (eventSet) {
        PyObject *iterator = PyObject_GetIter(eventSet);  // new
        if (iterator) {
            Event *e = (Event *)PyIter_Next(iterator);
            while (e) {
                SaltoEventWrapper *event = [SaltoEventWrapper wrapperForEvent:e];
                // TODO: Assign event colours.
                event.color = CGColorCreateGenericRGB(0.0, 0.0, 0.0, 0.3);
                NSTimeInterval eventStart = event.startTime;
                NSTimeInterval eventEnd = event.endTime;
                if (appDelegate.alignment != SaltoAlignCalendarDate) {
                    eventStart -= self.visibleRangeStart;
                    eventEnd -= self.visibleRangeStart;
                }
                CGFloat xMin = [self xForTimeInterval:event.startTime];
                CGFloat xMax = [self xForTimeInterval:event.endTime];
                CGRect frame = CGRectMake(xMin, 0, xMax - xMin, NSHeight(self.view.frame));
                [self.view setFrame:frame forEvent:event];
                Py_DECREF(e);
                e = (Event *)PyIter_Next(iterator);
            }
            Py_DECREF(iterator);
        }
        Py_DECREF(eventSet);
    }
    PyGILState_Release(state);
}

- (NSUInteger)numberOfRecordsForPlot:(CPTPlot *)plot {
    const double maxPointsPerPixel = 3.0;
    SaltoGuiDelegate *appDelegate = [NSApp delegate];
    NSUInteger start = round(MAX(self.visibleRangeStart - self.startTime, 0.0) * self.samplerate);
    NSUInteger end = round(MIN(start + appDelegate.visibleRange, self.duration) * self.samplerate);
    double pixelsPerSecond = [self pixelsPerSecond];
    __block NSUInteger pointCount = 0;
    NSMutableArray *segments = [NSMutableArray arrayWithArray:self.segments];
    [segments enumerateObjectsUsingBlock:^(NSValue *obj, NSUInteger idx, BOOL *stop) {
        SaltoChannelSegment segment = [obj channelSegmentValue];
        if (segment.expandedLocation + segment.expandedLength >= start) {
            if (segment.length > 0) {
                // Data segment
                NSUInteger validPointCount;
                if (segment.expandedLocation < start) {
                    // The (first) segment starts from the middle.
                    validPointCount = segment.expandedLocation + segment.expandedLength - start;
                } else if (segment.location + segment.length > end) {
                    // The (last) segment ends in the middle.
                    validPointCount = end - segment.expandedLocation + 1;
                } else {
                    // The entire segment is visible.
                    validPointCount = segment.length;
                }
                NSUInteger pixelCount = ceil(validPointCount / self.samplerate * pixelsPerSecond);
                if (validPointCount / pixelCount <= maxPointsPerPixel) {
                    // Show all data points.
                    segment.pointCount = validPointCount;
                } else {
                    // Aggregate data points to two points (min and max) per pixel.
                    segment.pointCount = 2 * pixelCount;
                }
            } else {
                // Fill segment
                segment.pointCount = 2;
            }
            [self.segments replaceObjectAtIndex:idx withObject:[NSValue valueWithChannelSegment:segment]];
            pointCount += segment.pointCount;
        } else if (segment.expandedLocation > end) {
            *stop = YES;
        }
    }];
    
    return pointCount;
}

- (double *)doublesForPlot:(CPTPlot *)plot field:(NSUInteger)fieldEnum recordIndexRange:(NSRange)range {
    BOOL errorOccurred = NO;
    double *values = calloc(range.length, sizeof(double));
    if (!values)
        errorOccurred = YES;
    
    SaltoGuiDelegate *appDelegate = [NSApp delegate];
    NSTimeInterval startTime = MAX(self.visibleRangeStart - self.startTime, 0.0);
    NSTimeInterval endTime = MIN(startTime + appDelegate.visibleRange, self.duration);
    NSUInteger start = round(startTime * self.samplerate);
    NSUInteger end = round(endTime * self.samplerate);
    
    PyArrayObject *dataArray = NULL;
    double *data = NULL;
    NSUInteger dataOffset = 0;
    if (!errorOccurred && fieldEnum == CPTScatterPlotFieldY) {
        // Convert the relevant slice of data to double.
        PyObject *startPyObj = NULL;
        PyObject *endPyObj = NULL;
        PyGILState_STATE state = PyGILState_Ensure();
        npy_intp dataLength = PyArray_DIM(self.channel->data, 0);
        for (NSValue *value in self.segments) {
            SaltoChannelSegment segment = [value channelSegmentValue];
            if (!startPyObj && segment.expandedLocation + segment.expandedLength >= start) {
                dataOffset = segment.location + start - segment.expandedLocation;
                startPyObj = PyLong_FromUnsignedLong(dataOffset);  // new
            }
            if (segment.expandedLocation + segment.expandedLength >= end) {
                if (segment.length > 0) {
                    NSUInteger unusedSampleCount = segment.expandedLocation + segment.expandedLength - end - 1;
                    // The endPyObj needs to point to end + 1 due to the way slicing works.
                    endPyObj = PyLong_FromUnsignedLong(segment.location + segment.length - unusedSampleCount);  // new
                } else {
                    // The endPyObj needs to point to end + 1 due to the way slicing works.
                    endPyObj = PyLong_FromUnsignedLong(segment.location);  // new
                }
                break;
            } else if (segment.location + segment.length == dataLength) {
                // End is out of bounds. Fix it.
                end = segment.expandedLength;
                // The endPyObj needs to point to end + 1 due to the way slicing works.
                endPyObj = PyLong_FromUnsignedLong(dataLength);  // new
            }
        }
        if (startPyObj && endPyObj) {
            PyObject *slice = PySlice_New(startPyObj, endPyObj, NULL);  // new
            PyObject *part = NULL;
            if (slice) {
                part = PyObject_GetItem((PyObject *)self.channel->data, slice);  //new
                Py_DECREF(slice);
            }
            if (part) {
                dataArray = (PyArrayObject *)PyArray_Cast((PyArrayObject *)part, NPY_DOUBLE);  // new
                Py_DECREF(part);
            }
            if (dataArray) {
                data = PyArray_DATA(dataArray);
            }
            if (!data) {
                errorOccurred = YES;
                NSLog(@"Failed to get Python objects in %@", NSStringFromSelector(_cmd));
            }
        }
        Py_XDECREF(startPyObj);
        Py_XDECREF(endPyObj);
        PyGILState_Release(state);
    }
    
    if (!errorOccurred) {
        // Compute the data points.
        const double maxPointsPerPixel = 3.0;
        double pixelsPerSecond = [self pixelsPerSecond];
        NSUInteger pointIndex = 0;
        for (NSValue *value in self.segments) {
            SaltoChannelSegment segment = [value channelSegmentValue];
            if (segment.pointCount == 0) {
                continue;
            } else if (pointIndex + segment.pointCount < range.location) {
                pointIndex += segment.pointCount;
                continue;
            } else if (pointIndex >= range.location + range.length) {
                break;
            } else if (segment.expandedLocation > end) {
                break;
            }
            if (segment.length > 0) {
                // Data segment
                NSUInteger sampleCount;
                if (segment.expandedLocation < start) {
                    // The (first) segment starts from the middle.
                    sampleCount = segment.expandedLocation + segment.expandedLength - start;
                } else if (segment.location + segment.length > end) {
                    // The (last) segment ends in the middle.
                    sampleCount = end - segment.expandedLocation + 1;
                } else {
                    // The entire segment is visible.
                    sampleCount = segment.length;
                }
                NSUInteger pixelCount = ceil(sampleCount / self.samplerate * pixelsPerSecond);
                NSUInteger pointCount = MIN(range.location + range.length - pointIndex,
                                            segment.pointCount);
                NSUInteger skipCount = 0;
                if (pointIndex < range.location) {
                    skipCount = range.location - pointIndex;
                    pointIndex = range.location;
                }
               if (fieldEnum == CPTScatterPlotFieldY) {
                    // Y axis (data values)
                    if (sampleCount / pixelCount <= maxPointsPerPixel) {
                        // Show all data points.
                        for (NSUInteger i = skipCount; i < pointCount + skipCount; i++) {
                            values[pointIndex - range.location] = data[segment.location + i - dataOffset] * self.channel->scale + self.channel->offset;
                            pointIndex++;
                        }
                    } else {
                        // Aggregate data points to two points (min and max) per pixel.
                        NSUInteger i = 0;
                        for (NSUInteger p = skipCount; p < pointCount + skipCount; p++) {
                            double max = data[segment.location + i - dataOffset];
                            double min = max;
                            while (i < p / 2 * sampleCount / pixelCount && i < sampleCount) {
                                if (data[segment.location + i - dataOffset] > max) {
                                    max = data[segment.location + i - dataOffset];
                                } else if (data[segment.location + i - dataOffset] < min) {
                                    min = data[segment.location + i - dataOffset];
                                }
                                i++;
                            }
                            values[pointIndex - range.location] = (p % 2) ? max : min;
                            values[pointIndex - range.location] *= self.channel->scale;
                            values[pointIndex - range.location] += self.channel->offset;
                            pointIndex++;
                        }
                    }
                } else {
                    // X axis (time)
                    if (sampleCount / pixelCount <= maxPointsPerPixel) {
                        // Show all data points.
                        for (NSUInteger i = skipCount; i < pointCount + skipCount; i++) {
                            values[pointIndex - range.location] = (segment.expandedLocation + i) / self.samplerate;
                            pointIndex++;
                        }
                    } else {
                        // Aggregate data points to two points (min and max) per pixel.
                        for (NSUInteger p = skipCount; p < pointCount + skipCount; p++) {
                            values[pointIndex - range.location] = (segment.expandedLocation + (p / 2 + 0.5) * sampleCount / pixelCount) / self.samplerate;
                            pointIndex++;
                        }
                    }
                }
            } else {
                // Fill segment
                if (fieldEnum == CPTScatterPlotFieldY) {
                    // Y axis (data values)
                    for (NSUInteger i = 0; i < 2; i++) {
                        if (pointIndex >= range.location && pointIndex < range.location + range.length) {
                            values[pointIndex - range.location] = segment.value;
                        }
                        pointIndex++;
                    }
                } else {
                    // X axis (time)
                    if (pointIndex >= range.location && pointIndex < range.location + range.length) {
                        values[pointIndex - range.location] = segment.expandedLocation / self.samplerate;
                    }
                    pointIndex++;
                    if (pointIndex >= range.location && pointIndex < range.location + range.length) {
                        values[pointIndex - range.location] = (segment.expandedLocation + segment.expandedLength) / self.samplerate;
                    }
                    pointIndex++;
                    // TODO: Convert times if necessary
                    // (appDelegate.alignment == SaltoAlignTimeOfDay ||
                    //  appDelegate.alignment == SaltoAlignCalendarDate)
                }
            }
        }
    }
    if (dataArray) {
        // Clean up.
        PyGILState_STATE state = PyGILState_Ensure();
        Py_DECREF(dataArray);
        PyGILState_Release(state);
    }

    return values;
}

- (double)pixelsPerSecond {
    SaltoGuiDelegate *appDelegate = [NSApp delegate];
    NSTimeInterval start = MAX(self.visibleRangeStart, self.startTime);
    NSTimeInterval end = MIN(self.visibleRangeStart + appDelegate.visibleRange, self.endTime);
    CGFloat xMin = [self xForTimeInterval:start];
    CGFloat xMax = [self xForTimeInterval:end];
    double pixelsPerSecond = (xMax - xMin) / (end - start);
    
    return pixelsPerSecond;
}


- (void)setupPlot {
    if (!self.graph) {
        // Create a graph object.
        CGRect frame = NSRectToCGRect(self.view.frame);
        self.graph = [[CPTXYGraph alloc] initWithFrame:frame];

        // Add some padding to the graph.
        self.graph.plotAreaFrame.paddingTop = 0.0;
        if (self.yVisibleRangeMin == 0.0) {
            self.graph.plotAreaFrame.paddingBottom = 12.0;
        } else {
            self.graph.plotAreaFrame.paddingBottom = 0.0;            
        }
        self.graph.plotAreaFrame.paddingLeft = 25.0;
        self.graph.plotAreaFrame.paddingRight = 0.0;
        self.graph.paddingTop = 0.0;
        self.graph.paddingBottom = 0.0;
        self.graph.paddingLeft = 0.0;
        self.graph.paddingRight = 0.0;
        
        // Create a line style for the axes.
        CPTMutableLineStyle *axisLineStyle = [CPTMutableLineStyle lineStyle];
        axisLineStyle.lineColor = [CPTColor blackColor];
        axisLineStyle.lineWidth = 1.0;

        // Create a line style for the plot.
        CPTMutableLineStyle *plotLineStyle = [CPTMutableLineStyle lineStyle];
        plotLineStyle.lineColor = [CPTColor blueColor];
        plotLineStyle.lineWidth = 1.0;
        
        // Create a text style for the axis labels.
        CPTMutableTextStyle *textStyle = [CPTMutableTextStyle textStyle];
        textStyle.fontName = @"Lucida Grande";
        textStyle.fontSize = 8;
        textStyle.color = [CPTColor blackColor];
    
        // Modify plot space to accommodate the range of y values.
        CPTXYPlotSpace *plotSpace = (CPTXYPlotSpace *)self.graph.defaultPlotSpace;
        CPTPlotRange *yVisibleRange = [CPTPlotRange plotRangeWithLocation:CPTDecimalFromDouble(self.yVisibleRangeMin) length:CPTDecimalFromDouble(self.yVisibleRangeMax - self.yVisibleRangeMin)];
        CPTMutablePlotRange *yRange =[yVisibleRange mutableCopy];
        [yRange expandRangeByFactor:CPTDecimalFromDouble(1.1)];
        plotSpace.yRange = yRange;
        
        // Set the axis labels, line styles, etc.
        CPTXYAxisSet *axisSet = (CPTXYAxisSet *)self.graph.axisSet;
        CPTXYAxis *x = axisSet.xAxis;
        CPTXYAxis *y = axisSet.yAxis;
        
        x.titleTextStyle = textStyle;
        x.axisLineStyle = axisLineStyle;
        x.majorTickLineStyle = axisLineStyle;
        x.minorTickLineStyle = axisLineStyle;
        x.minorTickLength = 1.0;
        x.majorTickLength = 2.0;
        x.labelTextStyle = textStyle;
        
        y.title = [NSString stringWithUTF8String:self.channel->unit];
        y.titleTextStyle = textStyle;
        y.titleOffset = 15.0;
        y.titleLocation = CPTDecimalFromDouble([plotSpace.yRange midPointDouble] / 1.2);
        y.axisLineStyle = axisLineStyle;
        y.majorTickLineStyle = axisLineStyle;
        y.minorTickLineStyle = axisLineStyle;
        y.labelingPolicy = CPTAxisLabelingPolicyAutomatic;
        y.labelTextStyle = textStyle;
        y.labelOffset = 1.0;
        y.orthogonalCoordinateDecimal = CPTDecimalFromDouble(0.0);
        y.minorTickLength = 1.0;
        y.majorTickLength = 2.0;
        y.visibleRange = yVisibleRange;
        NSNumberFormatter *formatter = [[NSNumberFormatter alloc] init];
        [formatter setMaximumFractionDigits:0];
        y.labelFormatter = formatter;
        [formatter release];
        
        // Add a plot to our graph and axis. We give it an identifier so that we
        // could add multiple plots to the same graph if necessary.
        CPTScatterPlot *plot = [[CPTScatterPlot alloc] init];
        plot.dataSource = self;
        plot.identifier = self.label;
        plot.dataLineStyle = plotLineStyle;
        [self.graph addPlot:plot];
        plotSpace.delegate = self;
    }
    // Modify plot space to accommodate the range of x values.
    SaltoGuiDelegate *appDelegate = [NSApp delegate];
    CPTXYPlotSpace *plotSpace = (CPTXYPlotSpace *)self.graph.defaultPlotSpace;
    CPTXYAxisSet *axisSet = (CPTXYAxisSet *)self.graph.axisSet;
    CPTXYAxis *x = axisSet.xAxis;
    if (appDelegate.alignment == SaltoAlignCalendarDate) {
        plotSpace.globalXRange = [CPTPlotRange plotRangeWithLocation:CPTDecimalFromDouble(appDelegate.rangeStart) length:CPTDecimalFromDouble(appDelegate.range)];
        x.visibleRange = [CPTPlotRange plotRangeWithLocation:CPTDecimalFromDouble(self.startTime) length:CPTDecimalFromDouble(self.duration)];
        x.orthogonalCoordinateDecimal = CPTDecimalFromDouble(appDelegate.rangeStart);
        x.axisTitle = nil;
        x.labelingPolicy = CPTAxisLabelingPolicyAutomatic;
        x.labelOffset = 1.0;
    } else if (appDelegate.alignment == SaltoAlignTimeOfDay) {
        plotSpace.globalXRange = [CPTPlotRange plotRangeWithLocation:CPTDecimalFromDouble(0.0) length:CPTDecimalFromDouble(86400.0)];
        x.visibleRange = [CPTPlotRange plotRangeWithLocation:CPTDecimalFromDouble(0.0) length:CPTDecimalFromDouble(self.duration)];
        x.orthogonalCoordinateDecimal = CPTDecimalFromDouble(0.0);
        x.axisTitle = nil;
        x.labelingPolicy = CPTAxisLabelingPolicyAutomatic;
        x.labelOffset = 1.0;
    } else {
        plotSpace.globalXRange = [CPTPlotRange plotRangeWithLocation:CPTDecimalFromDouble(0.0) length:CPTDecimalFromDouble(appDelegate.maxVisibleRange)];
        x.visibleRange = [CPTPlotRange plotRangeWithLocation:CPTDecimalFromDouble(0.0) length:CPTDecimalFromDouble(self.duration)];
        x.orthogonalCoordinateDecimal = CPTDecimalFromDouble(0.0);
        x.title = @"ms";
        x.titleOffset = 10.0;
        x.labelingPolicy = CPTAxisLabelingPolicyAutomatic;
        x.labelOffset = 1.0;
    }
    plotSpace.xRange = [CPTPlotRange plotRangeWithLocation:CPTDecimalFromDouble(appDelegate.visibleRangeStart) length:CPTDecimalFromDouble(appDelegate.visibleRange)];
    if (x.axisTitle) {
        x.titleLocation = CPTDecimalFromDouble([plotSpace.xRange midPointDouble]);
    }
}

- (CPTPlotRange *)plotSpace:(CPTPlotSpace *)space
      willChangePlotRangeTo:(CPTPlotRange *)newRange
              forCoordinate:(CPTCoordinate)coordinate {
    // Keep X axis at the left edge of the screen.
    CPTXYAxisSet *axisSet = (CPTXYAxisSet *)self.graph.axisSet;
    if (coordinate == CPTCoordinateX) {
        axisSet.yAxis.orthogonalCoordinateDecimal = newRange.location;
        axisSet.xAxis.titleLocation = CPTDecimalFromFloat(newRange.locationDouble +
                                                          (newRange.lengthDouble / 2.0));
    }
    
    return newRange;
}


#pragma mark - Conversions

- (NSTimeInterval)timeForPoint:(CGPoint)point {
    NSDecimal plotPoint[2];
    
    CPTPlotSpace *plotSpace = self.graph.defaultPlotSpace;
    // Convert from view coordinates to plot area coordinates.
    CGPoint areaPoint = [self.graph convertPoint:point
                                        toLayer:self.graph.plotAreaFrame.plotArea];
    // Convert from plot area coordinates to data coordinates.
    [plotSpace plotPoint:plotPoint numberOfCoordinates:2 forPlotAreaViewPoint:areaPoint];
    
    return [[NSDecimalNumber decimalNumberWithDecimal:plotPoint[0]] doubleValue];
}

- (CGFloat)xForTimeInterval:(NSTimeInterval)time {
    double plotPoint[2] = {time, 0.0};
    // Convert from data coordinates to plot area coordinates.
    CPTPlotSpace *plotSpace = self.graph.defaultPlotSpace;
    CGPoint areaPoint = [plotSpace plotAreaViewPointForDoublePrecisionPlotPoint:plotPoint
                                                            numberOfCoordinates:2];
    // Convert from plot area coordinates to view coordinates.
    CGPoint graphPoint = [self.graph convertPoint:areaPoint
                                        fromLayer:self.graph.plotAreaFrame.plotArea];


    return graphPoint.x;
}

- (CGFloat)xForTimespec:(struct timespec)t {
    NSTimeInterval time = t.tv_sec + t.tv_nsec / 1e9;
    return [self xForTimeInterval:time];
}

@end