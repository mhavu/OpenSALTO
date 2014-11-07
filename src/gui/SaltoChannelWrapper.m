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
            _visibleRangeStart = appDelegate.rangeStart;
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
    }

    return self;
}

- (void)dealloc {
    [_graph release];
    [_eventViewArray release];
    [_label release];
    [_signalType release];
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
                // Draw the event as a Core Animation layer.
                CALayer *eventLayer = [CALayer layer];
                // TODO: Assign event colours.
                eventLayer.backgroundColor = CGColorCreateGenericRGB(1.0, 1.0, 0.0, 0.3);
                eventLayer.borderWidth = 1.0;
                eventLayer.borderColor = CGColorCreateGenericRGB(0.0, 0.0, 0.0, 0.3);
                struct timespec eventStart = {e->start_sec, e->start_nsec};
                struct timespec eventEnd = {e->end_sec, e->end_nsec};
                if (appDelegate.alignment != SaltoAlignCalendarDate) {
                    eventStart = endTimeFromDuration(e->end_sec, e->end_nsec, -self.visibleRangeStart);
                    eventEnd = endTimeFromDuration(e->end_sec, e->end_nsec, -self.visibleRangeStart);
                }
                CGFloat xMin = [self xForTimespec:eventStart];
                CGFloat xMax = [self xForTimespec:eventEnd];
                eventLayer.frame = CGRectMake(xMin, 0, xMax - xMin, NSHeight(self.view.frame));
                [self.view addEventLayer:eventLayer];
                [self.view addTrackingAreasForEvent:eventLayer];
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
    SaltoGuiDelegate *appDelegate = [NSApp delegate];
    NSTimeInterval start = MAX(self.visibleRangeStart, self.startTime);
    NSTimeInterval end = MIN(self.visibleRangeStart + appDelegate.visibleRange, self.endTime);
    NSInteger nPoints = (end > start) ? (end - start) * self.samplerate : 0;
    if (nPoints > 0) {
        CGFloat xMin = [self xForTimeInterval:start];
        CGFloat xMax = [self xForTimeInterval:end];
        NSUInteger nPixels = xMax - xMin + 1;
        if (nPoints > 3 * nPixels) {
            // Aggregate data points.
            double pixelsPerSecond = (xMax - xMin) / (end - start);
            nPoints = nPoints * pixelsPerSecond / self.samplerate;
            if (nPoints < 2)
                nPoints = 2;
            nPoints = 3 * nPoints - 2;
        }
    }
    
    return nPoints;
}

- (double *)doublesForPlot:(CPTPlot *)plot field:(NSUInteger)fieldEnum recordIndexRange:(NSRange)range {
    double *values = calloc(range.length, sizeof(double));
    if (values) {
        SaltoGuiDelegate *appDelegate = [NSApp delegate];
        NSTimeInterval start = MAX(self.visibleRangeStart, self.startTime);
        NSTimeInterval end = MIN(self.visibleRangeStart + appDelegate.visibleRange, self.endTime);
        struct timespec start_t = endTimeFromDuration(0, 0, start);
        struct timespec end_t = endTimeFromDuration(0, 0, end);
        CGFloat xMin = [self xForTimeInterval:start];
        CGFloat xMax = [self xForTimeInterval:end];
        double pixelsPerSecond = (xMax - xMin) / (end - start);
        
        if (fieldEnum == CPTScatterPlotFieldY) {
            PyGILState_STATE state = PyGILState_Ensure();
            PyArrayObject *data = (PyArrayObject *)PyObject_CallMethod((PyObject *)_channel,
                                                                       "resampledData", "dLlLlis",
                                                                       pixelsPerSecond,
                                                                       start_t.tv_sec, start_t.tv_nsec,
                                                                       end_t.tv_sec, end_t.tv_nsec,
                                                                       typenum, "VRA");
            memcpy(values, PyArray_DATA(data), range.length * sizeof(double));
            PyGILState_Release(state);
        } else {
            NSUInteger nPoints = [self numberOfRecordsForPlot:plot];
            if (appDelegate.alignment == SaltoAlignMarkers) {
                for (NSUInteger i = 0; i < range.length; i++) {
                    values[i] = (range.location + i) * (end - start) / (nPoints - 1);
                }
            } else {
                for (NSUInteger i = 0; i < range.length; i++) {
                    values[i] = start + (range.location + i) * (end - start) / (nPoints - 1);
                }
            }
        }
    }
    
    return values;
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
    NSDecimal origin = CPTDecimalFromDouble(0.0);
    if (appDelegate.alignment == SaltoAlignCalendarDate) {
        origin = CPTDecimalFromDouble(self.visibleRangeStart);
        x.title = @"";
        x.titleOffset = 10.0;
        x.titleLocation = CPTDecimalFromDouble([plotSpace.xRange midPointDouble]);
        x.labelingPolicy = CPTAxisLabelingPolicyAutomatic;
        x.labelOffset = 1.0;
    } else {
        x.title = @"ms";
        x.titleOffset = 10.0;
        x.titleLocation = CPTDecimalFromDouble([plotSpace.xRange midPointDouble]);
        x.labelingPolicy = CPTAxisLabelingPolicyAutomatic;
        x.labelOffset = 1.0;
    }
    plotSpace.globalXRange = [CPTPlotRange plotRangeWithLocation:origin length:CPTDecimalFromDouble(appDelegate.visibleRange)];
    x.orthogonalCoordinateDecimal = origin;
    plotSpace.xRange = [plotSpace.globalXRange copy];
    x.visibleRange = [plotSpace.globalXRange copy];
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