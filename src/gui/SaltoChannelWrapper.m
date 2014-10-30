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
    [_graph release];
    [_eventViewArray release];
    [_label release];
    [_signalType release];
    [_samplerate release];
    PyGILState_STATE state = PyGILState_Ensure();
	Py_XDECREF(_channel);
	PyGILState_Release(state);
    [super dealloc];
}

- (void)updateEventViews {
    SaltoGuiDelegate *appDelegate = [NSApp delegate];
    NSTimeInterval visibleInterval = appDelegate.visibleRangeEnd - appDelegate.visibleRangeStart;
    struct timespec start_t = endTimeFromDuration(0, 0, appDelegate.visibleRangeStart);
    struct timespec end_t = endTimeFromDuration(self.channel->start_sec, self.channel->start_nsec, visibleInterval);
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
                // TODO: Create events.
                // Draw the event as a Core Animation layer.
                CALayer *eventLayer = [CALayer layer];
                eventLayer.backgroundColor = CGColorCreateGenericRGB(1.0, 1.0, 0.0, 0.3);
                eventLayer.borderWidth = 1.0;
                eventLayer.borderColor = CGColorCreateGenericRGB(0.0, 0.0, 0.0, 0.3);
                eventLayer.frame = CGRectMake(NSMidX(self.view.frame) - 40, 0, 80, NSHeight(self.view.frame));
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
    double plotPoint[2];
    
    NSTimeInterval xMin = MAX(-self.alignment, 0);
    NSTimeInterval xMax = MIN(channelDuration(self.channel) - self.alignment, self.visibleRange);
    plotPoint[0] = xMin;
    plotPoint[1] = 0.0;
    CGPoint startPoint = [plot.plotSpace plotAreaViewPointForDoublePrecisionPlotPoint:plotPoint numberOfCoordinates:2];
    plotPoint[0] = xMax;
    CGPoint endPoint = [plot.plotSpace plotAreaViewPointForDoublePrecisionPlotPoint:plotPoint numberOfCoordinates:2];
    NSUInteger nPixels = endPoint.x - startPoint.x + 1;
    
    return 3 * nPixels;
}

- (double *)doublesForPlot:(CPTPlot *)plot field:(NSUInteger)fieldEnum recordIndexRange:(NSRange)range {
    double *values = calloc(range.length, sizeof(double));
    if (values) {
        // Determine start time, end time, and samplerate based on the shown area.
        SaltoGuiDelegate *appDelegate = [NSApp delegate];
        struct timespec start_t = endTimeFromDuration(0, 0, appDelegate.visibleRangeStart);
        struct timespec end_t = endTimeFromDuration(self.channel->start_sec, self.channel->start_nsec, _visibleRange);
        NSTimeInterval xMin = MAX(-self.alignment, 0);
        NSTimeInterval xMax = MIN(channelDuration(self.channel) - self.alignment, self.visibleRange);
        
        if (fieldEnum == CPTScatterPlotFieldY) {
            PyGILState_STATE state = PyGILState_Ensure();
            PyArrayObject *data = (PyArrayObject *)PyObject_CallMethod((PyObject *)_channel, "resampledData",
                                                                       "dLlLlis", [self numberOfRecordsForPlot:plot] / 3 /(xMax - xMin),
                                                                       start_t.tv_sec, start_t.tv_nsec,
                                                                       end_t.tv_sec, end_t.tv_nsec,
                                                                       typenum, "VRA");
            memcpy(values, PyArray_DATA(data), range.length * sizeof(double));
            PyGILState_Release(state);
        } else {
            for (NSUInteger i = 0; i < range.length; i++) {
                //values[i] = (range.location + i) * (xMax - xMin) / [self numberOfRecordsForPlot:plot];
                values[i] = i;
            }
        }
    }
    
    return values;
}

- (void)setupPlot {
    if (self.view.hostingView != nil && !self.graph) {
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
        
        // Create a line style that we will apply to the axes.
        CPTMutableLineStyle *axisLineStyle = [CPTMutableLineStyle lineStyle];
        axisLineStyle.lineColor = [CPTColor blackColor];
        axisLineStyle.lineWidth = 1.0;

        // Create a line style that we will apply to the plot.
        CPTMutableLineStyle *plotLineStyle = [CPTMutableLineStyle lineStyle];
        plotLineStyle.lineColor = [CPTColor blueColor];
        plotLineStyle.lineWidth = 1.0;
        
        // Create a text style that we will use for the axis labels.
        CPTMutableTextStyle *textStyle = [CPTMutableTextStyle textStyle];
        textStyle.fontName = @"Lucida Grande";
        textStyle.fontSize = 8;
        textStyle.color = [CPTColor blackColor];
        
        // We modify the graph's plot space to setup the axis' min / max values.
        SaltoGuiDelegate *appDelegate = [NSApp delegate];
        CPTXYPlotSpace *plotSpace = (CPTXYPlotSpace *)self.graph.defaultPlotSpace;
        plotSpace.globalXRange = [CPTPlotRange plotRangeWithLocation:CPTDecimalFromDouble(appDelegate.rangeStart) length:CPTDecimalFromDouble(appDelegate.range)];
        plotSpace.xRange = [plotSpace.globalXRange copy];
        CPTPlotRange *yVisibleRange = [CPTPlotRange plotRangeWithLocation:CPTDecimalFromDouble(self.yVisibleRangeMin) length:CPTDecimalFromDouble(self.yVisibleRangeMax - self.yVisibleRangeMin)];
        CPTMutablePlotRange *yRange =[yVisibleRange mutableCopy];
        [yRange expandRangeByFactor:CPTDecimalFromDouble(1.1)];
        plotSpace.yRange = yRange;
        
        // Modify the graph's axis with a label, line style, etc.
        CPTXYAxisSet *axisSet = (CPTXYAxisSet *)self.graph.axisSet;
        CPTXYAxis *x = axisSet.xAxis;
        CPTXYAxis *y = axisSet.yAxis;
        
        x.title = @"ms";
        x.titleTextStyle = textStyle;
        x.titleOffset = 10.0;
        x.titleLocation = CPTDecimalFromDouble([plotSpace.xRange midPointDouble]);
        x.axisLineStyle = axisLineStyle;
        x.majorTickLineStyle = axisLineStyle;
        x.minorTickLineStyle = axisLineStyle;
        x.labelingPolicy = CPTAxisLabelingPolicyAutomatic;
        x.labelTextStyle = textStyle;
        x.labelOffset = 1.0;
        x.orthogonalCoordinateDecimal = CPTDecimalFromDouble(0.0);
        x.minorTickLength = 1.0;
        x.majorTickLength = 2.0;
        x.visibleRange = [plotSpace.globalXRange copy];
        
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

@end