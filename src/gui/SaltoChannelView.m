//
//  SaltoChannelView.m
//  OpenSALTO GUI
//
//  Copyright 2014 Marko Havu. Released under the terms of
//  GNU General Public License version 3 or later.
//

#import "SaltoChannelView.h"
#import "SaltoChannelWrapper.h"
#import "SaltoEventWrapper.h"
#import "SaltoGuiDelegate.h"

@implementation SaltoChannelView

- (void)awakeFromNib {
    _eventLayerArray = [[NSMutableArray array] retain];
    _trackingAreaArray = [[NSMutableArray array] retain];
}

- (void)dealloc {
    [_eventLayerArray release];
    [_trackingAreaArray release];
    [super dealloc];
}

- (void)refreshView {
    [self.objectValue setupPlot];
    self.hostingView.hostedGraph = [self.objectValue graph];
    [self.hostingView.hostedGraph setNeedsDisplay];
    [self.objectValue updateEventViews];
}

- (void)viewDidMoveToSuperview {
    SaltoChannelView *linkedView = self.superview ? self : nil;
    [self.objectValue setView:linkedView];
    if (linkedView) {
        [self refreshView];
    } else {
        [self clearEventLayers];
        [self clearTrackingAreas];
    }
}

- (void)clearEventLayers {
    for (CALayer *layer in self.eventLayerArray) {
        [layer removeFromSuperlayer];
    }
    [self.eventLayerArray removeAllObjects];
}

- (void)setFrame:(CGRect)rect forEvent:(SaltoEventWrapper *)event {
    CALayer *layer = [CALayer layer];
    layer.frame = rect;
    layer.backgroundColor = CGColorCreateGenericRGB(1.0, 1.0, 0.0, 0.3);
    layer.borderWidth = 1.0;
    layer.borderColor = event.color;
    [layer setValue:event forKey:@"eventObject"];
    [self.hostingView.layer addSublayer:layer];
    [self.eventLayerArray addObject:layer];
    [self addTrackingAreasForLayer:layer];
}

- (void)clearTrackingAreas {
    for (NSTrackingArea *area in self.trackingAreaArray) {
        [self removeTrackingArea:area];
    }
    [self.trackingAreaArray removeAllObjects];
}

- (void)addTrackingArea:(NSTrackingArea *)trackingArea {
    [super addTrackingArea:trackingArea];
    [self.trackingAreaArray addObject:trackingArea];
}

- (void)addTrackingAreasForLayer:(CALayer *)layer {
    // Add tracking areas to the left and right edge of the event
    // to allow resizing by dragging.
    CGFloat width = 4.0;
    NSRect lRect = NSMakeRect(NSMinX(layer.frame) - width / 2, NSMinY(layer.frame),
                              width, NSHeight(layer.frame));
    NSRect rRect = NSMakeRect(NSMaxX(layer.frame) - width / 2, NSMinY(layer.frame),
                              width, NSHeight(layer.frame));
    NSMutableDictionary *userInfoDict = [NSMutableDictionary dictionaryWithObject:layer
                                                                           forKey:@"layer"];
    if (!NSIntersectsRect(lRect, rRect)) {
        [userInfoDict setObject:@"left" forKey:@"edge"];
        NSTrackingArea *leftEdge = [[NSTrackingArea alloc] initWithRect:lRect
                                                                options:(NSTrackingCursorUpdate | NSTrackingActiveInActiveApp)
                                                                  owner:self
                                                               userInfo:[userInfoDict copy]];
        [self addTrackingArea:leftEdge];
        [leftEdge release];
        [userInfoDict setObject:@"right" forKey:@"edge"];
        NSTrackingArea *rightEdge = [[NSTrackingArea alloc] initWithRect:rRect
                                                                 options:(NSTrackingCursorUpdate | NSTrackingActiveInActiveApp)
                                                                   owner:self
                                                                userInfo:userInfoDict];
        [self addTrackingArea:rightEdge];
        [rightEdge release];
    } else {
        // Edges overlap. Add a single combined tracking area.
        NSRect rect = NSIntersectionRect(lRect, rRect);
        [userInfoDict setObject:@"both" forKey:@"edge"];
        NSTrackingArea *area = [[NSTrackingArea alloc] initWithRect:rect
                                                                options:(NSTrackingCursorUpdate | NSTrackingActiveInActiveApp)
                                                                  owner:self
                                                               userInfo:userInfoDict];
        [self addTrackingArea:area];
        [area release];
    }
}

#pragma mark - NSEvent handling

- (void)updateTrackingAreas {
    [self clearTrackingAreas];
    for (CALayer *layer in self.eventLayerArray) {
        [self addTrackingAreasForLayer:layer];
    }
    [super updateTrackingAreas];
}

- (BOOL)acceptsFirstResponder {
    return YES;
}

- (BOOL)acceptsFirstMouse:(NSEvent *)event {
    return YES;
}

- (void)cursorUpdate:(NSEvent *)event {
    if (event.trackingArea) {
        NSPoint hitPoint = [self convertPoint:event.locationInWindow fromView:nil];
        if ([self mouse:hitPoint inRect:event.trackingArea.rect]) {
            [[NSCursor resizeLeftRightCursor] set];
            _activeTrackingArea = [event.trackingArea retain];
        } else {
            [[NSCursor arrowCursor] set];
            [_activeTrackingArea release];
            _activeTrackingArea = nil;
        }
    }
}

- (void)mouseDown:(NSEvent *)event {
    NSPoint hitPoint = [self convertPoint:event.locationInWindow fromView:nil];
    NSLog(@"time: %f s", [self.objectValue timeForPoint:NSPointToCGPoint(hitPoint)]);
    if (_activeTrackingArea) {
        [self removeTrackingArea:_activeTrackingArea];
        _resizing = YES;
    } else {
        [self.nextResponder mouseDown:event];
    }
}

- (void)mouseDragged:(NSEvent *)event {
    if (self.isResizing) {
        // Resize the event rectangle.
        // TODO: If value for the key edge is @"both", move left or right edge based on the initial
        // direction of dragging.
        CALayer *eventLayer = [self.activeTrackingArea.userInfo objectForKey:@"layer"];
        NSPoint hitPoint = [self convertPoint:event.locationInWindow fromView:nil];
        [CATransaction begin];
        [CATransaction setDisableActions:YES];
        if ([[self.activeTrackingArea.userInfo objectForKey:@"edge"] isEqualToString:@"left"]) {
            if (hitPoint.x < NSMaxX(eventLayer.frame)) {
                eventLayer.frame = NSMakeRect(hitPoint.x, 0.0, NSMaxX(eventLayer.frame) - hitPoint.x, NSHeight(eventLayer.frame));
            } else {
                eventLayer.frame = NSMakeRect(NSMaxX(eventLayer.frame), 0.0, 0.5, NSHeight(eventLayer.frame));
            }
        } else {
            if (hitPoint.x > NSMinX(eventLayer.frame)) {
                eventLayer.frame = NSMakeRect(NSMinX(eventLayer.frame), 0.0, hitPoint.x - NSMinX(eventLayer.frame), NSHeight(eventLayer.frame));
            } else {
                eventLayer.frame = NSMakeRect(NSMinX(eventLayer.frame), 0.0, 0.5, NSHeight(eventLayer.frame));
            }
        }
        [CATransaction commit];
        // Restrict mouse cursor movement to channel boundaries.
        // TODO: Scroll when cursor is near the edge.
        NSRect visibleRect = [self convertRect:self.visibleRect toView:nil];
        if (!NSPointInRect(event.locationInWindow, visibleRect)) {
            CGAssociateMouseAndMouseCursorPosition(false);
            NSPoint newLocation = event.locationInWindow;
            if (newLocation.x < NSMinX(visibleRect)) {
                newLocation.x = NSMinX(visibleRect);
            } else if (newLocation.x > NSMaxX(visibleRect)) {
                newLocation.x = NSMaxX(visibleRect);
            }
            if (newLocation.y < NSMinY(visibleRect)) {
                newLocation.y = NSMinY(visibleRect);
            } else if (newLocation.y > NSMaxY(visibleRect)) {
                newLocation.y = NSMaxY(visibleRect);
            }
            NSRect screenLocation = [self.window convertRectToScreen:(NSRect){.origin = newLocation}];
            screenLocation.origin.y = NSHeight([[NSScreen mainScreen] frame]) - screenLocation.origin.y;
            CGWarpMouseCursorPosition(NSPointToCGPoint(screenLocation.origin));
            CGAssociateMouseAndMouseCursorPosition(true);
        }
        // Make sure the cursor doesn't change until dragging ends.
        [[NSCursor resizeLeftRightCursor] set];
    } else {
        [self.nextResponder mouseDragged:event];
    }
}

- (void)mouseUp:(NSEvent *)event {
    if (self.isResizing) {
        // Set new start and end time for the event.
        CALayer *eventLayer = [self.activeTrackingArea.userInfo objectForKey:@"layer"];
        CGPoint point = NSPointToCGPoint(eventLayer.frame.origin);
        NSTimeInterval start = [self.objectValue timeForPoint:point];
        point.x += NSWidth(eventLayer.frame);
        NSTimeInterval end = [self.objectValue timeForPoint:point];
        SaltoEventWrapper *event = [eventLayer valueForKey:@"eventObject"];
        event.startTime = start;
        event.endTime = end;
        // Reset tracking areas.
        [self clearTrackingAreas];
        for (CALayer *layer in self.eventLayerArray) {
            [self addTrackingAreasForLayer:layer];
        }
        _resizing = NO;
    } else {
        [self.nextResponder mouseUp:event];
    }
}

- (void)scrollWheel:(NSEvent *)event {
    if (event.deltaX != 0.0) {
        SaltoGuiDelegate *appDelegate = [NSApp delegate];
        NSScroller *scroller = appDelegate.scroller;
        if (scroller.isEnabled) {
            double delta = event.deltaX / (NSWidth(scroller.bounds) * (1.0 - scroller.knobProportion));
            scroller.doubleValue = MIN(MAX(scroller.doubleValue - delta, 0.0), 1.0);
            [appDelegate moveVisibleRangeToScrollerPosition:scroller.doubleValue];
        }
    }
    if (event.deltaY != 0.0) {
        [self.nextResponder scrollWheel:event];
    }
}

@end