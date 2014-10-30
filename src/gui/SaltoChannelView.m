//
//  SaltoChannelView.m
//  OpenSALTO GUI
//
//  Copyright 2014 Marko Havu. Released under the terms of
//  GNU General Public License version 3 or later.
//

#import "SaltoChannelView.h"
#import "SaltoChannelWrapper.h"
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

- (void)viewDidMoveToSuperview {
    SaltoChannelView *linkedView = self.superview ? self : nil;
    [self.objectValue setView:linkedView];
    if (linkedView) {
        [self.objectValue setupPlot];
        self.hostingView.hostedGraph = [self.objectValue graph];
        [self.objectValue updateEventViews];
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

- (void)addEventLayer:(CALayer *)eventLayer {
    [self.hostingView.layer addSublayer:eventLayer];
    [self.eventLayerArray addObject:eventLayer];
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

- (void)addTrackingAreasForEvent:(CALayer *)eventLayer {
    // Add tracking areas to the left and right edge of the event
    // to allow resizing by dragging.
    CGFloat width = 4.0;
    NSRect lRect = NSMakeRect(NSMinX(eventLayer.frame) - width / 2, NSMinY(eventLayer.frame), width, NSHeight(eventLayer.frame));
    NSRect rRect = NSMakeRect(NSMaxX(eventLayer.frame) - width / 2, NSMinY(eventLayer.frame), width, NSHeight(eventLayer.frame));
    NSMutableDictionary *userInfoDict = [NSMutableDictionary dictionaryWithObject:eventLayer
                                                                           forKey:@"layer"];
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
}

#pragma mark - NSEvent handling

- (void)updateTrackingAreas {
    [self clearTrackingAreas];
    for (CALayer *layer in self.eventLayerArray) {
        [self addTrackingAreasForEvent:layer];
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
        // TODO: If event duration is 0, move left or right edge based on the starting
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
        [self clearTrackingAreas];
        for (CALayer *layer in self.eventLayerArray) {
            [self addTrackingAreasForEvent:layer];
        }
        _resizing = NO;
        // TODO: Edit event start and end times
    } else {
        [self.nextResponder mouseUp:event];
    }
}

@end