//
//  SaltoChannelView.h
//  OpenSALTO GUI
//
//  Copyright 2014 Marko Havu. Released under the terms of
//  GNU General Public License version 3 or later.
//

#import <Cocoa/Cocoa.h>

@class CPTGraphHostingView;

@interface SaltoChannelView : NSTableCellView

@property (unsafe_unretained) IBOutlet CPTGraphHostingView *hostingView;
@property (readonly) NSMutableArray *eventLayerArray;
@property (readonly) NSMutableArray *trackingAreaArray;
@property (readonly) NSTrackingArea *activeTrackingArea;
@property (readonly, getter = isResizing) BOOL resizing;

- (void)clearEventLayers;
- (void)addEventLayer:(CALayer *)eventLayer;
- (void)clearTrackingAreas;
- (void)addTrackingAreasForEvent:(CALayer *)eventLayer;


@end