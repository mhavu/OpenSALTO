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

@property (nonatomic, unsafe_unretained) IBOutlet CPTGraphHostingView *hostingView;
@property (nonatomic, readonly) NSMutableArray *eventLayerArray;
@property (nonatomic, readonly) NSMutableArray *trackingAreaArray;
@property (nonatomic, readonly) NSTrackingArea *activeTrackingArea;
@property (nonatomic, readonly, getter = isResizing) BOOL resizing;

- (void)clearEventLayers;
- (void)addEventLayer:(CALayer *)eventLayer;
- (void)clearTrackingAreas;
- (void)addTrackingAreasForEvent:(CALayer *)eventLayer;

@end