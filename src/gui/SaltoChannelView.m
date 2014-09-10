//
//  SaltoChannelView.m
//  OpenSALTO GUI
//
//  Copyright 2014 Marko Havu. Released under the terms of
//  GNU General Public License version 3 or later.
//

#import "SaltoChannelView.h"
#import "SaltoChannelWrapper.h"

@implementation SaltoChannelView

@synthesize channel;

- (instancetype)initWithFrame:(NSRect)frame {
    self = [super initWithFrame:frame];
    if (self) {
        // init
    }

    return self;
}

- (void)dealloc {
    // dealloc
    [super dealloc];
}

- (void)drawRect:(NSRect)dirtyRect {
    CGFloat hue = ( arc4random() % 256 / 256.0 );  //  0.0 to 1.0
    CGFloat saturation = ( arc4random() % 128 / 256.0 ) + 0.5;  //  0.5 to 1.0, away from white
    CGFloat brightness = ( arc4random() % 128 / 256.0 ) + 0.5;  //  0.5 to 1.0, away from black
    [[NSColor colorWithDeviceHue:hue saturation:saturation brightness:brightness alpha:1] set];
    [NSBezierPath fillRect:dirtyRect];
}

@end