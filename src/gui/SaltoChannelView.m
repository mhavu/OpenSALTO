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

- (void)viewDidMoveToSuperview {
    SaltoChannelView *linkedView = self.superview ? self : nil;
    [self.objectValue setView:linkedView];
}

- (void)drawRect:(NSRect)rect {
    // TODO: Only draw rect.
    [self.objectValue drawInContext:NSGraphicsContext.currentContext.graphicsPort];
}

@end