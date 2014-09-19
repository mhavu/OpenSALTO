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

- (instancetype)initWithFrame:(NSRect)frame {
    self = [super initWithFrame:frame];
    if (self) {
        [self setBackgroundStyle:NSBackgroundStyleLight];
    }

    return self;
}

- (void)dealloc {
    // dealloc
    [super dealloc];
}

- (void)drawRect:(NSRect)rect {
    // TODO: Only draw rect.
    [self.objectValue drawInContext:NSGraphicsContext.currentContext.graphicsPort size:self.frame.size];
}

@end