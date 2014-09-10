//
//  SaltoChannelViewController.m
//  OpenSALTO GUI
//
//  Copyright 2014 Marko Havu. Released under the terms of
//  GNU General Public License version 3 or later.
//

#import "SaltoChannelViewController.h"
#import "SaltoChannelView.h"
#include "salto.h"

@implementation SaltoChannelWrapper

@synthesize channel;
@synthesize view;
@synthesize label;
@synthesize alignment;
@synthesize visibleRangeStart;
@synthesize visibleRangeEnd;
@synthesize visible;

+ (instancetype)wrapperForChannel:(Channel *)ch {
    return [[[SaltoChannelWrapper alloc] initWithChannel:ch] autorelease];
}

- (instancetype)initWithChannel:(Channel *)ch {
    self = [super init];
    if (self) {
        Py_INCREF(ch);
        channel = ch;
        label = @"label";
    }

    return self;
}

- (void)dealloc {
    [view release];
    Py_XDECREF(channel);
    [super dealloc];
}

@end