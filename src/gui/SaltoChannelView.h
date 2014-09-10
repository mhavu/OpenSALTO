//
//  SaltoChannelView.h
//  OpenSALTO GUI
//
//  Copyright 2014 Marko Havu. Released under the terms of
//  GNU General Public License version 3 or later.
//

#import <Cocoa/Cocoa.h>

@class SaltoChannelWrapper;

@interface SaltoChannelView : NSTableCellView

@property (unsafe_unretained) SaltoChannelWrapper *channel;

@end