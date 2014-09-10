//
//  SaltoChannelView.h
//  OpenSALTO GUI
//
//  Copyright 2014 Marko Havu. Released under the terms of
//  GNU General Public License version 3 or later.
//

#import <Cocoa/Cocoa.h>

@class SaltoChannelViewController;
@class SaltoChannelWrapper;

@interface SaltoChannelCellView : NSTableCellView

@property (readonly) IBOutlet SaltoChannelViewController *viewController;
@property (unsafe_unretained) SaltoChannelWrapper *channel;

@end