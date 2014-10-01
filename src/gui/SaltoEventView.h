//
//  SaltoEventView.h
//  OpenSALTO GUI
//
//  Copyright 2014 Marko Havu. Released under the terms of
//  GNU General Public License version 3 or later.
//

#import <Cocoa/Cocoa.h>
#include "Event.h"

@interface SaltoEventView : NSView

@property (unsafe_unretained) Event *event;

@end
