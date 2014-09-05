//
//  AppDelegate.h
//  OpenSalto GUI
//
//  Copyright 2014 Marko Havu. Released under the terms of
//  GNU General Public License version 3 or later.
//

#import <Cocoa/Cocoa.h>
#import "SaltoMainWindowController.h"
#import "SaltoConsoleController.h"

@interface SaltoGuiAppDelegate : NSObject

@property (readonly) IBOutlet SaltoMainWindowController *mainWindowController;
@property (readonly) IBOutlet SaltoConsoleController *consoleController;
@property (nonatomic, assign) dispatch_queue_t queue;

- (IBAction)showConsoleWindow:(id)sender;

@end
