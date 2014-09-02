//
//  AppDelegate.m
//  OpenSalto GUI
//
//  Copyright 2014 Marko Havu. Released under the terms of
//  GNU General Public License version 3 or later.
//

#import "SaltoGuiAppDelegate.h"
#include "salto.h"

@implementation SaltoGuiAppDelegate

@synthesize mainWindowController;
@synthesize consoleController;

- (instancetype)init {
    self = [super init];
    if (self) {
        mainWindowController = [[SaltoMainWindowController alloc] init];
        consoleController = [[SaltoConsoleController alloc] init];
    }

    return self;
}

- (void)dealloc {
    [mainWindowController release];
    [consoleController release];
    [super dealloc];
}

- (void)applicationDidFinishLaunching:(NSNotification *)aNotification {
    // "Start Dictation..." and "Special Characters..." are not relevant to this application.
    // Remove these automatically added entries from the Edit menu.
    NSMenu *edit = [[[[NSApplication sharedApplication] mainMenu] itemWithTitle:@"Edit"] submenu];
    if ([[edit itemAtIndex:[edit numberOfItems] - 1] action] ==
        NSSelectorFromString(@"orderFrontCharacterPalette:"))
        [edit removeItemAtIndex:[edit numberOfItems] - 1];
    if ([[edit itemAtIndex: [edit numberOfItems] - 1] action] ==
        NSSelectorFromString(@"startDictation:"))
        [edit removeItemAtIndex:[edit numberOfItems] - 1];
    if ([[edit itemAtIndex:[edit numberOfItems] - 1] isSeparatorItem])
        [edit removeItemAtIndex:[edit numberOfItems] - 1];

    //  Start the Python backend.
    if (saltoInit() == 0) {
        dispatch_queue_t queue = dispatch_queue_create("com.mediavec.OpenSALTO.python", NULL);
        if (queue) {
            dispatch_set_finalizer_f(queue, &saltoEnd);
            dispatch_async(queue, ^{ [[consoleController console] run]; });
        }
    } else {
        NSLog(@"Failed to initialize Python interpreter");
        dispatch_async(dispatch_get_main_queue(),
                       ^{
                           NSAlert *alert = [[NSAlert alloc] init];
                           [alert setAlertStyle:NSCriticalAlertStyle];
                           [alert setMessageText:@"Failed to initialize Python"];
                           [alert setInformativeText:@"OpenSALTO failed to initialize the Python interpreter, and needs to quit."];
                           [alert addButtonWithTitle:@"OK"];
                           [alert runModal];
                           [alert release];
                           [NSApp terminate:self];
                       });
    }
    
    //  Show the main window.
    [mainWindowController showWindow: self];
}

- (IBAction)showConsoleWindow:(id)sender {
    [consoleController showWindow:sender];
}

@end
