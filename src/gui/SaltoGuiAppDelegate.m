//
//  AppDelegate.m
//  OpenSalto GUI
//
//  Copyright 2014 Marko Havu. Released under the terms of
//  GNU General Public License version 3 or later.
//

#import "SaltoGuiAppDelegate.h"

@implementation SaltoGuiAppDelegate

@synthesize window;

- (void)dealloc
{
    [super dealloc];
}

- (void)applicationDidFinishLaunching:(NSNotification *)aNotification
{
    // "Start Dictation..." and "Special Characters..." are not relevant to this application.
    // Remove these automatically added entries from the Edit menu.
    NSMenu *edit = [[[[NSApplication sharedApplication] mainMenu] itemWithTitle:@"Edit"] submenu];
    if ([[edit itemAtIndex:[edit numberOfItems] - 1] action] ==
        NSSelectorFromString(@"orderFrontCharacterPalette:"))
        [edit removeItemAtIndex:[edit numberOfItems] - 1];
    if ([[edit itemAtIndex:[edit numberOfItems] - 1] action] ==
        NSSelectorFromString(@"startDictation:"))
        [edit removeItemAtIndex:[edit numberOfItems] - 1];
    if ([[edit itemAtIndex:[edit numberOfItems] - 1] isSeparatorItem])
        [edit removeItemAtIndex:[edit numberOfItems] - 1];
}

@end
