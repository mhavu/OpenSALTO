//
//  SaltoGuiDelegate.m
//  OpenSALTO GUI
//
//  Copyright 2014 Marko Havu. Released under the terms of
//  GNU General Public License version 3 or later.
//

#import "SaltoGuiDelegate.h"
#import "SaltoChannelViewController.h"
#import "saltoGui.h"

@implementation SaltoGuiDelegate

@synthesize consoleController;
@synthesize channelView;
@synthesize channelArray;
@synthesize queue;

- (instancetype)init {
    self = [super init];
    if (self) {
        consoleController = [[SaltoConsoleController alloc] init];
        channelArray = [[NSMutableArray alloc] init];
    }

    return self;
}

- (void)dealloc {
    [consoleController release];
    [channelArray release];
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
    const char *saltoPyPath = [[[NSBundle mainBundle] pathForAuxiliaryExecutable:@"salto.py"] UTF8String];
    if (saltoInit(saltoPyPath, &PyInit_saltoGui) == 0) {
        queue = dispatch_queue_create("com.mediavec.OpenSALTO.python", NULL);
        if (queue) {
            dispatch_set_finalizer_f(queue, &saltoEnd);
        }
    } else {
        NSLog(@"Failed to initialize Python interpreter");
        NSAlert *alert = [[NSAlert alloc] init];
        [alert setAlertStyle:NSCriticalAlertStyle];
        [alert setMessageText:@"Failed to initialize Python"];
        [alert setInformativeText:@"OpenSALTO failed to initialize the Python interpreter, and needs to quit."];
        [alert addButtonWithTitle:@"OK"];
        [alert runModal];
        [alert release];
        [NSApp terminate:self];
    }
}

- (IBAction)showConsoleWindow:(id)sender {
    [consoleController showWindow:sender];
}

- (IBAction)toggleAlignment:(id)sender {
    if ([sender state] == NSOnState) {
        // TODO: fix
        // [channelView.delegate alignByTimeOfDay:NO];
        [sender setState:NSOffState];
    } else {
        // TODO: fix
        // [channelView.delegate alignByTimeOfDay:YES];
        [sender setState:NSOnState];
    }
}

- (void)addChannel:(Channel *)ch {
    SaltoChannelWrapper *channel = [SaltoChannelWrapper wrapperForChannel:ch];
    [channelArray addObject:channel];
}

- (void)removeChannel:(Channel *)ch {
    [channelArray enumerateObjectsUsingBlock:^(SaltoChannelWrapper *obj, NSUInteger idx, BOOL *stop) {
        if (obj.channel == ch) {
            [channelArray removeObjectAtIndex:idx];
            *stop = YES;
        }
    }];
}


@end
