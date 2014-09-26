//
//  SaltoGuiDelegate.m
//  OpenSALTO GUI
//
//  Copyright 2014 Marko Havu. Released under the terms of
//  GNU General Public License version 3 or later.
//

#import "SaltoGuiDelegate.h"
#import "SaltoChannelWrapper.h"
#import "saltoGui.h"

@implementation SaltoGuiDelegate

@synthesize consoleController;
@synthesize channelArray;
@synthesize queue;
@synthesize xRange;
@synthesize xVisibleRangeStart;
@synthesize xVisibleRangeEnd;
@synthesize alignment;

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

- (void)applicationDidFinishLaunching:(NSNotification *)notification {
    // "Start Dictation..." and "Special Characters..." are not relevant to this application.
    // Remove these automatically added entries from the Edit menu.
    NSMenu *edit = [[[NSApp mainMenu] itemWithTitle:@"Edit"] submenu];
    if ([[edit itemAtIndex:edit.numberOfItems - 1] action] ==
        NSSelectorFromString(@"orderFrontCharacterPalette:"))
        [edit removeItemAtIndex:edit.numberOfItems - 1];
    if ([[edit itemAtIndex:edit.numberOfItems - 1] action] ==
        NSSelectorFromString(@"startDictation:"))
        [edit removeItemAtIndex:edit.numberOfItems - 1];
    if ([[edit itemAtIndex:edit.numberOfItems - 1] isSeparatorItem])
        [edit removeItemAtIndex:edit.numberOfItems - 1];

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

- (void)tableView:(NSTableView *)view didAddRowView:(NSTableRowView *)rowView forRow:(NSInteger)row {
    [view setBackgroundColor:[NSColor whiteColor]];
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

- (IBAction)openDocument:(id)sender {
    NSOpenPanel *panel = [NSOpenPanel openPanel];
    // TODO: [panel setCanChooseDirectories:YES];
    // TODO: [panel setAllowedFileTypes:(NSArray *)types];
    [panel setDelegate:(id)self];
    [panel beginSheetModalForWindow:[NSApp mainWindow] completionHandler:^(NSInteger result) {
        if (result == NSFileHandlingPanelOKButton) {
            for (NSURL *file in [panel URLs]) {
                NSString *command = [NSString stringWithFormat:@"salto.open(\"%@\")", file.path];
                [consoleController insertInput:command];
                [consoleController.console execute:command];
                // TODO: [[NSDocumentController sharedDocumentController] noteNewRecentDocumentURL:file];
            }
        }
    }];
}

- (BOOL)application:(NSApplication *)theApplication openFile:(NSString *)filename {
    NSString *command = [NSString stringWithFormat:@"salto.open(\"%@\")", filename];
    [consoleController insertInput:command];
    [consoleController.console execute:command];

    return YES;  // TODO: Return no, if there is an error.
}

- (void)addChannel:(SaltoChannelWrapper *)channel {
    // TODO: Set the NSDate objects.
    // TODO: Set SaltoChannelView heights.
    [self willChangeValueForKey:@"channelArray"];
    [channelArray addObject:channel];
    [self didChangeValueForKey:@"channelArray"];
}

- (void)removeChannel:(SaltoChannelWrapper *)channel {
    [channelArray enumerateObjectsUsingBlock:^(SaltoChannelWrapper *obj, NSUInteger idx, BOOL *stop) {
        if (obj.channel == channel.channel) {
            [self willChangeValueForKey:@"channelArray"];
            [channelArray removeObjectAtIndex:idx];
            [self didChangeValueForKey:@"channelArray"];
            *stop = YES;
        }
    }];
}


@end
