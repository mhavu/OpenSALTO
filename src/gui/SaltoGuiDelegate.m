//
//  SaltoGuiDelegate.m
//  OpenSALTO GUI
//
//  Copyright 2014 Marko Havu. Released under the terms of
//  GNU General Public License version 3 or later.
//

#import "SaltoGuiDelegate.h"
#import "SaltoChannelView.h"
#import "SaltoChannelWrapper.h"
#import "saltoGui.h"

@implementation SaltoGuiDelegate

static const float zoomFactor = 1.3;

- (void)setRangeStart:(NSTimeInterval)start end:(NSTimeInterval)end {
    if (end > start) {
        _rangeStart = start;
        _rangeEnd = end;
    } else {
        _rangeStart = end;
        _rangeEnd = start;
    }
    if (!self.isZoomedIn)
        self.visibleRange = self.range;
}

- (NSTimeInterval)rangeStart {
    return _rangeStart;
}

- (NSTimeInterval)rangeEnd {
    return _rangeEnd;
}

- (NSTimeInterval)range {
    return _rangeEnd - _rangeStart;
}

- (void)setVisibleRange:(NSTimeInterval)newRange {
    if (newRange > self.range)
        newRange = self.range;
    if (newRange >= 0.0) {
        _visibleRange = newRange;
        _zoomedIn = (_visibleRange < self.range);
        [self.scrollView.documentView setNeedsDisplay:YES];
    } else {
        // TODO: log this
    }
}


- (instancetype)init {
    self = [super init];
    if (self) {
        _consoleController = [[SaltoConsoleController alloc] init];
        _channelArray = [[NSMutableArray alloc] init];
        _alignment = SaltoAlignStartTime;
    }

    return self;
}

- (void)dealloc {
    [_consoleController release];
    [_channelArray release];
    [_scrollView release];
    [super dealloc];
}


- (void)applicationDidFinishLaunching:(NSNotification *)notification {
    // Remove the automatically added items "Start Dictation..." and
    // "Special Characters..." from the Edit menu.
    NSMenu *edit = [[[NSApp mainMenu] itemWithTitle:@"Edit"] submenu];
    if ([[edit itemAtIndex:edit.numberOfItems - 1] action] ==
        NSSelectorFromString(@"orderFrontCharacterPalette:"))
        [edit removeItemAtIndex:edit.numberOfItems - 1];
    if ([[edit itemAtIndex:edit.numberOfItems - 1] action] ==
        NSSelectorFromString(@"startDictation:"))
        [edit removeItemAtIndex:edit.numberOfItems - 1];
    if ([[edit itemAtIndex:edit.numberOfItems - 1] isSeparatorItem])
        [edit removeItemAtIndex:edit.numberOfItems - 1];

    // Toggle the correct alignment mode in the Alignment menu.
    [[self.alignmentMenu itemWithTag:self.alignment] setState:NSOnState];
    
    // TODO: Snap to channels in vertical scroll.
    // Post notifications when the clip view's bounds change.
    // [_scrollView.contentView setPostsBoundsChangedNotifications:YES];
    // [NSNotificationCenter.defaultCenter addObserver:self
    //                                        selector:@selector(boundsDidChange:)
    //                                            name:NSViewBoundsDidChangeNotification
    //                                          object:scrollView.contentView];

    // Start the Python backend.
    // TODO: Use a dedicated thread instead of a GCD queue. Otherwise interrupts may not work.
    // (Use NSThread and performSelector:onThread:withObject:waitUntilDone:)
    _queue = dispatch_queue_create("com.mediavec.OpenSALTO.python", NULL);
    if (_queue) {
        NSString *saltoPyPath = [[NSBundle mainBundle] pathForAuxiliaryExecutable:@"salto.py"];
        dispatch_async(_queue,
                       ^{
                           if (saltoInit(saltoPyPath.UTF8String, &PyInit_saltoGui) == 0) {
                               dispatch_set_finalizer_f(_queue, &saltoEnd);
                           } else {
                               dispatch_async(dispatch_get_main_queue(),
                                              ^{ [self quitWithInitializationError:@"Python interpreter"]; });
                           }
                       });
    } else {
        [self quitWithInitializationError:@"Python interpreter thread"];
    }
}

- (void)quitWithInitializationError:(NSString *)string {
    NSLog(@"Failed to initialize %@", string);
    NSAlert *alert = [[NSAlert alloc] init];
    [alert setAlertStyle:NSCriticalAlertStyle];
    [alert setMessageText:@"Failed to initialize Python"];
    [alert setInformativeText:[NSString stringWithFormat:@"OpenSALTO failed to initialize the %@, and needs to quit.", string]];
    [alert addButtonWithTitle:@"OK"];
    [alert runModal];
    [alert release];
    [NSApp terminate:self];
}

- (void)tableView:(NSTableView *)view didAddRowView:(NSTableRowView *)rowView forRow:(NSInteger)row {
    [view setBackgroundColor:[NSColor whiteColor]];
}

- (void)tableView:(NSTableView *)view didRemoveRowView:(NSTableRowView *)rowView forRow:(NSInteger)row {
    if (view.numberOfRows == 0) {
        [view setBackgroundColor:[NSColor gridColor]];
    }
}

- (void)boundsDidChange:(NSNotification *)notification {
    // Snap to channels in vertical scroll.
    CGFloat y = _scrollView.contentView.bounds.origin.y;
    NSInteger row = [_scrollView.documentView rowAtPoint:_scrollView.contentView.bounds.origin];
    NSRect frame = [[_scrollView.documentView rowViewAtRow:row makeIfNecessary:NO] frame];
    CGFloat snap = (y - NSMinY(frame) > NSMaxY(frame) - y) ? NSMaxY(frame) : NSMinY(frame);
    NSLog(@"bounds changed: y = %0.0lf, (%0.0lf ... %0.0lf)", y, NSMinY(frame), NSMaxY(frame));
    [_scrollView.contentView scrollToPoint:NSMakePoint(0, snap)];
}

- (IBAction)showConsoleWindow:(id)sender {
    [_consoleController showWindow:sender];
}

- (IBAction)setAlignmentMode:(id)sender {
    [[[sender menu] itemWithTag:self.alignment] setState:NSOffState];
    switch ([sender tag]) {
        case SaltoAlignStartTime:
            self.alignment = SaltoAlignStartTime;
            self.visibleRange = 0.0;
            for (SaltoChannelWrapper *channel in self.channelArray) {
                if (channel.duration > self.visibleRange) {
                    self.visibleRange = channel.duration;
                }
                channel.visibleRangeStart = 0.0;
            }
            break;
        case SaltoAlignCalendarDate:
            self.alignment = SaltoAlignCalendarDate;
            self.visibleRange = self.range;
            for (SaltoChannelWrapper *channel in self.channelArray) {
                channel.visibleRangeStart = self.rangeStart;
            }
            break;
        case SaltoAlignTimeOfDay:
            self.alignment = SaltoAlignTimeOfDay;
            self.visibleRange = 86400.0;
            for (SaltoChannelWrapper *channel in self.channelArray) {
                channel.visibleRangeStart = 0.0;
            }
            break;
        default:
            NSLog(@"Unknown alignment mode %ld", (long)[sender tag]);
    }
    [sender setState:NSOnState];
    [[self.scrollView documentView] reloadData];
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
                [_consoleController insertInput:command];
                [_consoleController.console execute:command];
                // TODO: [[NSDocumentController sharedDocumentController] noteNewRecentDocumentURL:file];
            }
        }
    }];
}

- (BOOL)application:(NSApplication *)theApplication openFile:(NSString *)filename {
    NSString *command = [NSString stringWithFormat:@"salto.open(\"%@\")", filename];
    [_consoleController insertInput:command];
    [_consoleController.console execute:command];

    return YES;  // TODO: Return no if there is an error.
}

- (void)addChannel:(SaltoChannelWrapper *)channel {
    // TODO: Set SaltoChannelView heights.
    [self willChangeValueForKey:@"channelArray"];
    [_channelArray addObject:channel];
    [self didChangeValueForKey:@"channelArray"];
    if (channel.startTime < self.rangeStart)
        self.rangeStart = channel.startTime;
    if (channel.endTime > self.rangeEnd)
        self.rangeEnd = channel.endTime;
    [[self.scrollView documentView] reloadData];
}

- (void)removeChannel:(SaltoChannelWrapper *)channel {
    NSTimeInterval __block startTime = INFINITY;
    NSTimeInterval __block endTime = -INFINITY;
    [_channelArray enumerateObjectsUsingBlock:^(SaltoChannelWrapper *obj, NSUInteger idx, BOOL *stop) {
        if (obj.channel == channel.channel) {
            [self willChangeValueForKey:@"channelArray"];
            [_channelArray removeObjectAtIndex:idx];
            [self didChangeValueForKey:@"channelArray"];
        } else {
            if (obj.startTime < startTime)
                startTime = channel.startTime;
            if (obj.endTime > endTime)
                endTime = channel.endTime;
        }
    }];
    self.rangeStart = startTime;
    self.rangeEnd = endTime;
    if (self.visibleRange > self.range)
        self.visibleRange = self.range;
    [[self.scrollView documentView] reloadData];
}

- (IBAction)zoomIn:(id)sender {
    // Keep the beginning of the visible range stationary when zooming in.
    self.visibleRange /= zoomFactor;
    [[self.scrollView documentView] reloadData];
}

- (IBAction)zoomOut:(id)sender {
    // Keep the beginning of the visible range stationary when zooming out.
    self.visibleRange *= zoomFactor;
    [[self.scrollView documentView] reloadData];
}

- (IBAction)showAll:(id)sender {
    self.visibleRange = self.range;
    [[self.scrollView documentView] reloadData];
}

- (IBAction)interruptExecution:(id)sender {
    saltoGuiInterrupt();
}

@end
