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

@interface SaltoGuiDelegate() {
    NSTimeInterval _maxVisibleRange;
}

@end


@implementation SaltoGuiDelegate

static const float zoomFactor = 1.3;

- (NSTimeInterval)range {
    return _rangeEnd - _rangeStart;
}

- (void)setVisibleRange:(NSTimeInterval)newRange {
    if (newRange > _maxVisibleRange)
        newRange = _maxVisibleRange;
    if (newRange >= 0.0) {
        _visibleRange = newRange;
        _zoomedIn = (_visibleRange < _maxVisibleRange);
        [self.scrollView.documentView setNeedsDisplay];
    } else {
        _visibleRange = 0.0;
        NSLog(@"Visible range %f is negative", newRange);
    }
}


- (instancetype)init {
    self = [super init];
    if (self) {
        _consoleController = [[SaltoConsoleController alloc] init];
        _channelArray = [[NSMutableArray alloc] init];
        _alignment = SaltoAlignStartTime;
        _rangeStart = INFINITY;
        _rangeEnd = -INFINITY;
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
        _rangeStart = INFINITY;
        _rangeEnd = -INFINITY;
        _maxVisibleRange = 0.0;
        _visibleRange = 0.0;
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
    [[[sender menu] itemWithTag:_alignment] setState:NSOffState];
    switch ([sender tag]) {
        case SaltoAlignStartTime:
            _alignment = SaltoAlignStartTime;
            _maxVisibleRange = 0.0;
            _visibleRange = 0.0;
            for (SaltoChannelWrapper *channel in _channelArray) {
                if (channel.duration > _maxVisibleRange) {
                    _maxVisibleRange = channel.duration;
                    _visibleRange = _maxVisibleRange;
                }
                channel.visibleRangeStart = 0.0;
            }
            break;
        case SaltoAlignCalendarDate:
            _alignment = SaltoAlignCalendarDate;
            _maxVisibleRange = self.range;
            _visibleRange = _maxVisibleRange;
            for (SaltoChannelWrapper *channel in _channelArray) {
                channel.visibleRangeStart = _rangeStart;
            }
            break;
        case SaltoAlignTimeOfDay:
            _alignment = SaltoAlignTimeOfDay;
            _maxVisibleRange = 86400.0;
            _visibleRange = _maxVisibleRange;
            for (SaltoChannelWrapper *channel in _channelArray) {
                channel.visibleRangeStart = 0.0;
            }
            break;
        default:
            NSLog(@"Unknown alignment mode %ld", (long)[sender tag]);
    }
    [sender setState:NSOnState];
    [self.scrollView.documentView setNeedsDisplay];
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
    if (channel.startTime < _rangeStart || channel.endTime > _rangeEnd) {
        _rangeStart = channel.startTime;
        _rangeEnd = channel.endTime;
        if (_alignment == SaltoAlignCalendarDate) {
            _maxVisibleRange = self.range;
        } else if (_alignment == SaltoAlignStartTime && channel.duration > _maxVisibleRange) {
            _maxVisibleRange = channel.duration;
            if (!_zoomedIn) {
                _visibleRange = _maxVisibleRange;
            }
        }
    }
    
    [self.scrollView.documentView setNeedsDisplay];
}

- (void)removeChannel:(SaltoChannelWrapper *)channel {
    NSTimeInterval __block startTime = INFINITY;
    NSTimeInterval __block endTime = -INFINITY;
    double __block maxDuration = 0.0;
    [_channelArray enumerateObjectsUsingBlock:^(SaltoChannelWrapper *obj, NSUInteger idx, BOOL *stop) {
        if (obj.channel == channel.channel) {
            [self willChangeValueForKey:@"channelArray"];
            [_channelArray removeObjectAtIndex:idx];
            [self didChangeValueForKey:@"channelArray"];
        } else {
            if (obj.startTime < startTime) {
                startTime = obj.startTime;
            }
            if (obj.endTime > endTime) {
                endTime = obj.endTime;
            }
            if (obj.duration > maxDuration) {
                maxDuration = obj.duration;
            }
        }
    }];
    if (_rangeStart != startTime || _rangeEnd != endTime) {
        _rangeStart = startTime;
        _rangeEnd = endTime;
        if (_alignment == SaltoAlignCalendarDate) {
            _maxVisibleRange = self.range;
        }
    }
    if (_alignment == SaltoAlignStartTime && _maxVisibleRange != maxDuration) {
        _maxVisibleRange = maxDuration;
    }
    if (!_zoomedIn) {
        _visibleRange = _maxVisibleRange;
    }
    [self.scrollView.documentView setNeedsDisplay];
}

- (IBAction)zoomIn:(id)sender {
    // Keep the beginning of the visible range stationary when zooming in.
    self.visibleRange /= zoomFactor;
    [self.scrollView.documentView setNeedsDisplay];
}

- (IBAction)zoomOut:(id)sender {
    // Keep the beginning of the visible range stationary when zooming out.
    self.visibleRange *= zoomFactor;
    [self.scrollView.documentView setNeedsDisplay];
}

- (IBAction)showAll:(id)sender {
    self.visibleRange = _maxVisibleRange;
    [self.scrollView.documentView setNeedsDisplay];
}

- (IBAction)interruptExecution:(id)sender {
    saltoGuiInterrupt();
}

@end
