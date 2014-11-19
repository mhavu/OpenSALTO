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
    double _visibleRangePosition;
}

@end


@implementation SaltoGuiDelegate

static const double zoomFactor = 1.3;

- (NSTimeInterval)range {
    return _rangeEnd - _rangeStart;
}

- (void)setVisibleRange:(NSTimeInterval)newRange {
    if (newRange > _maxVisibleRange)
        newRange = _maxVisibleRange;
    if (newRange >= 0.0) {
        _visibleRange = newRange;
        _zoomedIn = (_visibleRange < _maxVisibleRange);
    } else {
        _visibleRange = 0.0;
        NSLog(@"Visible range %f is negative", newRange);
    }
    if (_zoomedIn) {
        _scrollerHeightConstraint.constant = [NSScroller scrollerWidthForControlSize:NSRegularControlSize scrollerStyle:NSScrollerStyleOverlay];
        [_scroller setKnobProportion:(_visibleRange / _maxVisibleRange)];
        [_scroller setDoubleValue:_visibleRangePosition];
        [_scroller setEnabled:YES];
    } else {
        _scrollerHeightConstraint.constant = 0.0;
    }
    [self updateGraphs];
}


- (instancetype)init {
    self = [super init];
    if (self) {
        _consoleController = [[SaltoConsoleController alloc] init];
        _channelArray = [[NSMutableArray alloc] init];
        _alignment = SaltoAlignCalendarDate;
        _rangeStart = INFINITY;
        _rangeEnd = -INFINITY;
    }

    return self;
}

- (void)dealloc {
    [_consoleController release];
    [_channelArray release];
    [_scrollView release];
    [_scroller release];
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
    
    // Hide the horizontal scroller.
    self.scrollerHeightConstraint.constant = 0.0;
    
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

- (void)updateGraphs {
    for (SaltoChannelWrapper *channel in _channelArray) {
        if (channel.view) {
            [channel setupPlot];
        }
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

- (void)addChannel:(SaltoChannelWrapper *)channel {
    // TODO: Set SaltoChannelView heights.
    [self willChangeValueForKey:@"channelArray"];
    [_channelArray addObject:channel];
    if (channel.startTime < _rangeStart || channel.endTime > _rangeEnd) {
        _rangeStart = channel.startTime;
        _rangeEnd = channel.endTime;
        if (_alignment == SaltoAlignCalendarDate) {
            _maxVisibleRange = self.range;
        } else if (_alignment == SaltoAlignStartTime && channel.duration > _maxVisibleRange) {
            _maxVisibleRange = channel.duration;
        }
        if (!_zoomedIn) {
            _visibleRange = _maxVisibleRange;
        }
    }
    [self didChangeValueForKey:@"channelArray"];
}

- (void)removeChannel:(SaltoChannelWrapper *)channel {
    [self willChangeValueForKey:@"channelArray"];
    if ([_channelArray count] == 1) {
        [_channelArray removeAllObjects];
        _rangeStart = INFINITY;
        _rangeEnd = -INFINITY;
        _maxVisibleRange = 0.0;
        _visibleRange = 0.0;
    } else {
        NSTimeInterval __block startTime = INFINITY;
        NSTimeInterval __block endTime = -INFINITY;
        double __block maxDuration = 0.0;
        [_channelArray enumerateObjectsUsingBlock:^(SaltoChannelWrapper *obj, NSUInteger idx, BOOL *stop) {
            if (obj.channel == channel.channel) {
                [_channelArray removeObjectAtIndex:idx];
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
    }
    [self didChangeValueForKey:@"channelArray"];
}

#pragma mark - Menu actions

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
                channel.visibleRangeStart = channel.startTime;
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
                // TODO: fix
                channel.visibleRangeStart = 0.0;
            }
            break;
        default:
            NSLog(@"Unknown alignment mode %ld", (long)[sender tag]);
    }
    _visibleRangePosition = 0.0;
    [sender setState:NSOnState];
    [self updateGraphs];
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

- (IBAction)zoomIn:(id)sender {
    // Keep the beginning of the visible range stationary when zooming in.
    self.visibleRange /= zoomFactor;
}

- (IBAction)zoomOut:(id)sender {
    // Keep the beginning of the visible range stationary when zooming out.
    self.visibleRange *= zoomFactor;
}

- (IBAction)showAll:(id)sender {
    self.visibleRange = _maxVisibleRange;
}

- (IBAction)interruptExecution:(id)sender {
    saltoGuiInterrupt();
}

- (IBAction)scrollAction:(id)sender {
    switch (self.scroller.hitPart) {
        case NSScrollerNoPart:
            break;
        case NSScrollerDecrementPage:
            _visibleRangePosition = MAX(_visibleRangePosition - _visibleRange / _maxVisibleRange, 0.0);
            self.scroller.doubleValue = _visibleRangePosition;
            break;
        case NSScrollerIncrementPage:
            _visibleRangePosition = MIN(_visibleRangePosition + _visibleRange / _maxVisibleRange, 1.0);
            self.scroller.doubleValue = _visibleRangePosition;
            break;
        case NSScrollerKnob:
        case NSScrollerKnobSlot:
            _visibleRangePosition = self.scroller.doubleValue;
            break;
        default:
            NSLog(@"unsupported scroller part code %lu", (unsigned long)self.scroller.hitPart);
    }
    // TODO: Change plot range.
}

#pragma mark - Other delegate messages

- (void)tableView:(NSTableView *)view didAddRowView:(NSTableRowView *)rowView forRow:(NSInteger)row {
    [view setBackgroundColor:[NSColor whiteColor]];
}

- (void)tableView:(NSTableView *)view didRemoveRowView:(NSTableRowView *)rowView forRow:(NSInteger)row {
    if (view.numberOfRows == 0) {
        [view setBackgroundColor:[NSColor gridColor]];
        self.scrollerHeightConstraint.constant = 0.0;
    }
}

- (BOOL)application:(NSApplication *)theApplication openFile:(NSString *)filename {
    NSString *command = [NSString stringWithFormat:@"salto.open(\"%@\")", filename];
    [_consoleController insertInput:command];
    [_consoleController.console execute:command];
    
    return YES;  // TODO: Return NO if there is an error.
}

@end