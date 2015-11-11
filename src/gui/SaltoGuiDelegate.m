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

@synthesize visibleRangeStart = _visibleRangeStart;
@synthesize visibleRangeEnd = _visibleRangeEnd;

static const double zoomFactor = 1.3;

- (NSTimeInterval)range {
    return _rangeEnd - _rangeStart;
}

- (NSTimeInterval)visibleRange {
    return _visibleRangeEnd - _visibleRangeStart;
}

- (void)setVisibleRangeStart:(NSTimeInterval)start {
    [self setVisibleRangeStart:start end:(start + self.visibleRange)];
}

- (void)setVisibleRangeEnd:(NSTimeInterval)end {
    [self setVisibleRangeStart:(end - self.visibleRange) end:end];
}

- (void)setVisibleRangeStart:(NSTimeInterval)start end:(NSTimeInterval)end {
    double position;
    if (_alignment == SaltoAlignCalendarDate) {
        _visibleRangeStart = MAX(start, _rangeStart);
        _visibleRangeEnd = MIN(end, _rangeEnd);
        position = (_visibleRangeStart - _rangeStart) / (_maxVisibleRange - self.visibleRange);
    } else {
        _visibleRangeStart = MAX(start, 0.0);
        _visibleRangeEnd = MIN(end, start + _maxVisibleRange);
        position = _visibleRangeStart / (_maxVisibleRange - self.visibleRange);
    }
    _zoomedIn = (self.visibleRange < _maxVisibleRange);
    if (_zoomedIn) {
        _scrollerHeightConstraint.constant = [NSScroller scrollerWidthForControlSize:NSRegularControlSize scrollerStyle:NSScrollerStyleOverlay];
        [_scroller setKnobProportion:(self.visibleRange / _maxVisibleRange)];
        [_scroller setDoubleValue:position];
        [_scroller setEnabled:YES];
    } else {
        _scrollerHeightConstraint.constant = 0.0;
    }
    [self updateGraphsWithReload:NO];
}

- (void)moveVisibleRangeToScrollerPosition:(double)position {
    if (self.alignment == SaltoAlignCalendarDate) {
        self.visibleRangeStart = _rangeStart + position * (_maxVisibleRange - self.visibleRange);
    } else {
        self.visibleRangeStart = position * (_maxVisibleRange - self.visibleRange);
    }
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

- (void)updateGraphsWithReload:(BOOL)reload {
    for (SaltoChannelWrapper *channel in _channelArray) {
        if (channel.view) {
            [channel setupPlot];
            if (reload) {
                [channel.graph reloadData];
            }
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
            if (_alignment == SaltoAlignCalendarDate) {
                _visibleRangeStart = _rangeStart;
                _visibleRangeEnd = _rangeEnd;
            } else {
                _visibleRangeStart = 0.0;
                _visibleRangeEnd = _maxVisibleRange;
            }
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
        _visibleRangeStart = 0.0;
        _visibleRangeEnd = 0.0;
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
            if (_alignment == SaltoAlignCalendarDate) {
                _visibleRangeStart = _rangeStart;
                _visibleRangeEnd = _rangeEnd;
            } else {
                _visibleRangeStart = 0.0;
                _visibleRangeEnd = _maxVisibleRange;
            }
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
            for (SaltoChannelWrapper *channel in _channelArray) {
                if (channel.duration > _maxVisibleRange) {
                    _maxVisibleRange = channel.duration;
                }
                channel.visibleRangeStart = channel.startTime;
            }
            [self setVisibleRangeStart:0.0 end:_maxVisibleRange];
            break;
        case SaltoAlignCalendarDate:
            _alignment = SaltoAlignCalendarDate;
            _maxVisibleRange = self.range;
            [self setVisibleRangeStart:_rangeStart end:_rangeEnd];
            for (SaltoChannelWrapper *channel in _channelArray) {
                channel.visibleRangeStart = _rangeStart;
            }
            break;
        case SaltoAlignTimeOfDay:
            _alignment = SaltoAlignTimeOfDay;
            _maxVisibleRange = 86400.0;
            [self setVisibleRangeStart:0.0 end:_maxVisibleRange];
            for (SaltoChannelWrapper *channel in _channelArray) {
                // TODO: fix
                channel.visibleRangeStart = 0.0;
            }
            break;
        default:
            NSLog(@"Unknown alignment mode %ld", (long)[sender tag]);
    }
    [sender setState:NSOnState];
    [self updateGraphsWithReload:YES];
}

- (IBAction)openDocument:(id)sender {
    NSOpenPanel *panel = [NSOpenPanel openPanel];
    // TODO: Get file types from plugin manager.
    NSArray *allowedTypes = [NSArray arrayWithObjects:@"csv", @"dat", @"ats", nil];
    [panel setCanChooseDirectories:YES];
    [panel setAllowedFileTypes:allowedTypes];
    [panel setDelegate:(id)self];
    [panel beginSheetModalForWindow:[NSApp mainWindow] completionHandler:^(NSInteger result) {
        if (result == NSFileHandlingPanelOKButton) {
            for (NSURL *file in [panel URLs]) {
                NSString *command = [NSString stringWithFormat:@"salto.open(\"%@\")", file.path];
                [_consoleController insertInput:command];
                [_consoleController.console execute:command];
                [[NSDocumentController sharedDocumentController] noteNewRecentDocumentURL:file];
            }
        }
    }];
}

- (IBAction)zoomIn:(id)sender {
    NSTimeInterval newVisibleRange = self.visibleRange / zoomFactor;
    if (self.isZoomedIn && _scroller.doubleValue == 0.0) {
        // Stay zoomed in on the beginning of the visible range.
        [self setVisibleRangeEnd:(self.visibleRangeStart + newVisibleRange)];
    } else if (self.isZoomedIn && _scroller.doubleValue == 1.0) {
        // Stay zoomed in on the end of the visible range.
        [self setVisibleRangeStart:(self.visibleRangeStart + self.visibleRange - newVisibleRange)];
    } else {
        // Keep the center of the visible range stationary when zooming in.
        NSTimeInterval start = self.visibleRangeStart + (self.visibleRange - newVisibleRange) / 2.0;
        [self setVisibleRangeStart:start end:(start + newVisibleRange)];
    }
}

- (IBAction)zoomOut:(id)sender {
    if (self.isZoomedIn) {
        NSTimeInterval newVisibleRange = self.visibleRange * zoomFactor;
        if ((self.alignment == SaltoAlignCalendarDate &&
             self.visibleRangeStart == self.rangeStart) ||
            self.visibleRangeStart == 0.0) {
            // If we are zoomed in on the beginning of the visible range,
            // keep it stationary when zooming out.
            [self setVisibleRangeStart:self.visibleRangeStart
                                   end:(self.visibleRangeStart + newVisibleRange)];
        } else if ((self.alignment == SaltoAlignCalendarDate &&
                    self.visibleRangeEnd == self.rangeEnd) ||
                   self.visibleRangeEnd == _maxVisibleRange) {
            // If we are zoomed in on the end of the visible range,
            // keep it stationary when zooming out.
            NSTimeInterval start = self.visibleRangeStart + self.visibleRange - newVisibleRange;
            [self setVisibleRangeStart:start end:self.visibleRangeEnd];
        } else {
            // Elsewhere keep the center of the visible range stationary when zooming out.
            NSTimeInterval start = self.visibleRangeStart + (self.visibleRange - newVisibleRange) / 2.0;
            [self setVisibleRangeStart:start end:(start + newVisibleRange)];
        }
    }
}

- (IBAction)showAll:(id)sender {
    if (self.alignment == SaltoAlignCalendarDate) {
        [self setVisibleRangeStart:_rangeStart end:_rangeEnd];
    } else {
        [self setVisibleRangeStart:0.0 end:_maxVisibleRange];
    }
}

- (IBAction)interruptExecution:(id)sender {
    saltoGuiInterrupt();
}

- (IBAction)scrollAction:(id)sender {
    double position = 0.0;
    switch (self.scroller.hitPart) {
        case NSScrollerNoPart:
            break;
        case NSScrollerDecrementPage:
            position = MAX(self.scroller.doubleValue - self.visibleRange / (_maxVisibleRange - self.visibleRange), 0.0);
            self.scroller.doubleValue = position;
            break;
        case NSScrollerIncrementPage:
            position = MIN(self.scroller.doubleValue + self.visibleRange / (_maxVisibleRange - self.visibleRange), 1.0);
            self.scroller.doubleValue = position;
            break;
        case NSScrollerKnob:
        case NSScrollerKnobSlot:
            position = self.scroller.doubleValue;
            break;
        default:
            NSLog(@"unsupported scroller part code %lu", (unsigned long)self.scroller.hitPart);
    }
    [self moveVisibleRangeToScrollerPosition:position];
}

- (IBAction)refresh:(id)sender {
    NSTableView *tableView = self.scrollView.documentView;
    NSRange rowsInRect = [tableView rowsInRect:self.scrollView.contentView.bounds];
    NSInteger column = [tableView columnWithIdentifier:@"Channel"];
    for (NSInteger row = rowsInRect.location; row < rowsInRect.length; row++) {
        SaltoChannelView *view = [tableView viewAtColumn:column row:row makeIfNecessary:NO];
        [view refreshView];
    }
}

- (IBAction)useTool:(id)sender {
    if ([[(NSToolbarItem *)sender itemIdentifier] isEqual:@"saltoMarkInactivity"]) {
        NSString *command = @"chTables = salto.channelTables\nm = chTables['main']\npm = salto.pluginManager\npm.compute('inclination', {'channelTable': 'main'})\nchTables['x'] = salto.ChannelTable()\nchTables['x'].add('X inclination', chTables['inclination'].channels['X inclination'])\nchTables['z'] = salto.ChannelTable()\nchTables['z'].add('Z inclination', chTables['inclination'].channels['Z inclination'])\npm.compute('threshold', {'channelTable': 'x', 'upper': 0.75*np.pi, 'minduration': 5.0})\npm.compute('threshold', {'channelTable': 'z', 'upper': 0.25*np.pi, 'minduration': 5.0})\nchTables['inclination'].channels['norm'].events = salto.Channel.eventUnion(chTables['inclination'].channels['X inclination'], chTables['inclination'].channels['Z inclination'])\nm.add('norm', chTables['inclination'].channels['norm'])\nevents = list(m.channels['norm'].events)\nevents.sort()\nprint(\"%s  1  %f\" % (m.channels['norm'].start().isoformat(' '), (events[0].start() - m.channels['norm'].start()).total_seconds()))\nprint(\"%s  0  %f\" % (events[0].start().isoformat(' '), events[0].duration()))\nfor i, e in enumerate(events[1:]):\n  print(\"%s  1  %f\" % (events[i].end().isoformat(' '), (e.start() - events[i].end()).total_seconds()))\n  print(\"%s  0  %f\" % (e.start().isoformat(' '), e.duration()))\nprint(\"%s  1  %f\" % (e.end().isoformat(' '), (m.channels['norm'].end() - e.end()).total_seconds()))";//\ndel(chTables['x'])\ndel(chTables['z'])
        [_consoleController.console execute:command];
    }
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