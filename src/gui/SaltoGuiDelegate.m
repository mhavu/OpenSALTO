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

@synthesize rangeStart = _rangeStart;
@synthesize rangeEnd = _rangeEnd;
@synthesize visibleRangeStart = _visibleRangeStart;
@synthesize visibleRangeEnd = _visibleRangeEnd;

- (void)setRangeStart:(NSTimeInterval)start end:(NSTimeInterval)end {
    if (end > start) {
        _rangeStart = start;
        _rangeEnd = end;
    } else {
        _rangeStart = end;
        _rangeEnd = start;
    }
    if (_visibleRangeStart < start || _visibleRangeEnd > end)
        [self setVisibleRangeStart:_visibleRangeStart end:_visibleRangeEnd];
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

- (void)setVisibleRangeStart:(NSTimeInterval)start end:(NSTimeInterval)end {
    if (self.range > 0 && end > start) {
        _visibleRangeStart = (start >= _rangeStart) ? start : _rangeStart;
        _visibleRangeEnd = (end <= _rangeEnd) ? end : _rangeEnd;
        NSTableColumn *column = [_scrollView.documentView tableColumnWithIdentifier:@"Channel"];
        _pixelsPerSecond = column.width / (end - start);
        // Adjust ruler.
        [NSRulerView registerUnitWithName:@"Seconds" abbreviation:@"s" unitToPointsConversionFactor:_pixelsPerSecond stepUpCycle:@[@2.0] stepDownCycle:@[@0.5, @0.2]];
        [_scrollView.horizontalRulerView setMeasurementUnits:@"Seconds"];
        // Adjust column width.
        [column setWidth:self.range * _pixelsPerSecond];
        // Update the view in background.
        NSGraphicsContext *gc = [[NSApp mainWindow] graphicsContext];
        NSInteger c = [_scrollView.documentView columnWithIdentifier:@"Channel"];
        [_scrollView.documentView enumerateAvailableRowViewsUsingBlock:^(NSTableRowView *rowView, NSInteger row){
            SaltoChannelView *view = [rowView viewAtColumn:c];
            SaltoChannelWrapper *channel = view.objectValue;
            channel.operation = [NSBlockOperation blockOperationWithBlock:
                                 ^{ [channel drawLayerForContext:gc.graphicsPort frame:view.frame]; }];
        }];
        // TODO: Set scrollers

        [_scrollView.documentView setNeedsDisplay:YES];
    }
}

- (double)pixelsPerSecond {
    return _pixelsPerSecond;
}

- (NSTimeInterval)visibleRangeStart {
    return _visibleRangeStart;
}

- (NSTimeInterval)visibleRangeEnd {
    return _visibleRangeEnd;
}

- (NSTimeInterval)visibleRange {
    return _visibleRangeEnd - _visibleRangeStart;
}


- (instancetype)init {
    self = [super init];
    if (self) {
        _consoleController = [[SaltoConsoleController alloc] init];
        _channelArray = [[NSMutableArray alloc] init];
        _opQueue = [[NSOperationQueue alloc] init];
        _opQueue.name = @"Channel drawing queue";
    }

    return self;
}

- (void)dealloc {
    [_consoleController release];
    [_channelArray release];
    [_scrollView release];
    [_opQueue release];
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

    // Set up the horizontal ruler.
    [_scrollView setHasHorizontalRuler:YES];
    NSInteger column = [_scrollView.documentView columnWithIdentifier:@"Channel"];
    [_scrollView.horizontalRulerView setOriginOffset:NSMinX([_scrollView.documentView rectOfColumn:column])];
    [_scrollView.horizontalRulerView setReservedThicknessForMarkers:0.0];
    [_scrollView.horizontalRulerView setReservedThicknessForAccessoryView:16.0];

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
    [_scrollView setRulersVisible:YES];
    [view setBackgroundColor:[NSColor whiteColor]];
    NSGraphicsContext *gc = [[NSApp mainWindow] graphicsContext];
    NSInteger c = [_scrollView.documentView columnWithIdentifier:@"Channel"];
    SaltoChannelView *channelView = [rowView viewAtColumn:c];
    SaltoChannelWrapper *channel = view.objectValue;
    channel.operation = [NSBlockOperation blockOperationWithBlock:
                         ^{ [channel drawLayerForContext:gc.graphicsPort frame:channelView.frame]; }];
}

- (void)tableView:(NSTableView *)view didRemoveRowView:(NSTableRowView *)rowView forRow:(NSInteger)row {
    if (view.numberOfRows == 0) {
        [_scrollView setRulersVisible:NO];
        [view setBackgroundColor:[NSColor gridColor]];
    }
}

- (void)tableViewColumnDidResize:(NSNotification *)notification {
    [self setVisibleRangeStart:_rangeStart end:_rangeEnd];
}

- (void)boundsDidChange:(NSNotification *)notification {
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

    return YES;  // TODO: Return no, if there is an error.
}

- (void)addChannel:(SaltoChannelWrapper *)channel {
    // TODO: Set the NSDate objects.
    // TODO: Set SaltoChannelView heights.
    [self willChangeValueForKey:@"channelArray"];
    [_channelArray addObject:channel];
    [self didChangeValueForKey:@"channelArray"];
}

- (void)removeChannel:(SaltoChannelWrapper *)channel {
    [_channelArray enumerateObjectsUsingBlock:^(SaltoChannelWrapper *obj, NSUInteger idx, BOOL *stop) {
        if (obj.channel == channel.channel) {
            [self willChangeValueForKey:@"channelArray"];
            [_channelArray removeObjectAtIndex:idx];
            [self didChangeValueForKey:@"channelArray"];
            *stop = YES;
        }
    }];
}

- (IBAction)zoomIn:(id)sender {
    // Keep the beginning of the visible range stationary when zooming in.
    NSTimeInterval interval = self.visibleRange / zoomFactor;
    [self setVisibleRangeStart:_visibleRangeStart end:_visibleRangeStart + interval];
}

- (IBAction)zoomOut:(id)sender {
    // Keep the middle of the visible range stationary when zooming out,
    // except when in the beginning or end of time range.
    NSTimeInterval delta = self.visibleRange * (zoomFactor - 1.0) / 2.0;
    NSTimeInterval start = _visibleRangeStart - delta;
    NSTimeInterval end = _visibleRangeEnd + delta;
    if (start < _rangeStart) {
        end += _rangeStart - start;
        start = _rangeStart;
    } else if (end > _rangeEnd) {
        start -= end - _rangeEnd;
        end = _rangeEnd;
    }
    [self setVisibleRangeStart:start end:end];
}

- (IBAction)showAll:(id)sender {
    [self setVisibleRangeStart:_rangeStart end:_rangeEnd];
}

- (IBAction)interruptExecution:(id)sender {
    saltoGuiInterrupt();
}

@end
