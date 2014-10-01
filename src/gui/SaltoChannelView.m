//
//  SaltoChannelView.m
//  OpenSALTO GUI
//
//  Copyright 2014 Marko Havu. Released under the terms of
//  GNU General Public License version 3 or later.
//

#import "SaltoChannelView.h"
#import "SaltoChannelWrapper.h"
#import "SaltoGuiDelegate.h"

@implementation SaltoChannelView

- (void)viewDidMoveToSuperview {
    SaltoChannelView *linkedView = self.superview ? self : nil;
    [self.objectValue setView:linkedView];
}

- (void)drawRect:(NSRect)rect {
    // TODO: Only draw rect.
    [self.objectValue drawInContext:NSGraphicsContext.currentContext.graphicsPort];
}

@end


@implementation SaltoXScaleView

- (void)awakeFromNib {
    _frame.size.height = 26;
}

- (void)drawRect:(NSRect)rect {
    [[NSColor whiteColor] set];
    [NSBezierPath fillRect:rect];
    [[NSColor blackColor] set];
    
    SaltoGuiDelegate *appDelegate = [NSApp delegate];
    NSDateFormatter *dateFormatter = [[NSDateFormatter alloc] init];
        if (appDelegate.alignment == SaltoAlignCalendarDate) {
            NSDate *start = [NSDate dateWithTimeIntervalSince1970:appDelegate.visibleRangeStart];
            [dateFormatter setDateFormat:@"yyyy-MM-dd HH:mm"];
            NSString *dateString = [dateFormatter stringFromDate:start];
            [dateString drawAtPoint:NSMakePoint(60.0, 5.0) withAttributes:nil];
        } else if (appDelegate.alignment == SaltoAlignTimeOfDay) {
            NSDate *start = [NSDate dateWithTimeIntervalSince1970:appDelegate.visibleRangeStart];
            [dateFormatter setDateFormat:@"yyyy-MM-dd HH:mm"];
            NSString *dateString = [dateFormatter stringFromDate:start];
            [dateString drawAtPoint:NSMakePoint(60.0, 5.0) withAttributes:nil];
        } else {
            NSDate *start = [NSDate dateWithTimeIntervalSince1970:appDelegate.visibleRangeStart];
            [dateFormatter setDateFormat:@"yyyy-MM-dd HH:mm"];
            NSString *dateString = [dateFormatter stringFromDate:start];
            [dateString drawAtPoint:NSMakePoint(60.0, 5.0) withAttributes:nil];
        }
    NSBezierPath *path = [[NSBezierPath alloc] init];
    [path setLineWidth:1.0];
    [path moveToPoint:NSMakePoint(50.0, 0.0)];
    [path lineToPoint:NSMakePoint(50.0, 5.0)];
    [path closePath];
    [NSGraphicsContext.currentContext setShouldAntialias:NO];
    [path stroke];
    [NSGraphicsContext.currentContext setShouldAntialias:YES];
    [path release];
    [dateFormatter release];
}

@end


@implementation SaltoYScaleView

- (void)viewDidMoveToSuperview {
    SaltoYScaleView *linkedView = self.superview ? self : nil;
    [self.objectValue setScaleView:linkedView];
}

- (void)drawRect:(NSRect)rect {
    [self.objectValue drawScaleInContext:NSGraphicsContext.currentContext.graphicsPort];
}

@end