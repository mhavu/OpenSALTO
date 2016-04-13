#
#  threshold.py
#  OpenSALTO
#
#  A filter that creates events for the sections of the channel
#  that are in the given range.
#
#  Copyright 2015, 2016 Marko Havu. Released under the terms of
#  GNU General Public License version 3 or later.
#

import salto, math, warnings
import numpy as np

class Plugin(salto.Plugin):
    """OpenSALTO filter plugin for creating events where values
       are in given range or above given threshold"""
    def __init__(self, manager):
        super(Plugin, self).__init__(manager)
        inputs = [('channelTable', 'S', 1, 1),
                  ('lower', 'f', "lower threshold", -np.inf),
                  ('upper', 'f', "upper threshold", np.inf),
                  ('includelower', 'i', "include lower threshold", 1),
                  ('includeupper', 'i', "include upper threshold", 0),
                  ('minduration', 'f', "minimum event duration (s)", 0.0),
                  ('minbreak', 'f', "minimum time between events (s)", 0.0)]
        self.registerComputation('threshold',
                                 self._threshold,
                                 inputs = inputs,
                                 outputs = [('channelTable', 'S', 0, 0)])
    def _position(self, data, lower, upper, includelower, includeupper):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            if (includelower and includeupper):
                result = np.where((data >= lower) & (data <= upper))
            elif includeupper:
                result = np.where((data > lower) & (data <= upper))
            elif includelower:
                result = np.where((data >= lower) & (data < upper))
            else:
                result = np.where((data > lower) & (data < upper))
        return result[0]
    def _createEvent(self, channel, startpos, endpos = None, duration = None):
        neither = endpos is None and duration is None
        both = endpos is not None and duration is not None
        if neither or both:
            raise ValueError("Specify either endpos or duration, but not both.")
        start = channel.timecodes(startpos, startpos)[0]
        if duration:
            end = start + duration
        else:
            end = channel.timecodes(endpos, endpos)[0]
        event = salto.Event(type = salto.CALCULATED_EVENT,
                            subtype = 'threshold',
                            start_sec = int(start),
                            start_nsec = int(math.fmod(start, 1.0) * 1e9),
                            end_sec = int(end),
                            end_nsec = int(math.fmod(end, 1.0) * 1e9))
        return event
    def _removeShortBreaks(self, events, minbreak):
        event1 = events[0]
        rest = events[1:]
        for event2 in rest:
            if (event2.start() - event1.end()).total_seconds() < minbreak:
                event1.end_sec = event2.end_sec
                event1.end_nsec = event2.end_nsec
            else:
                break
        return event1
    def _threshold(self, inputs):
        iChannels = salto.channelTables[inputs['channelTable']].channels
        for channel in iChannels.values():
            lower = (inputs['lower'] - channel.offset) / channel.scale
            upper = (inputs['upper'] - channel.offset) / channel.scale
            positions = self._position(channel.data, lower, upper,
                                       inputs['includelower'],
                                       inputs['includeupper'])
            if positions.size > 0:
                # Recode the positions as slices.
                starts = np.insert(np.where(np.diff(positions) != 1)[0] + 1, 0, 0)
                lengths = np.diff(np.append(starts, len(positions)) - 1)
                positions = list(zip(positions[starts], lengths))
            fill = 0
            events = []
            for start, length in positions:
                # Create events.
                extra = 0
                while (fill < channel.fills.size) and (channel.fills[fill]['pos'] < start + length):
                    if channel.fills[fill]['pos'] == start + length - 1:
                        extra = channel.fills[fill]['len']
                    fill += 1
                duration = (length + extra) / channel.samplerate
                events.append(self._createEvent(channel, start, duration = duration))
            if inputs['minduration']:
                # Make sure all events are at least minduration in length.
                valid = [i for i, e in enumerate(events)
                         if e.duration() >= inputs['minduration']]
                events = [events[slice(i1, i2)]
                          for i1, i2 in zip(valid, valid[1:] + [len(events)])]
                if inputs['minbreak']:
                    # Extend events at least minduration in length, for each
                    # event less than minbreak apart.
                    events = [self._removeShortBreaks(elist, inputs['minbreak'])
                              for elist in events]
            if events and inputs['minbreak']:
                # Merge events that are less than minbreak apart.
                rest = events[:]
                events = []
                while rest:
                    events.append(self._removeShortBreaks(rest, inputs['minbreak']))
                    rest = [e for e in rest if e.end() > events[-1].end()]
            for e in events:
                channel.events.add(e)
        return {}
