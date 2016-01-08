#
#  threshold.py
#  OpenSALTO
#
#  A filter that creates events for the sections of the channel
#  that are in the given range.
#
#  Copyright 2015 Marko Havu. Released under the terms of
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
                  ('lower', 'f', 'lower threshold', -np.inf),
                  ('upper', 'f', 'upper threshold', np.inf),
                  ('includelower', 'i', 'include lower threshold', 1),
                  ('includeupper', 'i', 'include upper threshold', 0),
                  ('minduration', 'f', 'minimum event duration (s)', None),
                  ('minbreak', 'f', 'minimum time between events (s)', None)]
        self.registerComputation("threshold",
                                 self._threshold,
                                 inputs = inputs,
                                 outputs = [('channelTable', 'S', 0, 0)])
    def _position(self, data, lower, upper, includelower, includeupper):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            if (includelower and includeupper):
                result = np.where((data >= lower) & (data <= upper))
            elif includeupper:
                result = np.where((data > lower) & (data <= upper))
            elif includelower:
                result = np.where((data >= lower) & (data < upper))
            else:
                result = np.where((data > lower) & (data < upper))
        return result[0]
    def _addEvent(self, channel, startpos, endpos = None, duration = None):
        if (endpos is None and duration is None) or (endpos is not None and duration is not None):
            raise ValueError("Specify either endpos or duration, but not both.")
        start = channel.timecodes(startpos, startpos)[0]
        if duration:
            end = start + duration
        else:
            end = channel.timecodes(endpos, endpos)[0]
        event = salto.Event(type = salto.CALCULATED_EVENT,
                            subtype = 'threshold',
                            start_sec = int(start), start_nsec = int(math.fmod(start, 1.0) * 1e9),
                            end_sec = int(end), end_nsec = int(math.fmod(end, 1.0) * 1e9))
        channel.events.add(event)
    def _threshold(self, inputs):
        iChannels = salto.channelTables[inputs['channelTable']].channels
        for channel in iChannels.values():
            lower = (inputs['lower'] - channel.offset) / channel.scale
            upper = (inputs['upper'] - channel.offset) / channel.scale
            positions = self._position(channel.data, lower, upper, inputs['includelower'], inputs['includeupper'])
            if positions.size > 0:
                # Recode the positions as slices.
                starts = np.insert(np.where(np.diff(positions) != 1)[0] + 1, 0, 0)
                lengths = np.diff(np.append(starts, len(positions)) - 1)
                positions = list(zip(positions[starts], lengths))
            fill = 0
            for start, length in positions:
                # TODO: Implement minbreak.
                extra = 0
                while (fill < channel.fills.size) and (channel.fills[fill]['pos'] < start + length):
                    if channel.fills[fill]['pos'] == start + length - 1:
                        extra = channel.fills[fill]['len']
                    fill += 1
                duration = (length + extra) / channel.samplerate
                if inputs['minduration'] is None or duration >= inputs['minduration']:
                    self._addEvent(channel, start, duration = duration)
        return {}
