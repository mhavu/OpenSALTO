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
                  ('lower', 'f', 'lower threshold', None),
                  ('upper', 'f', 'upper threshold', None),
                  ('includelower', 'i', 'include lower threshold', 1),
                  ('includeupper', 'i', 'include upper threshold', 0),
                  ('minduration', 'f', 'minimum event duration (s)', None),
                  ('minbreak', 'f', 'minimum time between events (s)', None)]
        self.registerComputation("threshold",
                                 self._threshold,
                                 inputs = inputs,
                                 outputs = [('channelTable', 'S', 0, 0)])
    def _position(self, data, lower, upper, includelower, includeupper):
        if ((lower is not None) and (upper is not None)):
            if (includelower and includeupper):
                result = np.where((data >= lower) & (data <= upper))
            elif includeupper:
                result = np.where((data > lower) & (data <= upper))
            elif includelower:
                result = np.where((data >= lower) & (data < upper))
            else:
                result = np.where((data > lower) & (data < upper))
        elif (lower is not None):
            if includelower:
                result = np.where(data >= lower)
            else:
                result = np.where(data > lower)
        elif (upper is not None):
            if includeupper:
                result = np.where(data <= upper)
            else:
                result = np.where(data < upper)
        else:
            raise ValueError("At least one threshold needs to be specified")
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
                            start_sec = int(start), start_nsec = int(math.fmod(start, 1.0)),
                            end_sec = int(end), end_nsec = int(math.fmod(end, 1.0)))
        channel.events.add(event)
    def _threshold(self, inputs):
        iChannels = salto.channelTables[inputs['channelTable']].channels
        for channel in iChannels.values():
            positions = self._position(channel.data, inputs['lower'], inputs['upper'], inputs['includelower'], inputs['includeupper'])
            fills = self._position(channel.fill_values, inputs['lower'], inputs['upper'], inputs['includelower'], inputs['includeupper'])
            if positions.size > 0:
                # Recode the positions as slices.
                starts = np.insert(np.where(np.diff(positions) != 1)[0] + 1, 0, positions[0])
                lengths = np.diff(np.append(starts, len(positions)) - 1)
                positions = list(zip(positions[starts], lengths))
            pos = 0
            fill = 0
            while fill < channel.fill_positions.size and pos < len(positions):
                if (channel.fill_positions[fill] >= positions[pos][0]) and (channel.fill_positions[fill] < sum(positions[pos])):
                    if fill not in fills:
                        duration = (start + length - 1) / channel.samplerate
                        if duration >= inputs['minduration']:
                            self._addEvent(channel, positions[pos][0], channel.fill_positions[fill] - 1)
                        positions[pos][0] = channel.fill_positions[fill]
                    fill += 1
                elif channel.fill_positions[fill] < positions[pos][0]:
                    if fill in fills:
                        duration = channel.fill_lengths[fill] / channel.samplerate
                        if duration >= inputs['minduration']:
                            self._addEvent(channel, channel.fill_positions[fill], duration = duration)
                    fill += 1
                else:
                    pos += 1
            for start, length in positions:
                # TODO: Implement minbreak.
                duration = (length - 1) / channel.samplerate
                if duration >= inputs['minduration']:
                    self._addEvent(channel, start, start + length - 1)
        return {}
