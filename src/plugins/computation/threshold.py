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
        # TODO: Implement minduration and minbreak.
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
    def _addEvent(self, channel, startpos, endpos):
        start = channel.timecodes(startpos, startpos)[0]
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
            # TODO: Check fill values too.
            if positions.size > 0:
                start = positions[0]
                prev = start
                for i in positions[1:]:
                    if i > prev + 1:
                        self._addEvent(channel, start, prev)
                        start = i
                    prev = i
                self._addEvent(channel, start, prev)
