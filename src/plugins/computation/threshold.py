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
        self.registerComputation("threshold",
                                 self.threshold_,
                                 inputs = [('channelTable', 'S', 1, 1),
                                           ('lower', 'f', 'lower threshold', None),
                                           ('upper', 'f', 'upper threshold', None)],
                                 outputs = [('channelTable', 'S', 0, 0)])
    def position_(self, data, lower = None, upper = None):
        if ((lower is not None) and (upper is not None)):
            result = np.where((data > lower) & (data < upper))
        elif (lower is not None):
            result = np.where(data > lower)
        elif (upper is not None):
            result = np.where(data < upper)
        else:
            raise ValueError("At least one threshold needs to be specified")
        return result[0]
    def addEvent_(self, channel, startpos, endpos):
        t0 = channel.start_sec + channel.start_nsec / 1e9
        (start, end) = (t0 + pos / channel.samplerate for pos in (startpos, endpos))
        event = salto.Event(type = salto.CALCULATED_EVENT,
                            subtype = 'threshold',
                            start_sec = int(start), start_nsec = int(math.fmod(start, 1.0)),
                            end_sec = int(end), end_nsec = int(math.fmod(end, 1.0)))
        channel.events.add(event)
    def threshold_(self, inputs):
        iChannels = salto.channelTables[inputs['channelTable']].channels
        for channel in iChannels.values():
            if (channel.collection):
                positions = np.array([])
                for part in channel.data:
                    p = self.position_(channel.data, inputs['lower'], inputs['upper'])
                    t = (part.start_sec + part.start_nsec / 1e9) - (channel.start_sec + channel.start_nsec / 1e9)
                    positions = np.concatenate(positions, p + round(t * channel.samplerate))
            else:
                positions = self.position_(channel.data, inputs['lower'], inputs['upper'])
            if positions.size > 0:
                start = positions[0]
                prev = start
                for i in positions[1:]:
                    if i > prev + 1:
                        self.addEvent_(channel, start, prev)
                        start = i
                    prev = i
                self.addEvent_(channel, start, prev)
