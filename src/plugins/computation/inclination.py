#
#  inclination.py
#  OpenSALTO
#
#  A filter that computes a norm channel and inclination
#  angle channels for orthonormal channels.
#
#  Copyright 2014 Marko Havu. Released under the terms of
#  GNU General Public License version 3 or later.
#

import salto
import numpy as np
import warnings

class Plugin(salto.Plugin):
    """OpenSALTO filter plugin for computing norm and
        inclination angle channels"""
    def __init__(self, manager):
        super(Plugin, self).__init__(manager)
        self.registerComputation("inclination",
                                 self._inclination,
                                 inputs = [('channelTable', 'S', 2, 3)],
                                 outputs = [('channelTable', 'S', 3, 4)])
    def _inclination(self, inputs):
        tableName = salto.makeUniqueKey(salto.channelTables, "inclination")
        iChannels = salto.channelTables[inputs['channelTable']].channels
        chIter = iter(iChannels.values())
        channel = next(chIter)
        for another in chIter:
            if not channel.matches(another):
                raise TypeError("Input channels must be of same type and from same time period")
        chTable = salto.ChannelTable()
        normArray = np.linalg.norm([ch.scale * ch.data + ch.offset
                                    for ch in iChannels.values()], axis = 0)
        fillNormArray = np.linalg.norm([ch.scale * ch.fill_values + ch.offset
                                        for ch in iChannels.values()], axis = 0)
        allEvents = set()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for name, ch in iChannels.items():
                allEvents = allEvents.union(ch.events)
                inclination = salto.Channel(np.arccos((ch.scale * ch.data + ch.offset) / normArray),
                                            ch.samplerate,
                                            ch.fill_positions, ch.fill_lengths,
                                            np.arccos((ch.scale * ch.fill_values + ch.offset) / fillNormArray),
                                            unit = "rad", type = "angle",
                                            start_sec = ch.start_sec, start_nsec = ch.start_nsec,
                                            events = ch.events)
                chTable.add(name + " inclination", inclination)
        norm = salto.Channel(normArray, channel.samplerate,
                             ch.fill_positions, ch.fill_lengths, fillNormArray,
                             unit = channel.unit, type = channel.type,
                             start_sec = channel.start_sec, start_nsec = channel.start_nsec,
                             events = allEvents)
        chTable.add("norm", norm)
        salto.channelTables[tableName] = chTable
        outputs = {'channelTable': tableName}
        return outputs
