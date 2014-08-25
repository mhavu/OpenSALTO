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
                                 self.inclination_,
                                 inputs = [('channelTable', 'S', 2, 3)],
                                 outputs = [('channelTable', 'S', 3, 4)])
    def inclination_(self, inputs):
        tableName = salto.makeUniqueKey(salto.channelTables, "inclination")
        iChannels = salto.channelTables[inputs['channelTable']].channels
        chTable = salto.ChannelTable()
        normArray = np.linalg.norm([ch.data for ch in iChannels.values()], axis = 0)
        norm = salto.Channel(normArray)
        chTable.add("norm", norm)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for name, ch in iChannels.items():
                inclination = salto.Channel(np.arccos(ch.data / normArray))
                chTable.add(name + " inclination", inclination)
        salto.channelTables[tableName] = chTable
        outputs = {'channelTable': tableName}
        return outputs
