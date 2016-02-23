#
#  coordinates.py
#  OpenSALTO
#
#  A filter that converts between rectangular, cylindrical, and spherical
#  coordinate systems.
#
#  Copyright 2016 Marko Havu. Released under the terms of
#  GNU General Public License version 3 or later.
#

import salto
import numpy as np

def checkChannels(channels):
    chIter = iter(channels)
    ch = next(chIter)
    for another in chIter:
        if not ch.matches(another):
            raise TypeError("Input channels must be of same type and from same time period")

def init(chTable):
    tableName = salto.makeUniqueKey(salto.channelTables, "coordinates")
    iChannels = salto.channelTables[chTable].channels
    checkChannels(iChannels.values())
    values = [ch.values() for ch in iChannels.values()]
    first = list(iChannels.values())[0]
    return (tableName, values, first)

class Plugin(salto.Plugin):
    """OpenSALTO filter plugin for computing coordinate transformations"""
    def __init__(self, manager):
        super(Plugin, self).__init__(manager)
        io2 = [('channelTable', 'S', 2, 2)]
        io23 = [('channelTable', 'S', 2, 3)]
        io3 = [('channelTable', 'S', 3, 3)]
        self.registerComputation('rectangular to polar', self._r2c, io2, io2)
        self.registerComputation('polar to rectangular', self._c2r, io2, io2)
        self.registerComputation('rectangular to spherical', self._r2s, io3, io3)
        self.registerComputation('rectangular to cylindrical', self._r2c, io23, io23)
        self.registerComputation('spherical to rectangular', self._s2r, io3, io3)
        self.registerComputation('spherical to cylindrical', self._s2c, io3, io3)
        self.registerComputation('cylindrical to rectangular', self._c2r, io23, io23)
        self.registerComputation('cylindrical to spherical', self._c2s, io3, io3)
    def _r2s(self, inputs):
        """Convert (x, y, z) to (R, 𝜃, 𝜑)"""
        tableName, values, x = init(inputs['channelTable'])
        chTable = salto.ChannelTable()
        norm = np.linalg.norm(values, axis = 0)
        # R = norm[x;y;z]
        R = salto.Channel(norm, x.samplerate, x.fills, unit = x.unit, type = x.type,
                          start_sec = x.start_sec, start_nsec = x.start_nsec)
        chTable.add("R", R)
        # theta = arctan(y / x)
        theta = salto.Channel(np.arctan2(values[1], values[0]),
                              x.samplerate, x.fills, unit = "rad", type = "angle",
                              start_sec = x.start_sec, start_nsec = x.start_nsec)
        chTable.add("𝜃", theta)
        # phi = arctan(norm[x;y] / z)
        phi = salto.Channel(np.arctan2(np.linalg.norm(values[0:2], axis = 0), values[2]),
                            x.samplerate, x.fills, unit = "rad", type = "angle",
                            start_sec = x.start_sec, start_nsec = x.start_nsec)
        chTable.add("𝜑", phi)
        salto.channelTables[tableName] = chTable
        outputs = {'channelTable': tableName}
        return outputs
    def _s2r(self, inputs):
        """Convert (R, 𝜃, 𝜑) to (x, y, z)"""
        tableName, values, R = init(inputs['channelTable'])
        chTable = salto.ChannelTable()
        # x = R * cos(theta) * sin(phi)
        x = salto.Channel(values[0] * np.cos(values[1]) * np.sin(values[2]),
                          R.samplerate, R.fills, unit = R.unit, type = R.type,
                          start_sec = R.start_sec, start_nsec = R.start_nsec)
        chTable.add("x", x)
        # y = R * sin(theta) * sin(phi)
        y = salto.Channel(values[0] * np.sin(values[1]) * np.sin(values[2]),
                          R.samplerate, R.fills, unit = "rad", type = "angle",
                          start_sec = R.start_sec, start_nsec = R.start_nsec)
        chTable.add("y", y)
        # z = R * cos(phi)
        z = salto.Channel(values[0] * np.cos(values[2]),
                          R.samplerate, R.fills, unit = "rad", type = "angle",
                          start_sec = R.start_sec, start_nsec = R.start_nsec)
        chTable.add("z", z)
        salto.channelTables[tableName] = chTable
        outputs = {'channelTable': tableName}
        return outputs
    def _r2c(self, inputs):
        """Convert (x, y, z) to (r, 𝜃, z) or (x, y) to (r, 𝜃)"""
        tableName, values, x = init(inputs['channelTable'])
        chTable = salto.ChannelTable()
        # r = norm[x;y]
        r = salto.Channel(np.linalg.norm(values[0:2], axis = 0),
                          x.samplerate, x.fills, unit = x.unit, type = x.type,
                          start_sec = x.start_sec, start_nsec = x.start_nsec)
        chTable.add("r", r)
        # theta = arctan(y / x)
        theta = salto.Channel(np.arctan2(values[1], values[0]),
                              x.samplerate, x.fills, unit = "rad", type = "angle",
                              start_sec = x.start_sec, start_nsec = x.start_nsec)
        chTable.add("𝜃", theta)
        # z = z
        if len(values) == 3:
            z = salto.channelTables[inputs['channelTable']].channels[2]
            chTable.add("z", z)
        salto.channelTables[tableName] = chTable
        outputs = {'channelTable': tableName}
        return outputs
    def _c2r(self, inputs):
        """Convert (r, 𝜃, z) to (x, y, z) or (r, 𝜃) to (x, y)"""
        tableName, values, r = init(inputs['channelTable'])
        chTable = salto.ChannelTable()
        # x = r * cos(theta)
        x = salto.Channel(values[0] * np.cos(values[1]),
                          r.samplerate, r.fills, unit = r.unit, type = r.type,
                          start_sec = r.start_sec, start_nsec = r.start_nsec)
        chTable.add("x", x)
        # y = r * sin(theta)
        y = salto.Channel(values[0] * np.sin(values[1]),
                          r.samplerate, r.fills, unit = r.unit, type = r.type,
                          start_sec = r.start_sec, start_nsec = r.start_nsec)
        chTable.add("y", y)
        # z = z
        if len(values) == 3:
            z = salto.channelTables[inputs['channelTable']].channels[2]
            chTable.add("z", z)
        salto.channelTables[tableName] = chTable
        outputs = {'channelTable': tableName}
        return outputs
    def _s2c(self, inputs):
        """Convert (R, 𝜃, 𝜑) to (r, 𝜃, z)"""
        tableName, values, R = init(inputs['channelTable'])
        chTable = salto.ChannelTable()
        # r = R * sin(phi)
        r = salto.Channel(values[0] * np.sin(values[2]),
                          R.samplerate, R.fills, unit = R.unit, type = R.type,
                          start_sec = R.start_sec, start_nsec = R.start_nsec)
        chTable.add("r", r)
        # theta = theta
        theta = salto.channelTables[inputs['channelTable']].channels[1]
        chTable.add("𝜃", theta)
        # z = R * cos(phi)
        z = salto.Channel(values[0] * np.cos(values[2]),
                          R.samplerate, R.fills, unit = R.unit, type = R.type,
                          start_sec = R.start_sec, start_nsec = R.start_nsec)
        chTable.add("z", z)
        salto.channelTables[tableName] = chTable
        outputs = {'channelTable': tableName}
        return outputs
    def _c2s(self, inputs):
        """Convert (r, 𝜃, z) to (R, 𝜃, 𝜑)"""
        tableName, values, r = init(inputs['channelTable'])
        chTable = salto.ChannelTable()
        # R = norm[r;z]
        R = salto.Channel(np.linalg.norm(values[0:3:2], axis = 0),
                          r.samplerate, r.fills, unit = r.unit, type = r.type,
                          start_sec = r.start_sec, start_nsec = r.start_nsec)
        chTable.add("R", R)
        # theta = theta
        theta = salto.channelTables[inputs['channelTable']].channels[1]
        chTable.add("𝜃", theta)
        # phi = arctan(r / z)
        phi = salto.Channel(np.arctan2(values[0], values[2]),
                              r.samplerate, r.fills, unit = "rad", type = "angle",
                              start_sec = r.start_sec, start_nsec = r.start_nsec)
        chTable.add("𝜑", phi)
        salto.channelTables[tableName] = chTable
        outputs = {'channelTable': tableName}
        return outputs