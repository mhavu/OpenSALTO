#
#  mad.py
#  OpenSALTO
#
#  A filter that computes mean amplitude deviation as introduced in
#  Vähä-Ypyä et al. (2015). (The formula in the article is missing a
#  sum symbol.)
#
#  Vähä-Ypyä, H., Vasankari, T., Husu, P., Suni. J. & Sievänen, H. 2015.
#    A universal, accurate intensity-based classification of different
#    physical activities using raw data of accelerometer.
#    Clin Physiol Funct Imaging 35(1): 64–70.
#
#  Copyright 2016 Marko Havu. Released under the terms of
#  GNU General Public License version 3 or later.
#

import salto
import numpy as np

class BufferOperation:
    """Buffer operation class"""
    def __init__(self, size, callback):
        self._buffer = np.empty(size)
        self._size = size
        self._pos = 0
        self._callback = callback
    def write(self, data):
        """Write data to buffer calling callback each time the buffer is full"""
        pos = 0
        left = len(data)
        n = self._size - self._pos
        while left >= n:
            self._buffer[self._pos:self._size] = data[pos:pos+n]
            pos += n
            left -= n
            self._callback(self._buffer)
            self._pos = 0
            n = self._size
        self._buffer[self._pos:self._pos+left] = data[pos:pos+left]
        self._pos += left
    def fill(self, value):
        """Fill the rest of the buffer with value and call callback"""
        n = self._size - self._pos
        self._buffer[self._pos:self._size] = value
        self._callback(self._buffer)
        self._pos = 0
        return n

class MadResult:
    """MAD result array"""
    def __init__(self, size):
        self._buffer = np.empty(size)
        self._size = size
        self._pos = 0
    @property
    def array(self):
        return self._buffer[0:self._pos]
    def put(self, value):
        """Put a value into the result array and increment position counter"""
        self._buffer[self._pos] = value
        self._pos += 1
    def compute(self, data):
        """Compute mean amplitude deviation from data"""
        mean = np.mean(data)
        self._buffer[self._pos] = np.mean(np.abs(data - mean))
        self._pos += 1

class Plugin(salto.Plugin):
    """OpenSALTO filter plugin for computing mean amplitude deviation"""
    def __init__(self, manager):
        super(Plugin, self).__init__(manager)
        inputs = [('channelTable', 'S', 1, 1),
                  ('epoch', 'i', "length of the MAD computation window (samples)", None)]
        self.registerComputation('MAD', self._mad,
                                 inputs = inputs,
                                 outputs = [('channelTable', 'S', 1, 0)])
    def _mad(self, inputs):
        """Compute mean amplitude deviation"""
        tableName = salto.makeUniqueKey(salto.channelTables, "MAD")
        iChannels = salto.channelTables[inputs['channelTable']].channels
        chTable = salto.ChannelTable()
        epoch = inputs['epoch']
        for name, ch in iChannels.items():
            nSamples = ch.data.size + np.sum(ch.fills['len'])
            nSurvivingFills = np.sum(ch.fills['len'] >= 3 * epoch)
            size = nSamples // epoch + nSurvivingFills
            data = MadResult(size)
            bop = BufferOperation(epoch, data.compute)
            fills = ch.fills.copy()
            pos = 0
            extra = 0
            dropped = 0
            for i, fill in enumerate(ch.fills):
                bop.write(ch.values(pos, fill['pos']))
                if fill['len'] >= 3 * epoch:
                    # Fill buffer.
                    n = bop.fill(ch.values(fill['pos']))
                    # Adjust fill size.
                    div, mod = divmod(fill['len'] - n, epoch)
                    extra += n
                    data.put(0)
                    fills[i - dropped]['pos'] = (fill['pos'] + extra) / epoch + 1
                    fills[i - dropped]['len'] = div - 1
                    # Put remaining fill values to the buffer.
                    bop.write(ch.values(fill['pos']).repeat(mod))
                    extra += mod + epoch
                else:
                    # Replace the fill with individual values.
                    extra += fill['len']
                    bop.write(ch.values(fill['pos']).repeat(fill['len']))
                    fills = np.delete(fills, i - dropped)
                    dropped += 1
                pos = fill['pos'] + 1
            bop.write(ch.values(pos, -1))
            mad = salto.Channel(data.array, ch.samplerate / epoch, fills,
                                unit = ch.unit, type = ch.type,
                                start_sec = ch.start_sec,
                                start_nsec = ch.start_nsec)
            chTable.add(name + " MAD", mad)
        salto.channelTables[tableName] = chTable
        outputs = {'channelTable': tableName}
        return outputs