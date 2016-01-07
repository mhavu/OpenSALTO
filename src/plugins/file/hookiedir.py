#
#  hookiedir.py
#  OpenSALTO
#
#  Imports directories with Hookie AM20 Activity Meter data files.
#
#  Copyright 2016 Marko Havu. Released under the terms of
#  GNU General Public License version 3 or later.
#

import salto, os, math, json

class Plugin(salto.Plugin):
    """OpenSALTO plugin for reading Hookie AM20 Activity Meter
        data file directories"""
    def __init__(self, manager):
        super(Plugin, self).__init__(manager)
        self.registerFormat('Hookie AM20 Directory', [''])
        self.setImportFunc('Hookie AM20 Directory', self._read)
    def _read(self, dirname, chTable):
        if os.path.isdir(dirname):
            filelist = os.listdir(dirname)
            if not filelist:
                raise FileNotFoundError("Empty directory")
            pm = salto.pluginManager
            fmt = 'Hookie AM20 Activity Meter'
            # Read the files, each to its own channel table.
            tableList = []
            channels = []
            tmpName = salto.makeUniqueKey(salto.channelTables, "temporary")
            for file in filelist:
                tmpTable = salto.ChannelTable()
                salto.channelTables[tmpName] = tmpTable
                tableList.append(tmpTable)
                pm.read(os.path.join(dirname, file), fmt, tmpName)
                salto.channelTables.pop(tmpName, None)
                for name, ch in tmpTable.channels.items():
                    if name not in channels:
                        channels.append(name)
                    # Create an event for the file.
                    e = salto.Event(salto.MARKER_EVENT, file, ch.start_sec,
                                    ch.start_nsec, ch.start_sec, ch.start_nsec,
                                    os.path.join(dirname, file))
                    ch.events.add(e)
            # Combine the files to sparse channels.
            for chName in channels:
                parts = [table.channels[chName] for table in tableList
                         if chName in table.channels]
                # Correct the samplerate.
                drift = max([json.loads(ch.json)['Clock drift'] for ch in parts])
                for ch in parts:
                    ch.samplerate = ch.samplerate * (drift + 1.0)
                    # Set the event end time.
                    e = ch.events.pop()
                    end = ch.end().timestamp()
                    e.end_sec = int(end)
                    e.end_nsec = int(math.fmod(end, 1.0) * 1e9)
                    ch.events.add(e)
                ch = salto.Channel.collate(*parts)
                chTable.add(chTable.getUnique(chName), ch)
        else:
            raise NotADirectoryError
