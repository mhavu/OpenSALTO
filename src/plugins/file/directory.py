#
#  directory.py
#  OpenSALTO
#
#  Reads data file directories.
#
#  Copyright 2014 Marko Havu. Released under the terms of
#  GNU General Public License version 3 or later.
#

import salto, os, math

class Plugin(salto.Plugin):
    """OpenSALTO plugin for reading data file directories"""
    def __init__(self, manager):
        super(Plugin, self).__init__(manager)
        self.registerFormat('directory', [''])
        self.setImportFunc('directory', self._read)
    def _read(self, dirname, chTable):
        if os.path.isdir(dirname):
            filelist = os.listdir(dirname)
            if not filelist:
                raise FileNotFoundError("Empty directory")
            extdict = {}
            for file in filelist:
                (_, ext) = os.path.splitext(file)
                extdict.setdefault(ext, [])
                extdict[ext].append(file)
            pm = salto.pluginManager
            # For directories with more than one type of file, process
            # the type with most files.
            dominantExt = max(extdict, key = lambda x: len(extdict[x]))
            formats = pm.query(ext = dominantExt, mode = 'r')
            file = extdict[dominantExt][0]
            # Try to detect the file format.
            fmt = pm.detect(os.path.join(dirname, file))
            if fmt is None:
                raise RuntimeError("Detecting import file type failed")
            # Read the files, each to its own channel table.
            tableList = []
            channels = []
            tmpName = salto.makeUniqueKey(salto.channelTables, "temporary")
            for file in extdict[dominantExt]:
                tmpTable = salto.ChannelTable()
                salto.channelTables[tmpName] = tmpTable
                tableList.append(tmpTable)
                pm.read(os.path.join(dirname, file), fmt, tmpName)
                salto.channelTables.pop(tmpName, None)
                for name, ch in tmpTable.channels.items():
                    if name not in channels:
                        channels.append(name)
                    # Create an event for the file.
                    end = ch.end().timestamp()
                    end_sec = int(end)
                    end_nsec = int(math.fmod(end, 1.0))
                    e = salto.Event(salto.MARKER_EVENT, file, ch.start_sec,
                                    ch.start_nsec, end_sec, end_nsec,
                                    os.path.join(dirname, file))
                    ch.events.add(e)
            # Combine the files to sparse channels.
            for chName in channels:
                parts = [table.channels.get(chName) for table in tableList
                         if chName in table.channels]
                ch = salto.Channel.collate(*parts)
                chTable.add(chTable.getUnique(chName), ch)
        else:
            raise NotADirectoryError
