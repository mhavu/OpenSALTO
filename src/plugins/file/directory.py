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
            dominantext = max(extdict, key = lambda x: len(extdict[x]))
            formats = pm.query(ext = dominantext, mode = 'r')
            file = extdict[dominantext].pop()
            # If file extension matches more than one format,
            # detect the format by trial and error.
            tmpName = salto.makeUniqueKey(salto.channelTables, "temporary")
            tmpTable = salto.ChannelTable()
            salto.channelTables[tmpName] = tmpTable
            for fmt in formats:
                try:
                    pm.read(os.path.join(dirname, file), fmt, tmpName)
                    break
                except IOError:
                    pass
            else:
                salto.channelTables.pop(tmpName, None)
                raise RuntimeError("Detecting the import file type failed")
            # Read the files, each to its own channel table.
            tableList = []
            channels = []
            for file in extdict[dominantext]:
                tmpTable = salto.ChannelTable()
                salto.channelTables[tmpName] = tmpTable
                tableList.append(tmpTable)
                pm.read(os.path.join(dirname, file), fmt, tmpName)
                salto.channelTables.pop(tmpName, None)
                for name, ch in tmpTable.channels.items():
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
                         if chName in table.channels.keys()]
                ch = salto.Channel.collate(parts)
                chTable.add(chTable.getUnique(chName), ch)
        else:
            raise NotADirectoryError
