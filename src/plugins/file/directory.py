#
#  directory.py
#  OpenSALTO
#
#  Reads data file directories.
#
#  Copyright 2014 Marko Havu. Released under the terms of
#  GNU General Public License version 3 or later.
#

import salto, os

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
            for name, table in salto.channelTables.items():
                if table == chTable:
                    chTableName = name
                    break
            # For directories with more than one type of file, process
            # the type with most files.
            dominantext = max(extdict, key = lambda x: len(extdict[x]))
            formats = pm.query(ext = dominantext, mode = 'r')
            file = extdict[dominantext].pop()
            for fmt in formats:
                try:
                    pm.read(os.path.join(dirname, file), fmt, chTableName)
                    break
                except:
                    pass
            else:
                raise RuntimeError("Detecting the import file type failed")
            for file in extdict[dominantext]:
                pm.read(os.path.join(dirname, file), fmt, chTableName)
            # TODO: Combine the channels
            # Create an event for each file
        else:
            raise NotADirectoryError
