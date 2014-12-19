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
        self.registerFormat("directory", [""])
        self.setImportFunc("directory", self.read_)
    def read_(self, dirname, chTable):
        if os.path.isdir(dirname):
            filelist = os.listdir(dirname)
            extdict = {}
            for file in filelist:
                (_, ext) = os.path.splitext(file)
                extdict.setdefault(ext, [])
                extdict[ext].append(file)
            # For directories with more than one type of file, process
            # the type with most files.
            dominantext = max(extdict, key = lambda x: len(extdict[x]))
            for file in extdict[dominantext]:
                salto.pluginManager.read(file, chTable)
            # TODO: Combine the channels
        else:
            raise NotADirectoryError
