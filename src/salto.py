import re, json, os
from datetime import datetime
from ctypes import CDLL, RTLD_LOCAL, c_char_p, c_void_p, pointer

# Load the OpenSALTO C extension module and extend it with native Python classes.
import salto
moduleName = __name__
__name__ = 'salto'

class ChannelTable:
    "OpenSALTO channel table"
    ordinal = re.compile(r'\d+$')
    def __init__(self):
        self.channels = {}
    def add(self, name, ch):
        self.channels.setdefault(name, ch)
    def remove(self, name):
        self.channels.pop(name, None)
    def findKeyForPointer(self, ptr):
        return next(key for key, ch in self.channels.iteritems() if getDataPtr(ch) == ptr)
    def getUnique(self, name):
        """Get a unique channel name by adding or
            incrementing an ordinal"""
        if name not in self.channels:
            unique = name
        else:
            match = ChannelTable.ordinal.search(name)
            if match:
                n = int(match.group())
                n += 1
                unique = name[0:match.start()] + str(n)
            else:
                unique = '%s %d' % (name, 2)
            unique = self.getUnique(unique)
        return unique

class Plugin:
    "OpenSALTO plugin"
    def __init__(self, manager, cdll = None):
        self.manager = manager
        self.cdll = cdll
        self.formats = {}
        if self.cdll:
            self.cdll.describeError.restype = c_char_p
            self.cdll.initPlugin.argtypes = [c_void_p]
            self.cdll.initPlugin(pointer(self))
    def registerFormat(self, name, exts):
        self.formats.setdefault(name, {'exts':exts, 'readfunc':None, 'writefunc':None})
    def unregisterFormat(self, name):
        self.formats.pop(name, None)
    def setImportFunc(self, format, func):
        self.formats.get(format)['readfunc'] = func
    def setExportFunc(self, format, func):
        self.formats.get(format)['writefunc'] = func
    def read(self, filename, format, chTable):
        registered = self.formats.get(format)
        if registered:
            readfunc = registered.get('readfunc')
        if readfunc:
            err = readfunc(filename, chTable)
        if err != 0:
            raise(IOError, self.cdll.describeError(err))
    def write(self, filename, chTable):
        registered = self.formats.get(format)
        if registered:
            writefunc = registered.get('writefunc')
        if readfunc:
            err = writefunc(filename, chTable)
        if err != 0:
            raise(IOError, self.cdll.describeError(err))
    def filter(self, ch):
        return ch

class PluginManager:
    "OpenSALTO plugin manager"
    def __init__(self):
        self.plugins = []
    def discover(self, path):
        for filename in os.listdir(path):
            plugin, ext = os.path.splitext(filename)
            isLoadable = ext in ('.dylib', '.so', '.dll')
            #isNew = plugin not in self.plugins
            if isLoadable: #and isNew:
                self.load(filename)
    def load(self, filename):
        cdll = CDLL(filename, mode = RTLD_LOCAL)
        self.register(salto.Plugin(self, cdll))
    def register(self, plugin):
        if plugin not in self.plugins:
            self.plugins.append(plugin)
    def unregister(self, plugin):
        self.plugins.pop(plugin, None)
    def query(self, **kwargs):
        # TODO: implement
        "Query for plugins capable of doing x"
        for plugin in self.plugins:
            pass

setattr(salto, 'ChannelTable', ChannelTable)
setattr(salto, 'Plugin', Plugin)
setattr(salto, 'PluginManager', PluginManager)
del ChannelTable
del Plugin
del PluginManager
__name__ = moduleName


if __name__ == "__main__":
    salto.channelTables = {'main': salto.ChannelTable()}
    salto.pluginManager = salto.PluginManager()
    salto.pluginManager.discover('.')