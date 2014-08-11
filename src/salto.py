import re, json, os, importlib.machinery
from datetime import datetime
import ctypes as c
import numpy as np

# Load the OpenSALTO C extension module and extend it with native Python classes.
import salto
moduleName = __name__
__name__ = 'salto'

class ChannelTable:
    """OpenSALTO channel table"""
    ordinal = re.compile(r"\d+$")
    def __init__(self):
        self.channels = {}
    def add(self, name, ch):
        assert isinstance(ch, salto.Channel), "%r is not a Channel object" % ch
        self.channels.setdefault(name, ch)
    def remove(self, name):
        self.channels.pop(name, None)
    def getUnique(self, name):
        """Get a unique channel name by adding or
            incrementing an ordinal"""
        if name not in self.channels:
            unique = name
        else:
            match = salto.ChannelTable.ordinal.search(name)
            if match:
                n = int(match.group())
                n += 1
                unique = name[0:match.start()] + str(n)
            else:
                unique = "%s %d" % (name, 2)
            unique = self.getUnique(unique)
        return unique

class Plugin:
    """OpenSALTO plugin"""
    def __init__(self, manager, cdll = None):
        self.manager = manager
        self.cdll = cdll
        self.formats = {}
        if self.cdll:
            self.cdll.describeError.restype = c.c_char_p
            self.cdll.initPlugin.argtypes = [c.py_object]
            self.cdll.initPlugin(self)
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
            err = readfunc(filename.encode('utf-8'), chTable.encode('utf-8'))
        if err != 0 and self.cdll:
            raise(IOError, self.cdll.describeError(err))
    def write(self, filename, format, chTable):
        registered = self.formats.get(format)
        if registered:
            writefunc = registered.get('writefunc')
        if readfunc:
            err = writefunc(filename.encode('utf-8'), chTable.encode('utf-8'))
        if err != 0 and self.cdll:
            raise(IOError, self.cdll.describeError(err))
    def filter(self, name, ch):
        return ch

class PluginManager:
    """OpenSALTO plugin manager"""
    def __init__(self):
        self.plugins = []
        self.importFormats = {}
        self.exportFormats = {}
    def discover(self, path):
        for filename in os.listdir(path):
            plugin, ext = os.path.splitext(filename)
            isLoadable = ext.lower() in (".dylib", ".so", ".dll")
            isPy = ext.lower() in (".py", ".pyc", ".pyo") 
            if isLoadable:
                self.load(os.path.join(path, filename))
            elif isPy:
                filepath = os.path.join(path, filename)
                loader = importlib.machinery.SourceFileLoader("salto." + plugin, filepath)
                setattr(salto, plugin, loader.load_module())
                self.register(getattr(salto, plugin).Plugin(self))
    def load(self, filepath):
        cdll = c.CDLL(filepath, mode = c.RTLD_LOCAL)
        self.register(salto.Plugin(self, cdll))
    def register(self, plugin):
        if plugin not in self.plugins:
            self.plugins.append(plugin)
        for format, attrs in plugin.formats.items():
            if attrs['readfunc']:
                self.importFormats.setdefault(format, plugin)
            if attrs['writefunc']:
                self.exportFormats.setdefault(format, plugin)
    def unregister(self, plugin):
        self.plugins.pop(plugin, None)
        self.importFormats = {key: value
            for key, value in self.importFormats.items()
            if value is not plugin}
        self.exportFormats = {key: value
            for key, value in self.exportFormats.items()
            if value is not plugin}
    def query(self, **kwargs):
        ext = kwargs.get('ext')
        if ext:
            return [format for format, plugin in self.importFormats.items()
                           if ext.lower() in map(str.lower, plugin.formats[format]['exts'])]

setattr(salto, 'ChannelTable', ChannelTable)
setattr(salto, 'Plugin', Plugin)
setattr(salto, 'PluginManager', PluginManager)
del ChannelTable
del Plugin
del PluginManager
__name__ = moduleName
del moduleName


if __name__ == '__main__':
    salto.channelTables = {'main': salto.ChannelTable()}
    salto.pluginManager = salto.PluginManager()
    salto.pluginManager.discover("plugins")