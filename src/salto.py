#
#  salto.py
#  OpenSALTO
#
#  The parts of the OpenSALTO extension module that are implemented in Python.
#
#  Copyright 2014 Marko Havu. Released under the terms of
#  GNU General Public License version 3 or later.
#

import re, json, os, pint
import importlib.machinery
from datetime import datetime
import ctypes as c
import numpy as np

# Load the OpenSALTO C extension module and extend it with native Python classes.
import salto
moduleName = __name__
__name__ = 'salto'

def makeUniqueKey(dict, key):
    """Make a dictionary key unique by adding or incrementing an ordinal"""
    if 'ordinal' not in salto.makeUniqueKey.__dict__:
        salto.makeUniqueKey.ordinal = re.compile(r"\d+$")
    if key not in dict:
        unique = key
    else:
        match = salto.makeUniqueKey.ordinal.search(key)
        if match:
            n = int(match.group())
            n += 1
            unique = key[0:match.start()] + str(n)
        else:
            unique = "%s %d" % (key, 2)
        unique = salto.makeUniqueKey(dict, unique)
    return unique

class ChannelTable:
    """OpenSALTO channel table"""
    def __init__(self):
        self.channels = {}
    def add(self, name, ch):
        assert isinstance(ch, salto.Channel), "%r is not a Channel object" % ch
        self.channels.setdefault(name, ch)
    def remove(self, name):
        self.channels.pop(name, None)
    def getUnique(self, name):
        return salto.makeUniqueKey(self.channels, name)

class Plugin:
    """OpenSALTO plugin"""
    def __init__(self, manager, cdll = None):
        self.manager = manager
        self.cdll = cdll
        self.formats = {}
        self.computations = {}
        if self.cdll:
            self.cdll.describeError.restype = c.c_char_p
            self.cdll.initPlugin.argtypes = [c.py_object]
            err = self.cdll.initPlugin(self)
            if err != 0:
                raise RuntimeError(self.cdll.describeError(err).decode('utf-8'))
    # reading and writing files
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
            if self.cdll:
                err = readfunc(filename.encode('utf-8'), chTable.encode('utf-8'))
                if err != 0:
                    raise IOError(self.cdll.describeError(err).decode('utf-8'))
            else:
                readfunc(filename, salto.channelTables[chTable])
    def write(self, filename, format, chTable):
        registered = self.formats.get(format)
        if registered:
            writefunc = registered.get('writefunc')
        if writefunc:
            if self.cdll:
                err = writefunc(filename.encode('utf-8'), chTable.encode('utf-8'))
                if err != 0:
                    raise IOError(self.cdll.describeError(err).decode('utf-8'))
            else:
                writefunc(filename, salto.channelTables[chTable])
    # computations
    @staticmethod
    def convertOutputPtr(value, format):
        if format == 'S':
            value = c.c_char_p(value).value
        return value
    @staticmethod
    def convertInputPtr(value, format):
        if format == 'S':
            value = c.cast(value, c.c_void_p).value if value else 0
        return value
    @staticmethod
    def convertPtrFormat(format):
        if format == 'S':
            format = 'u' + str(c.sizeof(c.c_void_p))
        return format
    def nOutputChannels(self, name, nInputChannels):
        pass
    def registerComputation(self, name, func, inputs, outputs):
        if self.cdll:
            dtypes = (np.dtype([(i[0], salto.Plugin.convertPtrFormat(i[1])) for i in inputs], align = True),
                      np.dtype([(o[0], salto.Plugin.convertPtrFormat(o[1])) for o in outputs], align = True))
            func.argtypes = [np.ctypeslib.ndpointer(dtype = dtypes[0], flags = ('C', 'A')),
                             np.ctypeslib.ndpointer(dtype = dtypes[1], flags = ('C', 'A'))]
            self.nOutputChannels = self.cdll.nOutputChannels
            self.cdll.nOutputChannels.restype = c.c_size_t
            self.cdll.nOutputChannels.argtypes = [c.c_char_p, c.c_size_t]
        else:
            dtypes = None
        self.computations.setdefault(name, (func, inputs, outputs, dtypes))
    def unregisterComputation(self, name):
        self.computations.pop(name, None)
    def compute(self, name, inputs):
        func, inputSpec, outputSpec, dtypes = self.computations.get(name)
        minChannels, maxChannels = inputSpec[0][2:4]
        chTable = inputs.get('channelTable')
        nChannels = len(salto.channelTables[chTable].channels) if chTable else 0
        if nChannels < minChannels or nChannels > maxChannels:
            raise TypeError("Number of input channels must be between " +
                            str(minChannels) + " and " + str(maxChannels))
        if self.cdll:
            inputs = {k: v.encode('utf-8') if isinstance(v, str) else v for k, v in inputs.items()}
            values = tuple([salto.Plugin.convertInputPtr(inputs.get(i[0], i[3]), i[1]) for i in inputSpec])
            inputArgs = np.array(values, dtypes[0])
            outputs = np.empty((), dtypes[1])
            err = func(inputArgs, outputs)
            if err != 0:
                raise RuntimeError(self.cdll.describeError(err).decode('utf-8'))
            outputs = {o[0]: salto.Plugin.convertOutputPtr(outputs[o[0]].item(), o[1]) for o in outputSpec}
            outputs = {k: v.decode('utf-8') if isinstance(v, bytes) else v for k, v in outputs.items()}
        else:
            outputs = func(inputs)
        return outputs

class PluginManager:
    """OpenSALTO plugin manager"""
    def __init__(self):
        self.plugins = []
        self.importFormats = {}
        self.exportFormats = {}
        self.computations = {}
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
        for computation in plugin.computations:
            self.computations.setdefault(computation, plugin)
    def unregister(self, plugin):
        self.plugins.pop(plugin, None)
        self.importFormats = {key: value
            for key, value in self.importFormats.items()
            if value is not plugin}
        self.exportFormats = {key: value
            for key, value in self.exportFormats.items()
            if value is not plugin}
        self.computations = {key: value
            for key, value in self.computations.items()
            if value is not plugin}
    def query(self, **kwargs):
        ext = kwargs.get('ext')
        if ext:
            return [format for format, plugin in self.importFormats.items()
                           if ext.lower() in map(str.lower, plugin.formats[format]['exts'])]
    # convenience functions for calling plugins
    def compute(self, compname, inputs):
        plugin = self.computations.get(compname)
        if plugin:
            return plugin.compute(compname, inputs)
    def read(self, filename, format, chTable):
        plugin = self.importFormats.get(format)
        if plugin:
            return plugin.read(filename, format, chTable)
    def write(self, filename, format, chTable):
        plugin = self.exportFormats.get(format)
        if plugin:
            return plugin.write(filename, format, chTable)

setattr(salto, 'makeUniqueKey', makeUniqueKey)
setattr(salto, 'ChannelTable', ChannelTable)
setattr(salto, 'Plugin', Plugin)
setattr(salto, 'PluginManager', PluginManager)
del makeUniqueKey
del ChannelTable
del Plugin
del PluginManager
__name__ = moduleName
del moduleName


if __name__ == '__main__':
    salto.__dict__.update({'CUSTOM_EVENT': 0, 'ACTION_EVENT': 1, 'ARTIFACT_EVENT': 2, 'CALCULATED_EVENT': 3, 'MARKER_EVENT': 4, 'TIMER_EVENT': 5})
    salto.units = pint.UnitRegistry()
    salto.Q = salto.units.Quantity
    salto.channelTables = {'main': salto.ChannelTable()}
    salto.sessionData = {}
    salto.pluginManager = salto.PluginManager()
    salto.pluginManager.discover("plugins")
