#
#  salto.py
#  OpenSALTO
#
#  The parts of the OpenSALTO extension module that are implemented in Python.
#
#  Copyright 2014 Marko Havu. Released under the terms of
#  GNU General Public License version 3 or later.
#

import re, json, pint
import code
import os, platform, sys
import importlib.machinery
from datetime import datetime
import ctypes as c
import _ctypes
import numpy as np
from collections import OrderedDict

# Extend the OpenSALTO C extension module with native Python classes.
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
    def __init__(self, gui = False):
        self.showsInGui = gui if hasattr(salto, 'gui') else False
        self._channels = OrderedDict()
    def __iter__(self):
        yield from self._channels
    @property
    def channels(self):
        return self._channels
    def add(self, name, ch):
        if not isinstance(ch, salto.Channel):
            raise TypeError("%r is not a Channel object" % ch)
        if name not in self._channels:
            self._channels[name] = ch
            if self.showsInGui:
                salto.gui.addChannel(ch, name)
    def remove(self, name):
        removed = self._channels.pop(name, None)
        if self.showsInGui and removed:
            salto.gui.removeChannel(removed)
    def getUnique(self, name):
        return salto.makeUniqueKey(self._channels, name)
    def name(self):
        result = None
        for name, table in salto.channelTables.items():
            if table == self:
                result = name
                break
        return result

class Plugin:
    """OpenSALTO plugin"""
    def __init__(self, manager, cdll = None):
        self.manager = manager
        self._cdll = cdll
        self._formats = {}
        self._computations = {}
        if self._cdll:
            self._cdll.describeError.restype = c.c_char_p
            self._cdll.initPlugin.argtypes = [c.py_object]
            err = self._cdll.initPlugin(self)
            if err != 0:
                raise RuntimeError(self._cdll.describeError(err).decode('utf-8'))
    @property
    def file(self):
        if self._cdll:
            filepath = self._cdll._name
        else:
            filepath = sys.modules[self.__module__].__file__
        return filepath
    @property
    def cdll(self):
        return self._cdll
    @property
    def formats(self):
        return self._formats
    @property
    def computations(self):
        return self._computations
    # reading and writing files
    def registerFormat(self, name, exts):
        self._formats.setdefault(name, {'exts':exts, 'readfunc':None, 'writefunc':None})
    def unregisterFormat(self, name):
        self._formats.pop(name, None)
    def setImportFunc(self, format, func):
        self._formats.get(format)['readfunc'] = func
    def setExportFunc(self, format, func):
        self._formats.get(format)['writefunc'] = func
    def read(self, filename, format, chTableName):
        registered = self._formats.get(format)
        if registered:
            readfunc = registered.get('readfunc')
        if readfunc:
            if self._cdll:
                err = readfunc(filename.encode('utf-8'), chTableName.encode('utf-8'))
                if err != 0:
                    raise IOError(self._cdll.describeError(err).decode('utf-8'))
            else:
                readfunc(filename, salto.channelTables[chTableName])
    def write(self, filename, format, chTableName):
        registered = self._formats.get(format)
        if registered:
            writefunc = registered.get('writefunc')
        if writefunc:
            if self._cdll:
                err = writefunc(filename.encode('utf-8'), chTableName.encode('utf-8'))
                if err != 0:
                    raise IOError(self._cdll.describeError(err).decode('utf-8'))
            else:
                writefunc(filename, salto.channelTables[chTableName])
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
        if self._cdll:
            dtypes = (np.dtype([(i[0], salto.Plugin.convertPtrFormat(i[1])) for i in inputs], align = True),
                      np.dtype([(o[0], salto.Plugin.convertPtrFormat(o[1])) for o in outputs], align = True))
            func.argtypes = [np.ctypeslib.ndpointer(dtype = dtypes[0], flags = ('C', 'A')),
                             np.ctypeslib.ndpointer(dtype = dtypes[1], flags = ('C', 'A'))]
            self.nOutputChannels = self._cdll.nOutputChannels
            self._cdll.nOutputChannels.restype = c.c_size_t
            self._cdll.nOutputChannels.argtypes = [c.c_char_p, c.c_size_t]
        else:
            dtypes = None
        self._computations.setdefault(name, (func, inputs, outputs, dtypes))
    def unregisterComputation(self, name):
        self._computations.pop(name, None)
    def compute(self, name, inputs):
        func, inputSpec, outputSpec, dtypes = self._computations[name]
        minChannels, maxChannels = inputSpec[0][2:4]
        if minChannels and not maxChannels:
            maxChannels = np.inf
        chTable = inputs.get('channelTable')
        nChannels = len(salto.channelTables[chTable].channels) if chTable else 0
        if nChannels < minChannels or nChannels > maxChannels:
            raise TypeError("Number of input channels must be between %i and %i" % (minChannels, maxChannels))
        if self._cdll:
            inputs = {k: v.encode('utf-8') if isinstance(v, str) else v for k, v in inputs.items()}
            values = tuple([salto.Plugin.convertInputPtr(inputs.get(i[0], i[3]), i[1]) for i in inputSpec])
            inputArgs = np.array(values, dtypes[0])
            outputs = np.empty((), dtypes[1])
            err = func(inputArgs, outputs)
            if err != 0:
                raise RuntimeError(self._cdll.describeError(err).decode('utf-8'))
            outputs = {o[0]: salto.Plugin.convertOutputPtr(outputs[o[0]].item(), o[1]) for o in outputSpec}
            outputs = {k: v.decode('utf-8') if isinstance(v, bytes) else v for k, v in outputs.items()}
        else:
            inputArgs = {i[0]:inputs.get(i[0], i[3]) for i in inputSpec}
            inputArgs = {k:v for k, v in inputArgs.items() if v is not None}
            outputs = func(inputArgs)
        return outputs

class PluginManager:
    """OpenSALTO plugin manager"""
    def __init__(self):
        self._plugins = []
        self._importFormats = {}
        self._exportFormats = {}
        self._computations = {}
    @property
    def plugins(self):
        return self._plugins
    @property
    def importFormats(self):
        return self._importFormats
    @property
    def exportFormats(self):
        return self._exportFormats
    @property
    def computations(self):
        return self._computations
    def discover(self, path, check_loaded = True):
        for filename in os.listdir(path):
            try:
                self.load(os.path.join(path, filename), check_loaded)
            except TypeError:
                pass
    def isLoaded(self, filepath):
        return any(os.path.samefile(filepath, p.file) for p in self._plugins)
    def load(self, filepath, check_loaded = True):
        if check_loaded and self.isLoaded(filepath):
            raise ValueError("Plugin %s is already loaded" % filepath)
        filename = os.path.basename(filepath)
        plugin, ext = os.path.splitext(filename)
        isLibrary = ext.lower() in ('.dylib', '.so', '.dll')
        isModule = ext.lower() in ('.py', '.pyc', '.pyo')
        if isLibrary:
            self._loadLibrary(filepath)
        elif isModule:
            self._loadModule(filepath, plugin)
        else:
            raise TypeError("File %s is not a dynamically linked library or a Python file" % filename)
    def _loadLibrary(self, filepath):
        cdll = c.CDLL(filepath, mode = c.RTLD_LOCAL)
        self.register(salto.Plugin(self, cdll))
    def _loadModule(self, filepath, name):
        loader = importlib.machinery.SourceFileLoader('salto.' + name, filepath)
        setattr(salto, name, loader.load_module())
        self.register(getattr(salto, name).Plugin(self))
    def unload(self, plugin):
        filepath = plugin.file
        cdll = plugin.cdll
        self.unregister(plugin)
        if cdll:
            if platform.system() == 'Windows':
                _ctypes.FreeLibrary(cdll._handle)
            else:
                _ctypes.dlclose(cdll._handle)
        return filepath
    def reload(self, plugin):
        filepath = self.unload(plugin)
        self.load(filepath)
    def register(self, plugin):
        if plugin not in self._plugins:
            self._plugins.append(plugin)
            for format, attrs in plugin.formats.items():
                if attrs['readfunc']:
                    self._importFormats.setdefault(format, plugin)
                if attrs['writefunc']:
                    self._exportFormats.setdefault(format, plugin)
            for computation in plugin.computations:
                self._computations.setdefault(computation, plugin)
    def unregister(self, plugin):
        self._plugins.remove(plugin)
        self._importFormats = {key: value
            for key, value in self._importFormats.items()
            if value is not plugin}
        self._exportFormats = {key: value
            for key, value in self._exportFormats.items()
            if value is not plugin}
        self._computations = {key: value
            for key, value in self._computations.items()
            if value is not plugin}
    def query(self, **kwargs):
        result = set()
        ext = kwargs.get('ext')
        if not isinstance(ext, str):
            raise TypeError("Invalid extension %r" % ext)
        if ext is not None:
            mode = kwargs.get('mode')
            if mode is None:
                mode = 'rw'
            if not isinstance(mode, str):
                raise TypeError("Invalid mode %r" % mode)
            if mode.lower() not in ('r', 'w', 'rw', 'wr'):
                raise ValueError("Mode %r not 'r', 'w', or 'rw'" % mode)
            if mode.lower() in ('r', 'rw', 'wr'):
                result = {format for format, plugin in self._importFormats.items()
                          if ext.lower() in map(str.lower, plugin.formats[format]['exts'])}
            if mode.lower() in ('w', 'rw', 'wr'):
                result.union({format for format, plugin in self._exportFormats.items()
                              if ext.lower() in map(str.lower, plugin.formats[format]['exts'])})
        return result
    # convenience functions for calling plugins
    def compute(self, compname, inputs):
        plugin = self._computations[compname]
        return plugin.compute(compname, inputs)
    def detect(self, filename):
        _, ext = os.path.splitext(filename)
        formatsWithExt = self.query(ext = ext, mode = 'r')
        if len(formatsWithExt) == 1:
            format = formatsWithExt.pop()
        elif os.path.isdir(filename):
            format = 'directory'
        else:
            # If file extension matches more than one format,
            # detect the format by trial and error.
            chTableName = salto.makeUniqueKey(salto.channelTables, "temporary")
            tmpTable = salto.ChannelTable()
            salto.channelTables[chTableName] = tmpTable
            for fmt in formatsWithExt:
                try:
                    plugin = self._importFormats[fmt]
                    plugin.read(filename, fmt, chTableName)
                    format = fmt
                    break
                except:
                    pass
            else:
                format = None
            salto.channelTables.pop(chTableName, None)
        return format
    def read(self, filename, format, chTableName = None):
        # The optional argument is format. Switch if necessary. 
        if chTableName is None:
            format, chTableName = (None, format)
        if format is None:
            format = self.detect(filename)
        if format is None:
            raise KeyError("Could not determine file format by file extension. Try specifying the format explicitly.")
        plugin = self._importFormats[format]
        plugin.read(filename, format, chTableName)
    def write(self, filename, format, chTableName = None):
        if chTableName is None:
            format, chTableName = (None, format)
        if format is None:
            _, ext = os.path.splitext(filename)
            formatsWithExt = self.query(ext = ext, mode = 'w')
            if len(formatsWithExt) == 1:
                format = formatsWithExt.pop()
            else:
                raise KeyError("Could not determine file format by file extension. Try specifying the format explicitly.")
        plugin = self._exportFormats[format]
        plugin.write(filename, format, chTableName)

def open(filename, format = None):
    salto.pluginManager.read(filename, format, 'main')

def setquit():
    """Define new built-ins 'quit' and 'exit'.
        These are simply strings that display a hint on how to exit."""
    class Quitter(object):
        def __repr__(self):
            return "Close the OpenSALTO application to exit"
        def __call__(self, code=None):
            print("Close the OpenSALTO application to exit")
    __builtins__.exit = Quitter()
    __builtins__.quit = __builtins__.exit

# Move everything from the global namespace to salto.
for o in [makeUniqueKey, ChannelTable, Plugin, PluginManager, open, setquit]:
    setattr(salto, o.__name__, o)
del [o, makeUniqueKey, ChannelTable, Plugin, PluginManager, open, setquit]
__name__ = moduleName
del moduleName

def main():
    salto.__dict__.update({'CUSTOM_EVENT': 0, 'ACTION_EVENT': 1, 'ARTIFACT_EVENT': 2, 'CALCULATED_EVENT': 3, 'MARKER_EVENT': 4, 'TIMER_EVENT': 5})
    salto.units = pint.UnitRegistry()
    salto.channelTables = {'main': salto.ChannelTable(gui = True)}
    salto.sessionData = {}
    salto.pluginManager = salto.PluginManager()
    salto.pluginManager.discover('plugins', False)
    if hasattr(salto, 'gui'):
        salto.setquit()
    del salto.setquit

if __name__ == '__main__':
    main()