import salto
import re
from datetime import datetime
import json

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

if __name__ == "__main__":
    salto.channelTables = {'main': ChannelTable()}
