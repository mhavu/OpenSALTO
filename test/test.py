from ctypes import *

gcdc = CDLL("gcdc.dylib")
gcdcRead = gcdc.readFile
gcdcRead.restype = c_int
gcdcRead.argtypes = [c_char_p, c_char_p]

atsf = CDLL("atsf.dylib")
atsfRead = atsf.readFile
atsfRead.restype = c_int
atsfRead.argtypes = [c_char_p, c_char_p]

m = salto.channelTables["main"]
atsfRead("TINY.ATS", "main")
gcdcRead("GCDC.CSV", "main")
z = m.channels["Z"]
salto.metadata(z)
