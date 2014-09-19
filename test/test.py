#
#  test.py
#  OpenSALTO
#
#  Python tests for OpenSALTO
#
#  Copyright 2014 Marko Havu. Released under the terms of
#  GNU General Public License version 3 or later.
#

chTables = salto.channelTables
m = chTables['main']
pm = salto.pluginManager
rf = pm.importFormats
wf = pm.exportFormats
comp = pm.computations
alive = pm.query(ext=".ats")[0]
gcdc = pm.query(ext=".csv")[0]
hookie = pm.query(ext=".dat")[0]
test = 'Plugin test'
pm.read("TINY.ATS", alive, 'main')
pm.read("HUGE.ATS", alive, 'main')
pm.read("GCDC.CSV", gcdc, 'main')
pm.read("/Volumes/homes3/mhavu/My Documents/Testidata/DATA0001.DAT", hookie, 'main')
pm.read("/Volumes/homes3/mhavu/My Documents/Testidata/koeM1.h5", 'GViewer', 'main')
pm.write("/Volumes/homes3/mhavu/My Documents/Testidata/writetest.h5", 'GViewer', 'main')
meta = [[getattr(m.channels[z], metadata) for metadata in ('device', 'serial_no', 'resolution', 'samplerate', 'scale', 'offset', 'unit', 'start_sec', 'start_nsec', 'json')] for z in ('Z', 'Z 2', 'Z 3')]
z = m.channels['Z 2']
z.start()
z.end()
z.duration()
[ch.json for ch in m.channels.values()]
pm.compute(test, {'channelTable': 'main', 'iarg1': 100, 'iarg2': "testi", 'iarg3': -3.8})
chTables['a'] = salto.ChannelTable()
len([chTables['a'].add(name, ch) for name, ch in m.channels.items() if ch.type == 'acceleration'])
pm.compute('inclination', {'channelTable': 'a'})
m.channels['X'].resampledData(10,method="VRA")
m.channels['X'].resampledData(100,method="VRA")