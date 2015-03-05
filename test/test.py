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
alive = pm.query(ext=".ats").pop()
gcdc = 'Gulf Coast Data Concepts'
hookie = 'Hookie AM20 Activity Meter'
test = 'Plugin test'
pm.read("/Volumes/Scratch/EMG24/Ying/07_ACC 4-5 days", 'directory', 'main')
pm.read("TINY.ATS", 'main')
pm.read("HUGE.ATS", 'main')
pm.read("GCDC.CSV", gcdc, 'main')
pm.read("/Volumes/Scratch/EMG24/Katri/03_7.10/DATA0019.DAT", hookie, 'main')
pm.read("/Volumes/Scratch/EMG24/Katri/03_7.10/DATA0020.DAT", hookie, 'main')
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


chTables = salto.channelTables
m = chTables['main']
pm = salto.pluginManager
rf = pm.importFormats
wf = pm.exportFormats
comp = pm.computations
alive = pm.query(ext=".ats").pop()
gcdc = 'Gulf Coast Data Concepts'
hookie = 'Hookie AM20 Activity Meter'
test = 'Plugin test'
pm.read("/Volumes/Scratch/EMG24/Ying/07_ACC 4-5 days/DATA-011.CSV", gcdc, 'main')
chTables['a'] = salto.ChannelTable()
len([chTables['a'].add(name, ch) for name, ch in m.channels.items() if ch.type == 'acceleration'])
pm.compute('inclination', {'channelTable': 'a'})
chTables['i'] = salto.ChannelTable()
chTables['i'].add('i', chTables['inclination'].channels['Z inclination'])
pm.compute('threshold', {'channelTable': 'i', 'lower': 0.0, 'upper': None})

salto.datetimeFromTimespec(1415375739,429999999)

chTables = salto.channelTables
m = chTables['main']
pm = salto.pluginManager
rf = pm.importFormats
wf = pm.exportFormats
comp = pm.computations
alive = pm.query(ext=".ats").pop()
gcdc = 'Gulf Coast Data Concepts'
hookie = 'Hookie AM20 Activity Meter'
test = 'Plugin test'
pm.read("/Volumes/Scratch/EMG24/Katri/03_7.10/DATA0019.DAT", hookie, 'main')
pm.read("/Volumes/Scratch/EMG24/Katri/03_7.10/DATA0020.DAT", hookie, 'main')
salto.Channel.collate([m.channels['X'], m.channels['X 2']])