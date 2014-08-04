npa = np.ndarray(shape=(10), dtype=float)
ch = salto.Channel(npa, samplerate=40)
m = salto.channelTables['main']
p = salto.pluginManager.plugins
rf = salto.pluginManager.importFormats
wf = salto.pluginManager.exportFormats
alive = salto.pluginManager.query(ext=".ats")[0]
gcdc = salto.pluginManager.query(ext=".csv")[0]
rf[alive].read("TINY.ATS", alive, 'main')
rf[alive].read("HUGE.ATS", alive, 'main')
rf[gcdc].read("GCDC.CSV", gcdc, 'main')
meta = [[getattr(m.channels[z], metadata) for metadata in ('device', 'serial_no', 'resolution', 'samplerate', 'scale', 'offset', 'unit', 'start_sec', 'start_nsec', 'json')] for z in ('Z', 'Z 2', 'Z 3')]
z = m.channels['Z 2']
z.start()
z.end()
z.duration()