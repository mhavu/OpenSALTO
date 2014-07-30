m = salto.channelTables['main']
p = salto.pluginManager.plugins
rf = salto.pluginManager.importFormats
wf = salto.pluginManager.exportFormats
alive = salto.pluginManager.query(ext=".ats")
gcdc = salto.pluginManager.query(ext=".csv")
rf[alive].read("TINY.ATS", alive, 'main')
rf[gcdc].read("GCDC.CSV", gcdc, 'main')
z = m.channels['Z']
salto.metadata(z)
