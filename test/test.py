m = salto.channelTables["main"]
p = salto.pluginManager.plugins
rf = salto.pluginManager.importFormats
wf = salto.pluginManager.exportFormats
alive = "Alive Heart and Activity Monitor"
gcdc = "Gulf Coast Data Concepts"
rf[alive].read("TINY.ATS", alive, "main")
rf[gcdc].read("GCDC.CSV", gcdc, "main")
z = m.channels["Z"]
salto.metadata(z)
salto.pluginManager.register(salto.Plugin(salto.pluginManager))
salto.pluginManager.query(ext='.ats')
salto.pluginManager.query(ext='.csv')
