m = salto.channelTables["main"]
salto.readFile("GCDC.CSV", "main")
z = m.channels["Z"]
salto.metadata(z)
