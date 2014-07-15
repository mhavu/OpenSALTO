m = salto.channelTables["main"]
salto.readFile("TINY.ATS", "main")
z = m.channels["Z"]
salto.metadata(z)
