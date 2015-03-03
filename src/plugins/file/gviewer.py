#
#  gviewer.py
#  OpenSALTO
#
#  Reads and writes GViewer HDF5 files.
#
#  Copyright 2014 Marko Havu. Released under the terms of
#  GNU General Public License version 3 or later.
#

import salto, h5py, json, math, getpass
import numpy as np
from datetime import datetime, timedelta

def datetimeFromMatlabDatenum(datenum):
    """Converts MATLAB datenum to a datetime object"""
    dt = datetime.fromordinal(int(datenum))
    dt = dt - timedelta(days = 366)
    dt = dt + timedelta(days = math.fmod(datenum, 1.0))
    epsilon = round(dt.microsecond / 1000.0) * 1000 - dt.microsecond
    dt = dt + timedelta(microseconds = epsilon)
    return dt
    
def datenumFromTimespec(sec, nsec):
    """Converts a timespec to MATLAB datenum"""
    dt = datetime.fromtimestamp(sec) + timedelta(days = 366)
    datenum = dt.toordinal()
    partial = dt - datetime.fromordinal(datenum)
    fraction = (partial.seconds + nsec / 1e9) / 86400.0
    datenum = datenum + fraction
    return datenum

def timespecFromString(datestr):
    """Converts date string to a timespec tuple (s, ns)"""
    dt = datetime.strptime(datestr, "%Y-%m-%d %H:%M:%S.%f")
    frac_sec, sec = math.modf(dt.timestamp())
    return (round(sec), round(1e9 * frac_sec))

class Plugin(salto.Plugin):
    """OpenSALTO plugin for reading and writing GViewer HDF5 files"""
    def __init__(self, manager):
        super(Plugin, self).__init__(manager)
        self.registerFormat("GViewer", [".h5"])
        self.setImportFunc("GViewer", self._read)
        self.setExportFunc("GViewer", self._write)
    def _read(self, filename, chTable):
        with h5py.File(filename, 'r') as f:
            metadata = dict(subjectID = f['measurement'].attrs['subject'].item().decode('utf-8'),
                            notes = "\n".join([l.decode('utf-8')
                                               for l in f['measurement'].attrs['notes'].tolist()]),
                            measurement = int(f['measurement'].value.item()))
            samplerate = float(f.attrs['samplerate'].item())
            device = (f['source'].attrs['device'].item().decode('utf-8')
                      if 'device' in f['source'].attrs
                      else "unknown")
            serial = (f['source'].attrs['serialno'].item().decode('utf-8')
                      if 'serialno' in f['source'].attrs
                      else "unknown")
            resolution = f['source'].attrs['resolution'].item()
            datasets = [f[d] for d in f.keys() if d.startswith('dataset')]
            # Multi-channel variables 
            name = ["\n".join([l.decode('utf-8') for l in d.attrs['name'].tolist()]) for d in datasets]
            starttime = [timespecFromString(d.attrs['starttime'].item().decode('utf-8'))
                         for d in datasets]
            data = [9.81 * d[()] for d in datasets]
            events = [d.attrs['events'] if 'events' in d.attrs else [] for d in datasets]
            for n, d, t, elist in zip(name, data, starttime, events):
                ch = salto.Channel(d,
                                   samplerate = samplerate,
                                   unit = "m/s^2",
                                   type = "acceleration",
                                   start_sec = t[0],
                                   start_nsec = t[1],
                                   device = device,
                                   serial_no = serial,
                                   resolution = resolution,
                                   json = json.dumps(metadata))
                for e in elist:
                    start = datetimeFromMatlabDatenum(e['startTime']).timestamp()
                    end = datetimeFromMatlabDatenum(e['endTime']).timestamp()
                    event = salto.Event(type = salto.ACTION_EVENT,
                                        subtype = e['description'].decode('utf-8'),
                                        start_sec = int(start), start_nsec = int(math.fmod(start, 1.0)),
                                        end_sec = int(end), end_nsec = int(math.fmod(end, 1.0)))
                    ch.events.add(event)
                chTable.add(chTable.getUnique(n), ch)
    def _write(self, filename, chTable):
        timestr = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        username = getpass.getuser()
        samplerate = 0.0
        with h5py.File(filename, 'w') as f:
            f.attrs['format'] = np.array(["GViewer"], dtype = 'S')
            f.attrs['majorversion'] = np.array([1], dtype = 'u1')
            f.attrs['minorversion'] = np.array([3], dtype = 'u1')
            f.attrs['programversion'] = np.array(["OpenSALTO"], dtype = 'S')
            f.attrs['creator'] = np.array([username], dtype = 'S')
            f.attrs['created'] = np.array([timestr], dtype = 'S')
            f.attrs['lastmodifier'] = np.array([username], dtype = 'S')
            f.attrs['lastmodified'] = np.array([timestr], dtype = 'S')
            f.attrs['filtering'] = np.array(["unknown"], dtype = 'S')
            f.attrs['samplerate'] = np.array([samplerate])
            dset = None
            chNum = 0
            for name, ch in chTable.channels.items():
                chNum += 1
                if ch.type == "acceleration" and not ch.fill_values:
                    # TODO: Add support for sparse channels.
                    if ch.samplerate > samplerate: samplerate = ch.samplerate
                    dsname = "dataset" + str(chNum)
                    dset = f.create_dataset(dsname, (len(ch.data),), dtype = 'f4')
                    dset[...] = (ch.scale * ch.data + ch.offset) / 9.81
                    dset.attrs['name'] = np.array([name], dtype = 'S')
                    start_str = ch.start().strftime("%Y-%m-%d %H:%M:%S.%f")[0:23]
                    end_str = ch.end().strftime("%Y-%m-%d %H:%M:%S.%f")[0:23]
                    dset.attrs['starttime'] = np.array([start_str], dtype = 'S')
                    dset.attrs['endtime'] = np.array([end_str], dtype = 'S')
                    events = [(type, e.subtype, [1.0, 1.0, 0.8],
                               datenumFromTimespec(e.start_sec, e.start_nsec),
                               datenumFromTimespec(e.end_sec, e.end_nsec))
                              for type, e in enumerate(ch.events, 1000)]
                    maxdescrlen = max([len(e.subtype) for e in ch.events]) if ch.events else 8
                    descrtype = 'S' + str(8 * math.ceil(maxdescrlen / 8.0))
                    eventtype = [('type', '<f8'),
                                 ('description', descrtype),
                                 ('color', '<f8', (3,)),
                                 ('startTime', '<f8'),
                                 ('endTime', '<f8')]
                    dset.attrs['events'] = np.array(events, dtype = eventtype)
            measurementNum = salto.sessionData.get('measurement', 0)
            measurement = f.create_dataset("measurement", data = np.array([measurementNum]))
            subject = salto.sessionData.get('subject', "unknown")
            measurement.attrs['subject'] = np.array([subject], dtype = 'S')
            notes = salto.sessionData.get('notes', "")
            measurement.attrs['notes'] = np.array([notes], dtype = 'S')
            start_str = min([d.attrs['starttime'].item().decode('utf-8')
                             for d in f.values() if 'starttime' in d.attrs]) if dset else timestr
            src = np.array([("unknown", start_str)], dtype=[('filename', 'S8'), ('starttime', 'S24')])
            source = f.create_dataset("source", data = src)
            source.attrs['device'] = np.array([""], dtype = 'S')
            source.attrs['serialno'] = np.array([""], dtype = 'S')
            source.attrs['samplerate'] = np.array([samplerate])
            source.attrs['resolution'] = np.array([32], dtype = 'u2')
            source.attrs['channels'] = np.array([chNum], dtype = 'u2')

