#
#  gviewer.py
#
#  OpenSALTO
#
#  Reads GViewer HDF5 files.
#
#  Created by Marko Havu on 2014-08-05.
#  Released under the terms of GNU General Public License version 3.
#

import salto, h5py, json, math, datetime, time
import numpy as np

def datetimeFromMatlabDatenum(datenum):
    """Converts MATLAB datenum to a datetime object"""
    dt = datetime.datetime.fromordinal(int(datenum))
    dt = dt - datetime.timedelta(days = 366)
    dt = dt + datetime.timedelta(days = math.fmod(datenum, 1.0))
    epsilon = round(dt.microsecond / 1000.0) * 1000 - dt.microsecond
    dt = dt + datetime.timedelta(microseconds = epsilon)
    return dt
    
def timespecFromString(datestr):
    """Converts date string to a timespec tuple (s, ns)"""
    dt = datetime.datetime.strptime(datestr, "%Y-%m-%d %H:%M:%S.%f")
    frac_sec, sec = math.modf(dt.timestamp())
    return (round(sec), round(1e9 * frac_sec))

class Plugin(salto.Plugin):
    """OpenSALTO plugin for reading and writing GViewer HDF5 files"""
    def __init__(self, manager):
        super(Plugin, self).__init__(manager)
        self.registerFormat("GViewer", [".h5"])
        self.setImportFunc("GViewer", self.read_)
        self.setExportFunc("GViewer", self.write_)
    def read_(self, filename, chTable):
        with h5py.File(filename, 'r') as f:
            metadata = dict(subjectID = f['measurement'].attrs['subject'].item().decode('utf-8'),
                            notes = "\n".join([l.decode('utf-8')
                                               for l in f['measurement'].attrs['notes'].tolist()]),
                            measurement = int(f['measurement'].value.item()))
            samplerate = float(f.attrs['samplerate'].item())
            device = f['source'].attrs.get('device', np.array(b"unknown", dtype='S')).item().decode('utf-8')
            serial = f['source'].attrs.get('serialno', np.array(b"unknown", dtype='S')).item().decode('utf-8')
            resolution = f['source'].attrs['resolution'].item()
            datasets = [f[d] for d in f.keys() if d.startswith('dataset')]
            # Multi-channel variables 
            name = ["\n".join([l.decode('utf-8') for l in d.attrs['name'].tolist()]) for d in datasets]
            starttime = [timespecFromString(d.attrs['starttime'].item().decode('utf-8'))
                         for d in datasets]
            data = [9.81 * d[()] for d in datasets]
            events = [d.attrs.get('events') for d in datasets]
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
                    start = datetimeFromMatlabDatenum(e['startTime'])
                    start_sec = round(time.mktime(start.timetuple()))
                    start_nsec = 1000 * start.microsecond
                    end = datetimeFromMatlabDatenum(e['endTime'])
                    end_sec = round(time.mktime(end.timetuple()))
                    end_nsec = 1000 * end.microsecond
                    event = salto.Event(type = salto.ACTION_EVENT,
                                        subtype = e['description'].decode('utf-8'),
                                        start_sec = start_sec, start_nsec = start_nsec,
                                        end_sec = end_sec, end_nsec = end_nsec)
                    ch.events.add(event)
                chTable.add(chTable.getUnique(n), ch)
    def write_(self, filename, chTable):
        with h5py.File(file, 'w') as f:
            pass
