OpenSALTO
=========

OpenSALTO is an open source signal analysis tool. It is released under
the GNU General Public License version 3 or later.

OpenSALTO includes an extensible backend for reading in signals,
exporting them in a different file format, performing computations on
them, and reporting the results of the analysis. A GUI frontend is
also being developed.

#APIs

OpenSALTO has plugin APIs for adding support for new file formats,
filtering and performing calculations, and reporting the results of
the analysis.

##Import and Export API

Support for new file formats can be added via plugins. Support can be
added for any kind of signal. Current and planned plugins add support
for signals including accelerometry, actigraphy, audio, ECG, EEG, EMG,
force, pressure, heartrate, and so on.

Plugins read or write one or more channels with corresponding metadata.
The metadata includes:
- channel names
- sample rates
- original resolution
- start time
- end time
- mark and model of the device that was used to acquire the data
- serial number of the device
- units (V, A, m, m/s, m/s/s, N, Pa, %, etc.)
- any other metadata as a JSON dictionary.
A list of events with event types, start times and end times may also
be included. Channels may have different sample rates, durations and
start times.

The plugins are expected to implement the following functions:
```C
int registerFileFormat(const char *format, const char **exts);
int readFile(const char *filename, const char *chTable);
int writeFile(const char *filename, const char *chTable);
const char *describeError(int err);
```

The following functions are available for the plugins:
```C
const char *getUniqueName(const char *chTable, const char *name);
uint8_t *newUInt8Channel(const char *chTable, const char *name, size_t length);
uint16_t *newUInt16Channel(const char *chTable, const char *name, size_t length);
uint32_t *newUInt32Channel(const char *chTable, const char *name, size_t length);
float *newFloatChannel(const char *chTable, const char *name, size_t length);
double *newDoubleChannel(const char *chTable, const char *name, size_t length);

void deleteChannel(const char *chTable, const char *name);

void *getChannelData(const char *chTable, const char *name, size_t *length);
const char *getChannelName(const char *chTable, void *ptr);

int setScaleAndOffset(const char *chTable, const char *ch, double scale, double offset);
double scale(const char *chTable, const char *ch);
double offset(const char *chTable, const char *ch);
int setUnit(const char *chTable, const char *ch, const char *unit);
const char *unit(const char *chTable, const char *ch);

int setSampleRate(const char *chTable, const char *ch, double samplerate);
double sampleRate(const char *chTable, const char *ch);
int setStartTime(const char *chTable, const char *ch, struct timespec start);
struct timespec startTime(const char *chTable, const char *ch);
struct timespec endTime(const char *chTable, const char *ch);
size_t length(const char *chTable, const char *ch);
double duration(const char *chTable, const char *ch);

int setDevice(const char *chTable, const char *ch, const char *device, const char *serial);
const char *device(const char *chTable, const char *ch);
const char *serial(const char *chTable, const char *ch);
```

##Filter API

Filter plugins are modules that perform computations on one or more
channels and return one or more new channels as a result.

##Report API

Report plugins are modules that perform computations on one or more
channels and return descriptive values, graphs and so on.