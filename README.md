OpenSALTO
=========

OpenSALTO is an open source signal analysis tool. It is released under
the GNU General Public License version 3 or later.

OpenSALTO includes an extensible backend for reading in signals,
exporting them in a different file format, performing computations on
them, and reporting the results of the analysis. A GUI frontend is
also being developed.

#Plugin API

OpenSALTO has plugin APIs for adding support for new file formats,
filtering and performing calculations, and reporting the results of
the analysis.

##Import and Export Plugins

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

##Computation Plugins

Filters and calculation and report plugins are modules that perform
computations on one or more channels. Filters return one or more new
channels as a result, whereas calculation and report plugins return
descriptive values, graphs and so on.

##API

The plugins are expected to implement the following functions:
```C
int initPlugin(void *handle);
const char *describeError(int err);
size_t nOutputChannels(const char* computation, size_t nInputChannels); // Computation plugins only
```

The plugins register themselves using the following functions:
```C
int registerFileFormat(void *handle, const char *format, const char **exts, size_t n_exts);
int registerImportFunc(void *handle, const char *format, const char *funcname); // funcname is of type ReadFileFunc
int registerExportFunc(void *handle, const char *format, const char *funcname); // funcname is of type WriteFileFunc
int registerComputation(void *handle, const char *name, const char *funcname,
                        ComputationArgs *inputs, ComputationArgs *outputs); // funcname is of type ComputeFunc
typedef int (*ReadFileFunc)(const char *filename, const char *chTable);
typedef int (*WriteFileFunc)(const char *filename, const char *chTable);
typedef int (*ComputeFunc)(void *inputs, void *outputs);
```

The following functions are available for the plugins:
```C
const char *newChannelTable(const char *name);
void deleteChannelTable(const char *name);
const char *getUniqueName(const char *chTable, const char *name);
const char **getChannelNames(const char *chTable, size_t *size);

uint8_t *newUInt8Channel(const char *chTable, const char *name, size_t length);
uint16_t *newUInt16Channel(const char *chTable, const char *name, size_t length);
uint32_t *newUInt32Channel(const char *chTable, const char *name, size_t length);
int8_t *newInt8Channel(const char *chTable, const char *name, size_t length);
int16_t *newInt16Channel(const char *chTable, const char *name, size_t length);
int32_t *newInt32Channel(const char *chTable, const char *name, size_t length);
float *newFloatChannel(const char *chTable, const char *name, size_t length);
double *newDoubleChannel(const char *chTable, const char *name, size_t length);
int newCombinationChannel(const char *chTable, const char *name, const char *fromChannelTable, void *fillValues);
void deleteChannel(const char *chTable, const char *name);
int moveChannel(const char *fromChannelTable, const char *name, const char *toChannelTable);
int copyChannel(const char *fromChannelTable, const char *name, const char *toChannelTable);

void *getChannelData(const char *chTable, const char *name, size_t *length);
const char *getChannelName(const char *chTable, void *dataPtr);

int setScaleAndOffset(const char *chTable, const char *ch, double scale, double offset);
double scale(const char *chTable, const char *ch);
double offset(const char *chTable, const char *ch);
int setUnit(const char *chTable, const char *ch, const char *unit);
const char *unit(const char *chTable, const char *ch);
int setSignalType(const char *chTable, const char *ch, const char *type);
const char *signalType(const char *chTable, const char *ch);
int setResolution(const char *chTable, const char *ch, int resolution);
int resolution(const char *chTable, const char *ch);
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
int setMetadata(const char *chTable, const char *ch, const char *json);
const char *metadata(const char *chTable, const char *ch, MetadataFields fields);

int addEvent(const char *chTable, const char *ch, Event *event);
int removeEvent(const char *chTable, const char *ch, Event *event);
Event **getEvents(const char *chTable, const char *ch, size_t *size);
void clearEvents(const char *chTable, const char *ch);

Event *newEvent(EventVariety type, const char *subtype, struct timespec start, struct timespec end, const char *description);
void discardEvent(Event *event);
const char *channelsWithEventType(const char *chTable, EventVariety type, const char *subtype);
int setEventType(Event *event, EventVariety type, const char *subtype);
int moveEvent(Event *event, struct timespec start, struct timespec end);
int setEventDescription(Event *event, const char *description);
```