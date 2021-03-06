//
//  salto_api.h
//  OpenSALTO
//
//  Copyright 2014 Marko Havu. Released under the terms of
//  GNU General Public License version 3 or later.
//

#ifndef _salto_api_h
#define _salto_api_h

#include <stdint.h>
#include <time.h>

struct Event;
typedef struct Event Event;

typedef enum {
    STANDARD_FIELDS = 1 << 0,
    CUSTOM_FIELDS = 1 << 1
} MetadataFields;

#define ALL_METADATA (STANDARD_FIELDS | CUSTOM_FIELDS)

typedef enum {
    CUSTOM_EVENT = 0,
    ACTION_EVENT = 1,
    ARTIFACT_EVENT = 2,
    CALCULATED_EVENT = 3,
    MARKER_EVENT = 4,
    TIMER_EVENT = 5
} EventVariety;

typedef struct {
    size_t n_args;
    size_t min_channels;
    size_t max_channels;
    const char **name;
    const char **format;
    const char **description;
    void **default_value;
} ComputationArgs;

// Application hooks to which plugins attach
typedef int (*ReadFileFunc)(const char *filename, const char *chTable);
typedef int (*WriteFileFunc)(const char *filename, const char *chTable);
typedef const char (*DescribeErrorFunc)(int err);
typedef int (*InitPluginFunc)(void *handle);
typedef int (*ComputeFunc)(void *inputs, void *outputs);
typedef size_t (*NumberOfOutputChannelsFunc)(const char* computation, size_t nInputChannels);

// Registration
int registerFileFormat(void *handle, const char *format, const char **exts, size_t n_exts);
int registerImportFunc(void *handle, const char *format, const char *funcname);
int registerExportFunc(void *handle, const char *format, const char *funcname);
int registerComputation(void *handle, const char *name, const char *funcname,
                        ComputationArgs *inputs, ComputationArgs *outputs);

// Exposing application capabilities back to plugins
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
uint8_t *newSparseUInt8Channel(const char *chTable, const char *name, size_t length, size_t nParts);
uint16_t *newSparseUInt16Channel(const char *chTable, const char *name, size_t length, size_t nParts);
uint32_t *newSparseUInt32Channel(const char *chTable, const char *name, size_t length, size_t nParts);
int8_t *newSparseInt8Channel(const char *chTable, const char *name, size_t length, size_t nParts);
int16_t *newSparseInt16Channel(const char *chTable, const char *name, size_t length, size_t nParts);
int32_t *newSparseInt32Channel(const char *chTable, const char *name, size_t length, size_t nParts);
float *newSparseFloatChannel(const char *chTable, const char *name, size_t length, size_t nParts);
double *newSparseDoubleChannel(const char *chTable, const char *name, size_t length, size_t nParts);
void deleteChannel(const char *chTable, const char *name);
int moveChannel(const char *fromChannelTable, const char *name, const char *toChannelTable);
int copyChannel(const char *fromChannelTable, const char *name, const char *toChannelTable);

void *getChannelData(const char *chTable, const char *name, size_t *length);
const char *getChannelName(const char *chTable, void *dataPtr);

int setFills(const char *chTable, const char *ch, size_t *positions, size_t *lengths);
int getFills(const char *chTable, const char *ch, size_t *positions, size_t *lengths);
size_t numberOfFills(const char *chTable, const char *ch);
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
struct timespec endTimeFromDuration(time_t start_sec, long start_nsec, double duration);
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

#endif
