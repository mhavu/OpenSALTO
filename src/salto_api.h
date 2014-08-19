//
//  salto_api.h
//  OpenSALTO
//
//  Created by Marko Havu on 2014-07-03.
//  Released under the terms of GNU General Public License version 3.
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

// Application hooks to which plugins attach
typedef int (*ReadFileFunc)(const char *filename, const char *chTable);
typedef int (*WriteFileFunc)(const char *filename, const char *chTable);
typedef const char (*DescribeErrorFunc)(int err);
typedef int (*InitPluginFunc)(void *handle);

// Registration
int registerFileFormat(void *handle, const char *format, const char **exts, size_t n_exts);
int registerImportFunc(void *handle, const char *format, const char *funcname);
int registerExportFunc(void *handle, const char *format, const char *funcname);

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

// TODO: Expose plugin settings to application


#endif
