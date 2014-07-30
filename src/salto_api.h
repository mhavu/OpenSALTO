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
const char *getUniqueName(const char *chTable, const char *name);
uint8_t *newUInt8Channel(const char *chTable, const char *name, size_t length);
uint16_t *newUInt16Channel(const char *chTable, const char *name, size_t length);
uint32_t *newUInt32Channel(const char *chTable, const char *name, size_t length);
int8_t *newInt8Channel(const char *chTable, const char *name, size_t length);
int16_t *newInt16Channel(const char *chTable, const char *name, size_t length);
int32_t *newInt32Channel(const char *chTable, const char *name, size_t length);
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

// TODO: Add these
// int setMetadata(const char *chTable, const char *ch, const char *json, int custom);
// const char *metadata(const char *chTable, const char *ch, int custom);

// int setType(const char *chTable, const char *ch, const char *type);  // electrical, mechanical, biosignal, audio, etc.
// const char *type(const char *chTable, const char *ch);

// int addEvent(const char *name, const char *type, struct timespec start, struct timespec end, const char *description);
// Event *eventsByName(const char *name, size_t *size);
// Event *eventsByType(const cahr *type, size_t *size);

#endif
