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

// Discovery

// Registration
int registerFileFormat(const char *format, const char **exts);

// Application hooks to which plugins attach
int readFile(const char *filename);
int writeFile(const char *filename, const char **ch);
const char *describeError(int err);
// initPlugin();

// Exposing application capabilities back to plugins
const char *getUniqueName(const char *name);
uint8_t *newUInt8Channel(const char *name, size_t length);
uint16_t *newUInt16Channel(const char *name, size_t length);
uint32_t *newUInt32Channel(const char *name, size_t length);
float *newFloatChannel(const char *name, size_t length);
double *newDoubleChannel(const char *name, size_t length);
void deleteChannel(const char *name);
void *getChannelData(const char *name);
const char *getChannelName(void *ptr);
int setSampleRate(const char *name, double samplerate);
int setScaleAndOffset(const char *name, double scale, double offset);
int setStartTime(const char *name, struct timespec start);
int setUnit(const char *name, const char *unit);
int setDevice(const char *name, const char *device, const char *serial);
double sampleRate(const char *name);
double scale(const char *name);
double offset(const char *name);
struct timespec startTime(const char *name);
struct timespec endTime(const char *name);
size_t length(const char *name);
double duration(const char *name);
const char *unit(const char *name);
const char *device(const char *name);
const char *serial(const char *name);

#endif
