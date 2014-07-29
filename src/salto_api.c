//
//  salto_api.c
//  OpenSALTO
//
//  Created by Marko Havu on 2014-07-03.
//  Released under the terms of GNU General Public License version 3.
//

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "salto_api.h"
#include "salto.h"


int registerImportFunc(void *handle, const char *format, const char *funcname, const char **exts, size_t n_exts)
{
    // TODO: hide implementation details
    return setCallback(handle, "setImportFunc", format, funcname, exts, n_exts);
}

int registerExportFunc(void *handle, const char *format, const char *funcname, const char **exts, size_t n_exts)
{
    // TODO: hide implementation details
    return setCallback(handle, "setExportFunc", format, funcname, exts, n_exts);
}

static void *newIntegerChannel(const char *chTable, const char *name, size_t length, size_t size, int is_signed) {
    Channel ch = {
        .length = length,
        .bytes_per_sample = size,
        .scale = 1.0,
        .is_signed = is_signed
    };

    ch.ptr = calloc(length, size);
    if (ch.ptr)
        addChannel(chTable, name, &ch);  // TODO: Check return value

    return ch.ptr;
}

static void *newRealChannel(const char *chTable, const char *name, size_t length, size_t size) {
    Channel ch = {
        .length = length,
        .bytes_per_sample = size,
        .scale = nan(NULL),
        .offset = nan(NULL),
        .is_signed = 1
    };

    ch.ptr = calloc(length, size);
    if (ch.ptr)
        addChannel(chTable, name, &ch);  // TODO: Check return value

    return ch.ptr;
}

uint8_t *newUInt8Channel(const char *chTable, const char *name, size_t length) {
    return newIntegerChannel(chTable, name, length, sizeof(uint8_t), 0);
}

uint16_t *newUInt16Channel(const char *chTable, const char *name, size_t length) {
    return newIntegerChannel(chTable, name, length, sizeof(uint16_t), 0);
}

uint32_t *newUInt32Channel(const char *chTable, const char *name, size_t length) {
    return newIntegerChannel(chTable, name, length, sizeof(uint32_t), 0);
}

int8_t *newInt8Channel(const char *chTable, const char *name, size_t length) {
    return newIntegerChannel(chTable, name, length, sizeof(int8_t), 1);
}

int16_t *newInt16Channel(const char *chTable, const char *name, size_t length) {
    return newIntegerChannel(chTable, name, length, sizeof(int16_t), 1);
}

int32_t *newInt32Channel(const char *chTable, const char *name, size_t length) {
    return newIntegerChannel(chTable, name, length, sizeof(int32_t), 1);
}

float *newFloatChannel(const char *chTable, const char *name, size_t length) {
    return newRealChannel(chTable, name, length, sizeof(float));
}

double *newDoubleChannel(const char *chTable, const char *name, size_t length) {
    return newRealChannel(chTable, name, length, sizeof(double));
}

void deleteChannel(const char *chTable, const char *name) {
    Channel *ch = getChannel(chTable, name);
    if (ch) {
        removeChannel(chTable, name);
        free(ch->ptr);
    }
}


void *getChannelData(const char *chTable, const char *name, size_t *length) {
    Channel *ch = getChannel(chTable, name);
    *length = ch->length;
    return ch->ptr;
}

const char *getChannelName(const char *chTable, void *ptr) {
    return getNameForData(chTable, ptr);
}


int setSampleRate(const char *chTable, const char *name, double samplerate) {
    Channel *ch;

    ch = getChannel(chTable, name);
    if (ch) {
        ch->samplerate = samplerate;
    } else {
         return -1;
    }

    return 0;
}

int setScaleAndOffset(const char *chTable, const char *name, double scale, double offset) {
    Channel *ch;

    ch = getChannel(chTable, name);
    if (ch) {
        ch->scale = scale;
        ch->offset = offset;
    } else {
        return -1;
    }

    return 0;
}

int setStartTime(const char *chTable, const char *name, struct timespec start) {
    Channel *ch;

    ch = getChannel(chTable, name);
    if (ch) {
        ch->start_sec = start.tv_sec;
        ch->start_nsec = start.tv_nsec;
    } else {
        return -1;
    }

    return 0;
}

int setUnit(const char *chTable, const char *name, const char *unit) {
    Channel *ch;

    ch = getChannel(chTable, name);
    if (ch) {
        ch->unit = (char *)unit;
    } else {
        return -1;
    }

    return 0;
}

int setResolution(const char *chTable, const char *name, int resolution) {
    Channel *ch;

    ch = getChannel(chTable, name);
    if (ch) {
        ch->resolution = resolution;
    } else {
        return -1;
    }

    return 0;
}

int setDevice(const char *chTable, const char *name, const char *device, const char *serial) {
    Channel *ch;

    ch = getChannel(chTable, name);
    if (ch) {
        ch->device = (char *)device;
        ch->serial_no = (char *)serial;
    } else {
        return -1;
    }

    return 0;
}

double sampleRate(const char *chTable, const char *name) {
    Channel *ch;
    double samplerate;

    ch = getChannel(chTable, name);
    if (ch) {
        samplerate = ch->samplerate;
    } else {
        return NAN;
    }

    return samplerate;
}

double scale(const char *chTable, const char *name) {
    Channel *ch;
    double scale;

    ch = getChannel(chTable, name);
    if (ch) {
        scale = ch->scale;
    } else {
        return NAN;
    }

    return scale;
}

double offset(const char *chTable, const char *name) {
    Channel *ch;
    double offset;

    ch = getChannel(chTable, name);
    if (ch) {
        offset = ch->offset;
    } else {
        return NAN;
    }

    return offset;
}

struct timespec startTime(const char *chTable, const char *name) {
    struct timespec t;
    Channel *ch;

    ch = getChannel(chTable, name);
    if (ch) {
        t.tv_sec = ch->start_sec;
        t.tv_nsec = ch->start_nsec;
    } else {
        t.tv_sec = -1;
        t.tv_nsec = -1;
    }
    
    return t;
}

struct timespec endTime(const char *chTable, const char *name) {
    struct timespec t;
    double duration;
    time_t s;
    long ns;
    Channel *ch;

    ch = getChannel(chTable, name);
    if (ch) {
        duration = ch->length / ch->samplerate;
        s = duration;
        ns = (long)((duration - s) * 1.0e9);
        t.tv_sec = ch->start_sec + s + (ch->start_nsec + ns) / 1000000000;
        t.tv_nsec = (ch->start_nsec + ns) % 1000000000;
    } else {
        t.tv_sec = -1;
        t.tv_nsec = -1;
    }
    
    return t;
}

size_t length(const char *chTable, const char *name) {
    Channel *ch;
    size_t len = 0;

    ch = getChannel(chTable, name);
    if (ch) {
        len = ch->length;
    }

    return len;
}

double duration(const char *chTable, const char *name) {
    Channel *ch;
    double duration;

    ch = getChannel(chTable, name);
    if (ch) {
        duration = (ch->length - 1) / ch->samplerate;
    } else {
        duration = NAN;
    }
    
    return duration;
}

const char *unit(const char *chTable, const char *name) {
    Channel *ch;
    char *unit = NULL;

    ch = getChannel(chTable, name);
    if (ch) {
        unit = ch->unit;
    }
    
    return unit;
}

int resolution(const char *chTable, const char *name) {
    Channel *ch;
    int resolution = 0;

    ch = getChannel(chTable, name);
    if (ch) {
        resolution = ch->resolution;
    }

    return resolution;
}

const char *device(const char *chTable, const char *name) {
    Channel *ch;
    char *device = NULL;

    ch = getChannel(chTable, name);
    if (ch) {
        device = ch->device;
    }

    return device;
}

const char *serial(const char *chTable, const char *name) {
    Channel *ch;
    char *serial = NULL;

    ch = getChannel(chTable, name);
    if (ch) {
        serial = ch->serial_no;
    }

    return serial;
}

