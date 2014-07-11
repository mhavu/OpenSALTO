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


static void *newIntegerChannel(const char *name, size_t length, size_t size) {
    Channel ch = {
        .length = length,
        .bytes_per_sample = size,
        .scale = 1.0
    };

    ch.ptr = calloc(length, size);
    if (ch.ptr)
        addChannel(name, &ch);

    return ch.ptr;
}

static void *newRealChannel(const char *name, size_t length, size_t size) {
    Channel ch = {
        .length = length,
        .bytes_per_sample = size,
        .scale = nan(NULL),
        .offset = nan(NULL)
    };

    ch.ptr = calloc(length, size);
    if (ch.ptr)
        addChannel(name, &ch);

    return ch.ptr;
}

uint8_t *newUInt8Channel(const char *name, size_t length) {
    return newIntegerChannel(name, length, sizeof(uint8_t));
}

uint16_t *newUInt16Channel(const char *name, size_t length) {
    return newIntegerChannel(name, length, sizeof(uint16_t));
}

uint32_t *newUInt32Channel(const char *name, size_t length) {
    return newIntegerChannel(name, length, sizeof(uint32_t));
}

float *newFloatChannel(const char *name, size_t length) {
    return newIntegerChannel(name, length, sizeof(float));
}

double *newDoubleChannel(const char *name, size_t length) {
    return newIntegerChannel(name, length, sizeof(double));
}

void deleteChannel(const char *name) {
    Channel *ch = getChannel(name);
    removeChannel(name);
    free(ch->ptr);
}


void *getChannelData(const char *name) {
    return getChannel(name)->ptr;
}

const char *getChannelName(void *ptr) {
    return getNameForData(ptr);
}


int setSampleRate(const char *name, double samplerate) {
    Channel *ch;

    ch = getChannel(name);
    if (ch) {
        ch->samplerate = samplerate;
    } else {
         return -1;
    }

    return 0;
}

int setScaleAndOffset(const char *name, double scale, double offset) {
    Channel *ch;

    ch = getChannel(name);
    if (ch) {
        ch->scale = scale;
        ch->offset = offset;
    } else {
        return -1;
    }

    return 0;
}

int setStartTime(const char *name, struct timespec start) {
    Channel *ch;

    ch = getChannel(name);
    if (ch) {
        ch->start_sec = start.tv_sec;
        ch->start_nsec = start.tv_nsec;
    } else {
        return -1;
    }

    return 0;
}

int setUnit(const char *name, const char *unit) {
    Channel *ch;

    ch = getChannel(name);
    if (ch) {
        ch->unit = (char *)unit;
    } else {
        return -1;
    }

    return 0;
}

int setDevice(const char *name, const char *device, const char *serial) {
    Channel *ch;

    ch = getChannel(name);
    if (ch) {
        ch->device = (char *)device;
        ch->serial_no = (char *)serial;
    } else {
        return -1;
    }

    return 0;
}

double sampleRate(const char *name) {
    Channel *ch;
    double samplerate;

    ch = getChannel(name);
    if (ch) {
        samplerate = ch->samplerate;
    } else {
        return NAN;
    }

    return samplerate;
}

double scale(const char *name) {
    Channel *ch;
    double scale;

    ch = getChannel(name);
    if (ch) {
        scale = ch->scale;
    } else {
        return NAN;
    }

    return scale;
}

double offset(const char *name) {
    Channel *ch;
    double offset;

    ch = getChannel(name);
    if (ch) {
        offset = ch->offset;
    } else {
        return NAN;
    }

    return offset;
}

struct timespec startTime(const char *name) {
    struct timespec t;
    Channel *ch;

    ch = getChannel(name);
    if (ch) {
        t.tv_sec = ch->start_sec;
        t.tv_nsec = ch->start_nsec;
    } else {
        t.tv_sec = -1;
        t.tv_nsec = -1;
    }
    
    return t;
}

struct timespec endTime(const char *name) {
    struct timespec t;
    double duration;
    time_t s;
    long ns;
    Channel *ch;

    ch = getChannel(name);
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

size_t length(const char *name) {
    Channel *ch;
    size_t len = 0;

    ch = getChannel(name);
    if (ch) {
        len = ch->length;
    }

    return len;
}

double duration(const char *name) {
    Channel *ch;
    double duration;

    ch = getChannel(name);
    if (ch) {
        duration = ch->length / ch->samplerate;
    } else {
        duration = NAN;
    }
    
    return duration;
}

const char *unit(const char *name) {
    Channel *ch;
    char *unit = NULL;

    ch = getChannel(name);
    if (ch) {
        unit = ch->unit;
    }
    
    return unit;
}

const char *device(const char *name) {
    Channel *ch;
    char *device = NULL;

    ch = getChannel(name);
    if (ch) {
        device = ch->device;
    }

    return device;
}

const char *serial(const char *name) {
    Channel *ch;
    char *serial = NULL;

    ch = getChannel(name);
    if (ch) {
        serial = ch->serial_no;
    }

    return serial;
}

