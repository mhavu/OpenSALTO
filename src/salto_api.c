//
//  salto_api.c
//  OpenSALTO
//
//  Created by Marko Havu on 2014-07-03.
//  Released under the terms of GNU General Public License version 3.
//

#include "salto_api.h"
#include "salto.h"


int registerImportFunc(void *handle, const char *format, const char *funcname) {
    return setCallback(handle, "Import", format, funcname);
}

int registerExportFunc(void *handle, const char *format, const char *funcname) {
    return setCallback(handle, "Export", format, funcname);
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


int setSampleRate(const char *chTable, const char *name, double samplerate) {
    Channel *ch;
    int result = 0;

    ch = getChannel(chTable, name);
    if (ch) {
        ch->samplerate = samplerate;
    } else {
         result = -1;
    }

    return result;
}

int setScaleAndOffset(const char *chTable, const char *name, double scale, double offset) {
    Channel *ch;
    int result = 0;

    ch = getChannel(chTable, name);
    if (ch) {
        ch->scale = scale;
        ch->offset = offset;
    } else {
        result = -1;
    }

    return result;
}

int setStartTime(const char *chTable, const char *name, struct timespec start) {
    Channel *ch;
    int result = 0;

    ch = getChannel(chTable, name);
    if (ch) {
        ch->start_sec = start.tv_sec;
        ch->start_nsec = start.tv_nsec;
    } else {
        result = -1;
    }

    return result;
}

int setUnit(const char *chTable, const char *name, const char *unit) {
    Channel *ch;
    char *ptr;
    size_t length;
    int result = 0;

    ch = getChannel(chTable, name);
    if (ch) {
        ptr = ch->unit;
        length = strlen(unit) + 1;
        ch->unit = malloc(length);
        strlcpy(ch->unit, unit, length);
        free(ptr);
    } else {
        result = -1;
    }

    return result;
}

int setResolution(const char *chTable, const char *name, int resolution) {
    Channel *ch;
    int result = 0;

    ch = getChannel(chTable, name);
    if (ch) {
        ch->resolution = resolution;
    } else {
        result = -1;
    }

    return result;
}

int setDevice(const char *chTable, const char *name, const char *device, const char *serial) {
    Channel *ch;
    char *ptr;
    size_t length;
    int result = 0;

    ch = getChannel(chTable, name);
    if (ch) {
        ptr = ch->device;
        length = strlen(device) + 1;
        ch->device = malloc(length);
        strlcpy(ch->device, device, length);
        free(ptr);
        ptr = ch->serial_no;
        length = strlen(serial) + 1;
        ch->serial_no = malloc(length);
        strlcpy(ch->serial_no, serial, length);
        free(ptr);
    } else {
        result = -1;
    }

    return result;
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
    size_t length;
    Channel *ch;

    ch = getChannel(chTable, name);
    if (ch) {
        channelData(ch, &length);
        if (length > 0) {
            duration = (length - 1) / ch->samplerate;
            s = duration;
            ns = (long)((duration - s) * 1.0e9);
            t.tv_sec = ch->start_sec + s + (ch->start_nsec + ns) / 1000000000;
            t.tv_nsec = (ch->start_nsec + ns) % 1000000000;
        } else {
            t.tv_sec = -1;
            t.tv_nsec = -1;
        }
    } else {
        t.tv_sec = -1;
        t.tv_nsec = -1;
    }
    
    return t;
}

size_t length(const char *chTable, const char *name) {
    size_t len;

    getChannelData(chTable, name, &len);

    return len;
}

const char *unit(const char *chTable, const char *name) {
    Channel *ch;
    const char *unit = NULL;

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
    const char *device = NULL;

    ch = getChannel(chTable, name);
    if (ch) {
        device = ch->device;
    }

    return device;
}

const char *serial(const char *chTable, const char *name) {
    Channel *ch;
    const char *serial = NULL;

    ch = getChannel(chTable, name);
    if (ch) {
        serial = ch->serial_no;
    }

    return serial;
}

int setMetadata(const char *chTable, const char *name, const char *json) {
    Channel *ch;
    char *ptr;
    size_t length;
    int result = 0;

    ch = getChannel(chTable, name);
    if (ch) {
        ptr = ch->json;
        length = strlen(json) + 1;
        ch->json = malloc(length);
        strlcpy(ch->json, json, length);
        free(ptr);
        // TODO: handle standard fields
    } else {
        result = -1;
    }

    return result;
}

const char *metadata(const char *chTable, const char *name, MetadataFields fields) {
    Channel *ch;
    const char *json = "{}";

    ch = getChannel(chTable, name);
    if (ch) {
        if (fields & CUSTOM_FIELDS) {
            json = ch->json;
        }
        // TODO: handle standard fields
    }

    return json;
}

int setSignalType(const char *chTable, const char *name, const char *type) {
    Channel *ch;
    char *ptr;
    size_t length;
    int result = 0;

    ch = getChannel(chTable, name);
    if (ch) {
        ptr = ch->type;
        length = strlen(type) + 1;
        ch->type = malloc(length);
        strlcpy(ch->type, type, length);
        free(ptr);
    } else {
        result = -1;
    }

    return result;
}

const char *signalType(const char *chTable, const char *name) {
    Channel *ch;
    const char *type = NULL;

    ch = getChannel(chTable, name);
    if (ch) {
        type = ch->type;
    }

    return type;
}

int moveChannel(const char *fromChannelTable, const char *name, const char *toChannelTable) {
    int result;

    result = copyChannel(fromChannelTable, name, toChannelTable);
    if (result == 0)
        deleteChannel(fromChannelTable, name);

    return result;
}

int copyChannel(const char *fromChannelTable, const char *name, const char *toChannelTable) {
    Channel *ch;
    int result = 0;

    ch = getChannel(fromChannelTable, name);
    if (ch) {
        result = addChannel(toChannelTable, name, ch);
    }

    return result;
}

const char *channelsWithEventType(const char *chTable, EventVariety type, const char *subtype) {
    const char *resultChTable;
    const char **chNames;
    size_t size, i;

    resultChTable = newChannelTable("Channels with events");
    if (resultChTable) {
        chNames = getChannelNames(chTable, &size);
        if (chNames) {
            for (i = 0; i < size; i++) {
                if (copyChannel(chTable, chNames[i], resultChTable) != 0) {
                    deleteChannelTable(resultChTable);
                    resultChTable = NULL;
                    break;
                }
            }
            free(chNames);
        }
    }

    return resultChTable;
}