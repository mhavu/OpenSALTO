//
//  hookie.c
//  OpenSALTO
//
//  Imports Hookie AM20 Activity Meter data files.
//  Check http://www.hookiemeter.com/support for further specifications.
//
//  Copyright 2014 Marko Havu. Released under the terms of
//  GNU General Public License version 3 or later.
//

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <sys/stat.h>
#include <math.h>
#include "salto_api.h"

typedef struct {
    const char *name;
    int16_t *buffer;
    int16_t *data;
} Channel;

typedef enum {
    SUCCESS = 0,
    FOPEN_FAILED,
    INVALID_FORMAT,
    INVALID_HEADER,
    INVALID_FILE,
    UNSUPPORTED_COMPRESSION,
    ALLOCATION_FAILED
} Error;

static const char headerTop[] = "\
*******************************\x0A\
*  Hookie Technologies Ltd    *\x0A\
*  Activity Sensor            *\x0A";

static uint16_t letoh16(uint8_t *buffer) {
    return (uint16_t)buffer[0] | (uint16_t)buffer[1] << 8;
}

static off_t fsize(const char *filename) {
    struct stat st;

    if (stat(filename, &st) != 0) {
        perror("stat()");
        return -1;
    }

    return st.st_size;
}

static time_t bcdToTime(uint8_t *buffer) {
    struct tm time;

    time.tm_mday = 10 * (buffer[0] >> 4) + (buffer[0] & 0x0F);
    time.tm_mon = 10 * (buffer[1] >> 4) + (buffer[1] & 0x0F);
    time.tm_year = 100 + 10 * (buffer[2] >> 4) + (buffer[2] & 0x0F);
    time.tm_hour = 10 * (buffer[3] >> 4) + (buffer[3] & 0x0F);
    time.tm_min = 10 * (buffer[4] >> 4) + (buffer[4] & 0x0F);
    time.tm_sec = 10 * (buffer[5] >> 4) + (buffer[5] & 0x0F);

    return mktime(&time);
}

int readFile(const char *filename, const char *chTable) {
    uint8_t buffer[512], *sample;
    long long blk, nBlocks, fileLength, part, nParts;
    size_t length, i, *partLen;
    int headerIsValid, isDynamic = 0;
    int16_t *fillValues;
    int ch;
    const int samplesPerBlock = 84;
    const int nChannels = 3;
    Channel channel[3];
    const char *names[3] = {"X", "Y", "Z"};
    char serialno[28], tag[28], value[28], json[512] = "{ ";
    double samplerate;
    Error err = SUCCESS;
    struct timespec startTime, t;
    time_t latestStart, *timecodes = NULL;
    const char *tmpTable, *partTable;
    char **name;

    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        perror("fopen()");
        err = FOPEN_FAILED;
    }

    if (!err) {
        // Read the file header (512 bytes).
        fileLength = fsize(filename);
        headerIsValid = (fread(buffer, 1, 512, fp) == 512 &&
                         memcmp((const char *)buffer, headerTop, strlen(headerTop)) == 0);
        if (headerIsValid && fileLength % 512 == 0) {
            for (i = strlen(headerTop); i < 480; i += 32) {
                if (sscanf((const char *)&buffer[i], "* %28[^\x0A:*]: %28s *\x0A", tag, value) > 0) {
                    if (strcmp(tag, "S/N") == 0) {
                        strlcpy(serialno, value, sizeof(serialno));
                    } else if (strcmp(tag, "Data rate") == 0) {
                        if (sscanf(value, "%lfHz", &samplerate) <= 0) {
                            fclose(fp);
                            fprintf(stderr, "readFile(): Invalid sample rate %s\n", value);
                            err = INVALID_HEADER;
                            break;
                        }
                    } else if (strcmp(tag, "Data compression") == 0 ||
                               strcmp(tag, "Activity threshold") == 0 ||
                               strcmp(tag, "Inactivity threshold") == 0 ||
                               strcmp(tag, "Inactivity time") == 0 ||
                               strcmp(tag, "Acceleration coupling") == 0) {
                        snprintf(strchr(json, 0), sizeof(json) - strlen(json), "\"%s\": %s,", tag, value);
                        if (strcmp(tag, "Data compression") == 0 && atoi(value) != 0) {
                            fclose(fp);
                            fprintf(stderr, "readFile(): Data compression %s not supported\n", value);
                            err = UNSUPPORTED_COMPRESSION;
                            break;
                        }
                        if (strcmp(tag, "Acceleration coupling") == 0 && strcmp(value, "AC") == 0) {
                            isDynamic = 1;
                        }
                   }
                }
            }
        } else {
            fclose(fp);
            fprintf(stderr, "readFile(): Unknown file format (not Hookie .DAT)\n");
            err = INVALID_FORMAT;
        }
    }

    if (!err) {
        // Allocate resources.
        nBlocks = fileLength / 512 - 1;
        length = samplesPerBlock * nBlocks;
        for (ch = 0; ch < nChannels; ch++) {
            channel[ch].buffer = calloc(length, sizeof(uint16_t));
            if (!channel[ch].buffer) {
                err = ALLOCATION_FAILED;
            } else {
                channel[ch].name = getUniqueName(chTable, names[ch]);
                if (!channel[ch].name) {
                    err = ALLOCATION_FAILED;
                }
            }
        }
        timecodes = calloc(nBlocks, sizeof(time_t));
        fillValues = calloc(nBlocks, sizeof(uint16_t));
        partLen = calloc(nBlocks, sizeof(size_t));
        if (timecodes && fillValues && partLen) {
            tmpTable = newChannelTable(NULL);
            if (tmpTable) {
                partTable = newChannelTable("collection");
                if (!partTable) {
                    deleteChannelTable(tmpTable);
                    free((void *)tmpTable);
                    err = ALLOCATION_FAILED;
                }
                bzero(partLen, nBlocks * sizeof(size_t));
            } else {
                err = ALLOCATION_FAILED;
            }
        } else {
            free(timecodes);
            free(fillValues);
            free(partLen);
            err = ALLOCATION_FAILED;
        }
    }
    
    if (!err) {
        nParts = 1;
        for (blk = 0; blk < nBlocks; blk++) {
            if (fread(buffer, 1, 512, fp) == 512 && buffer[0] == 0xAA && buffer[1] == 0xAA) {
                // Read the data.
                timecodes[blk] = bcdToTime(&buffer[2]);
                for (i = 0; i < samplesPerBlock; i++) {
                    for (ch = 0; ch < nChannels; ch++) {
                        sample = &buffer[8 + 2 * (nChannels * i + ch)];
                        channel[ch].buffer[samplesPerBlock * blk + i] = letoh16(sample);
                    }
                }
                // Check whether the device has entered sleep mode.
                if (blk == 0) {
                    latestStart = timecodes[0];
                    partLen[0] = samplesPerBlock;
                } else if (timecodes[blk] - latestStart > round(partLen[nParts - 1] / samplerate)) {
                    partLen[nParts++] += samplesPerBlock;
                    latestStart = timecodes[blk];
                } else {
                    partLen[nParts - 1] += samplesPerBlock;
                }
            } else {
                for (ch = 0; ch < nChannels; ch++) {
                    free(channel[ch].buffer);
                }
                free(timecodes);
                fprintf(stderr, "readFile(): Corrupt data packet or premature end of file\n");
                err = INVALID_FILE;
                break;
            }
        }
        fclose(fp);
    } else if (err == ALLOCATION_FAILED) {
        fclose(fp);
        fprintf(stderr, "readFile(): Resource allocation failed\n");
        for (ch = 0; ch < nChannels; ch++) {
            free(channel[ch].buffer);
            free((void *)channel[ch].name);
        }
        free(timecodes);
        free((void *)tmpTable);
    }

    if (!err) {
        // Create the channels.
        startTime.tv_sec = timecodes[0];
        startTime.tv_nsec = 0;
        json[strlen(json) - 1] = '}';
        for (ch = 0; ch < nChannels; ch++) {
            // Copy data.
            if (nParts == 1) {
                channel[ch].data = newInt16Channel(tmpTable, channel[ch].name, length);
                memcpy(channel[ch].data, channel[ch].buffer, length);
            } else {
                blk = 0;
                t.tv_nsec = 0;
                name = calloc(nParts, sizeof(char *));
                // TODO: Check for allocation errors.
                for (part = 0; part < nParts; part++) {
                    name[part] = malloc(20);
                    // TODO: Check for allocation errors.
                    snprintf(name[part], 20, "%lld", part);
                    // TODO: Check for errors.
                    channel[ch].data = newInt16Channel(partTable, name[part], partLen[part]);
                    setSampleRate(partTable, name[part], samplerate);
                    t.tv_sec = timecodes[blk];
                    setStartTime(partTable, name[part], t);
                    memcpy(channel[ch].data, &channel[ch].buffer[samplesPerBlock * blk], partLen[part]);
                    blk += partLen[part] / samplesPerBlock;
                    fillValues[part] = isDynamic ? channel[ch].buffer[samplesPerBlock * blk - 1] : 0;
                }
                collateChannelsFromTable(tmpTable, channel[ch].name, partTable, fillValues);
            }
            // range: [-16 16] * 9.81 m/s^2
            setScaleAndOffset(tmpTable, channel[ch].name, 16.0 / 4096 * 9.81, 0.0);
            setResolution(tmpTable, channel[ch].name, 13);
            setUnit(tmpTable, channel[ch].name, "m/s^2");
            setSignalType(tmpTable, channel[ch].name, "acceleration");
            setSampleRate(tmpTable, channel[ch].name, samplerate);
            setDevice(tmpTable, channel[ch].name, "Hookie AM20", serialno);
            setStartTime(tmpTable, channel[ch].name, startTime);
            setMetadata(tmpTable, channel[ch].name, json);
            moveChannel(tmpTable, channel[ch].name, chTable);
            free(channel[ch].buffer);
            free((void *)channel[ch].name);
        }
        free(timecodes);
        free(fillValues);
        free(partLen);
        for (part = 0; part < nParts; part++) {
            free(name[part]);
        }
        free(name);
        deleteChannelTable(partTable);
        deleteChannelTable(tmpTable);
        free((void *)partTable);
        free((void *)tmpTable);
    }

    return err;
}

const char *describeError(int err) {
    char *str;

    switch (err) {
        case SUCCESS:
            str = NULL;
            break;
        case FOPEN_FAILED:
            str = "Could not open file";
            break;
        case INVALID_FORMAT:
            str = "Invalid file format";
            break;
        case INVALID_HEADER:
            str = "Invalid file header";
            break;
        case INVALID_FILE:
            str = "Corrupt file";
            break;
        case UNSUPPORTED_COMPRESSION:
            str = "Unsupported data compression";
            break;
        case ALLOCATION_FAILED:
            str = "Resource allocation failed";
            break;
        default:
            str = "Unknown error";
    }

    return str;
}

int initPlugin(void *handle) {
    const char *name = "Hookie AM20 Activity Meter";
    const char *exts[1] = {".DAT"};
    int err;

    err = registerFileFormat(handle, name, exts, 1);
    if (!err)
        err = registerImportFunc(handle, name, "readFile");
    
    return err;
}