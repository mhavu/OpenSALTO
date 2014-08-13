//
//  hookie.c
//  OpenSALTO
//
//  Imports Hookie AM20 Activity Meter data files.
//  Check http://www.hookiemeter.com/support for further specifications.
//
//  Created by Marko Havu on 2014-07-30.
//  Released under the terms of GNU General Public License version 3.
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
    UNSUPPORTED_COMPRESSION
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
    long long blk, nBlocks, fileLength;
    size_t length, i, position;
    int headerIsValid, isDynamic = 0;
    int16_t fill;
    int ch;
    const int samplesPerBlock = 84;
    const int nChannels = 3;
    Channel channel[3];
    const char *names[3] = {"X", "Y", "Z"};
    char serialno[28], tag[28], value[28], json[512] = "{ ";
    double samplerate, t;
    Error err = SUCCESS;
    struct timespec startTime;
    time_t *latestStart, *timecodes;


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
        nBlocks = fileLength / 512 - 1;
        length = samplesPerBlock * nBlocks;
        for (ch = 0; ch < nChannels; ch++) {
            channel[ch].name = getUniqueName(chTable, names[ch]);
            channel[ch].buffer = calloc(length, sizeof(uint16_t));
        }
        timecodes = calloc(nBlocks, sizeof(time_t));
        latestStart = timecodes;
        t = 0;
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
                // Reserve space for missing samples if the device has entered sleep mode.
                // TODO: Use sparse channels. Adding the missing samples may take terabytes of storage.
                t += samplesPerBlock / samplerate;
                if (timecodes[blk] - *latestStart > round(t)) {
                    length += (timecodes[blk] - *latestStart - t) * samplerate;
                    t = 0;
                    latestStart = &timecodes[blk];
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
    }

    if (!err) {
        // Create the channels.
        startTime.tv_sec = timecodes[0];
        startTime.tv_nsec = 0;
        json[strlen(json) - 1] = '}';
        for (ch = 0; ch < nChannels; ch++) {
            channel[ch].data = newInt16Channel(chTable, channel[ch].name, length);
            // range: [-16 16] * 9.81 m/s^2
            setScaleAndOffset(chTable, channel[ch].name, 16.0 / 4096 * 9.81, 0.0);
            setResolution(chTable, channel[ch].name, 13);
            setUnit(chTable, channel[ch].name, "m/s^2");
            setSampleRate(chTable, channel[ch].name, samplerate);
            setDevice(chTable, channel[ch].name, "Hookie AM20", serialno);
            setStartTime(chTable, channel[ch].name, startTime);
            setMetadata(chTable, channel[ch].name, json);
            // Copy data filling in blanks if the device has entered sleep mode.
            latestStart = timecodes;
            t = 0;
            position = 0;
            for (blk = 0; blk < nBlocks; blk++) {
                t += samplesPerBlock / samplerate;
                if (timecodes[blk] - *latestStart > round(t)) {
                    fill = isDynamic ? channel[ch].buffer[samplesPerBlock * blk - 1] : 0;
                    for (i = 0; i < (timecodes[blk] - *latestStart - t) * samplerate; i++) {
                        channel[ch].data[position++] = fill;
                    }
                    t = 0;
                    latestStart = &timecodes[blk];
                }
                for (i = 0; i < samplesPerBlock; i++) {
                    channel[ch].data[position++] = channel[ch].buffer[samplesPerBlock * blk + i];
                }
            }
            free(channel[ch].buffer);
        }
        free(timecodes);
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