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
    uint8_t buffer[512];
    long long blk, nBlocks, fileLength, latestStartBlock;
    size_t length, i, position;
    int headerIsValid, isDynamic = 0;
    int16_t *ptr, fill[3];
    int ch, nChannels = 3;
    Channel channel[3];
    const char *names[3] = {"X", "Y", "Z"};
    char serialno[28], tag[28], value[28], json[512] = "{ ";
    double samplerate;
    Error err = SUCCESS;
    struct timespec startTime;
    time_t t, blockTime, latestStartTime, lastBlockTime;


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
        // Determine data length.
        // Get time stamp for the first data block.
        if (fread(buffer, 1, 512, fp) == 512 && buffer[0] == 0xAA && buffer[1] == 0xAA) {
            startTime.tv_sec = bcdToTime(&buffer[2]);
            startTime.tv_nsec = 0;
        } else {
            err = INVALID_FILE;
        }
        // Get time stamp for the last data block.
        if (!err && fseek(fp, -512, SEEK_END) == 0 &&
            fread(buffer, 1, 512, fp) == 512 &&
            buffer[0] == 0xAA && buffer[1] == 0xAA)
        {
            lastBlockTime = bcdToTime(&buffer[2]);
            length = ceil((lastBlockTime - startTime.tv_sec + 0.5) * samplerate + 84);
        } else {
            err = INVALID_FILE;
        }
        if (!err && fseek(fp, 512, SEEK_SET) != 0)
            err = INVALID_FILE;
        if (err) {
            fclose(fp);
            fprintf(stderr, "readFile(): Corrupt data packet or premature end of file\n");
        }
    }
    
    if (!err) {
        for (ch = 0; ch < nChannels; ch++) {
            channel[ch].name = names[ch];
            channel[ch].data = calloc(length, sizeof(uint16_t));
        }
        latestStartTime = startTime.tv_sec;
        latestStartBlock = 0;
        position = 0;
        nBlocks = fileLength / 512 - 1;
        for (blk = 0; blk < nBlocks; blk++) {
            if (fread(buffer, 1, 512, fp) == 512 && buffer[0] == 0xAA && buffer[1] == 0xAA) {
                blockTime = bcdToTime(&buffer[2]);
                // Fill in blanks if the device has entered sleep mode.
                t = latestStartTime + round(84 * (blk - latestStartBlock) / samplerate);
                if (blockTime > t) {
                    for (ch = 0; ch < nChannels; ch++) {
                        fill[ch] = isDynamic ? channel[ch].data[position - 1] : 0;
                    }
                    for (i = 0; i / samplerate < blockTime - t; i++) {
                        for (ch = 0; ch < nChannels; ch++) {
                            channel[ch].data[position] = fill[ch];
                        }
                        position++;
                    }
                    latestStartBlock = blk;
                    latestStartTime = blockTime;
                }
                // Read the data.
                for (i = 0; i < 84; i++) {
                    for (ch = 0; ch < nChannels; ch++) {
                        channel[ch].data[position] = letoh16(&buffer[8 + 2 * (nChannels * i + ch)]);
                    }
                    position++;
                }
            } else {
                for (ch = 0; ch < nChannels; ch++) {
                    free(channel[ch].data);
                }
                fprintf(stderr, "readFile(): Corrupt data packet or premature end of file\n");
                err = INVALID_FILE;
                break;
            }
        }
        fclose(fp);
    }

    if (!err) {
        // Create the channels.
        length = position;
        json[strlen(json) - 1] = '}';
        for (ch = 0; ch < nChannels; ch++) {
            ptr = newInt16Channel(chTable, channel[ch].name, length);
            memcpy(ptr, channel[ch].data, length);
            free(channel[ch].data);
            // range: [-16 16] * 9.81 m/s^2
            setScaleAndOffset(chTable, channel[ch].name, 16.0 / 4096 * 9.81, 0.0);
            setResolution(chTable, channel[ch].name, 13);
            setUnit(chTable, channel[ch].name, "m/s^2");
            setSampleRate(chTable, channel[ch].name, samplerate);
            setDevice(chTable, channel[ch].name, "Hookie AM20", serialno);
            setStartTime(chTable, channel[ch].name, startTime);
            // TODO: set json
        }
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