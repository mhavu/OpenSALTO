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
    time.tm_mon = 10 * (buffer[1] >> 4) + (buffer[1] & 0x0F) - 1;
    time.tm_year = 100 + 10 * (buffer[2] >> 4) + (buffer[2] & 0x0F);
    time.tm_hour = 10 * (buffer[3] >> 4) + (buffer[3] & 0x0F);
    time.tm_min = 10 * (buffer[4] >> 4) + (buffer[4] & 0x0F);
    time.tm_sec = 10 * (buffer[5] >> 4) + (buffer[5] & 0x0F);

    return mktime(&time);
}

int readFile(const char *filename, const char *chTable) {
    uint8_t buffer[512], *sample;
    off_t fileLength;
    size_t length, i, *partLen, longestPart, blkForLongestPart, blk, nBlocks;
    size_t fill, nParts, pos, blklen, corrupt = 0;
    int headerIsValid, isDynamic = 0;
    int ch;
    const int samplesPerBlock = 84;
    const int nChannels = 3;
    Channel channel[3];
    const char *names[3] = {"X", "Y", "Z"};
    char serialno[28], tag[28], value[28], json[512] = "{ ";
    double samplerate, partDuration, timedelta;
    Error err = SUCCESS;
    struct timespec startTime;
    time_t t, *timecodes;
    const char *tmpTable;
    size_t *posArray = NULL, *lenArray = NULL;

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
                        if (strlen(value) + 1 > sizeof(serialno)) {
                            memcpy(serialno, value, sizeof(serialno) - 1);
                            serialno[sizeof(serialno) - 1] = 0;
                        } else {
                            strcpy(serialno, value);
                        }
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
        if (!isDynamic) {
            length += nBlocks - 1;
        }
        for (ch = 0; ch < nChannels; ch++) {
            channel[ch].buffer = calloc(length, sizeof(uint16_t));
            if (channel[ch].buffer) {
                channel[ch].name = getUniqueName(chTable, names[ch]);
                if (!channel[ch].name) {
                    err = ALLOCATION_FAILED;
                }
            } else {
                err = ALLOCATION_FAILED;
            }
        }
        timecodes = calloc(nBlocks, sizeof(time_t));
        partLen = calloc(nBlocks, sizeof(size_t));
        if (timecodes && partLen) {
            tmpTable = newChannelTable(NULL);
            if (tmpTable) {
                bzero(partLen, nBlocks * sizeof(size_t));
            } else {
                free(timecodes);
                free(partLen);
                err = ALLOCATION_FAILED;
            }
        } else {
            free(timecodes);
            free(partLen);
            err = ALLOCATION_FAILED;
        }
        if (err == ALLOCATION_FAILED) {
            // Clean up.
            fclose(fp);
            fprintf(stderr, "readFile(): Resource allocation failed\n");
            for (ch = 0; ch < nChannels; ch++) {
                free(channel[ch].buffer);
                free((void *)channel[ch].name);
            }
        }
    }
    
    if (!err) {
        nParts = 1;
        longestPart = 0;
        blkForLongestPart = 0;
        for (blk = 0; blk < nBlocks; blk++) {
            blklen = fread(buffer, 1, 512, fp);
            if (blklen == 512 && buffer[0] == 0xAA && buffer[1] == 0xAA) {
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
                    partLen[0] = 1;
                } else if (timecodes[blk] - timecodes[blk - 1] > 2 * samplesPerBlock / samplerate) {
                    if (partLen[nParts - 1] > partLen[longestPart]) {
                        blkForLongestPart = blk - partLen[nParts - 1];
                        longestPart = nParts - 1;
                    }
                    partLen[nParts++] = 1;
                } else {
                    partLen[nParts - 1]++;
                }
            } else {
                // Ignore packet.
                if (blklen == 512) {
                    fprintf(stderr, "readFile(): Corrupt data packet %zu ignored in %s\n",
                            blk + corrupt++, filename);
                } else {
                    fprintf(stderr, "readFile(): Premature end of file in %s\n, corrupt data packet ignored", filename);
                    break;
                }
                nBlocks--;
                blk--;
            }
        }
        fclose(fp);
    }

    if (!err) {
        startTime.tv_sec = timecodes[0];
        startTime.tv_nsec = 0;
        json[strlen(json) - 1] = '}';
        // Correct samplerate, if necessary.
        partDuration = partLen[longestPart] * samplesPerBlock / samplerate;
        timedelta = timecodes[blkForLongestPart + partLen[longestPart] - 1] - timecodes[blkForLongestPart];
        if (timedelta + 1 < partDuration) {
            samplerate *= partDuration / timedelta;
        }
        // Add fills, if necessary.
        if (nParts > 1) {
            posArray = calloc(nParts - 1, sizeof(size_t));
            lenArray = calloc(nParts - 1, sizeof(size_t));
            if (posArray && lenArray) {
                // Calculate fill positions and lengths.
                blk = 0;  // block
                i = 0;    // sample
                pos = 0;  // sample (including skipped ones)
                for (fill = 0; fill < nParts - 1; fill++) {
                    i += partLen[fill] * samplesPerBlock;
                    blk += partLen[fill];
                    pos += partLen[fill] * samplesPerBlock;
                    t = timecodes[blk] - startTime.tv_sec;
                    lenArray[fill] = round(t * samplerate) - pos;
                    pos += lenArray[fill];
                    if (!isDynamic) {
                        i++;
                        lenArray[fill]--;
                    }
                    posArray[fill] = i - 1;
                }
            } else {
                err = ALLOCATION_FAILED;
            }
        }
    }
    
    if (!err) {
        for (ch = 0; ch < nChannels; ch++) {
            // Create the channels.
            channel[ch].data = newSparseInt16Channel(tmpTable, channel[ch].name, length, nParts);
            memcpy(channel[ch].data, channel[ch].buffer, length * sizeof(uint16_t));
            if (nParts > 1) {
                if (!isDynamic) {
                    for (fill = 0; fill < nParts - 1; fill++) {
                        channel[ch].buffer[posArray[fill]] = 0;
                    }
                }
                setFills(tmpTable, channel[ch].name, posArray, lenArray);
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
        free(posArray);
        free(lenArray);
        free(timecodes);
        free(partLen);
        deleteChannelTable(tmpTable);
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