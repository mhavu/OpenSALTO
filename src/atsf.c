//
//  atsf.c
//  OpenSALTO
//
//  Imports the .ATS files of Alive Heart and Activity Monitor by Alive
//  Technologies. Supports the 8-bit 2-axis HM120 and 3-axis HM131 models.
//
//  Created by Marko Havu on 2014-07-09.
//  Released under the terms of GNU General Public License version 3.
//

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <sys/stat.h>
#include "salto_api.h"

typedef struct {
    uint8_t type;
    uint8_t format;
    uint16_t pktlen;
    double samplerate;
    double scale;
    double offset;
    char *unit;
    uint8_t nsubs;
    uint16_t subpktlen;
    char **sub;
    uint8_t **dset;
    uint64_t length;
} Channel;

typedef enum {
    SUCCESS = 0,
    FOPEN_FAILED,
    INVALID_FORMAT,
    INVALID_FILE,
    INVALID_BLOCK_COUNT
} Error;

static uint16_t betoh16(uint8_t *buf)
{
    return (uint16_t)buf[1] | (uint16_t)buf[0] << 8;
}

static uint32_t betoh32(uint8_t *buf)
{
    return ((uint32_t)buf[0] << 24 | (uint32_t)buf[1] << 16 |
            (uint32_t)buf[2] << 8 | (uint32_t)buf[3]);
}

off_t fsize(const char *filename) {
    struct stat st;

    if (stat(filename, &st) != 0) {
        perror("stat()");
        return -1;
    }

    return st.st_size;
}

// init() {
//     const char *extensions[1] = {".ATS"};
//     success = registerFileFormat("Alive Heart and Activity Monitor", extensions);
// }

int readFile(const char *filename, const char *chTable) {
    uint8_t header[128];
    uint8_t *buffer;
    long long blk, nBlocks;
    uint16_t i, headerLength, blockLength, paddingLength, maxPktLength;
    uint8_t ch, nChannels, sub;
    int headerIsValid;
    Channel *channel;
    Error err = SUCCESS;
    char *device = "unknown";
    struct timespec startTime;
    struct tm time;


    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        perror("fopen()");
        err = FOPEN_FAILED;
    }

    if (!err) {
        // Read the file header (128 bytes).
        headerIsValid = (fread(header, 1, 128, fp) == 128 &&
                         strncmp((const char *)header, "ATSF", 5) == 0);
        if (!headerIsValid) {
            fclose(fp);
            fprintf(stderr, "readFile(): Unknown file format (not ATSF)\n");
            err = INVALID_FORMAT;
        }
    }
    
    if (!err) {
        headerLength = betoh16(&header[5]);
        nChannels = header[7];
        nBlocks = betoh32(&header[8]);
        blockLength = betoh16(&header[12]);
        time.tm_year = betoh16(&header[14]) - 1900;
        time.tm_mon = header[16];
        time.tm_mday = header[17];
        time.tm_hour = header[18];
        time.tm_min = header[19];
        time.tm_sec = header[20];
        startTime.tv_sec = mktime(&time);
        paddingLength = blockLength;
        maxPktLength = 0;
        channel = calloc(nChannels, sizeof(Channel));
        // The nBlocks stored in the file is often 0, in which case the
        // number of blocks has to be calculated from the file size.
        if (nBlocks == 0)
            nBlocks = ((fsize(filename) - headerLength) / blockLength);
        if (nBlocks < 1) {
            fclose(fp);
            fprintf(stderr, "readFile(): No data blocks or block count could not be determined\n");
            err = INVALID_BLOCK_COUNT;
        }

        // Read channel descriptions (32 bytes each).
        for (ch = 0; ch < nChannels; ch++) {
            if (!err && fread(header, 1, 32, fp) != 32) {
                fclose(fp);
                fprintf(stderr, "readFile(): Corrupt channel header\n");
                err = INVALID_FILE;
            }
            if (!err) {
                channel[ch].type = header[0];
                channel[ch].format = header[1];
                channel[ch].pktlen = betoh16(&header[2]);
                channel[ch].length = nBlocks * channel[ch].pktlen;

                switch (channel[ch].type) {
                    case 0x11:
                        // status channel
                        if (channel[ch].format != 0) {
                            // unknown data format
                            // TODO: handle nicely
                        }
                        // Byte 1, Bit 7 (LSB) = Button event
                        // Byte 2, Battery voltage (200 = 100%)
                        channel[ch].nsubs = 2;
                        channel[ch].length /= 2;
                        channel[ch].sub = calloc(2, sizeof(char *));
                        channel[ch].sub[0] = "Button event";
                        channel[ch].sub[1] = "Battery voltage";
                        channel[ch].dset = calloc(2, sizeof(uint8_t *));
                        channel[ch].dset[0] = newUInt8Channel(chTable, channel[ch].sub[0], channel[ch].length);
                        channel[ch].dset[1] = newUInt8Channel(chTable, channel[ch].sub[1], channel[ch].length);
                        channel[ch].scale = 0.5;
                        channel[ch].offset = 0.0;
                        channel[ch].unit = "%";
                        channel[ch].samplerate = 0.0; // TODO: set to NAN?
                        break;
                    case 0xAA:
                        // ECG channel
                        switch (channel[ch].format) {
                            case 1:
                                channel[ch].samplerate = 150.0;
                                break;
                            case 2:
                                channel[ch].samplerate = 300.0;
                                break;
                            default:
                                // unknown data format
                                // TODO: handle nicely
                                break;
                        }
                        channel[ch].nsubs = 1;
                        channel[ch].sub = calloc(1, sizeof(char *));
                        channel[ch].sub[0] = "ECG";
                        channel[ch].dset = calloc(1, sizeof(uint8_t *));
                        channel[ch].dset[0] = newUInt8Channel(chTable, channel[ch].sub[0], channel[ch].length);
                        // range: [-2.66 2.66] mV
                        channel[ch].scale = 2 * 2.66 / 256;
                        channel[ch].offset = -2.66;
                        channel[ch].unit = "mV";
                        break;
                    case 0x55:
                        // 2-axis accelerometer channel
                        if (channel[ch].format != 0) {
                            // unknown data format
                            // TODO: handle nicely
                        }
                        // range: [-2.0 2.0] * 9.81 m/s^2
                        channel[ch].scale = 2 * 2.0 / 256 * 9.81;
                        channel[ch].offset = -2.0 * 9.81;
                        channel[ch].unit = "m/s^2";
                        channel[ch].samplerate = 75.0;
                        device = "Alive HM120";
                        channel[ch].nsubs = 2;
                        channel[ch].length /= 2;
                        channel[ch].sub = calloc(2, sizeof(char *));
                        channel[ch].sub[0] = "X";
                        channel[ch].sub[1] = "Y";
                        channel[ch].dset = calloc(2, sizeof(uint8_t *));
                        channel[ch].dset[0] = newUInt8Channel(chTable, channel[ch].sub[0], channel[ch].length);
                        channel[ch].dset[1] = newUInt8Channel(chTable, channel[ch].sub[1], channel[ch].length);
                        break;
                    case 0x56:
                        // 3-axis accelerometer channel
                        if (channel[ch].format != 0) {
                            // unknown data format
                            // TODO: handle nicely
                        }
                        // range: [-2.7 2.7] * 9.81 m/s^2
                        channel[ch].scale = 2 * 2.7 / 256 * 9.81;
                        channel[ch].offset = -2.7 * 9.81;
                        channel[ch].unit = "m/s^2";
                        channel[ch].samplerate = 75.0;
                        device = "Alive HM131";
                        channel[ch].nsubs = 3;
                        channel[ch].sub = calloc(3, sizeof(char *));
                        channel[ch].length /= 3;
                        channel[ch].sub[0] = "X";
                        channel[ch].sub[1] = "Y";
                        channel[ch].sub[2] = "Z";
                        channel[ch].dset = calloc(3, sizeof(uint8_t *));
                        channel[ch].dset[0] = newUInt8Channel(chTable, channel[ch].sub[0], channel[ch].length);
                        channel[ch].dset[1] = newUInt8Channel(chTable, channel[ch].sub[1], channel[ch].length);
                        channel[ch].dset[2] = newUInt8Channel(chTable, channel[ch].sub[2], channel[ch].length);
                        break;
                    default:
                        // unknown channel type
                        // TODO: handle nicely
                        break;
                }
                channel[ch].subpktlen = channel[ch].pktlen / channel[ch].nsubs;
                paddingLength -= channel[ch].pktlen;
                if (channel[ch].pktlen > maxPktLength)
                    maxPktLength = channel[ch].pktlen;
            } 
        }

        // Read data blocks.
        if (!err && fseek(fp, headerLength, SEEK_SET) != 0) {
            fclose(fp);
            fprintf(stderr, "readFile(): Premature end of file\n");
            err = INVALID_FILE;
        }
        if (!err) {
            buffer = malloc(maxPktLength);
            for (blk = 0; blk < nBlocks; blk++) {
                for (ch = 0; ch < nChannels; ch++) {
                    if (fread(buffer, 1, channel[ch].pktlen, fp) == channel[ch].pktlen)

                        for (i = 0; i < channel[ch].subpktlen; i++) {
                            for (sub = 0; sub < channel[ch].nsubs; sub++) {
                                // The samples are interleaved.
                                channel[ch].dset[sub][i] = buffer[channel[ch].nsubs * i + sub];
                            }
                        }
                }

                if (paddingLength > 0 && fseek(fp, paddingLength, SEEK_CUR) != 0) {
                    fclose(fp);
                    fprintf(stderr, "readFile(): Premature end of file\n");
                    err = INVALID_FILE;
                }
            }
            free(buffer);
        }

        // Clean up.
        for (ch = 0; ch < nChannels; ch++) {
            for (sub = 0; sub < channel[ch].nsubs; sub++) {
                if (channel[ch].type == 0x11 && sub == 0) {
                    // TODO: add this as events to all channels
                    // Bit 7 (LSB) = Button Event
                    deleteChannel(chTable, channel[ch].sub[sub]);
                } else {
                    setScaleAndOffset(chTable, channel[ch].sub[sub], channel[ch].scale, channel[ch].offset);
                    setUnit(chTable, channel[ch].sub[sub], "m/s^2");
                    setSampleRate(chTable, channel[ch].sub[sub], channel[ch].samplerate);
                    setDevice(chTable, channel[ch].sub[sub], device, "unknown");
                    setStartTime(chTable, channel[ch].sub[sub], startTime);
                }
            }
        }
        free(channel);
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
        case INVALID_FILE:
            str = "Corrupt file";
            break;
        case INVALID_BLOCK_COUNT:
            str = "No data blocks or block count could not be determined";
            break;
        default:
            str = "Unknown error";
    }

    return str;
}