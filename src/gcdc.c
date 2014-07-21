//
//  gcdc.c
//  OpenSALTO
//
//  Imports .CSV files from Gulf Coast Data Concepts accelerometers.
//
//  Created by Marko Havu on 2014-07-15.
//  Released under the terms of GNU General Public License version 3.
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <math.h>
#include "salto_api.h"

#ifdef _WIN32
static char *strsep(char **stringp, const char *delim) {
    char *begin, *end;

    begin = *stringp;
    if(begin == NULL)
        return NULL;

    if(delim[0] == '\0' || delim[1] == '\0') {
        char ch = delim[0];

        if(ch == '\0')
            end = NULL;
        else {
            if(*begin == ch)
                end = begin;
            else if(*begin == '\0')
                end = NULL;
            else
                end = strchr(begin + 1, ch);
        }
    }
    else
        end = strpbrk(begin, delim);

    if(end) {
        *end++ = '\0';
        *stringp = end;
    }
    else
        *stringp = NULL;
    
    return begin;
}
#endif

typedef enum {
    SUCCESS = 0,
    FOPEN_FAILED,
    INVALID_FORMAT,
    INVALID_HEADER,
    INVALID_FILE,
    UNKNOWN_MODEL
} Error;

typedef struct {
    double t;
    int value[3];
} Sample;

static off_t fsize(const char *filename) {
    struct stat st;

    if (stat(filename, &st) != 0) {
        perror("stat()");
        return -1;
    }

    return st.st_size;
}

// init() {
//     const char *extensions[1] = {".CSV"};
//     success = registerFileFormat("Gulf Coast Data Concepts", extensions);
// }

int readFile(const char *filename, const char *chTable) {
    Error err = SUCCESS;
    char header[4096], gain[256] = "", serial[256] = "unknown", device[256] = "GCDC ";
    char *model, *buffer, *line, *token;
    size_t j, length, lines, nSamples, n;
    int i, resolution, result, divisor, nChannels;
    double samplerate, duration, first, t;
    struct tm tm;
    struct timespec start;
    Sample *sample, *data;
    nChannels = 3;
    int16_t *channel[3];
    char *name[3] = {"X", "Y", "Z"};

    FILE *fp = fopen(filename, "r");
    if (!fp) {
        perror("fopen()");
        err = FOPEN_FAILED;
    }

    if (!err) {
        // Read device model name.
        model = device + 5;
        if (!fgets(header, sizeof(header), fp) || sscanf(header, ";Title, %*s %250[^\x0A\x0D]", model) != 1) {
            fclose(fp);
            fprintf(stderr, "readFile(): Unknown file format (not GCDC CSV)\n");
            err = INVALID_FORMAT;
        }
    }

    if (!err) {
        // Read the other file header tags.
        resolution = 12;
        while (!err && getc(fp) == ';') {
            if (fgets(header, sizeof(header), fp)) {
                line = header;
                token = strsep(&line, ",");
                if (strcmp(token, "16 bit resolution") == 0) {
                    resolution = 16;
                } else if (strcmp(token, "Deadband") == 0) {
                    // A new sample from the sensor must exceed
                    // the last reading by the deadband value
                } else if (strcmp(token, "DeadbandTimeout") == 0) {
                    // The period in seconds when a sample is
                    // recorded regardless of the deadband setting
                } else if (strcmp(token, "Dwell") == 0) {
                    // The number of samples recorded after
                    // a threshold event
                } else if (strcmp(token, "Gain") == 0) {
                    token = strsep(&line, ",");
                    if (sscanf(token, "%255s", gain) == EOF) {
                        fclose(fp);
                        fprintf(stderr, "readFile(): Invalid gain value\n");
                        err = INVALID_HEADER;
                    }
                } else if (strcmp(token, "Headers") == 0) {
                    // TODO: headers
                } else if (strcmp(token, "HPF filter set to:") == 0) {
                    // TODO: high pass filter value
                } else if (strcmp(token, "SampleRate") == 0) {
                    token = strsep(&line, ",");
                    if (sscanf(token, "%lf", &samplerate) == EOF) {
                        fclose(fp);
                        fprintf(stderr, "readFile(): Invalid sample rate\n");
                        err = INVALID_HEADER;
                    }
                } else if (strcmp(token, "Start_time") == 0) {
                    token = strsep(&line, ",");
                    result = sscanf(token, "%i-%i-%i", &tm.tm_year, &tm.tm_mon, &tm.tm_mday);
                    if (result != EOF) {
                        token = strsep(&line, ",");
                        result = sscanf(token, "%i:%i:%i.%li",
                                        &tm.tm_hour, &tm.tm_min, &tm.tm_sec, &start.tv_nsec);
                    }
                    if (result != EOF) {
                        tm.tm_year -= 1900;
                        start.tv_sec = mktime(&tm);
                        start.tv_nsec = start.tv_nsec * 1000000;
                    } else {
                        fclose(fp);
                        fprintf(stderr, "readFile(): Invalid start time\n");
                        err = INVALID_HEADER;
                    }
                } else if (strcmp(token, "Switch") == 0) {
                    // TODO: Don't know what this is.
                } else if (strcmp(token, "Temperature") == 0) {
                    // TODO: temperature in °C and Vbat in mV
                } else if (strcmp(token, "Version") == 0) {
                    // TODO: get firmware version
                    while (line) {
                        token = strsep(&line, ",");
                        sscanf(token, " SN:%255s", serial);
                    }
                } else {
                    fprintf(stderr, "Unrecognized GCDC header tag: %s", token);
                }
            }
        }
    }
    
    if (!err) {
        if (strncmp(model, "X16-", 4) == 0) {
            // range = [-16.0 16.0]
            resolution = 16;
            divisor = 1024;
        } else if (strncmp(model, "X6-", 3) == 0) {
            if (strcmp(gain, "high") == 0) {
                // range = [-2.0 2.0]
                divisor = (resolution == 16) ? 16384 : 1024;
            } else {
                // range = [-6.0 6.0]
                divisor = (resolution == 16) ? 5440 : 340;
            }
        } else {
            // unknown model
            resolution = 0;
            divisor = 1;
        }

        // Read the data.
        fseek(fp, -1, SEEK_CUR);
        length = fsize(filename) - ftell(fp);
        buffer = malloc(length);
        duration = 0.0;
        nSamples = 0;
        if (fread(buffer, 1, length, fp) == length) {
            lines = 1;
            for (j = length; j > 0; j--) {
                if (buffer[j] == '\n')
                    lines++;
            }
            data = calloc(lines, sizeof(Sample));
            sample = data;
            line = buffer;
            while (sscanf(line, "%lf, %d, %d, %d%n", &sample->t, &sample->value[0],
                          &sample->value[1], &sample->value[2], &i) == 4) {
                // Adjust sample times and start time so that first sample is at start time.
                if (nSamples == 0) {
                    first = sample->t;
                    start.tv_sec += (int)first;
                    start.tv_nsec += fmod(first, 1.0) * 1000000000;
                    if (start.tv_nsec > 1000000000) {
                        start.tv_sec++;
                        start.tv_nsec -= 1000000000;
                    }
                }
                sample->t -= first;

                if (sample->t > duration)
                    duration = sample->t;

                sample++;
                nSamples++;
                line += i;
            }
        }
        free(buffer);
        fclose(fp);

        if (nSamples > 0) {
            // Adjust channel length to constant sample interval.
            length = lrint(duration * samplerate) + 1;
            // Create the channels.
            for (i = 0; i < nChannels; i++) {
                channel[i] = newInt16Channel(chTable, name[i], length);
                channel[i][j] = data[0].value[i];
                setScaleAndOffset(chTable, name[i], 9.81 / (2 * divisor), 0.0);
                setUnit(chTable, name[i], "m/s^2");
                setSampleRate(chTable, name[i], samplerate);
                setDevice(chTable, name[i], device, serial);
                setStartTime(chTable, name[i], start);
                setResolution(chTable, name[i], resolution);
            }
            // Resample data using constant sample interval.
            t = 0.5 / samplerate;
            n = 0;
            for (j = 1; j < length; j++) {
                for (i = 0; i < nChannels; i++) {
                    if (n < nSamples - 1 && (t - data[n].t) >= (data[n+1].t - t))
                        n++;
                    channel[i][j] = data[n].value[i];
                    t += 1.0 / samplerate;
                }
            }
        }
        free(data);
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
        case UNKNOWN_MODEL:
            str = "Unknown GCDC device model";
            break;
        default:
            str = "Unknown error";
    }

    return str;
}