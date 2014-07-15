//
//  salto.h
//  OpenSALTO
//
//  Created by Marko Havu on 2014-07-09.
//  Released under the terms of GNU General Public License version 3.
//

#ifndef _salto_h
#define _salto_h

typedef struct {
    void *ptr;
    size_t length;
    size_t bytes_per_sample;
    double samplerate;
    double scale;
    double offset;
    char *unit;
    long long start_sec;
    long start_nsec;
    char *device;
    char *serial_no;
    char *json;
} Channel;

Channel *getChannel(const char *chTable, const char *name);
const char *getNameForData(const char *chTable, void *ptr);
int addChannel(const char *chTable, const char *name, Channel *ch);
void removeChannel(const char *chTable, const char *name);
const char *getUniqueName(const char *chTable, const char *name);

#endif
