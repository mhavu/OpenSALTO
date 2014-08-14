//
//  salto.h
//  OpenSALTO
//
//  Created by Marko Havu on 2014-07-09.
//  Released under the terms of GNU General Public License version 3.
//

#ifndef _salto_h
#define _salto_h

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

typedef struct {
    PyObject_HEAD
    PyObject *dict;
    PyObject *data;
    PyArrayObject *fill_values;
    double samplerate;
    double scale;
    double offset;
    char *unit;
    char *type;
    long long start_sec;
    long start_nsec;
    char *device;
    char *serial_no;
    int resolution;
    int collection;
    char *json;
} Channel;

void *newIntegerChannel(const char *chTable, const char *name, size_t length, size_t size, int is_signed);
void *newRealChannel(const char *chTable, const char *name, size_t length, size_t size);
Channel *getChannel(const char *chTable, const char *name);
void *channelData(Channel *ch, size_t *length);
int addChannel(const char *chTable, const char *name, Channel *ch);
const char *getUniqueName(const char *chTable, const char *name);
int setCallback(void *obj, const char *type, const char *format, const char *funcname);

#endif
