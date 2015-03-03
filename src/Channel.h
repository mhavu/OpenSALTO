//
//  Channel.h
//  OpenSALTO
//
//  Copyright 2014 Marko Havu. Released under the terms of
//  GNU General Public License version 3 or later.
//

#ifndef OpenSALTO_Channel_h
#define OpenSALTO_Channel_h

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

PyTypeObject ChannelType;

typedef struct {
    PyObject_HEAD
    PyObject *dict;
    PyArrayObject *data;
    PyArrayObject *fill_positions;
    PyArrayObject *fill_lengths;
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
    char *json;
    PySetObject *events;
} Channel;

#endif
