//
//  Event.h
//  OpenSALTO
//
//  Copyright 2014 Marko Havu. Released under the terms of
//  GNU General Public License version 3 or later.
//

#ifndef OpenSALTO_Event_h
#define OpenSALTO_Event_h

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <structmember.h>
#include "salto_api.h"

PyTypeObject EventType;

struct Event {
    PyObject_HEAD
    PyObject *dict;
    EventVariety type;
    char *subtype;
    long long start_sec;
    long start_nsec;
    long long end_sec;
    long end_nsec;
    char *description;
};

void Event_dealloc(Event* self);
PyObject *Event_new(PyTypeObject *type, PyObject *args, PyObject *kwds);
int Event_init(Event *self, PyObject *args, PyObject *kwds);

#endif
