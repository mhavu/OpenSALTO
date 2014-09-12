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

#endif
