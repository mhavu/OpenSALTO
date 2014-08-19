//
//  Event.c
//  OpenSALTO
//
//  Copyright 2014 Marko Havu. Released under the terms of
//  GNU General Public License version 3 or later.
//

#define PY_ARRAY_UNIQUE_SYMBOL Py_Array_API_OpenSALTO
#define NO_IMPORT_ARRAY
#include "Event.h"
#include "salto.h"

static PyMethodDef Event_methods[] = {
    {NULL}  // sentinel
};

static PyMemberDef Event_members[] = {
    {"__dict__", T_OBJECT, offsetof(Event, dict), READONLY, "dictionary for instance variables"},
    {"type", T_INT, offsetof(Event, type), 0, "event type"},
    {"subtype", T_STRING, offsetof(Event, subtype), 0, "event subtype"},
    {"start_sec", T_LONGLONG, offsetof(Event, start_sec), 0, "start time (POSIX time)"},
    {"start_nsec", T_LONG, offsetof(Event, start_nsec), 0, "nanoseconds to add to the start time"},
    {"end_sec", T_LONGLONG, offsetof(Event, end_sec), 0, "end time (POSIX time)"},
    {"end_nsec", T_LONG, offsetof(Event, end_nsec), 0, "nanoseconds to add to the end time"},
    {"description", T_STRING, offsetof(Event, description), 0, "event description"},
    {NULL}  // sentinel
};

PyTypeObject EventType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "salto.Event",               // tp_name
    sizeof(Event),               // tp_basicsize
    0,                           // tp_itemsize
    (destructor)Event_dealloc,   // tp_dealloc
    0,                           // tp_print
    0,                           // tp_getattr
    0,                           // tp_setattr
    0,                           // tp_compare
    0,                           // tp_repr
    0,                           // tp_as_number
    0,                           // tp_as_sequence
    0,                           // tp_as_mapping
    0,                           // tp_hash
    0,                           // tp_call
    0,                           // tp_str
    0,                           // tp_getattro
    0,                           // tp_setattro
    0,                           // tp_as_buffer
    Py_TPFLAGS_DEFAULT |
    Py_TPFLAGS_BASETYPE,         // tp_flags
    "OpenSALTO Event",           // tp_doc
    0,		                     // tp_traverse
    0,		                     // tp_clear
    0,		                     // tp_richcompare
    0,		                     // tp_weaklistoffset
    0,		                     // tp_iter
    0,		                     // tp_iternext
    Event_methods,               // tp_methods
    Event_members,               // tp_members
    0,                           // tp_getset
    0,                           // tp_base
    0,                           // tp_dict
    0,                           // tp_descr_get
    0,                           // tp_descr_set
    offsetof(Event, dict),       // tp_dictoffset
    (initproc)Event_init,        // tp_init
    0,                           // tp_alloc
    Event_new                    // tp_new
};

void Event_dealloc(Event* self) {
    free(self->subtype);
    free(self->description);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

PyObject *Event_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    return (PyObject *)type->tp_alloc(type, 0);
}

int Event_init(Event *self, PyObject *args, PyObject *kwds) {
    int result;
    size_t size;
    char *tmp;
    static char *kwlist[] = {"type", "subtype", "start_sec", "start_nsec",
        "end_sec", "end_nsec", "description", NULL};

    self->subtype = NULL;
    self->description = NULL;
    result = !PyArg_ParseTupleAndKeywords(args, kwds, "|isLlLls", kwlist, &(self->type), &(self->subtype),
                                          &(self->start_sec), &(self->start_nsec),
                                          &(self->end_sec), &(self->end_nsec),
                                          &(self->description));
    if (self->subtype) {
        size = strlen(self->subtype) + 1;
        tmp = self->subtype;
        self->subtype = malloc(size);
        strlcpy(self->subtype, tmp, size);
    } else {
        self->subtype = malloc(8);
        strlcpy(self->subtype, "unknown", 8);
    }
    if (self->description) {
        size = strlen(self->description) + 1;
        tmp = self->description;
        self->description = malloc(size);
        strlcpy(self->description, tmp, size);
    } else {
        self->description = malloc(1);
        strlcpy(self->description, "", 1);
    }

    return result;
}

void discardEvent(Event *event) {
    PyGILState_STATE state;

    state = PyGILState_Ensure();
    Py_XDECREF(event);
    PyGILState_Release(state);
}

int setEventType(Event *event, EventVariety type, const char *subtype) {
    int result = 0;
    char *ptr;
    size_t length;
    PyGILState_STATE state;

    state = PyGILState_Ensure();
    if (event) {
        event->type = type;
        ptr = event->subtype;
        length = strlen(subtype) + 1;
        event->subtype = malloc(length);
        strlcpy(event->subtype, subtype, length);
        free(ptr);
    } else {
        result = -1;
    }
    PyGILState_Release(state);

    return result;
}

int moveEvent(Event *event, struct timespec start, struct timespec end) {
    int result = 0;
    PyGILState_STATE state;

    state = PyGILState_Ensure();
    if (event) {
        event->start_sec = start.tv_sec;
        event->start_nsec = start.tv_nsec;
        event->end_sec = end.tv_sec;
        event->end_nsec = end.tv_nsec;
    } else {
        result = -1;
    }
    PyGILState_Release(state);

    return result;
}

int setEventDescription(Event *event, const char *description) {
    int result = 0;
    char *ptr;
    size_t length;
    PyGILState_STATE state;

    state = PyGILState_Ensure();
    if (event) {
        ptr = event->description;
        length = strlen(description) + 1;
        event->description = malloc(length);
        strlcpy(event->description, description, length);
        free(ptr);
    } else {
        result = -1;
    }
    PyGILState_Release(state);

    return result;
}
