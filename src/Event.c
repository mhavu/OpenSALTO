//
//  Event.c
//  OpenSALTO
//
//  Copyright 2014 Marko Havu. Released under the terms of
//  GNU General Public License version 3 or later.
//

#include "Event.h"
#include <structmember.h>
#include "salto.h"

// Define API functions.

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


// Define Event class methods.

static void Event_dealloc(Event* self) {
    free(self->subtype);
    free(self->description);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject *Event_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    return (PyObject *)type->tp_alloc(type, 0);
}

static int Event_init(Event *self, PyObject *args, PyObject *kwds) {
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

static PyObject *Event_richcmp(Event *self, PyObject *other, int op) {
    // Compares event start times
    PyObject *result;
    long long o_sec;
    long o_nsec;
    
    if (PyObject_TypeCheck(other, &EventType)) {
        o_sec = ((Event *)other)->start_sec;
        o_nsec = ((Event *)other)->start_nsec;
        switch (op) {
            case Py_LT:
                if (self->start_sec > o_sec) {
                    result = Py_False;
                } else if (self->start_sec == o_sec && self->start_nsec >= o_nsec) {
                    result = Py_False;
                } else {
                    result = Py_True;
                }
                break;
            case Py_LE:
                if (self->start_sec > o_sec) {
                    result = Py_False;
                } else if (self->start_sec == o_sec && self->start_nsec > o_nsec) {
                    result = Py_False;
                } else {
                    result = Py_True;
                }
                break;
            case Py_EQ:
                result = (self->start_sec == o_sec && self->start_nsec == o_nsec) ? Py_True : Py_False;
                break;
            case Py_NE:
                result = (self->start_sec == o_sec && self->start_nsec == o_nsec) ? Py_False : Py_True;
                break;
            case Py_GT:
                if (self->start_sec < o_sec) {
                    result = Py_False;
                } else if (self->start_sec == o_sec && self->start_nsec <= o_nsec) {
                    result = Py_False;
                } else {
                    result = Py_True;
                }
                break;
            case Py_GE:
                if (self->start_sec < o_sec) {
                    result = Py_False;
                } else if (self->start_sec == o_sec && self->start_nsec < o_nsec) {
                    result = Py_False;
                } else {
                    result = Py_True;
                }
                break;
            default:
                result = Py_NotImplemented;
        }
    } else {
        result = Py_NotImplemented;
    }
    Py_INCREF(result);
    
    return result;
}

static PyObject *Event_start(Event *self) {
    PyObject *timespec, *result;
    
    timespec = Py_BuildValue("Ll", self->start_sec, self->start_nsec);  // new
    if (timespec) {
        result = datetimeFromTimespec(NULL, timespec);
        Py_DECREF(timespec);
    } else {
        result = NULL;
    }
    
    return result;
}

static PyObject *Event_end(Event *self) {
    PyObject *timespec, *result;
    
    timespec = Py_BuildValue("Ll", self->end_sec, self->end_nsec);  // new
    if (timespec) {
        result = datetimeFromTimespec(NULL, timespec);
        Py_DECREF(timespec);
    } else {
        result = NULL;
    }
    
    return result;
}

static PyObject *Event_duration(Event *self) {
    double duration = self->end_sec - self->start_sec + (self->end_nsec - self->start_nsec) / 1e9;
    return PyFloat_FromDouble(duration);
}

static PyObject *Event_union(Event *self, PyObject *args) {
    PyObject *list, *result;
    Event *e1, *e2;
    Py_ssize_t size, i;
    
    list = PySequence_List(args);  // new
    result = PySet_New(NULL);  // new
    if (list && result) {
        PyList_Sort(list);  // new
        size = PyList_GET_SIZE(list);
        e1 = (Event *)PyObject_CallFunction((PyObject *)&EventType, "isLlLls", self->type,
                                            self->subtype, self->start_sec, self->start_nsec,
                                            self->end_sec, self->end_nsec, self->description);  // new
        for (i = 0; i < size; i++) {
            e2 = (Event *)PyList_GET_ITEM(list, i);  // borrowed
            if (e2->type != self->type || strcmp(e2->subtype, self->subtype) ||
                e2->end_sec < e1->start_sec || e2->start_sec > e1->end_sec ||
                (e2->end_sec == e1->start_sec && e2->end_nsec < e1->start_nsec) ||
                (e2->start_sec == e1->end_sec && e2->start_nsec > e1->end_nsec)) {
                // Events are of different type, or they do not overlap. Keep both.
                PySet_Add(result, (PyObject *)e2);
            } else {
                // Events overlap. Modify e1, and discard e2.
                if (e2->start_sec < e1->start_sec || (e2->start_sec == e1->start_sec &&
                                                      e2->start_nsec < e1->start_nsec)) {
                    e1->start_sec = e2->start_sec;
                    e1->start_nsec = e2->start_nsec;
                }
                if (e2->end_sec > e1->end_sec || (e2->end_sec == e1->end_sec &&
                                                  e2->end_nsec > e1->end_nsec)) {
                    e1->end_sec = e2->end_sec;
                    e1->end_nsec = e2->end_nsec;
                }
            }
        }
        PySet_Add(result, (PyObject *)e1);
    } else {
        Py_XDECREF(result);
        result = NULL;
    }
    Py_XDECREF(list);
    
    return result;
}

static PyObject *Event_intersection(Event *self, PyObject *args) {
    PyObject *list, *result;
    Event *other, *copy;
    Py_ssize_t size, i;
    
    list = PySequence_List(args);  // new
    result = PySet_New(NULL);  // new
    if (list && result) {
        PyList_Sort(list);  // new
        size = PyList_GET_SIZE(list);
        for (i = 0; i < size; i++) {
            other = (Event *)PyList_GET_ITEM(list, i);  // borrowed
            if (other->end_sec < self->start_sec ||
                (other->end_sec == self->start_sec && other->end_nsec < self->start_nsec)) {
                // Events do not overlap.
                continue;
            } else if (other->start_sec > self->end_sec ||
                       (other->start_sec == self->end_sec && other->start_nsec > self->end_nsec)) {
                // Events do not overlap.
                break;
            } else {
                // Events overlap.
                Py_INCREF(other);
                if (other->start_sec < self->start_sec ||
                    (other->start_sec == self->start_sec && other->start_nsec < self->start_nsec)) {
                    copy = (Event *)PyObject_CallFunction((PyObject *)&EventType, "isLlLls",
                                                          other->type, other->subtype,
                                                          self->start_sec, self->start_nsec,
                                                          other->end_sec, other->end_nsec,
                                                          other->description);  // new
                    Py_DECREF(other);
                    other = copy;
                }
                if (other->end_sec > self->end_sec ||
                    (other->end_sec == self->end_sec && other->end_nsec > self->end_nsec)) {
                    copy = (Event *)PyObject_CallFunction((PyObject *)&EventType, "isLlLls",
                                                          other->type, other->subtype,
                                                          other->start_sec, other->start_nsec,
                                                          self->end_sec, self->end_nsec,
                                                          other->description);  // new
                    Py_DECREF(other);
                    other = copy;
                }
                PySet_Add(result, (PyObject *)other);
            }
        }
    } else {
        Py_XDECREF(result);
        result = NULL;
    }
    Py_XDECREF(list);
    
    return result;
}

static PyMethodDef Event_methods[] = {
    {"start", (PyCFunction)Event_start, METH_NOARGS, "event start time as a datetime object"},
    {"duration", (PyCFunction)Event_duration, METH_NOARGS, "event duration in seconds"},
    {"end", (PyCFunction)Event_end, METH_NOARGS, "event end time as a datetime object"},
    {"union", (PyCFunction)Event_union, METH_VARARGS, "Return the union of events as a new set of events."},
    {"intersection", (PyCFunction)Event_intersection, METH_VARARGS, "Return the intersection of events as a new set of events."},
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
    PyObject_HEAD_INIT(NULL)
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
    (richcmpfunc)Event_richcmp,  // tp_richcompare
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