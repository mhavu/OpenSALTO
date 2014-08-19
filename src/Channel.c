//
//  Channel.c
//  OpenSALTO
//
//  Copyright 2014 Marko Havu. Released under the terms of
//  GNU General Public License version 3 or later.
//

#define PY_ARRAY_UNIQUE_SYMBOL Py_Array_API_OpenSALTO
#define NO_IMPORT_ARRAY
#include "Channel.h"
#include "salto.h"

static PyMethodDef Channel_methods[] = {
    {"start", (PyCFunction)Channel_start, METH_NOARGS, "channel start time as a datetime object"},
    {"duration", (PyCFunction)Channel_duration, METH_NOARGS, "channel duration in seconds"},
    {"end", (PyCFunction)Channel_end, METH_NOARGS, "channel end time as a datetime object"},
    {NULL}  // sentinel
};

static PyMemberDef Channel_members[] = {
    {"__dict__", T_OBJECT, offsetof(Channel, dict), READONLY, "dictionary for instance variables"},
    {"data", T_OBJECT_EX, offsetof(Channel, data), READONLY, "Channel data as NumPy array or collection of Channel objects"},
    {"fill_values", T_OBJECT_EX, offsetof(Channel, fill_values), 0, "fill values for collection channels as NumPy array"},
    {"samplerate", T_DOUBLE, offsetof(Channel, samplerate), 0, "sample rate in Hz"},
    {"scale", T_DOUBLE, offsetof(Channel, scale), 0, "scale for integer channels"},
    {"offset", T_DOUBLE, offsetof(Channel, offset), 0, "offset for integer channels"},
    {"unit", T_STRING, offsetof(Channel, unit), 0, "channel units"},
    {"type", T_STRING, offsetof(Channel, type), 0, "channel type"},
    {"start_sec", T_LONGLONG, offsetof(Channel, start_sec), 0, "start time (POSIX time)"},
    {"start_nsec", T_LONG, offsetof(Channel, start_nsec), 0, "nanoseconds to add to the start time"},
    {"device", T_STRING, offsetof(Channel, device), 0, "device make and model"},
    {"serial_no", T_STRING, offsetof(Channel, serial_no), 0, "device serial number"},
    {"resolution", T_INT, offsetof(Channel, resolution), 0, "sampling resolution in bits"},
    {"collection", T_INT, offsetof(Channel, collection), 0, "indicates a collection channel"},
    {"json", T_STRING, offsetof(Channel, json), 0, "additional metadata in JSON format"},
    {"events", T_OBJECT_EX, offsetof(Channel, events), 0, "set of Event objects"},
    {NULL}  // sentinel
};

PyTypeObject ChannelType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "salto.Channel",             // tp_name
    sizeof(Channel),             // tp_basicsize
    0,                           // tp_itemsize
    (destructor)Channel_dealloc, // tp_dealloc
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
    "OpenSALTO channel",         // tp_doc
    0,		                     // tp_traverse
    0,		                     // tp_clear
    (richcmpfunc)Channel_richcmp,// tp_richcompare
    0,		                     // tp_weaklistoffset
    0,		                     // tp_iter
    0,		                     // tp_iternext
    Channel_methods,             // tp_methods
    Channel_members,             // tp_members
    0,                           // tp_getset
    0,                           // tp_base
    0,                           // tp_dict
    0,                           // tp_descr_get
    0,                           // tp_descr_set
    offsetof(Channel, dict),     // tp_dictoffset
    (initproc)Channel_init,      // tp_init
    0,                           // tp_alloc
    Channel_new                  // tp_new
};

void Channel_dealloc(Channel* self) {
    free(self->device);
    free(self->serial_no);
    free(self->unit);
    free(self->type);
    free(self->json);
    Py_XDECREF(self->data);
    Py_XDECREF(self->dict);
    Py_XDECREF(self->fill_values);
    Py_XDECREF(self->events);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

PyObject *Channel_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    Channel *self;
    PyObject *data;
    Py_ssize_t length, i;
    
    data = PyTuple_GetItem(args, 0);
    if (data && (PyArray_Check(data) || PyList_Check(data))) {
        self = (Channel *)type->tp_alloc(type, 0);
        if (self) {
            if (PyList_Check(data)) {
                self->collection = 1;
                length = PyList_Size(data);
                for (i = 0; i < length; i++) {
                    if (!PyObject_TypeCheck(data, &ChannelType)) {
                        PyErr_SetString(PyExc_TypeError, "Channel.init() takes a NumPy array or a list of Channel objects as an argument");
                        Py_DECREF(self);
                        self = NULL;
                        break;
                    }
                }
            } else {
                self->collection = 0;
            }
        }
        if (self) {
            self->dict = NULL;
            self->events = NULL;
            Py_INCREF(data);
            self->data = data;
            if (self->data == NULL) {
                Py_XDECREF(self->events);
                Py_DECREF(data);
                Py_DECREF(self);
                self = NULL;
            }
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Channel.init() takes a NumPy array or a list of Channel objects as an argument");
        self = NULL;
    }

    return (PyObject *)self;
}

int Channel_init(Channel *self, PyObject *args, PyObject *kwds) {
    PyObject *data = NULL, *events = NULL;
    int result;
    static char *kwlist[] = {"data", "fill_values", "samplerate", "scale", "offset", "unit", "type",
        "start_sec", "start_nsec", "device", "serial_no", "resolution", "json", "events", NULL};

    self->device = NULL;
    self->serial_no = NULL;
    self->unit = NULL;
    self->type = NULL;
    self->scale = 1.0;
    self->json = NULL;
    self->fill_values = NULL;
    result = !PyArg_ParseTupleAndKeywords(args, kwds, "O|OdddssLlssisO", kwlist, &data, &(self->fill_values),
                                          &(self->samplerate), &(self->scale), &(self->offset),
                                          &(self->unit), &(self->type), &(self->start_sec), &(self->start_nsec),
                                          &(self->device), &(self->serial_no), &(self->resolution), &(self->json),
                                          &events);
    if (PyArray_Check(self->fill_values)) {
        self->events = (PySetObject *)PySet_New(events);  // new
        if (!self->events)
            result = -1;
    } else {
        result = -1;
        PyErr_SetString(PyExc_TypeError, "fill_values argument must be a NumPy array");
    }
    if (result == 0) {
        if (!self->device) {
            self->device = malloc(8);
            strlcpy(self->device, "unknown", 8);
        }
        if (!self->serial_no) {
            self->serial_no = malloc(8);
            strlcpy(self->serial_no, "unknown", 8);
        }
        if (!self->type) {
            self->type = malloc(8);
            strlcpy(self->type, "unknown", 8);
        }
        if (!self->json) {
            self->json = malloc(3);
            strlcpy(self->json, "{}", 3);
        }
    }

    return result;
}

PyObject *Channel_richcmp(Channel *self, PyObject *other, int op) {
    PyObject *result;
    long long o_sec;
    long o_nsec;

    if (PyObject_TypeCheck(other, &ChannelType)) {
        o_sec = ((Channel *)other)->start_sec;
        o_nsec = ((Channel *)other)->start_nsec;
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

PyObject *Channel_start(Channel *self) {
    PyObject *timespec = Py_BuildValue("(Ll)", self->start_sec, self->start_nsec);
    return datetimeFromTimespec((PyObject *)self, timespec);
}

PyObject *Channel_duration(Channel *self) {
    PyObject *start, *end, *timedelta, *s;
    Channel *last;
    Py_ssize_t nParts;
    npy_intp length;
    double duration;

    if (self->collection) {
        start = Channel_start(self);  // new
        nParts = PyList_Size(self->data);
        last = (Channel *)PyList_GET_ITEM(self->data, nParts - 1);  // borrowed
        end = Channel_end(last);  // new
        timedelta = PyObject_CallMethod(start, "__sub__", "(O)", end);  // new
        s = PyObject_CallMethod(timedelta, "total_seconds", NULL);  // new
        duration = PyFloat_AsDouble(s);
        Py_XDECREF(start);
        Py_XDECREF(end);
        Py_XDECREF(timedelta);
        Py_XDECREF(s);
    } else {
        length = PyArray_DIM((PyArrayObject *)self->data, 0);
        if (length > 0) {
            duration = (length - 1) / self->samplerate;
        } else {
            duration = nan(NULL);
        }
    }

    return PyFloat_FromDouble(duration);
}

PyObject *Channel_end(Channel *self) {
    double t;
    PyObject *duration, *timespec, *datetime;

    duration = Channel_duration(self);  // new
    t = self->start_sec + self->start_nsec / 1.0e9 + PyFloat_AsDouble(duration);
    timespec = Py_BuildValue("(Ll)", (long long)t, (long)(fmod(t, 1.0) * 1.0e9));  // new
    datetime = datetimeFromTimespec((PyObject *)self, timespec);  // new
    Py_XDECREF(duration);
    Py_XDECREF(timespec);

    return datetime;
}