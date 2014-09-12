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
#include <structmember.h>
#include "salto.h"

// Define Channel class methods.

static void Channel_dealloc(Channel* self) {
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

static PyObject *Channel_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
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

static int Channel_init(Channel *self, PyObject *args, PyObject *kwds) {
    PyObject *data = NULL, *events = NULL;
    int result;
    size_t size;
    char *tmp;
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
    if (!self->fill_values || PyArray_Check(self->fill_values)) {
        self->events = (PySetObject *)PySet_New(events);  // new
        if (!self->events)
            result = -1;
    } else {
        result = -1;
        PyErr_SetString(PyExc_TypeError, "fill_values argument must be a NumPy array");
    }
    if (result == 0) {
        if (self->device) {
            size = strlen(self->device) + 1;
            tmp = self->device;
            self->device = malloc(size);
            strlcpy(self->device, tmp, size);
        } else {
            self->device = malloc(8);
            strlcpy(self->device, "unknown", 8);
        }
        if (self->serial_no) {
            size = strlen(self->serial_no) + 1;
            tmp = self->serial_no;
            self->serial_no = malloc(size);
            strlcpy(self->serial_no, tmp, size);
        } else {
            self->serial_no = malloc(8);
            strlcpy(self->serial_no, "unknown", 8);
        }
        if (self->unit) {
            size = strlen(self->unit) + 1;
            tmp = self->unit;
            self->unit = malloc(size);
            strlcpy(self->unit, tmp, size);
        }
        if (self->type) {
            size = strlen(self->type) + 1;
            tmp = self->type;
            self->type = malloc(size);
            strlcpy(self->type, tmp, size);
        } else {
            self->type = malloc(8);
            strlcpy(self->type, "unknown", 8);
        }
        if (self->json) {
            size = strlen(self->json) + 1;
            tmp = self->json;
            self->json = malloc(size);
            strlcpy(self->json, tmp, size);
        } else {
            self->json = malloc(3);
            strlcpy(self->json, "{}", 3);
        }
    }

    return result;
}

static PyObject *Channel_richcmp(Channel *self, PyObject *other, int op) {
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

static PyObject *Channel_start(Channel *self) {
    PyObject *timespec = Py_BuildValue("(Ll)", self->start_sec, self->start_nsec);
    return datetimeFromTimespec((PyObject *)self, timespec);
}

static PyObject *Channel_end(Channel *self) {
    struct timespec t = channelEndTime(self);
    PyObject *timespec = Py_BuildValue("(Ll)", t.tv_sec, t.tv_nsec);
    return datetimeFromTimespec((PyObject *)self, timespec);
}

static PyObject *Channel_duration(Channel *self) {
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

static PyObject *Channel_matches(Channel *self, PyObject *args) {
    PyObject *result = Py_False;
    Channel *other;
    Py_ssize_t length, i;

    if (PyArg_ParseTuple(args, "O!", &ChannelType, &other)) {
        if (strcmp(self->type, other->type) == 0 &&
            self->samplerate == other->samplerate &&
            self->collection == other->collection &&
            self->start_sec == other->start_sec &&
            self->start_nsec == other->start_nsec)
        {
            if (self->collection) {
                length = PyList_GET_SIZE(self->data);
                if (PyList_GET_SIZE(other->data) == length) {
                    result = Py_True;
                    for (i = 0; i < length; i++) {
                        result = Channel_matches((Channel *)PyList_GET_ITEM(self->data, i),
                                                 PyList_GET_ITEM(other->data, i));
                        if (result != Py_True)
                            break;
                    }
                }
            } else if (PyArray_DIM((PyArrayObject *)self->data, 0) ==
                       PyArray_DIM((PyArrayObject *)other->data, 0)) {
                result = Py_True;
            }
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Channel.matches() takes a Channel argument");
    }
    Py_INCREF(result);

    return result;
}

static PyObject *Channel_resampledData(Channel *self, PyObject *args, PyObject *kwds) {
    Channel *part;
    PyArrayObject *roundedArray, *newArray = NULL, *tmpArray = NULL;
    PyObject *numpy, *linspace, *rint, *slice, *idxArray, *start, *end;
    double samplerate;
    long long start_sec, end_sec, tmpvar;
    long start_nsec, end_nsec;
    int oldTypenum, newTypenum, flip = 0;
    struct timespec end_t;
    Py_ssize_t start_idx, end_idx, nParts, i;
    const char *method = "interpolate-decimate";
    static char *kwlist[] = {"samplerate", "start_sec", "start_nsec", "end_sec", "end_nsec", "typenum", "method", NULL};

    samplerate = self->samplerate;
    start_sec = self->start_sec;
    start_nsec = self->start_nsec;
    end_t = channelEndTime(self);
    end_sec = end_t.tv_sec;
    end_nsec = end_t.tv_nsec;
    if (!self->collection) {
        oldTypenum = PyArray_DTYPE((PyArrayObject *)self->data)->type_num;
    } else {
        oldTypenum = PyArray_DTYPE(self->fill_values)->type_num;
    }
    newTypenum = oldTypenum;
    if (PyArg_ParseTupleAndKeywords(args, kwds, "d|LlLlis", kwlist, &samplerate,
                                    &start_sec, &start_nsec, &end_sec, &end_nsec, &newTypenum, &method)) {
        // Check validity of the start and end times.
        if (start_sec < self->start_sec) {
            start_sec = self->start_sec;
            start_nsec = self->start_nsec;
        } else if (start_sec == self->start_sec && start_nsec < self->start_nsec) {
            start_nsec = self->start_nsec;
        }
        if (end_sec > end_t.tv_sec) {
            end_sec = end_t.tv_sec;
            end_nsec = end_t.tv_nsec;
        } else if (end_sec == end_t.tv_sec && end_nsec > end_t.tv_nsec) {
            end_nsec = end_t.tv_nsec;
        }
        if (end_sec < start_sec || (end_sec == start_sec && end_nsec < start_nsec)) {
            tmpvar = start_sec;
            start_sec = end_sec;
            end_sec = tmpvar;
            tmpvar = start_nsec;
            start_nsec = end_nsec;
            end_nsec = tmpvar;
            flip = 1;  // TODO: Flip data.
        }
        
        // Get references to necessary NumPy functions.
        numpy = PyImport_AddModule("numpy");  // borrowed
        linspace = PyObject_GetAttrString(numpy, "linspace");  // new
        rint = PyObject_GetAttrString(numpy, "rint");  // new

        // Resample the specified part of the array.
        if (!self->collection) {
            start_idx = (start_sec - self->start_sec + (start_nsec - self->start_nsec) / 1e9) * self->samplerate;
            end_idx = PyArray_DIM((PyArrayObject *)self->data, 0);
            end_idx -= (end_t.tv_sec - end_sec + (end_t.tv_nsec - end_nsec) / 1e9) * self->samplerate;

            if (strcmp(method, "interpolate-decimate") == 0) {
                // Get a view to the specified part of the array.
                start = PyLong_FromSsize_t(start_idx);
                end = PyLong_FromSsize_t(end_idx);
                slice = PySlice_New(start, end, NULL);  // new
                Py_DECREF(start);
                Py_DECREF(end);
                tmpArray = (PyArrayObject *)PyObject_GetItem(self->data, slice);  // new
                // TODO: FIR
                Py_XDECREF(slice);
            } else if (strcmp(method, "VRA") == 0) {
                if (samplerate < self->samplerate) {
                    slice = PyObject_CallFunction(linspace, "nnd", start_idx, end_idx - 1,
                                                  (end_idx - start_idx) * samplerate / self->samplerate);  // new
                    roundedArray = (PyArrayObject *)PyObject_CallFunctionObjArgs(rint, slice, NULL);  // new
                    Py_XDECREF(slice);
                    idxArray = PyArray_Cast(roundedArray, NPY_INT64);  // new
                    Py_XDECREF(roundedArray);
                    tmpArray = (PyArrayObject *)PyObject_GetItem(self->data, idxArray);
                    Py_XDECREF(idxArray);
                } else {
                    start = PyLong_FromSsize_t(start_idx);
                    end = PyLong_FromSsize_t(end_idx);
                    slice = PySlice_New(start, end, NULL);  // new
                    Py_DECREF(start);
                    Py_DECREF(end);
                    tmpArray = (PyArrayObject *)PyObject_GetItem(self->data, slice);  // new
                }
                // TODO: Aggregate min and max between each new sample.
            } else {
                PyErr_SetString(PyExc_AttributeError, "unknown resampling method");
            }
        } else {
            nParts = PyList_GET_SIZE(self->data);
            for (i = 0; i < nParts; i++) {
                part = (Channel *)PyList_GET_ITEM(self->data, i);
                PyArray_GETPTR1(self->fill_values, i);
                // TODO: implement
                // PyObject_RichCompareBool(o1, o2, Py_LT);
            }
        }
        Py_XDECREF(linspace);
        Py_XDECREF(rint);
        if (tmpArray) {
            if (newTypenum != oldTypenum) {
                newArray = (PyArrayObject *)PyArray_Cast(tmpArray, newTypenum);  // new
                // TODO: (TYPE)value * (double)scale + (double)offset * (char *)unit
                Py_DECREF(tmpArray);
            } else {
                newArray = tmpArray;
            }
        }
    }

    return (PyObject *)newArray;
}

static PyObject *Channel_resample(Channel *self, PyObject *args, PyObject *kwds) {
    Channel *newChannel = NULL;
    PyObject *array, *events, *iterator;
    Event *e, *eCopy;
    double samplerate, scale, offset;
    long long start_sec, end_sec, eStart_sec, eEnd_sec;
    long start_nsec, end_nsec, eStart_nsec, eEnd_nsec;
    int oldTypenum, newTypenum, resolution, copyEvents;
    struct timespec end_t;
    const char *method = "interpolate-decimate";
    static char *kwlist[] = {"samplerate", "start_sec", "start_nsec", "end_sec", "end_nsec", "typenum", "method", "copyEvents", NULL};

    samplerate = self->samplerate;
    start_sec = self->start_sec;
    start_nsec = self->start_nsec;
    end_t = channelEndTime(self);
    end_sec = end_t.tv_sec;
    end_nsec = end_t.tv_nsec;
    if (!self->collection) {
        oldTypenum = PyArray_DTYPE((PyArrayObject *)self->data)->type_num;
    } else {
        oldTypenum = PyArray_DTYPE(self->fill_values)->type_num;
    }
    newTypenum = oldTypenum;
    if (PyArg_ParseTupleAndKeywords(args, kwds, "d|LlLlisi:resample", kwlist, &samplerate,
                                    &start_sec, &start_nsec, &end_sec, &end_nsec, &newTypenum, &method, &copyEvents)) {
        // Check validity of the start and end times.
        if (start_sec < self->start_sec) {
            start_sec = self->start_sec;
            start_nsec = self->start_nsec;
        } else if (start_sec == self->start_sec && start_nsec < self->start_nsec) {
            start_nsec = self->start_nsec;
        }
        if (end_sec > end_t.tv_sec) {
            end_sec = end_t.tv_sec;
            end_nsec = end_t.tv_nsec;
        } else if (end_sec == end_t.tv_sec && end_nsec > end_t.tv_nsec) {
            end_nsec = end_t.tv_nsec;
        }
        // Resample and create a new Channel object.
        array = Channel_resampledData(self, args, kwds);  // new
        if (array) {
            if (newTypenum == oldTypenum) {
                scale = self->scale;
                offset = self->offset;
                resolution = self->resolution;
            } else {
                scale = 1.0;
                offset = 0.0;
                resolution = 0;
            }
            events = PySet_New(NULL);  // new
            if (copyEvents) {
                iterator = PyObject_GetIter((PyObject *)self->events);  // new
                if (iterator) {
                    while ((e = (Event *)PyIter_Next(iterator))) {
                        if (e->end_sec >= start_sec && e->start_sec <= end_sec) {
                            eStart_sec = e->start_sec;
                            eStart_nsec = e->start_nsec;
                            eEnd_sec = e->end_sec;
                            eEnd_nsec = e->end_nsec;
                            if (eStart_sec < start_sec) {
                                eStart_sec = start_sec;
                                eStart_nsec = start_nsec;
                            } else if (eStart_sec == start_sec && eStart_nsec < start_nsec){
                                eStart_nsec = start_nsec;
                            }
                            if (eEnd_sec > end_sec) {
                                eEnd_sec = end_sec;
                                eEnd_nsec = end_nsec;
                            } else if (eEnd_sec == end_sec && eEnd_nsec < end_nsec){
                                eEnd_nsec = end_nsec;
                            }
                            eCopy = (Event *)PyObject_CallFunction((PyObject *)&EventType, "isLlLls", e->type,
                                                                   e->subtype, eStart_sec, eStart_nsec,
                                                                   eEnd_sec, eEnd_nsec, e->description);  // new
                            if (eCopy) {
                                PySet_Add(events, (PyObject *)eCopy);
                                Py_DECREF(eCopy);
                            }
                        }
                        Py_DECREF(e);
                    }
                    Py_DECREF(iterator);
                }
            }
            PyObject_CallFunction((PyObject *)&ChannelType, "OOdddssLlssisO", array, Py_None,
                                  samplerate, scale, offset, self->unit, self->type, start_sec, start_nsec,
                                  self->device, self->serial_no, resolution, self->json, events);  // new
            Py_DECREF(array);
        }
    }
    
    return (PyObject *)newChannel;
}

static PyMethodDef Channel_methods[] = {
    {"start", (PyCFunction)Channel_start, METH_NOARGS, "channel start time as a datetime object"},
    {"duration", (PyCFunction)Channel_duration, METH_NOARGS, "channel duration in seconds"},
    {"end", (PyCFunction)Channel_end, METH_NOARGS, "channel end time as a datetime object"},
    {"matches", (PyCFunction)Channel_matches, METH_VARARGS, "check whether channel type and time match those of another channel"},
    {"resample", (PyCFunction)Channel_resample, METH_VARARGS | METH_KEYWORDS, "resample channel"},
    {"resampledData", (PyCFunction)Channel_resampledData, METH_VARARGS | METH_KEYWORDS, "resampled channel data as a NumPy array"},
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