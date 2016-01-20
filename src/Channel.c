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
#include <complex.h>
#include <fftw3.h>
#include <structmember.h>
#include "salto.h"

#define roundToNext(a, b) ((b) * ceil((double)(a) / (b)))

// Define Channel class methods.

static void Channel_dealloc(Channel* self) {
    free(self->device);
    free(self->serial_no);
    free(self->unit);
    free(self->type);
    free(self->json);
    Py_DECREF(self->data);
    Py_XDECREF(self->dict);
    Py_XDECREF(self->fills);
    Py_XDECREF(self->events);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static int Channel_init(Channel *self, PyObject *args, PyObject *kwds) {
    PyObject *events = NULL, *fills = NULL;
    size_t size;
    char *tmp;
    static char *kwlist[] = {"data", "samplerate", "fills",
        "scale", "offset", "unit", "type",
        "start_sec", "start_nsec", "device", "serial_no", "resolution",
        "json", "events", NULL};

    self->device = NULL;
    self->serial_no = NULL;
    self->unit = NULL;
    self->type = NULL;
    self->json = NULL;
    self->scale = 1.0;
    self->offset = 0.0;
    self->start_sec = 0;
    self->start_nsec = 0;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!d|OddssLlssisO", kwlist,
                                     &PyArray_Type, &self->data,
                                     &(self->samplerate),
                                     &fills,
                                     &(self->scale), &(self->offset),
                                     &(self->unit), &(self->type),
                                     &(self->start_sec), &(self->start_nsec),
                                     &(self->device), &(self->serial_no),
                                     &(self->resolution), &(self->json),
                                     &events)) {
        return -1;
    }
    if (self->start_nsec < 0 || self->start_nsec > 999999999) {
        PyErr_SetString(PyExc_ValueError, "start_nsec is out of range");
        return -1;
    }
    // Check that fills has the correct type descriptor.
    self->fills = (PyArrayObject *)newFillArray(fills, 0);  // new
    if (self->fills) {
        self->events = (PySetObject *)PySet_New(events);  // new
        if (!self->events) {
            Py_DECREF(self->fills);
            return -1;
        }
    } else {
        return -1;
    }

    Py_INCREF(self->data);
    if (self->device) {
        size = strlen(self->device) + 1;
        tmp = self->device;
        self->device = malloc(size);
        strcpy(self->device, tmp);
    } else {
        self->device = malloc(8);
        strcpy(self->device, "unknown");
    }
    if (self->serial_no) {
        size = strlen(self->serial_no) + 1;
        tmp = self->serial_no;
        self->serial_no = malloc(size);
        strcpy(self->serial_no, tmp);
    } else {
        self->serial_no = malloc(8);
        strcpy(self->serial_no, "unknown");
    }
    if (self->unit) {
        size = strlen(self->unit) + 1;
        tmp = self->unit;
        self->unit = malloc(size);
        strcpy(self->unit, tmp);
    } else {
        self->unit = malloc(1);
        self->unit[0] = 0;
    }
    if (self->type) {
        size = strlen(self->type) + 1;
        tmp = self->type;
        self->type = malloc(size);
        strcpy(self->type, tmp);
    } else {
        self->type = malloc(8);
        strcpy(self->type, "unknown");
    }
    if (self->json) {
        size = strlen(self->json) + 1;
        tmp = self->json;
        self->json = malloc(size);
        strcpy(self->json, tmp);
    } else {
        self->json = malloc(3);
        strcpy(self->json, "{}");
    }

    return 0;
}

static PyObject *Channel_copy(Channel *self, PyObject *args, PyObject *kwds) {
    PyArrayObject *data;
    PyArrayObject *fills;
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
    PyObject *copy = NULL;
    static char *kwlist[] = {"data", "samplerate", "fills",
        "scale", "offset", "unit", "type",
        "start_sec", "start_nsec", "device", "serial_no", "resolution",
        "json", "events", NULL};
    
    data = self->data;
    fills = self->fills;
    samplerate = self->samplerate;
    scale = self->scale;
    offset = self->offset;
    unit = self->unit;
    type = self->type;
    start_sec = self->start_sec;
    start_nsec = self->start_nsec;
    device = self->device;
    serial_no = self->serial_no;
    resolution = self->resolution;
    json = self->json;
    events = self->events;
    if (PyArg_ParseTupleAndKeywords(args, kwds, "|O!dOddssLlssisO", kwlist,
                                     &PyArray_Type, &data, &samplerate,
                                     &fills, &scale, &offset,
                                     &unit, &type, &start_sec, &start_nsec,
                                     &device, &serial_no, &resolution, &json,
                                     &events)) {
        copy = PyObject_CallFunction((PyObject *)&ChannelType, "OdOddssLlssisO",
                                     data, samplerate, fills, scale, offset,
                                     unit, type, start_sec, start_nsec,
                                     device, serial_no, resolution, json,
                                     events);  // new
    }
    
    return copy;
}

static PyObject *Channel_simpleCopy(Channel *self) {
    return PyObject_CallFunction((PyObject *)&ChannelType, "OdOddssLlssisO",
                                 self->data,
                                 self->samplerate,
                                 self->fills,
                                 self->scale, self->offset,
                                 self->unit, self->type,
                                 self->start_sec, self->start_nsec,
                                 self->device, self->serial_no,
                                 self->resolution, self->json,
                                 self->events);  // new
}

static PyObject *Channel_deepCopy(Channel *self, PyObject *args) {
    PyObject *data, *fills, *events = NULL, *ptr, *copy = NULL, *memo = NULL;
    PyObject *item, *iterator, *eCopy;

    if (!PyArg_ParseTuple(args, "|O!", &PyDict_Type, &memo)) {
        return NULL;
    }
    if (memo) {
        Py_INCREF(memo);
    } else {
        memo = PyDict_New();  // new
    }
    data = PyObject_CallMethod((PyObject *)self->data, "__deepcopy__", "O", memo);  // new
    fills = PyObject_CallMethod((PyObject *)self->fills, "__deepcopy__", "O", memo);  // new
    ptr = PyLong_FromVoidPtr(self->events);  // new
    if (ptr) {
        events = PyDict_GetItem(memo, ptr);  // borrowed
        if (!events) {
            events = PySet_New(NULL);  // new
            iterator = PyObject_GetIter((PyObject *)self->events);
            if (events && iterator) {
                while ((item = PyIter_Next(iterator))) {
                    eCopy = PyObject_CallMethod(item, "__copy__", NULL);  // new
                    if (eCopy) {
                        PySet_Add(events, eCopy);
                        Py_DECREF(eCopy);
                    } else {
                        break;
                    }
                    Py_DECREF(item);
                }
                Py_DECREF(iterator);
            } else if (events) {
                Py_DECREF(events);
                events = NULL;
            }
            if (!PyErr_Occurred()) {
                PyDict_SetItem(memo, ptr, events);
            }
        }
        Py_DECREF(ptr);
    }
    if (data && fills && events) {
        copy = PyObject_CallFunction((PyObject *)&ChannelType, "OdOddssLlssisO",
                                     data, self->samplerate, fills,
                                     self->scale, self->offset,
                                     self->unit, self->type,
                                     self->start_sec, self->start_nsec,
                                     self->device, self->serial_no,
                                     self->resolution, self->json,
                                     events);  // new
    }
    Py_XDECREF(memo);
    Py_XDECREF(data);
    Py_XDECREF(fills);
    Py_XDECREF(events);
    
    return copy;
}

static PyObject *Channel_richcmp(Channel *self, PyObject *other, int op) {
    // Compares channel start times
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

static PyObject *Channel_end(Channel *self) {
    struct timespec t;
    PyObject *timespec, *result;
    
    t = channelEndTime(self);
    timespec = Py_BuildValue("Ll", t.tv_sec, t.tv_nsec);  // new
    if (timespec) {
        result = datetimeFromTimespec(NULL, timespec);
        Py_DECREF(timespec);
    } else {
        result = NULL;
    }
    
    return result;
}

static PyObject *Channel_duration(Channel *self) {
    return PyFloat_FromDouble(channelDuration(self));
}

static PyObject *Channel_timecodes(Channel *self, PyObject *args, PyObject *kwds) {
    npy_intp i, start, end, length, nFills, offset, fill;
    Channel_Fill *fills;
    double *timecodes, t;
    PyObject *result = NULL;
    static char *kwlist[] = {"start", "end", NULL};
    
    start = 0;
    end = -1;
    if (PyArg_ParseTupleAndKeywords(args, kwds, "|nn:timecodes", kwlist, &start, &end)) {
        length = PyArray_DIM(self->data, 0);
        if (start < 0) {
            start = length + start;
        }
        if (end < 0) {
            end = length + end;
        }
        if (start <= end && end < length) {
            result = PyArray_Arange(start, end + 1, 1.0, NPY_DOUBLE);  // new
            timecodes = PyArray_DATA((PyArrayObject *)result);
            t = self->start_sec + self->start_nsec / 1e9;
            if (self->fills) {
                fills = PyArray_DATA(self->fills);
                nFills = PyArray_DIM(self->fills, 0);
                offset = 0;
                fill = 0;
                while (fills[fill].pos < start && fill < nFills) {
                    offset += fills[fill].len;
                    fill++;
                }
                for (i = 0; i <= end - start; i++) {
                    if (fill < nFills && fills[fill].pos == start + i) {
                        offset += fills[fill].len;
                        fill++;
                    }
                    timecodes[i] = (timecodes[i] + offset) / self->samplerate + t;
                }
            } else {
                for (i = 0; i <= end - start; i++) {
                    timecodes[i] = timecodes[i] / self->samplerate + t;
                }
            }
        } else {
            PyErr_SetString(PyExc_IndexError, "Index out of range");
        }
    }
    
    return result;
}

static PyObject *Channel_matches(Channel *self, PyObject *args) {
    PyObject *result = Py_False;
    Channel *other;
    npy_intp size;

    if (PyArg_ParseTuple(args, "O!", &ChannelType, &other)) {
        if (strcmp(self->type, other->type) == 0 &&
            self->samplerate == other->samplerate &&
            self->start_sec == other->start_sec &&
            self->start_nsec == other->start_nsec &&
            self->scale == other->scale &&
            self->offset == other->offset &&
            strcmp(self->unit, other->unit) == 0 &&
            PyArray_TYPE(self->data) == PyArray_TYPE(other->data) &&
            PyArray_DIM(self->data, 0) == PyArray_DIM(other->data, 0))
        {
            if (!self->fills && !other->fills) {
                result = Py_True;
            } else if (self->fills && other->fills) {
                size = PyArray_NBYTES(self->fills);
                if (memcmp(PyArray_DATA(self->fills),
                           PyArray_DATA(other->fills), size) == 0) {
                        result = Py_True;
                    }
            }
        }
    } else {
        result = NULL;
        PyErr_SetString(PyExc_TypeError, "Channel.matches() takes a Channel argument");
    }
    Py_XINCREF(result);

    return result;
}

static PyObject *Channel_eventUnion(Channel *self, PyObject *args) {
    PyObject *item, *iterator, *events, *o, *result = NULL;
    Channel *ch;
    Py_ssize_t i, nChannels;
    int error = 0;
    
    item = PyTuple_GET_ITEM(args, 0);  // borrowed
    if (item) {
        ch = (Channel *)item;
        events = PyObject_CallFunctionObjArgs((PyObject *)&PyTuple_Type,
                                              (PyObject *)ch->events, NULL);  // new
        nChannels = PyTuple_GET_SIZE(args);
        for (i = 1; i < nChannels; i++) {
            item = PyTuple_GET_ITEM(args, i);  // borrowed
            if (item && PyObject_TypeCheck(item, &ChannelType)) {
                ch = (Channel *)item;
                iterator = PyObject_GetIter((PyObject *)ch->events);  // new
                if (iterator) {
                    while ((item = PyIter_Next(iterator))) {  // new
                        o = PyObject_CallMethod(item, "union", "O", events);  // new
                        if (o) {
                            Py_DECREF(events);
                            events = PyObject_CallFunctionObjArgs((PyObject *)&PyTuple_Type,
                                                                  o, NULL);  // new
                            Py_DECREF(o);
                        } else {
                            error = -1;
                            break;
                        }
                        Py_DECREF(item);
                    }
                    Py_DECREF(iterator);
                } else {
                    error = -1;
                    break;
                }
            } else {
                PyErr_SetString(PyExc_TypeError, "eventUnion() takes a sequence of Channel objects as an argument");
                error = -1;
                break;
            }
        }
        if (!error) {
            result = PySet_New(events);  // new
        }
        Py_XDECREF(events);
    } else {
        PyErr_SetString(PyExc_TypeError, "eventUnion() takes a sequence of Channel objects as an argument");
        error = -1;
    }

    return result;
}

static PyObject *Channel_eventIntersection(Channel *self, PyObject *args) {
    PyObject *o, *item, *iterator, *events, *list, *firstEvents, *result = NULL;
    Channel *ch;
    Event *e;
    long long start_sec;
    long start_nsec;
    Py_ssize_t i, nChannels, latest;
    int error = 0, done = 0;
    
    nChannels = PySequence_Size(args);
    if (nChannels >= 0) {
        events = PyList_New(nChannels);  // new
        for (i = 0; i < nChannels; i++) {
            item = PyTuple_GET_ITEM(args, i);  // borrowed
            if (item && PyObject_TypeCheck(item, &ChannelType)) {
                ch = (Channel *)item;
                list = PySequence_List((PyObject *)ch->events);  // new
                if (list) {
                    PyList_Sort(list);
                    PyList_SET_ITEM(events, i, list);  // stolen
                } else {
                    PyErr_SetString(PyExc_RuntimeError, "Could not convert events to list");
                    error = -1;
                    break;
                }
            } else {
                PyErr_SetString(PyExc_TypeError, "eventIntersection() takes a sequence of Channel objects as an argument");
                error = -1;
                break;
            }
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "eventIntersection() takes a sequence of Channel objects as an argument");
        error = -1;
    }
    if (!error) {
        result = PyList_New(0);  // new
        firstEvents = PyList_New(nChannels);  // new
        if (!result || !firstEvents) {
            Py_XDECREF(result);
            PyErr_SetString(PyExc_RuntimeError, "Creating lists failed");
            error = -1;
        }
    }
    if (!error) {
        latest = 0;
        start_sec = -1;
        start_nsec = -1;
        while (!done && !error) {
            for (i = 0; i < nChannels; i++) {
                list = PyList_GET_ITEM(events, i);  // borrowed
                if (PyList_GET_SIZE(list) == 0) {
                    done = 1;
                    break;
                } else {
                    // Take the first event on each channel.
                    item = PyList_GET_ITEM(list, 0);  // borrowed
                    Py_INCREF(item);
                    PyList_SetItem(firstEvents, i, item);  // stolen
                    // Record the one with the latest start time.
                    e = (Event *)item;
                    if (e->start_sec > start_sec || (e->start_sec == start_sec && e->start_nsec > start_nsec)) {
                        latest = i;
                        start_sec = e->start_sec;
                        start_nsec = e->start_nsec;
                    }
                }
            }
            item = PyList_GET_ITEM(firstEvents, latest);  // borrowed
            e = (Event *)item;
            Py_INCREF(e);
            // Discard any events with end time before the start time of the latest first event.
            for (i = 0; i < nChannels; i++) {
                list = PyList_GET_ITEM(events, i);  // borrowed
                iterator = PyObject_GetIter(list);  // new
                if (iterator) {
                    while ((item = PyIter_Next(iterator))) {
                        if (((Event *)item)->end_sec < start_sec ||
                            (((Event *)item)->end_sec == start_sec && ((Event *)item)->end_nsec < start_nsec)) {
                            o = PyObject_CallMethod(list, "remove", "O", item);  // new
                            if (o) {
                                Py_DECREF(o);
                            } else {
                                PyErr_SetString(PyExc_RuntimeError, "Removing list item failed");
                                error = -1;
                                break;
                            }
                        }
                        Py_DECREF(item);
                    }
                    Py_DECREF(iterator);
                } else {
                    PyErr_SetString(PyExc_RuntimeError, "Getting iterator failed");
                    error = -1;
                    break;
                }
            }
            // 6. if any of the channels has 0 events left, stop
            // 7. check that e in 4 still has the latest start time
            //   - if not, goto 3
            // 8. get the earliest of end times of the first event of each channel
            // 9. create result event with start time from e in 4 and 8 as end time
            // 10. get the next event from channel with earliest end time in 8
            // 11. goto 4
        }
        if (!error) {
            item = result;
            result = PySet_New(item);  // new
            Py_DECREF(item);
        } else {
            Py_XDECREF(result);
            result = NULL;
        }
    }
    Py_XDECREF(firstEvents);
    
    return result;
}

static PyObject *Channel_validateTimes(Channel *self, PyObject *args) {
    // Check validity of the start and end times.
    long long start_sec, end_sec, tmpvar;
    long start_nsec, end_nsec;
    struct timespec end_t;
    int dir = 1;
    PyObject *validatedTimes = NULL;

    if (PyArg_ParseTuple(args, "LlLl", &start_sec, &start_nsec, &end_sec, &end_nsec)) {
        if (end_sec < start_sec || (end_sec == start_sec && end_nsec < start_nsec)) {
            // End time is before start time.
            tmpvar = start_sec;
            start_sec = end_sec;
            end_sec = tmpvar;
            tmpvar = start_nsec;
            start_nsec = end_nsec;
            end_nsec = tmpvar;
            dir = -1;
        }
        end_t = channelEndTime(self);
        if (end_sec < self->start_sec ||
            (end_sec == self->start_sec && end_nsec < self->start_nsec) ||
            start_sec > end_t.tv_sec ||
            (start_sec == end_t.tv_sec && start_nsec > end_t.tv_nsec)) {
            // Times are out of range.
            validatedTimes = Py_None;
            Py_INCREF(Py_None);
        } else {
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
        }
    }

    if (!validatedTimes)
        validatedTimes = Py_BuildValue("LlLli", start_sec, start_nsec, end_sec, end_nsec, dir);

    return validatedTimes;
}

static PyObject *Channel_getEvents(Channel *self, PyObject *args, PyObject *kwds) {
    PyObject *events, *times, *iterator = NULL;
    Event *e, *eCopy;
    long long start_sec, end_sec, eStart_sec, eEnd_sec;
    long start_nsec, end_nsec, eStart_nsec, eEnd_nsec;
    int dir;
    static char *kwlist[] = {"start_sec", "start_nsec", "end_sec", "end_nsec", NULL};

    start_sec = 0;
    start_nsec = 0;
    end_sec = 0;
    end_nsec = 0;
    events = PySet_New(NULL);  // new
    if (PyArg_ParseTupleAndKeywords(args, kwds, "|LlLl:getEvents", kwlist,
                                    &start_sec, &start_nsec, &end_sec, &end_nsec)) {
        times = PyObject_CallMethod((PyObject *)self, "validateTimes", "LlLl",
                                    start_sec, start_nsec, end_sec, end_nsec);  // new
        if (times) {
            if (times != Py_None) {
                PyArg_ParseTuple(times, "LlLl", &start_sec, &start_nsec, &end_sec, &end_nsec, &dir);
                iterator = PyObject_GetIter((PyObject *)self->events);  // new
            }
            Py_DECREF(times);
        }

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

    return (PyObject *)events;
}

static double timeAsOffset(Channel *self, PyObject *time) {
    // Returns offset from the beginning of the channel in seconds. If time
    // is float or timedelta, it is converted to double. If time is datetime,
    // the offset is calculated. Returns NAN if an error occurred.
    PyObject *obj, *start, *delta = NULL, *sec = NULL;
    double offset;
    
    offset = NAN;
    if (PyFloat_Check(time)) {
        offset = PyFloat_AsDouble(time);
    } else {
        start = Channel_start(self);  // new
        if (start) {
            delta = PyNumber_Subtract(time, start);  // new
            Py_DECREF(start);
            // If the subtraction succeeds, time argument is a datetime.
            // If not, let us assume for a while that it is a timedelta.
            if (!delta) {
                PyErr_Clear();
                delta = time;
                Py_INCREF(delta);
            }
        }
        obj = Py_BuildValue("(d)", 1.0);  // new
        if (obj) {
            sec = timedeltaFromFloat(NULL, obj);  // new
            Py_DECREF(obj);
            obj = NULL;
        }
        if (sec) {
            obj = PyNumber_TrueDivide(delta, sec);  // new
            Py_DECREF(sec);
        }
        Py_XDECREF(delta);
        if (obj) {
            offset = PyFloat_AsDouble(obj);
            Py_DECREF(obj);
        } else {
            PyErr_SetString(PyExc_TypeError, "Argument 'time' needs to be a datetime object, or a float or timedelta specifying time offset from channel start");
        }
    }
    
    return offset;
}

static PyObject *Channel_sampleIndex(Channel *self, PyObject *args, PyObject *kwds) {
    PyObject *time = NULL, *result = NULL;
    double n, t, offset, tolerance;
    npy_intp nFills, i, index, len;
    Channel_Fill *fills;
    int ok = 0;
    const char *method = "nearest";
    static char *kwlist[] = {"time", "method", "tolerance", NULL};
    
    // Default tolerance is ±10% of sample interval.
    tolerance = 0.1 / self->samplerate;
    if (PyArg_ParseTupleAndKeywords(args, kwds, "O|sd:sampleIndex", kwlist,
                                    &time, &method, &tolerance)) {
        t = timeAsOffset(self, time);
        if (!PyErr_Occurred()) {
            ok = 1;
        }
    }
    if (ok) {
        n = t * self->samplerate;
        if (self->fills) {
            nFills = PyArray_DIM(self->fills, 0);
            fills = PyArray_DATA(self->fills);
            for (i = 0; i < nFills; i++) {
                if (fills[i].pos + fills[i].len < n - 0.5) {
                    // The entire fill is before n.
                    n -= fills[i].len;
                } else if (fills[i].pos < n) {
                    // The fill contains n.
                    if (strcasecmp(method, "nearest") == 0) {
                        // At the end of the fill, the nearest sample is the
                        // next one after the fill. Everywhere else it is the
                        // one before, since that is the fill value.
                        n = (fills[i].pos + fills[i].len < n + 0.5) ? fills[i].pos + 1: fills[i].pos;
                    } else  if (strcasecmp(method, "next") == 0) {
                        n = fills[i].pos + 1;
                    } else  if (strcasecmp(method, "previous") == 0) {
                        n = fills[i].pos;
                    } else if (strcasecmp(method, "exact") == 0) {
                        if (fills[i].pos + fills[i].len / 2.0 < n) {
                            n = fills[i].pos;
                            offset = n - fills[i].pos;
                        } else {
                            n = fills[i].pos + 1;
                            offset = fills[i].pos + fills[i].len - n;
                        }
                    }
                    break;
                } else {
                    offset = n - round(n);
                    break;
                }
            }
        }
    }
    if (ok) {
        len = PyArray_DIM(self->data, 0);
        if (strcasecmp(method, "nearest") == 0) {
            index = round(n);
            if (index < 0) {
                index = 0;
            } else if (index >= len) {
                index = len - 1;
            }
        } else if (strcasecmp(method, "next") == 0) {
            index = (n < 0) ? 0 : ceil(n);
        } else if (strcasecmp(method, "previous") == 0) {
            index = (n < len) ? floor(n) : len - 1;
        } else if (strcasecmp(method, "exact") == 0) {
            if (abs(offset) <= tolerance) {
                index = round(n);
            } else {
                PyErr_SetString(PyExc_ValueError, "Sample time is not within tolerance");
                ok = 0;
            }
        } else {
            PyErr_SetString(PyExc_ValueError, "Valid values for method are nearest, next, previous, and exact");
            ok = 0;
        }
    }
    if (ok) {
        if (index < 0) {
            PyErr_SetString(PyExc_IndexError, "Sample time preceeds channel start");
            ok = 0;
        }
        if (index >= len) {
            PyErr_SetString(PyExc_IndexError, "Sample time exceeds channel end");
            ok = 0;
        }
    }
    if (ok) {
        result = PyLong_FromSsize_t(index);  // new
    }
    
    return result;
}

double Channel_sampleOffsetAsDouble(Channel *self, npy_intp index) {
    // Returns time offset since channel start in seconds.
    npy_intp n, nFills, i, len;
    Channel_Fill *fills;
    double result;
    
    len = PyArray_DIM(self->data, 0);
    index = (index < 0) ? len + index : index;
    if (index >= 0 && index < len) {
        n = index;
        if (self->fills) {
            nFills = PyArray_DIM(self->fills, 0);
            fills = PyArray_DATA(self->fills);
            for (i = 0; i < nFills; i++) {
                if (fills[i].pos < index) {
                    n += fills[i].len;
                } else {
                    break;
                }
            }
        }
        result = n / self->samplerate;  // new
    } else {
        PyErr_SetString(PyExc_IndexError, "Sample index out of range");
        result = NAN;
    }
    
    return result;
}

static PyObject *Channel_sampleOffset(Channel *self, PyObject *args) {
    // Returns time offset since channel start in seconds.
    npy_intp index;
    double offset;
    PyObject *result = NULL;
    
    if (PyArg_ParseTuple(args, "n:sampleOffset", &index)) {
        offset = Channel_sampleOffsetAsDouble(self, index);
        result = PyFloat_FromDouble(offset);  // new
    }
    
    return result;
}

static PyObject *Channel_sampleTime(Channel *self, PyObject *args) {
    PyObject *offset, *argTuple, *delta = NULL, *start, *result = NULL;
    
    offset = Channel_sampleOffset(self, args);
    if (offset) {
        argTuple = PyTuple_Pack(1, offset);  // new
        if (argTuple) {
            delta = timedeltaFromFloat(NULL, argTuple);  // new
            Py_DECREF(argTuple);
        }
        start = Channel_start(self);  // new
        if (start && delta) {
            result = PyNumber_Add(start, delta);  // new
        } else {
            PyErr_SetString(PyExc_RuntimeError, "Failed to instantiate a datetime object");
        }
        Py_XDECREF(delta);
        Py_XDECREF(start);
    }
    
    return result;
}

static PyObject *Channel_getSample(Channel *self, PyObject *args, PyObject *kwds) {
    PyObject *index, *sample = NULL;
    npy_intp i;
    double value;
    
    index = Channel_sampleIndex(self, args, kwds);  // new
    if (index) {
        i = PyLong_AsSize_t(index);
        Py_DECREF(index);
        sample = PyArray_GETITEM(self->data, PyArray_GETPTR1(self->data, i));  // new(?)
        if (sample) {
            value = PyFloat_AsDouble(sample);
            Py_DECREF(sample);
            sample = NULL;
            value *= self->scale;
            value += self->offset;
            sample = PyObject_CallMethod(unitRegistry(), "Quantity", "ds", value, self->unit);  // new
            if (!sample) {
                PyErr_SetString(PyExc_RuntimeError, "Failed to get sample quantity");
            }
        } else {
            PyErr_SetString(PyExc_RuntimeError, "Failed to get sample value");
        }
    } else {
        PyErr_SetString(PyExc_RuntimeError, "Failed to get sample index");
    }
    
    return sample;
}

static PyObject *Channel_valuesFromIndex(Channel *self,
                                         Py_ssize_t start, Py_ssize_t stop,
                                         int include_fills, Py_ssize_t *fill_pos) {
    // Returns the sample values between start and stop, optionally including
    // fills. If fills are included, fill_pos (array of size 2) can be used for
    // specifying the positions where to start the beginning fill and end the
    // end fill.
    PyObject *data, *castData = NULL, *result = NULL;
    PyArrayObject *fillCopy = NULL;
    npy_intp size, fill, nFills, extra;
    Channel_Fill *fills = NULL;
    double *in = NULL, *out = NULL;
    int error = 0;
    
    nFills = 0;
    data = PySequence_GetSlice((PyObject *)self->data, start, stop);  // new
    if (data) {
        size = PyArray_DIM((PyArrayObject *)data, 0);
        if (include_fills) {
            // Get pointer to fills that are inside the slice.
            fillCopy = (PyArrayObject *)PyArray_Copy(self->fills);  // new
            if (fillCopy) {
                fills = PyArray_DATA(fillCopy);
            }
            if (fills) {
                nFills = PyArray_DIM(self->fills, 0);
                fill = 0;
                while (fill < nFills) {
                    if (fills[fill].pos < start) {
                        // Discard fills before starting position.
                        fills++;
                        nFills--;
                    } else if (fills[fill].pos < stop) {
                        if (fills[fill].pos == stop - 1) {
                            // Ends in a fill. Set to specified length.
                            fills[fill].len = fill_pos[1];
                        }
                        if (fills[fill].pos == start) {
                            // Begins with a fill. Set to specified length.
                            fills[fill].len -= fill_pos[0];
                        }
                        size += fills[fill].len;
                        fill++;
                    } else {
                        // The fill is after the end position. Discard and stop here.
                        nFills = fill;
                    }
                }
            } else {
                PyErr_SetString(PyExc_RuntimeError, "Failed to get pointers to fill data");
                error = -1;
            }
        }
    } else {
        PyErr_SetString(PyExc_RuntimeError, "Failed to get a view to the data slice");
        error = -1;
    }
    if (!error) {
        // Cast the view to data array to double.
        castData = PyArray_Cast((PyArrayObject *)data, NPY_DOUBLE);  // new
        Py_DECREF(data);
        if (castData) {
            in = PyArray_DATA((PyArrayObject *)castData);
        } else {
            PyErr_SetString(PyExc_RuntimeError, "Failed to cast data array to double");
            error = -1;
        }
    }
    if (!error) {
        // Create the result array.
        result = PyArray_EMPTY(1, &size, NPY_DOUBLE, 0);  // new
        if (result) {
            out = PyArray_DATA((PyArrayObject *)result);
        } else {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create result array");
            error = -1;
        }
    }
    if (!error && in && out) {
        fill = 0;
        extra = 0;
        for (npy_intp i = 0; i < stop - start; i++) {
            out[i + extra] = self->scale * in[i] + self->offset;
            if (fill < nFills && i == fills[fill].pos - start) {
                for (npy_intp j = 0; j < fills[fill].len; j++) {
                    out[i + extra + j + 1] = out[i + extra];
                }
                extra += fills[fill].len;
                fill++;
            }
        }
    }
    Py_XDECREF(castData);
    Py_XDECREF(fillCopy);
    if (result && !out) {
        Py_DECREF(result);
        result = NULL;
        PyErr_SetString(PyExc_RuntimeError, "Failed to get a pointer to result array");
    }
    
    return result;
}

static Py_ssize_t convertToIndex(Channel *self, PyObject *o, Py_ssize_t *fill_pos) {
    // Checks that object o is an integer or converts it to integer, if it is
    // a datetime or timedelta. If the offset specified by o is inside a fill,
    // fill_pos is set to the number samples between the fill position and the
    // specified offset. Returns -1 if an error occurs.
    npy_intp len, fill, nFills;
    Py_ssize_t index;
    Channel_Fill *fills;
    double tGiven, tFill;
    
    index = -1;
    len = PyArray_DIM(self->data, 0);
    if (PyLong_Check(o)) {
        // Argument o is already an index.
        Py_INCREF(o);
        nFills = 0;
    } else {
        // Convert float, timedelta, or datetime to an index.
        o = PyObject_CallMethod((PyObject *)self, "sampleIndex", "O", o);  // new
        nFills = PyArray_DIM(self->fills, 0);
        fills = PyArray_DATA(self->fills);
    }
    if (o) {
        index = PyLong_AsSsize_t(o);
        Py_DECREF(o);
        index = (index < 0) ? len + index : index;
        // Calculate fill length.
        *fill_pos = 0;
        for (fill = 0; fill < nFills; fill++) {
            if (fills[fill].pos == index) {
                tGiven = timeAsOffset(self, o);
                tFill = Channel_sampleOffsetAsDouble(self, index);
                *fill_pos = round((tGiven - tFill) * self->samplerate);
                break;
            }
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument o in convertToIndex() must be integer, float, timedelta, or datetime");
    }

    return index;
}

static PyObject *Channel_values(Channel *self, PyObject *args, PyObject *kwds) {
    // Returns the sample values between start and stop. Start and stop can be
    // array indices, datetimes, or timedeltas or floats that specify the offset
    // from the beginning of the channel.
    PyObject *result = NULL;
    PyObject *startObj = NULL, *stopObj = NULL;
    Py_ssize_t start, stop, fillPos[2] = {0, 0};
    int include_fills;
    static char *kwlist[] = {"start", "stop", "include_fills", NULL};
    int error = 0;
    
    start = 0;
    stop = PyArray_SIZE(self->data);
    include_fills = 0;
    if (PyArg_ParseTupleAndKeywords(args, kwds, "|OOp:values", kwlist, &startObj, &stopObj, &include_fills)) {
        if (startObj) {
            start = convertToIndex(self, startObj, &fillPos[0]);
            if (start < 0) {
                error = -1;
            }
        }
        if (!error && stopObj) {
            stop = convertToIndex(self, stopObj, &fillPos[1]);
            if (stop < 0) {
                error = -1;
            } else {
                stop++;
            }
        }
        if (!error) {
            result = Channel_valuesFromIndex(self, start, stop, include_fills, fillPos);  // new
        }
    }
    
    return result;
}

static PyObject *Channel_quantities(Channel *self, PyObject *args, PyObject *kwds) {
    PyObject *values, *result = NULL;
    
    values = Channel_values(self, args, kwds);
    if (values) {
        result = PyObject_CallMethod(unitRegistry(), "Quantity", "Os", values, self->unit);  // new
        if (!result) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to get channel quantities");
        }
    } else {
        PyErr_SetString(PyExc_RuntimeError, "Failed to get sample values");
    }

    return result;
}

static PyObject *Channel_filteredLength(Channel *self, PyObject *args) {
    // Returns the length and fills of a filtered channel, given fftSize,
    // windowSize, and stepSize.
    npy_intp length, nFills, fill, extra, rem, start, margin;
    npy_intp fftSize, windowSize, stepSize;
    PyArrayObject *fillArray;
    PyObject *valid, *fillObj = NULL;
    Channel_Fill *fills;
    
    if (!PyArg_ParseTuple(args, "nnn:filteredLength", &fftSize, &windowSize, &stepSize)) {
        return NULL;
    }
    if (fftSize < 1 || windowSize < 1 || stepSize < 1) {
        PyErr_SetString(PyExc_ValueError, "Size arguments must be positive integers");
        return NULL;
    }
    if (fftSize < windowSize) {
        PyErr_SetString(PyExc_ValueError, "Window size may not exceed FFT size");
        return NULL;
    }
    if (stepSize > windowSize) {
        PyErr_SetString(PyExc_ValueError, "Step size may not exceed window size");
        return NULL;
    }
    fillArray = (PyArrayObject *)PyArray_Copy(self->fills);  // new
    if (fillArray) {
        fills = PyArray_DATA(fillArray);
        nFills = PyArray_DIM(fillArray, 0);
        length = PyArray_DIM(self->data, 0);
    } else {
        PyErr_SetString(PyExc_RuntimeError, "Error copying fills in filteredLength()");
        return NULL;
    }
    
    // Calculate new fill lengths.
    margin = roundToNext(fftSize - windowSize, stepSize);
    extra = 0;
    rem = 0;
    start = 0;
    for (fill = 0; fill < nFills; fill++) {
        rem = (rem + fills[fill].pos + 1 - start) % stepSize;
        rem = rem ? stepSize - rem : 0;
        start = fills[fill].pos;
        if (fills[fill].len > rem + 2 * margin + windowSize) {
            // Fill is sufficiently long. Adjust length.
            fills[fill].len -= rem + 2 * margin;
            extra += rem + margin;
            fills[fill].pos += extra;
            rem = margin;
            extra += margin;
        } else {
            // Fill is too short. Replace with individual samples.
            extra += fills[fill].len;
            rem += fills[fill].len;
            fills[fill].pos = 0;
            fills[fill].len = 0;
        }
    }
    // Add padding to the end.
    rem = (rem + length - start) % stepSize;
    extra += rem ? fftSize - rem : fftSize - stepSize;
    
    // Remove zero-length fills.
    valid = PyArray_Nonzero(fillArray);
    if (valid) {
        fillObj = PyObject_GetItem((PyObject *)fillArray, valid);  // new
        Py_DECREF(valid);
    }
    Py_DECREF(fillArray);
    if (!fillObj) {
        PyErr_SetString(PyExc_RuntimeError, "Error removing zero-length fills in filteredLength()");
    }
    
    return Py_BuildValue("nO", length + extra, fillObj);  // new
}

struct stft {
    size_t n;         // current position
    size_t wlen;      // window length
    size_t size;      // FFT size
    size_t step;      // step size
    size_t op;        // overlap-add position
    double *t_in;     // time domain
    double *t_out;    // result of inverse FFT
    double *w;        // window
    double *ola;      // overlap-add buffer
    fftw_complex *F;  // frequency domain
    fftw_complex *H;  // transfer function
    fftw_plan forward;
    fftw_plan inverse;
    void (*callback_post)(struct stft *);
};

static void stft_destroy(struct stft *stft) {
    fftw_destroy_plan(stft->forward);
    fftw_destroy_plan(stft->inverse);
    fftw_free(stft->t_in);
    if (stft->t_out != stft->t_in) {
        fftw_free(stft->t_out);
    }
    fftw_free(stft->F);
    fftw_free(stft->H);
    fftw_free(stft->w);
    stft->forward = NULL;
    stft->inverse = NULL;
    stft->t_in = NULL;
    stft->t_out = NULL;
    stft->F = NULL;
    stft->H = NULL;
    stft->w = NULL;
    stft->ola = NULL;
    stft->callback_post = NULL;
    stft->size = 0;
    stft->n = 0;
    stft->op = 0;
}

static int stft_init(struct stft *stft, size_t size, double *kernel, size_t klen,
                     double *w, size_t wlen, size_t step) {
    int success = 1;
    
    if (size < 1 || klen < 1 || wlen < 1 || step < 1 || step > wlen || wlen >= size) {
        return 0;
    }
    stft->n = 0;
    stft->wlen = wlen;
    stft->size = size;
    stft->step = step;
    stft->op = 0;
    stft->ola = NULL;
    stft->callback_post = NULL;
    stft->t_in = (double *)fftw_alloc_real(size);
    stft->F = (fftw_complex *)fftw_alloc_complex(size / 2 + 1);
    if (step < wlen) {
        stft->t_out = (double *)fftw_alloc_real(size);
    } else {
        stft->t_out = stft->t_in;
    }
    if (stft->t_in && stft->t_out && stft->F) {
        stft->forward = fftw_plan_dft_r2c_1d((int)size, stft->t_in, stft->F, FFTW_MEASURE);
        stft->inverse = fftw_plan_dft_c2r_1d((int)size, stft->F, stft->t_out, FFTW_MEASURE);
        if (!stft->forward || !stft->inverse) {
            success = 0;
        }
    } else {
        success = 0;
    }
    
    if (success && kernel) {
        stft->H = (fftw_complex *)fftw_alloc_complex(size / 2 + 1);
        if (stft->H) {
            memcpy(stft->t_in, kernel, klen * sizeof(double));
            memset(stft->t_in + klen, 0, (stft->size - klen) * sizeof(double));
            fftw_execute(stft->forward);
            for (size_t i = 0; i < size / 2 + 1; i++) {
                stft->H[i] = stft->F[i] / size;
            }
        } else {
            success = 0;
        }
    } else {
        stft->w = NULL;
    }
    if (success && w) {
        stft->w = (double *)fftw_alloc_real(wlen);
        if (stft->w) {
            memcpy(stft->w, w, wlen * sizeof(double));
        } else {
            success = 0;
        }
    } else {
        stft->w = NULL;
    }
    
    if (!success) {
        stft_destroy(stft);
    }
    
    return success;
}

static void stft_execute(struct stft *stft) {
    npy_intp i;
    
    if (stft->w) {
        // Apply the window.
        for (i = 0; i < stft->wlen; i++) {
            stft->t_in[i] *= stft->w[i];
        }
    }
    fftw_execute(stft->forward);
    if (stft->callback_post) {
        stft->callback_post(stft);
    }
    if (stft->H) {
        // Compute the circular convolution.
        for (i = 0; i < stft->size / 2 + 1; i++) {
            stft->F[i] = stft->F[i] * stft->H[i];
        }
        fftw_execute(stft->inverse);
        
        // Overlap-add.
        if (stft->ola) {
            for (i = 0; i < stft->size; i++) {
                stft->ola[stft->op + i] += stft->t_out[i];
            }
        }
    }
    // Step forward reusing samples from input buffer, if possible.
    stft->op += stft->step;
    stft->n = stft->wlen - stft->step;
    if (stft->n) {
        memmove(stft->t_in, stft->t_in + stft->step, stft->n * sizeof(double));
    }
}

static void stft_zeropad(struct stft *stft) {
    memset(stft->t_in + stft->n, 0, (stft->size - stft->n) * sizeof(double));
    stft_execute(stft);
}

static size_t stft_copy(struct stft *stft, double *src, size_t len) {
    if (stft->n + len > stft->wlen) {
        len = stft->wlen - stft->n;
    }
    memcpy(stft->t_in + stft->n, src, len * sizeof(double));
    stft->n += len;
    if (stft->n == stft->wlen) {
        stft_zeropad(stft);
    }
    return len;
}

static size_t stft_add(struct stft *stft, double value, size_t n) {
    if (stft->n + n > stft->wlen) {
        n = stft->wlen - stft->n;
    }
    for (size_t i = 0; i < n; i++) {
        stft->t_in[stft->n++] = value;
    }
    if (stft->n == stft->wlen) {
        stft_zeropad(stft);
    }
    return n;
}

static PyObject *stft_fun = NULL;

static void stft_apply(struct stft *stft) {
    npy_intp dims[1];
    PyObject *F;
    
    if (stft_fun) {
        // Apply the optional STFT function.
        dims[0] = stft->size;
        F = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, stft->F);  // new
        if (F) {
            // TODO: Save the result.
            PyObject_CallFunctionObjArgs(stft_fun, F, NULL);  // new
        } else {
            PyErr_SetString(PyExc_RuntimeError, "Wrapping STFT result as PyArray_Object failed");
        }
    }
}

static PyObject *Channel_filter(Channel *self, PyObject *args, PyObject *kwds) {
    // Fast convolution using overlap-add (OLA) algorithm.
    // The optional argument stft_function can be used for frequency-domain
    // computations. The stft result is passed to the stft_function as
    // PyArray_Object. The call to stft_function must not alter the reference
    // count of this array object, since the memory it points to will be freed
    // upon exiting Channel_filter regardless of the reference count.
    Channel *newChannel = NULL;
    Channel_Fill *fills;
    PyObject *kernel, *obj, *constant, *values, *newArray, *newFills;
    PyObject *window = NULL, *fun = NULL;
    npy_intp chLen, fill;
    npy_intp margin, left;
    npy_intp  nFills, pos;
    double *data, *kernelData, *newData, *windowData, fillValue;
    int b, error = 0;
    static char *kwlist[] = {"kernel", "window", "overlap", "stft_function", "ensure_cola", NULL};
    npy_intp kernelSize = 0, windowSize, stepSize, overlap = 0;
    int fftSize, isCola = 0, ensureCola = 1;
    Py_ssize_t dims[1];
    struct stft stft;
    
    if (PyArg_ParseTupleAndKeywords(args, kwds, "O|OiOp:filter", kwlist,
                                    &kernel, &window, &overlap, &fun, &ensureCola)) {
        if (fun && !PyCallable_Check(fun)) {
            PyErr_SetString(PyExc_ValueError, "Argument stft_function must be callable");
            return NULL;
        }
        kernel = PyArray_FROMANY(kernel, NPY_DOUBLE, 1, 1, NPY_ARRAY_CARRAY_RO);  // new
        if (!kernel) {
            PyErr_SetString(PyExc_ValueError, "Kernel in filter() must be array-like");
            return NULL;
        }
        kernelSize = PyArray_SIZE((PyArrayObject *)kernel);
        if (overlap > kernelSize - 1) {
            PyErr_SetString(PyExc_ValueError, "Overlap in filter() must be less than kernel size");
            return NULL;
        }
    } else {
        return NULL;
    }
    
    // Make fftSize a power of two 4 - 8 times kernelSize.
    // TODO: Find optimal fftSize.
    b = 1;
    while (b < 4 * kernelSize) {
        b <<= 1;
    }
    fftSize = b;
    // Set the maximum windowSize.
    windowSize = fftSize - kernelSize + 1;
    stepSize = windowSize - overlap;
    
    if (window && window != Py_None) {
        window = PyArray_FROMANY(window, NPY_DOUBLE, 1, 1, NPY_ARRAY_CARRAY_RO);  // new
        if (window) {
            windowData = PyArray_DATA((PyArrayObject *)window);
            windowSize = PyArray_DIM((PyArrayObject *)window, 0);
            stepSize = windowSize - overlap;
            while (windowSize > fftSize - kernelSize + 1) {
                // The window is too long. Double fftSize until the convolution fits.
                // TODO: Warn about unoptimal fftSize.
                fftSize <<= 1;
            }
            // Check if the window is COLA.
            obj = Py_BuildValue("OO", window, stepSize);  //new
            if (obj) {
                constant = cola(NULL, obj);  // new
                Py_DECREF(obj);
                if (constant) {
                    if (PyObject_IsTrue(constant)) {
                        isCola = 1;
                    } else if (ensureCola) {
                        PyErr_SetString(PyExc_ValueError, "Window does not satisfy constant-overlap-add constraint.");
                        error = -1;
                    }
                    Py_DECREF(constant);
                }
            } else {
                PyErr_SetString(PyExc_RuntimeError, "Failed to build argument tuple in filter()");
                error = -1;
            }
        } else {
            PyErr_SetString(PyExc_RuntimeError, "Window conversion in filter() failed");
            error = -1;
        }
    } else {
        // Use rectangular window.
        windowData = NULL;
    }
    
    if (!error) {
        // Allocate space for the result.
        obj = PyObject_CallMethod((PyObject *)self, "filteredLength", "nnn",
                                  fftSize, windowSize, stepSize);  // new
        if (obj) {
            if (PyArg_ParseTuple(obj, "nO", dims, &newFills)) {
                newArray = PyArray_ZEROS(1, dims, NPY_DOUBLE, 0);
                newData = PyArray_DATA((PyArrayObject *)newArray);
            } else {
                error = -1;
            }
            Py_DECREF(obj);
        } else {
            error = -1;
        }
    }
    
    if (!error) {
        // Extract values.
        // TODO: Optimize memory usage.
        values = PyObject_CallMethod((PyObject *)self, "values", "", NULL);  // new
        if (values) {
            data = PyArray_DATA((PyArrayObject *)values);
        } else {
            PyErr_SetString(PyExc_RuntimeError, "Could not extract data in filter()");
            error = -1;
        }
    }
    
    if (!error) {
        // Initialize FFTW.
        kernelData = PyArray_DATA((PyArrayObject *)kernel);
        error = !stft_init(&stft, fftSize, kernelData, kernelSize,
                           windowData, windowSize, stepSize);
    }
    Py_DECREF(kernel);
    
    if (!error) {
        // Compute STFT.
        stft.ola = newData;
        chLen = PyArray_DIM(self->data, 0);
        fills = PyArray_DATA(self->fills);
        nFills = PyArray_DIM(self->fills, 0);
        margin = roundToNext(fftSize - windowSize, stepSize);
        pos = 0;
        for (fill = 0; fill < nFills; fill++) {
            while (pos <= fills[fill].pos) {
                left = fills[fill].pos - pos + 1;
                pos += stft_copy(&stft, data + pos, left);
            }
            fillValue = data[fills[fill].pos];
            left = (windowSize - stft.n) % windowSize;
            if (fills[fill].len > left + 2 * margin + windowSize) {
                // Fill is sufficiently long to keep after adjusting length.
                left += margin;
                while (left > 0) {
                    left -= stft_add(&stft, fillValue, left);
                }
                stft.n = 0;
                left = margin;
                while (left > 0) {
                    left -= stft_add(&stft, fillValue, left);
                }
            } else {
                // Fill is too short. Replace with individual samples.
                left = fills[fill].len;
                while (left > 0) {
                    left -= stft_add(&stft, fillValue, left);
                }
            }
        }
        while (pos < chLen) {
            left = chLen - pos;
            pos += stft_copy(&stft, data + pos, left);
        }
        left = stft.n / stepSize;
        do {
            stft_zeropad(&stft);
            left--;
        } while (left > 0);

        // Clean up after FFTW.
        stft_destroy(&stft);
    }
    if (!error) {
        // TODO: move start time by overlap, which is (fftSize - windowSize) samples
        // (NB causal vs zero-phase window)
        newChannel = (Channel *)PyObject_CallFunction((PyObject *)&ChannelType,
                                                      "OdOddssLlssisO",
                                                      newArray, self->samplerate,
                                                      newFills, 1.0, 0.0,
                                                      self->unit, self->type,
                                                      self->start_sec, self->start_nsec,
                                                      self->device, self->serial_no,
                                                      self->resolution,
                                                      self->json, self->events);  // new
    }
    Py_XDECREF(window);
    
    return (PyObject *)newChannel;
}

static PyObject *Channel_collate(Channel *self, PyObject *args, PyObject *kwds) {
    PyObject *chList, *item, *obj, *events, *argTuple;
    PyObject *unit = NULL, *dimensionality = NULL, *conversionFactorList =  NULL;
    PyObject *localUnit, *localDimensionality, *conversionFactor, *fillValues;
    PyArray_Dims dims;
    PyArrayObject *data = NULL, *values = NULL;
    PyArrayObject *fills = NULL;
    Channel *ch, *result = NULL;
    Py_ssize_t length, i, nParts, fillLen;
    int thisType, otherType, error = 0;
    npy_intp sample, j, fill, nFillSamples;
    npy_intp size, nFills = 0, nLocalFills, nNonlocalFills;
    Channel_Fill *fillSrc, *fillDst;
    double samplerate, t0, t, scale, offset, oldScale, oldOffset;
    struct timespec start;
    char *typestr, *unitstr;
    
    // Check keyword arguments.
    if (!kwds) {
        fillValues = PyLong_FromLong(0);  // new
    } else if (PyArg_ValidateKeywordArguments(kwds)) {
        fillValues = PyDict_GetItemString(kwds, "fill_values");  // borrowed
        if (fillValues) {
            Py_INCREF(fillValues);
        } else {
            fillValues = PyLong_FromLong(0);  // new
        }
    } else {
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_TypeError, "Invalid named argument in collate()");
        }
        error = -1;
    }
    // Interpret the other arguments as a list of channels.
    chList = PySequence_List(args);  // new
    if (!chList) {
        PyErr_SetString(PyExc_TypeError, "collate() takes Channel objects as arguments");
        error = -1;
    }
    if (!error) {
        nParts = PyList_Size(chList);
        PyList_Sort(chList);
        item = PyList_GET_ITEM(chList, 0);  // borrowed
        if (item && PyObject_TypeCheck(item, &ChannelType)) {
            // Check the first channel.
            ch = (Channel *)item;
            thisType = PyArray_TYPE(ch->data);
            length = PyArray_DIM(ch->data, 0);
            if (ch->fills) {
                nFills += PyArray_DIM(ch->fills, 0);
            }
            typestr = ch->type;
            samplerate = ch->samplerate;
            unitstr = ch->unit;
            start.tv_sec = ch->start_sec;
            start.tv_nsec = ch->start_nsec;
            t0 = start.tv_sec + start.tv_nsec / 1e9;
            scale = ch->scale;
            offset = ch->offset;
        } else {
            PyErr_SetString(PyExc_TypeError, "collate() takes a Channel objects as arguments");
            error = -1;
        }
    }
    if (!error) {
        unit = PyObject_CallMethod(unitRegistry(), "parse_expression", "s", unitstr);  // new
        dimensionality = unit ? PyObject_GetAttrString(unit, "dimensionality") : NULL;  // new
        conversionFactorList = PyList_New(nParts);  // new
        if (unit && dimensionality && conversionFactorList) {
            obj = PyFloat_FromDouble(1.0);  // new
            PyList_SET_ITEM(conversionFactorList, 0, obj);  // stolen
        } else {
            PyErr_SetString(PyExc_RuntimeError, "Unit check failed in collate()");
            error = -1;
        }
    }
    if (!error) {
        for (i = 1; i < nParts; i++) {
            // Check that the rest of the channels are
            // compatible with the first one.
            item = PyList_GET_ITEM(chList, i);  // borrowed
            if (PyObject_TypeCheck(item, &ChannelType)) {
                ch = (Channel *)item;
                otherType = PyArray_TYPE(ch->data);
                if (strcmp(typestr, ch->type) == 0) {
                    oldScale = ch->scale;
                    oldOffset = ch->offset;
                    if (ch->samplerate != samplerate) {
                        /*
                         // Resample to match the first channel.
                         // TODO: Allow setting samplerate.
                         argTuple = Py_BuildValue("(d)", samplerate);  // new
                         ch = (Channel *)Channel_resample((Channel *)item, argTuple, NULL);
                         Py_DECREF(argTuple);
                         PyList_SetItem(chList, i, (PyObject *)ch);
                         */
                        error = -1;
                        PyErr_SetString(PyExc_ValueError, "Samplerates do not match");
                        break;
                    }
                    localUnit = PyObject_CallMethod(unitRegistry(), "parse_expression", "s", ch->unit);  // new
                    conversionFactor = NULL;
                    obj = localUnit ? PyObject_CallMethod(localUnit, "to", "O", unit) : NULL;  // new
                    if (obj) {
                        conversionFactor = PyObject_GetAttrString(obj, "magnitude");  // new
                        if (conversionFactor) {
                            PyList_SET_ITEM(conversionFactorList, i, conversionFactor);  // stolen
                        }
                        Py_DECREF(obj);
                    }
                    if (!conversionFactor) {
                        PyErr_SetString(PyExc_RuntimeError, "Storing unit conversion factor in collate() failed");
                        error = -1;
                        Py_XDECREF(localUnit);
                        break;
                    }
                    localDimensionality = localUnit ? PyObject_GetAttrString(localUnit, "dimensionality") : NULL;  // new
                    if (localDimensionality) {
                        if (PyObject_RichCompareBool(dimensionality, localDimensionality, Py_EQ)) {
                            if (otherType != thisType ||
                                ch->scale != scale || ch->offset != offset ||
                                PyObject_RichCompareBool(unit, localUnit, Py_NE)) {
                                // Convert to double.
                                thisType = NPY_DOUBLE;
                                scale = 1.0;
                                offset = 0.0;
                            }
                            length += PyArray_DIM(ch->data, 0);
                            if (ch->fills) {
                                nFills += PyArray_DIM(ch->fills, 0);
                            }
                        } else {
                            PyErr_Format(PyExc_TypeError, "Channel objects with incompatible units %s and %s can not be collated", unitstr, ch->unit);
                            error = -1;
                            Py_DECREF(localDimensionality);
                            Py_DECREF(localUnit);
                            break;
                        }
                        Py_DECREF(localDimensionality);
                        Py_DECREF(localUnit);
                    } else {
                        PyErr_SetString(PyExc_RuntimeError, "Checking dimensionality in collate() failed");
                        error = -1;
                        Py_XDECREF(localUnit);
                        break;
                    }
                } else {
                    PyErr_SetString(PyExc_TypeError, "Channel objects in collate() need to be of same type");
                    error = -1;
                    break;
                }
            } else {
                PyErr_SetString(PyExc_TypeError, "collate() takes a list of Channel objects as an argument");
                error = -1;
                break;
            }
        }
    }
    if (!error) {
        // Reserve space for the fills between the collated channels.
        nFills += nParts - 1;
        length += nParts - 1;
        // Check that fillValues has the right shape.
        obj = PyArray_FROM_OTF(fillValues, thisType, NPY_ARRAY_CARRAY_RO);  // new
        Py_DECREF(fillValues);
        if (!obj) {
            PyErr_SetString(PyExc_TypeError, "fill_values must be scalar or array-like");
            error = -1;
        }
        if (PyArray_NDIM((PyArrayObject *)obj) == 0) {
            nNonlocalFills = nParts - 1;
            dims.ptr = &nNonlocalFills;
            dims.len = 1;
            fillValues = obj;
            obj = PyArray_Resize((PyArrayObject *)obj, &dims, 0, NPY_CORDER);  // new
            if (!obj) {
                error = -1;
            }
        } else if (PyArray_NDIM((PyArrayObject *)obj) > 1 ||
                   PyArray_DIM((PyArrayObject *)obj, 0) != nParts - 1) {
            Py_DECREF(obj);
            PyErr_SetString(PyExc_TypeError, "fill_values must be scalar or have a length one less than the number of channels");
            error = -1;
        }
        obj = NULL;
    }
    if (!error) {
        data = (PyArrayObject *)PyArray_EMPTY(1, &length, thisType, 0);  // new
        fills = (PyArrayObject *)newFillArray(NULL, nFills);
        if (!fills) {
            PyErr_SetString(PyExc_RuntimeError, "Error creating fill array in collate()");
            error = -1;
        }
    }
    if (!error) {
        fillDst = PyArray_DATA(fills);
        fill = 0;
        nFillSamples = 0;
        sample = 0;
        events = PySet_New(NULL);  // new
        argTuple = PyTuple_New(0);  // new
        for (i = 0; i < nParts; i++) {
            ch = (Channel *)PyList_GET_ITEM(chList, i);  // borrowed
            
            if (i > 0) {
                // Add a fill between files: calculate fill length.
                t = ch->start_sec + ch->start_nsec / 1e9;
                fillLen = round((t - t0) * samplerate - (sample + nFillSamples));
                // TODO: Specify minimum fill lengths.
                // (If the fill is shorter than minimum, store it as
                // ordinary samples instead of a fill.)
                if (fillLen > 0) {
                    fillDst[fill].pos = sample;
                    fillDst[fill].len = fillLen;
                    obj = PyArray_GETITEM((PyArrayObject *)fillValues,
                                          PyArray_GETPTR1((PyArrayObject *)fillValues, i - 1));
                    if (obj) {
                        error = PyArray_SETITEM(data, PyArray_GETPTR1(data, sample++), obj);
                        Py_DECREF(obj);
                        if (error) {
                            break;
                        }
                    } else {
                        error = -1;
                        break;
                    }
                } else if (fillLen == 0) {
                    // TODO: Allow some overlap to account for rounding errors.
                    // TODO: Remove the fill completely.
                    fillDst[fill].pos = sample;
                    fillDst[fill].len = 0;
                } else {
                    fprintf(stderr, "Channels overlap by %zd samples\n", -fillLen);
                    PyErr_SetString(PyExc_ValueError, "Can not collate() Channel objects that overlap");
                    error = -1;
                    break;
                }
                fill++;
                nFillSamples += fillLen;
            }
            
            if (thisType == NPY_DOUBLE) {
                // Get scaled values, and convert units.
                obj = Channel_values(ch, argTuple, NULL);  // new
                if (!obj) {
                    error = -1;
                    break;
                }
                conversionFactor = PyList_GET_ITEM(conversionFactorList, i);  // borrowed
                if (conversionFactor) {
                    values = (PyArrayObject *)PyNumber_Multiply(obj, conversionFactor);  // new
                }
                Py_DECREF(obj);
            } else {
                values = ch->data;
                Py_INCREF(values);
            }
            if (values) {
                size = PyArray_NBYTES(values);
                memcpy(PyArray_GETPTR1(data, sample), PyArray_DATA(values), size);
                Py_DECREF(values);
            } else {
                PyErr_SetString(PyExc_RuntimeError, "Failed to get channel data values");
                error = -1;
                break;
            }
            if (ch->fills) {
                // Copy the fills from all sparse channels.
                nLocalFills = PyArray_DIM(ch->fills, 0);
                fillSrc = PyArray_DATA(ch->fills);
                for (j = 0; j < nLocalFills; j++) {
                    fillDst[fill].pos = sample + fillSrc[j].pos;
                    fillDst[fill++].len = fillSrc[j].len;
                    nFillSamples += fillSrc[j].len;
                }
            }
            sample += PyArray_DIM(ch->data, 0);
            obj = PyObject_CallMethod(events, "update", "O", ch->events);  // new
            if (obj) {
                Py_DECREF(obj);
                obj = NULL;
            } else {
                PyErr_SetString(PyExc_ValueError, "Updating events in collate() failed");
                error = -1;
                break;
            }
        }
        Py_DECREF(argTuple);
    }
    if (!error) {
        // Create a new channel.
        result = (Channel *)PyObject_CallFunction((PyObject *)&ChannelType, "OdOddssLl",
                                                  data, samplerate, fills,
                                                  scale, offset, unitstr, typestr,
                                                  start.tv_sec, start.tv_nsec);  // new
        if (result) {
            Py_DECREF(result->events);
            result->events = (PySetObject *)events;
        }
    }
    Py_XDECREF(data);
    Py_XDECREF(fills);
    Py_XDECREF(unit);
    Py_XDECREF(dimensionality);
    Py_XDECREF(conversionFactorList);
    Py_DECREF(chList);

    return (PyObject *)result;
}

static PyMethodDef Channel_methods[] = {
    {"copy", (PyCFunction)Channel_copy, METH_VARARGS | METH_KEYWORDS, "copy of a channel object"},
    {"__copy__", (PyCFunction)Channel_simpleCopy, METH_NOARGS, "copy of a channel object"},
    {"__deepcopy__", (PyCFunction)Channel_deepCopy, METH_VARARGS, "deep copy of a channel object"},
    {"start", (PyCFunction)Channel_start, METH_NOARGS, "channel start time as a datetime object"},
    {"duration", (PyCFunction)Channel_duration, METH_NOARGS, "channel duration in seconds"},
    {"end", (PyCFunction)Channel_end, METH_NOARGS, "channel end time as a datetime object"},
    {"timecodes", (PyCFunction)Channel_timecodes, METH_VARARGS | METH_KEYWORDS, "timecodes for the samples in range"},
    {"matches", (PyCFunction)Channel_matches, METH_VARARGS, "check whether channel type and time match those of another channel"},
    {"collate", (PyCFunction)Channel_collate, METH_VARARGS | METH_KEYWORDS | METH_STATIC, "combine channels to a sparse channel"},
    {"eventUnion", (PyCFunction)Channel_eventUnion, METH_VARARGS | METH_STATIC, "form a union of events on given channels"},
    {"eventIntersection", (PyCFunction)Channel_eventIntersection, METH_VARARGS | METH_STATIC, "form an intersection of events on given channels"},
    {"validateTimes", (PyCFunction)Channel_validateTimes, METH_VARARGS, "validate start and end times"},
    {"getEvents", (PyCFunction)Channel_getEvents, METH_VARARGS | METH_KEYWORDS, "get channel events"},
    {"sampleIndex", (PyCFunction)Channel_sampleIndex, METH_VARARGS | METH_KEYWORDS, "sample index"},
    {"sampleOffset", (PyCFunction)Channel_sampleOffset, METH_VARARGS, "sample time as offset from the channel start"},
    {"sampleTime", (PyCFunction)Channel_sampleTime, METH_VARARGS, "sample time as a datetime object"},
    {"getSample", (PyCFunction)Channel_getSample, METH_VARARGS | METH_KEYWORDS, "get the sample at specified time"},
    {"filter", (PyCFunction)Channel_filter, METH_VARARGS | METH_KEYWORDS, "convolution of the channel and a NumPy array"},
    {"filteredLength", (PyCFunction)Channel_filteredLength, METH_VARARGS, "length and fills of the channel when filtered"},
    {"values", (PyCFunction)Channel_values, METH_VARARGS | METH_KEYWORDS, "channel data values"},
    {"quantities", (PyCFunction)Channel_quantities, METH_VARARGS | METH_KEYWORDS, "channel data quantities"},
    {NULL}  // sentinel
};

static PyMemberDef Channel_members[] = {
    {"__dict__", T_OBJECT, offsetof(Channel, dict), READONLY, "dictionary for instance variables"},
    {"data", T_OBJECT_EX, offsetof(Channel, data), READONLY, "Channel data as NumPy array"},
    {"fills", T_OBJECT_EX, offsetof(Channel, fills), 0, "fill positions and lengths for sparse channels as a NumPy array"},
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
    {"json", T_STRING, offsetof(Channel, json), 0, "additional metadata in JSON format"},
    {"events", T_OBJECT_EX, offsetof(Channel, events), 0, "set of Event objects"},
    {NULL}  // sentinel
};

PyTypeObject ChannelType = {
    PyObject_HEAD_INIT(NULL)
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
    PyType_GenericNew            // tp_new
};