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
    
    data = PyTuple_GetItem(args, 0);  // borrowed
    if (data && PyArray_Check(data)) {
        self = (Channel *)type->tp_alloc(type, 0);
        if (self) {
            self->dict = NULL;
            self->events = NULL;
            self->fill_positions = NULL;
            self->fill_values = NULL;
            Py_INCREF(data);
            self->data = (PyArrayObject *)data;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Channel.new() takes a NumPy array as an argument");
        self = NULL;
    }

    return (PyObject *)self;
}

static int Channel_init(Channel *self, PyObject *args, PyObject *kwds) {
    PyObject *data = NULL, *events = NULL;
    int error;
    size_t size;
    npy_intp nFills;
    PyArray_Descr *fillDescr;
    char *tmp;
    static char *kwlist[] = {"data", "samplerate",
        "fill_positions", "fill_lengths", "fill_values",
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
    error = !PyArg_ParseTupleAndKeywords(args, kwds, "Od|O!O!O!ddssLlssisO", kwlist,
                                         &data,
                                         &(self->samplerate),
                                         &PyArray_Type, &(self->fill_positions),
                                         &PyArray_Type, &(self->fill_lengths),
                                         &PyArray_Type, &(self->fill_values),
                                         &(self->scale), &(self->offset),
                                         &(self->unit), &(self->type),
                                         &(self->start_sec), &(self->start_nsec),
                                         &(self->device), &(self->serial_no),
                                         &(self->resolution), &(self->json),
                                         &events);
    if (!error && (self->start_nsec < 0 || self->start_nsec > 999999999)) {
        error = -1;
        PyErr_SetString(PyExc_ValueError, "start_nsec is out of range");
    }
    if (!error) {
        if (self->fill_positions && self->fill_lengths && self->fill_values) {
            if (PyArray_DTYPE(self->fill_values) == PyArray_DTYPE(self->data)) {
                if (PyArray_NDIM(self->fill_positions) == 1 &&
                    PyArray_NDIM(self->fill_lengths) == 1 &&
                    PyArray_NDIM(self->fill_values) == 1) {
                    nFills = PyArray_DIM(self->fill_positions, 0);
                    if (PyArray_DIM(self->fill_lengths, 0) == nFills &&
                        PyArray_DIM(self->fill_values, 0) == nFills) {
                        fillDescr = PyArray_DescrFromType(NPY_INTP);
                        if (PyArray_CanCastArrayTo(self->fill_positions, fillDescr, NPY_SAFE_CASTING) &&
                            PyArray_CanCastArrayTo(self->fill_lengths, fillDescr, NPY_SAFE_CASTING)) {
                            self->fill_positions = (PyArrayObject *)PyArray_Cast(self->fill_positions, NPY_INTP);
                            Py_INCREF(self->fill_positions);
                            self->fill_lengths = (PyArrayObject *)PyArray_Cast(self->fill_lengths, NPY_INTP);
                            Py_INCREF(self->fill_lengths);
                            Py_INCREF(self->fill_values);
                        } else {
                            error = -1;
                            PyErr_SetString(PyExc_ValueError, "fill_positions and fill_lengths need to be able to hold array indices");
                        }
                    } else {
                        error = -1;
                        PyErr_SetString(PyExc_ValueError, "fill_positions, fill_lengths and fill_values must be of equal length");
                    }
                } else {
                    error = -1;
                    PyErr_SetString(PyExc_ValueError, "fill_positions, fill_lengths and fill_values must have exactly one dimension");
                }
            } else {
                error = -1;
                PyErr_SetString(PyExc_TypeError, "fill_values must be of same type as data");
            }
        } else if (self->fill_positions || self->fill_values) {
            error = -1;
            PyErr_SetString(PyExc_ValueError, "fill_positions, fill_lengths and fill_values must be either all specified or all null");
        }
    }
    if (!error) {
        self->events = (PySetObject *)PySet_New(events);  // new
        if (!self->events) {
            Py_DECREF(self->fill_positions);
            Py_DECREF(self->fill_lengths);
            Py_DECREF(self->fill_values);
            error = -1;
            PyErr_SetString(PyExc_RuntimeError, "creating events failed");
        }
    }
    if (!error) {
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

    return error;
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

static PyObject *Channel_matches(Channel *self, PyObject *args) {
    PyObject *result = Py_False;
    Channel *other;

    if (PyArg_ParseTuple(args, "O!", &ChannelType, &other)) {
        if (strcmp(self->type, other->type) == 0 &&
            self->samplerate == other->samplerate &&
            self->start_sec == other->start_sec &&
            self->start_nsec == other->start_nsec &&
            self->fill_positions && other->fill_positions &&
            PyObject_RichCompareBool((PyObject *)self->fill_positions, (PyObject *)other->fill_positions, Py_EQ) &&
            PyArray_DTYPE(self->data)->type_num == PyArray_DTYPE(other->data)->type_num &&
            PyArray_DIM(self->data, 0) == PyArray_DIM(other->data, 0))
        {
            result = Py_True;
        }
    } else {
        result = NULL;
        PyErr_SetString(PyExc_TypeError, "Channel.matches() takes a Channel argument");
    }
    Py_XINCREF(result);

    return result;
}

static PyObject *Channel_collate(Channel *self, PyObject *args) {
    PyObject *chList, *item, *dataList, *numObj, *result = NULL, *data = NULL;
    PyArrayObject *fill_positions = NULL, *fill_lengths = NULL, *fill_values = NULL;
    Channel *ch;
    Py_ssize_t length, i, nParts, fillLen;
    size_t size;
    int thisType, otherType, error = 0;
    npy_intp num, fill, pos, nFills = 0;
    double samplerate, t0, t;
    char *typestr, *unit;
    
    if (PyArg_ParseTuple(args, "O!", &PyList_Type, &chList)) {
        nParts = PyList_Size(chList);
        PyList_Sort(chList);
        item = PyList_GET_ITEM(chList, 0);  // borrowed
        if (item && PyObject_TypeCheck(item, &ChannelType)) {
            // Check the first channel.
            ch = (Channel *)item;
            thisType = PyArray_DTYPE(ch->data)->type_num;
            length = PyArray_DIM(ch->data, 0);
            if (ch->fill_values) {
                nFills += PyArray_DIM(ch->fill_values, 0);
            }
            size = PyArray_ITEMSIZE(ch->data);
            typestr = ch->type;
            samplerate = ch->samplerate;
            unit = ch->unit;
            t0 = ch->start_sec + ch->start_nsec / 1e9;
        } else {
            PyErr_SetString(PyExc_TypeError, "collate() takes a list of Channel objects as an argument");
            error = -1;
        }
        if (!error) {
            for (i = 1; i < nParts; i++) {
                // Check that the rest of the channels are
                // compatible with the first one.
                item = PyList_GET_ITEM(chList, i);  // borrowed
                if (PyObject_TypeCheck(item, &ChannelType)) {
                    ch = (Channel *)item;
                    otherType = PyArray_DTYPE(ch->data)->type_num;
                    if (otherType == thisType && strcmp(typestr, ch->type) == 0) {
                        if (samplerate == ch->samplerate) {
                            // Channels can be combined (if they don't overlap).
                            length += PyArray_DIM(ch->data, 0);
                            if (ch->fill_values) {
                                nFills += PyArray_DIM(ch->fill_values, 0);
                            }
                        } else {
                            PyErr_SetString(PyExc_ValueError, "Channel objects in collate() need to have the same samplerate");
                            error = -1;
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
            dataList = PyList_New(nParts);  // new
            nFills += nParts - 1;
            fill_positions = (PyArrayObject *)PyArray_EMPTY(1, &nFills, NPY_INTP, 0);  // new
            fill_lengths = (PyArrayObject *)PyArray_EMPTY(1, &nFills, NPY_INTP, 0);  // new
            fill_values = (PyArrayObject *)PyArray_EMPTY(1, &nFills, thisType, 0);  // new
            fill = 0;
            pos = 0;
            for (i = 0; i < nParts; i++) {
                ch = (Channel *)PyList_GET_ITEM(chList, i);  // borrowed
                Py_INCREF(ch->data);
                PyList_SET_ITEM(dataList, i, (PyObject *)ch->data);  // stolen
                if (i > 0) {
                    // Calculate fill length.
                    t = ch->start_sec + ch->start_nsec / 1e9;
                    fillLen = round((t - (t0 + pos / samplerate)) * samplerate);
                    // TODO: Specify minimum fill lengths.
                    // If the fill is shorter than minimum, store it as
                    // ordinary samples instead of a fill.
                    if (fillLen > 0) {
                        numObj = PyLong_FromSsize_t(pos);  // new
                        error = PyArray_SETITEM(fill_positions, PyArray_GETPTR1(fill_positions, fill), numObj);
                        Py_DECREF(numObj);
                        if (!error) {
                            numObj = PyLong_FromSsize_t(fillLen);  // new
                            error = PyArray_SETITEM(fill_lengths, PyArray_GETPTR1(fill_lengths, fill), numObj);
                            Py_DECREF(numObj);
                        } else {
                            break;
                        }
                        // TODO: Take fill values as argument.
                        if (!error) {
                            numObj = PyFloat_FromDouble(0.0);  // new
                            error = PyArray_SETITEM(fill_values, PyArray_GETPTR1(fill_values, fill), numObj);
                            Py_DECREF(numObj);
                        } else {
                            break;
                        }
                    } else if (fillLen == 0) {
                        // TODO: Remove fill.
                    } else {
                        PyErr_SetString(PyExc_TypeError, "Can not collate() Channel objects that overlap");
                        error = -1;
                        Py_DECREF(dataList);
                        Py_DECREF(fill_positions);
                        Py_DECREF(fill_lengths);
                        Py_DECREF(fill_values);
                        break;
                    }
                    fill++;
                    pos += fillLen;
                }
                if (ch->fill_values) {
                    memcpy(PyArray_GETPTR1(fill_values, fill), ch->fill_values, PyArray_NBYTES(ch->fill_values));
                    memcpy(PyArray_GETPTR1(fill_positions, fill), ch->fill_positions, PyArray_NBYTES(ch->fill_positions));
                    memcpy(PyArray_GETPTR1(fill_lengths, fill), ch->fill_lengths, PyArray_NBYTES(ch->fill_lengths));
                    fill += PyArray_DIM(ch->fill_values, 0);
                    numObj = PyArray_Sum(ch->fill_lengths, 0, NPY_NOTYPE, NULL);  // new
                    if (numObj) {
                        PyArray_ScalarAsCtype(numObj, &num);
                        pos += PyArray_DIM(ch->data, 0);
                        pos += num;
                        Py_DECREF(numObj);
                    } else {
                        error = -1;
                        break;
                    }
                }
            }
        }
        if (!error) {
            data = PyArray_Concatenate(dataList, 0);  // new
            Py_DECREF(dataList);
            // Create a new channel.
            result = PyObject_CallFunction((PyObject *)&ChannelType, "OdOOOddss",
                                                  data, samplerate,
                                                  fill_positions, fill_lengths,
                                                  fill_values, 1.0, 0.0, "", "");  // new
            // TODO: Handle units, scaling and offsets.
            Py_DECREF(data);
            Py_DECREF(fill_positions);
            Py_DECREF(fill_lengths);
            Py_DECREF(fill_values);
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "collate() takes a list of Channel objects as an argument");
        error = -1;
    }
    
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

static PyObject *Channel_resampledData(Channel *self, PyObject *args, PyObject *kwds) {
    PyArrayObject *newArray = NULL, *tmpArray = NULL;
    PyObject *numpy, *linspace, *slice, *start, *end, *step = NULL, *times;
    double samplerate, *sliceData;
    long long start_sec, end_sec, k;
    long start_nsec, end_nsec;
    int oldTypenum, newTypenum, err = 0;
    struct timespec end_t;
    Py_ssize_t start_idx, end_idx, i;
    npy_intp size, dims[1], *index = NULL;
    const char *method = "interpolate-decimate";
    static char *kwlist[] = {"samplerate", "start_sec", "start_nsec", "end_sec", "end_nsec", "typenum", "method", NULL};

    samplerate = self->samplerate;
    start_sec = self->start_sec;
    start_nsec = self->start_nsec;
    end_t = channelEndTime(self);
    end_sec = end_t.tv_sec;
    end_nsec = end_t.tv_nsec;
    oldTypenum = PyArray_DTYPE((PyArrayObject *)self->data)->type_num;
    newTypenum = oldTypenum;
    if (PyArg_ParseTupleAndKeywords(args, kwds, "d|LlLlis", kwlist, &samplerate,
                                    &start_sec, &start_nsec, &end_sec, &end_nsec, &newTypenum, &method)) {
        times = PyObject_CallMethod((PyObject *)self, "validateTimes", "LlLl",
                                    start_sec, start_nsec, end_sec, end_nsec);  // new
        if (times) {
            if (times != Py_None) {
                PyArg_ParseTuple(times, "LlLlO", &start_sec, &start_nsec, &end_sec, &end_nsec, &step);
            } else {
                method = "empty";
            }
            Py_DECREF(times);
        } else {
            err = 1;
        }
    }

    // TODO: Check if fill_positions exists, and fill the data, if necessary.
    
    if (!err) {
        // Get references to necessary NumPy functions.
        numpy = PyImport_AddModule("numpy");  // borrowed
        linspace = PyObject_GetAttrString(numpy, "linspace");  // new

        // Resample the specified part of the array.
        start_idx = (start_sec - self->start_sec + (start_nsec - self->start_nsec) / 1e9) * self->samplerate;
        end_idx = PyArray_DIM((PyArrayObject *)self->data, 0);
        end_idx -= (end_t.tv_sec - end_sec + (end_t.tv_nsec - end_nsec) / 1e9) * self->samplerate;
        
        if (strcmp(method, "interpolate-decimate") == 0) {
            // Get a view to the specified part of the array.
            start = PyLong_FromSsize_t(start_idx);  // new
            end = PyLong_FromSsize_t(end_idx);  // new
            slice = PySlice_New(start, end, step);  // new
            Py_DECREF(start);
            Py_DECREF(end);
            tmpArray = (PyArrayObject *)PyObject_GetItem((PyObject *)self->data, slice);  // new
            // TODO: FIR
            Py_XDECREF(slice);
        } else if (strcmp(method, "VRA") == 0) {
            if (self->samplerate > 3.0 * samplerate) {
                size = llabs(end_idx - start_idx) * samplerate / self->samplerate;
                if (size < 2)
                size = 2;
                slice = PyObject_CallFunction(linspace, "nnn", start_idx, end_idx - 1, size);  // new
                if (slice) {
                    sliceData = PyArray_DATA((PyArrayObject *)slice);
                    index = calloc(size, sizeof(npy_intp));
                    if (sliceData && index) {
                        for (i = 0; i < size; i++) {
                            index[i] = rint(sliceData[i]);
                        }
                    }
                    Py_DECREF(slice);
                }
                if (index) {
                    dims[0] = 3 * size - 2;
                    tmpArray = (PyArrayObject *)PyArray_SimpleNew(1, dims, oldTypenum);
                    switch (oldTypenum) {
                        #define AGGREGATEMINMAX(type)                                   \
                        do {                                                            \
                            type *tmp = PyArray_DATA(tmpArray);                         \
                            type *data = PyArray_DATA((PyArrayObject *)self->data);     \
                            tmp[0] = data[*index];                                      \
                            for (i = 0; i < size - 1; i++) {                            \
                                tmp[3 * i + 1] = tmp[3 * i];                            \
                                tmp[3 * i + 2] = tmp[3 * i];                            \
                                for (k = index[i] + 1; k < index[i + 1]; k++) {         \
                                    if (data[k] < tmp[3 * i + 1])                       \
                                        tmp[3 * i + 1] = data[k];                       \
                                    if (data[k] > tmp[3 * i + 2])                       \
                                        tmp[3 * i + 2] = data[k];                       \
                                }                                                       \
                                tmp[3 * i + 3] = data[index[i + 1]];                    \
                                tmp[3 * i + 3] = data[index[i + 1]];                    \
                            }                                                           \
                        } while (0);
                        case NPY_BYTE: {
                            AGGREGATEMINMAX(int8_t);
                            break;
                        }
                        case NPY_UBYTE: {
                            AGGREGATEMINMAX(uint8_t);
                            break;
                        }
                        case NPY_INT16: {
                            AGGREGATEMINMAX(int16_t);
                            break;
                        }
                        case NPY_UINT16: {
                            AGGREGATEMINMAX(uint16_t);
                            break;
                        }
                        case NPY_INT32: {
                            AGGREGATEMINMAX(int32_t);
                            break;
                        }
                        case NPY_UINT32: {
                            AGGREGATEMINMAX(uint32_t);
                            break;
                        }
                        case NPY_FLOAT: {
                            AGGREGATEMINMAX(float);
                            break;
                        }
                        case NPY_DOUBLE: {
                            AGGREGATEMINMAX(double);
                            break;
                        }
                        default:
                            PyErr_SetString(PyExc_TypeError, "unsupported dtype");
                    }
                    free(index);
                }
            } else {
                // There is no need for aggregation. Return original samples.
                start = PyLong_FromSsize_t(start_idx);
                end = PyLong_FromSsize_t(end_idx);
                slice = PySlice_New(start, end, step);  // new
                Py_DECREF(start);
                Py_DECREF(end);
                tmpArray = (PyArrayObject *)PyObject_GetItem((PyObject *)self->data, slice);  // new
            }
        } else if (strcmp(method, "empty") == 0)  {
            dims[0] = 0;
            tmpArray = (PyArrayObject *)PyArray_SimpleNew(1, dims, newTypenum);
        } else {
            PyErr_SetString(PyExc_AttributeError, "unknown resampling method");
        }
        Py_XDECREF(linspace);
        Py_XDECREF(step);
        if (tmpArray) {
            if (newTypenum != oldTypenum) {
                newArray = (PyArrayObject *)PyArray_Cast(tmpArray, newTypenum);  // new
                Py_DECREF(tmpArray);
                if (newArray && self->scale != 1.0 && self->offset != 0.0) {
                    // Apply scale and offset.
                    if (newTypenum == NPY_DOUBLE) {
                        size = PyArray_DIM(newArray, 0);
                        double *data = PyArray_DATA(newArray);
                        if (data) {
                            for (i = 0; i < size; i++) {
                                data[i] = data[i] * self->scale + self->offset;
                            }
                        }
                    } else if (newTypenum == NPY_FLOAT) {
                        size = PyArray_DIM(newArray, 0);
                        float *data = PyArray_DATA(newArray);
                        if (data) {
                            for (i = 0; i < size; i++) {
                                data[i] = data[i] * self->scale + self->offset;
                            }
                        }
                    }
                }
            } else {
                newArray = tmpArray;
            }
        }
    }

    return (PyObject *)newArray;
}

static PyObject *Channel_resample(Channel *self, PyObject *args, PyObject *kwds) {
    Channel *newChannel = NULL;
    PyObject *tmpArray, *events, *iterator, *resampleArgs, *times, *array = NULL;
    Event *e, *eCopy;
    double samplerate, scale, offset, *data, max, min;
    long long start_sec, end_sec, eStart_sec, eEnd_sec;
    long start_nsec, end_nsec, eStart_nsec, eEnd_nsec;
    int oldTypenum, newTypenum, tmpTypenum, resolution, copyEvents, dir;
    PyArray_Descr *typeDescr;
    npy_intp size, i;
    struct timespec end_t;
    const char *method = "interpolate-decimate";
    static char *kwlist[] = {"samplerate", "start_sec", "start_nsec", "end_sec", "end_nsec", "typenum", "method", "copyEvents", NULL};

    samplerate = self->samplerate;
    start_sec = self->start_sec;
    start_nsec = self->start_nsec;
    end_t = channelEndTime(self);
    end_sec = end_t.tv_sec;
    end_nsec = end_t.tv_nsec;
    oldTypenum = PyArray_DTYPE((PyArrayObject *)self->data)->type_num;
    newTypenum = oldTypenum;
    if (PyArg_ParseTupleAndKeywords(args, kwds, "d|LlLlisi:resample", kwlist, &samplerate,
                                    &start_sec, &start_nsec, &end_sec, &end_nsec, &newTypenum, &method, &copyEvents)) {
        // Check that the new channel type is compatible with the old type.
        if (PyArray_CanCastSafely(oldTypenum, newTypenum)) {
            tmpTypenum = newTypenum;
        } else {
            // Types are not compatible. Resample to double, and convert to new type later.
            tmpTypenum = NPY_DOUBLE;
        }
        // Check validity of the start and end times.
        times = PyObject_CallMethod((PyObject *)self, "validateTimes", "LlLl",
                                    start_sec, start_nsec, end_sec, end_nsec);  // new
        if (times) {
            if (times != Py_None) {
                PyArg_ParseTuple(times, "LlLli", &start_sec, &start_nsec, &end_sec, &end_nsec, &dir);
                // Resample and create a new Channel object.
                resampleArgs = Py_BuildValue("dLlLlis", samplerate, start_sec, start_nsec,
                                             end_sec, end_nsec, tmpTypenum, method);  // new
                array = Channel_resampledData(self, resampleArgs, NULL);  // new
            }
            Py_DECREF(times);
        }
        if (array) {
            if (tmpTypenum != newTypenum) {
                typeDescr = PyArray_DescrFromType(newTypenum);
                if (typeDescr->kind == 'u') {
                    size = PyArray_DIM((PyArrayObject *)array, 0);
                    data = PyArray_DATA((PyArrayObject *)array);
                    max = data[0];
                    min = data[0];
                    for (i = 1; i < size; i++) {
                        if (max < data[i])
                            max = data[i];
                        if (min > data[i])
                            min = data[i];
                    }
                    scale = (max - min) / ((1ULL << typeDescr->elsize) - 1);
                    offset = min;
                    for (i = 0; i < size; i++) {
                        data[i] = rint((data[i] - min) / scale);
                    }
                } else if (typeDescr->kind == 'i') {
                    size = PyArray_DIM((PyArrayObject *)array, 0);
                    data = PyArray_DATA((PyArrayObject *)array);
                    max = data[0];
                    min = data[0];
                    for (i = 1; i < size; i++) {
                        if (max < data[i])
                            max = data[i];
                        if (min > data[i])
                            min = data[i];
                    }
                    max = (fabs(max) > fabs(min)) ? fabs(max) : fabs(min);
                    scale = max / ((1ULL << (typeDescr->elsize - 1)) - 1);
                    offset = 0.0;
                    for (i = 0; i < size; i++) {
                        data[i] = rint(data[i] / scale);
                    }
                } else {
                    scale = 1.0;
                    offset = 0.0;
                }
                resolution = (self->resolution > typeDescr->elsize) ? typeDescr->elsize : self->resolution;
                tmpArray = PyArray_Cast((PyArrayObject *)array, newTypenum);  // new
                Py_DECREF(array);
                array = tmpArray;
            } else if (newTypenum == NPY_FLOAT || newTypenum == NPY_DOUBLE) {
                scale = 1.0;
                offset = 0.0;
                resolution = self->resolution;
            } else {
                scale = self->scale;
                offset = self->offset;
                resolution = self->resolution;
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
            // TODO: Fix the call (cannot pass NULL values):
            newChannel = (Channel *)PyObject_CallFunction((PyObject *)&ChannelType, "OdOOOddssLlssisO",
                                                          array, samplerate,
                                                          NULL, NULL, NULL, scale, offset,
                                                          self->unit, self->type, start_sec, start_nsec,
                                                          self->device, self->serial_no, resolution,
                                                          self->json, events);  // new
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
    {"collate", (PyCFunction)Channel_collate, METH_VARARGS | METH_STATIC, "combine channels to a sparse channel"},
    {"validateTimes", (PyCFunction)Channel_validateTimes, METH_VARARGS, "validate start and end times"},
    {"getEvents", (PyCFunction)Channel_getEvents, METH_VARARGS | METH_KEYWORDS, "get channel events"},
    {"resample", (PyCFunction)Channel_resample, METH_VARARGS | METH_KEYWORDS, "resample channel"},
    {"resampledData", (PyCFunction)Channel_resampledData, METH_VARARGS | METH_KEYWORDS, "resampled channel data as a NumPy array"},
    {NULL}  // sentinel
};

static PyMemberDef Channel_members[] = {
    {"__dict__", T_OBJECT, offsetof(Channel, dict), READONLY, "dictionary for instance variables"},
    {"data", T_OBJECT_EX, offsetof(Channel, data), READONLY, "Channel data as NumPy array"},
    {"fill_positions", T_OBJECT_EX, offsetof(Channel, fill_positions), 0, "fill positions for sparse channels as a NumPy array"},
    {"fill_lengths", T_OBJECT_EX, offsetof(Channel, fill_lengths), 0, "fill lengths for sparse channels as a NumPy array"},
    {"fill_values", T_OBJECT_EX, offsetof(Channel, fill_values), 0, "fill values for sparse channels as a NumPy array"},
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