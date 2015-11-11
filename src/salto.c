//
//  salto.c
//  OpenSALTO
//
//  Copyright 2014 Marko Havu. Released under the terms of
//  GNU General Public License version 3 or later.
//

#include <stdlib.h>
#include <stdio.h>
#include "salto_api.h"
#define PY_ARRAY_UNIQUE_SYMBOL Py_Array_API_OpenSALTO
#include "salto.h"
#include <datetime.h>

static PyObject *mainDict = NULL;   // global namespace
static PyObject *saltoDict = NULL;  // salto namespace


// Define API functions.

double duration(const char *chTable, const char *name) {
    double duration;
    PyObject *channel, *value;
    PyGILState_STATE state;

    channel = (PyObject *)getChannel(chTable, name);
    state = PyGILState_Ensure();
    value = PyObject_CallMethod(channel, "duration", NULL);  // new
    duration = PyFloat_AsDouble(value);
    PyGILState_Release(state);

    return duration;
}

int setFills(const char *chTable, const char *name, size_t *positions, size_t *lengths) {
    Channel *ch;
    int result = 0;
    npy_intp nFills, i;
    Channel_Fill *fills;
    PyGILState_STATE state;
    
    ch = getChannel(chTable, name);
    if (ch) {
        state = PyGILState_Ensure();
        nFills = PyArray_DIM(ch->fills, 0);
        fills = PyArray_DATA(ch->fills);
        if (fills) {
            for (i = 0; i < nFills; i++) {
                fills[i].pos = positions[i];
                fills[i].len = lengths[i];
            }
        } else {
            result = -1;
        }
        PyGILState_Release(state);
    } else {
        result = -1;
    }
    
    return result;
}

int getFills(const char *chTable, const char *name, size_t *positions, size_t *lengths) {
    Channel *ch;
    int result = 0;
    npy_intp nFills, i;
    Channel_Fill *fills;
    PyGILState_STATE state;
    
    ch = getChannel(chTable, name);
    if (ch) {
        state = PyGILState_Ensure();
        nFills = PyArray_DIM(ch->fills, 0);
        fills = PyArray_DATA(ch->fills);
        if (fills) {
            for (i = 0; i < nFills; i++) {
                positions[i] = fills[i].pos;
                lengths[i] = fills[i].len;
            }
        } else {
            result = -1;
        }
        PyGILState_Release(state);
    } else {
        result = -1;
    }
    
    return result;
}

size_t numberOfFills(const char *chTable, const char *name) {
    size_t nFills = 0;
    Channel *ch;
    PyGILState_STATE state;
    
    ch = getChannel(chTable, name);
    if (ch) {
        state = PyGILState_Ensure();
        nFills = PyArray_DIM(ch->fills, 0);  // Casts npy_intp to size_t
        PyGILState_Release(state);
    }
    
    return nFills;
}

static int numpyType(size_t bytes_per_sample, int is_integer, int is_signed) {
    int typenum;

    if (is_integer) {
        switch (bytes_per_sample) {
            case 1:
                typenum = (is_signed ? NPY_BYTE : NPY_UBYTE);
                break;
            case 2:
                typenum = (is_signed ? NPY_INT16 : NPY_UINT16);
                break;
            case 4:
                typenum = (is_signed ? NPY_INT32 : NPY_UINT32);
                break;
            default:
                typenum = NPY_NOTYPE;
                break;
        }
    } else {
        switch (bytes_per_sample) {
            case 4:
                typenum = NPY_FLOAT;
                break;
            case 8:
                typenum = NPY_DOUBLE;
                break;
            default:
                typenum = NPY_NOTYPE;
                break;
        }
    }

    return typenum;
}

void *newIntegerChannel(const char *chTable, const char *name, size_t length, size_t size, int isSigned, size_t nParts) {
    int typenum;
    Channel *ch;
    void *ptr = NULL;
    PyObject *dataArray, *fillArray, *tempObj;
    PyArray_Descr *fillDescr;
    npy_intp nFills[1], nSamples[1];
    PyGILState_STATE state;

    if (nParts > 0) {
        typenum = numpyType(size, 1, isSigned);
        if (typenum != NPY_NOTYPE) {
            state = PyGILState_Ensure();
            nSamples[0] = length;
            dataArray = PyArray_ZEROS(1, nSamples, typenum, 0);  // new
            if (dataArray) {
                if (nParts == 1) {
                    ch = (Channel *)PyObject_CallFunction((PyObject *)&ChannelType, "Od",
                                                          dataArray, 0.0);  // new
                } else {
                    nFills[0] = nParts - 1;
                    // TODO: Move this to a single place.
                    tempObj = Py_BuildValue("[(s, s), (s, s)]", "pos", "p", "len", "p");  // new
                    PyArray_DescrConverter(tempObj, &fillDescr);  // new fillDescr
                    Py_DECREF(tempObj);
                    fillArray = PyArray_Zeros(1, nFills, fillDescr, 0);  // new, steals fillDescr
                    ch = (Channel *)PyObject_CallFunction((PyObject *)&ChannelType, "OdOOO",
                                                          dataArray, 0.0, fillArray);  // new
                }
                if (ch && addChannel(chTable, name, ch) == 0) {
                    ptr = PyArray_DATA((PyArrayObject *)ch->data);
                    Py_DECREF(ch);
                }
            }
            PyGILState_Release(state);
        } else {
            fprintf(stderr, "Invalid channel data type (%ssigned integer, %d bytes per sample)",
                    (isSigned ? "" : "un"), (int)size);
        }
    } else {
        fprintf(stderr, "newIntegerChannel called with nParts = 0");
    }

    return ptr;
}

void *newRealChannel(const char *chTable, const char *name, size_t length, size_t size, size_t nParts) {
    int typenum;
    Channel *ch;
    void *ptr = NULL;
    PyObject *dataArray, *fillArray, *tempObj;
    PyArray_Descr *fillDescr;
    npy_intp nFills[1], nSamples[1];
    PyGILState_STATE state;

    if (nParts > 0) {
        typenum = numpyType(size, 0, 1);
        if (typenum != NPY_NOTYPE) {
            state = PyGILState_Ensure();
            nSamples[0] = length;
            dataArray = PyArray_ZEROS(1, nSamples, typenum, 0);  // new
            if (dataArray) {
                if (nParts == 1) {
                    ch = (Channel *)PyObject_CallFunction((PyObject *)&ChannelType, "Od",
                                                          dataArray, 0.0);  // new
                    ch->scale = nan(NULL);
                    ch->offset = nan(NULL);
                } else {
                    nFills[0] = nParts - 1;
                    // TODO: Move this to a single place.
                    tempObj = Py_BuildValue("[(s, s), (s, s)]", "pos", "p", "len", "p");  // new
                    PyArray_DescrConverter(tempObj, &fillDescr);  // new fillDescr
                    Py_DECREF(tempObj);
                    fillArray = PyArray_Zeros(1, nFills, fillDescr, 0);  // new, steals fillDescr
                    ch = (Channel *)PyObject_CallFunction((PyObject *)&ChannelType, "OdOOOdd",
                                                          dataArray, 0.0, fillArray,
                                                          nan(NULL), nan(NULL));  // new
                }
                if (ch && addChannel(chTable, name, ch) == 0) {
                    ptr = PyArray_DATA((PyArrayObject *)ch->data);
                    Py_DECREF(ch);
                }
            }
            PyGILState_Release(state);
        } else {
            fprintf(stderr, "Invalid channel data type (real, %d bytes per sample)", (int)size);
        }
    } else {
        fprintf(stderr, "newRealChannel called with nParts = 0");
    }

    return ptr;
}

Channel *getChannel(const char *chTable, const char *name) {
    PyObject *chTableDict, *channelTable, *channels;
    Channel *ch = NULL;
    PyGILState_STATE state;

    state = PyGILState_Ensure();
    chTableDict = PyDict_GetItemString(saltoDict, "channelTables");  // borrowed
    channelTable = PyDict_GetItemString(chTableDict, chTable);  // borrowed
    if (channelTable) {
        channels = PyObject_GetAttrString(channelTable, "channels");  // new
        ch = (Channel *)PyDict_GetItemString(channels, name);  // borrowed
        if (ch && !PyObject_TypeCheck(ch, &ChannelType))
            ch = NULL;
        Py_DECREF(channels);
    }
    PyGILState_Release(state);

    return ch;
}

struct timespec channelEndTime(Channel *ch) {
    struct timespec t;
    double duration;
    size_t length;
    npy_intp nFills, i, total;
    Channel_Fill *fills;
    PyGILState_STATE state;

    if (ch) {
        total = 0;
        if (ch->fills) {
            state = PyGILState_Ensure();
            nFills = PyArray_DIM(ch->fills, 0);
            fills = PyArray_DATA(ch->fills);
            for (i = 0; i < nFills; i++) {
                total += fills[i].len;
            }
            PyGILState_Release(state);
        }
        channelData(ch, &length);
        if (length > 0) {
            total += length;
            duration = (total - 1) / ch->samplerate;
            t = endTimeFromDuration(ch->start_sec, ch->start_nsec, duration);
        } else {
            t.tv_sec = ch->start_sec;
            t.tv_nsec = ch->start_nsec;
        }
    } else {
        t.tv_sec = -1;
        t.tv_nsec = -1;
    }

    return t;
}

double channelDuration(Channel *ch) {
    struct timespec end;
    double duration;

    end = channelEndTime(ch);
    if (end.tv_nsec >= 0) {
        duration = end.tv_sec - ch->start_sec + (end.tv_nsec - ch->start_nsec) / 1e9;
    } else {
        duration = nan(NULL);
    }

    return duration;
}

void *channelData(Channel *ch, size_t *length) {
    void *ptr;
    PyGILState_STATE state;

    if (ch) {
        state = PyGILState_Ensure();
        ptr = PyArray_DATA((PyArrayObject *)ch->data);
        *length = (size_t)PyArray_DIM((PyArrayObject *)ch->data, 0);
        PyGILState_Release(state);
    } else {
        ptr = NULL;
        *length = 0;
    }

    return ptr;
}

void *getChannelData(const char *chTable, const char *name, size_t *length) {
    Channel *ch;
    void *ptr;

    ch = getChannel(chTable, name);
    if (ch) {
        ptr = channelData(ch, length);
    } else {
        ptr = NULL;
        *length = 0;
    }
    return ptr;
}

const char *getChannelName(const char *chTable, void *ptr) {
    PyObject *chTableDict, *channelTable, *name, *value;
    char *s = NULL;
    Py_ssize_t pos;
    PyGILState_STATE state;
    int isChannelObj;

    state = PyGILState_Ensure();
    chTableDict = PyDict_GetItemString(saltoDict, "channelTables");  // borrowed
    channelTable = PyDict_GetItemString(chTableDict, chTable);  // borrowed
    if (channelTable) {
        pos = 0;
        while (PyDict_Next(channelTable, &pos, &name, &value)) {  // borrowed
            isChannelObj = PyObject_TypeCheck(value, &ChannelType);
            if (isChannelObj &&
                PyArray_DATA((PyArrayObject *)((Channel *)value)->data) == ptr)
            {
                s = PyUnicode_AsUTF8(name);
                break;
            }
        }
    }
    PyGILState_Release(state);

    return s;
}

int addChannel(const char *chTable, const char *name, Channel *ch) {
    PyObject *chTableDict, *channelTable, *o;
    int result = -1;
    PyGILState_STATE state;

    state = PyGILState_Ensure();
    chTableDict = PyDict_GetItemString(saltoDict, "channelTables");  // borrowed
    channelTable = PyDict_GetItemString(chTableDict, chTable);  // borrowed
    if (channelTable) {
        o = PyObject_CallMethod(channelTable, "add", "(sO)", name, ch);  // new
        result = (o ? 0 : -1);
        Py_XDECREF(o);
    }
    PyGILState_Release(state);

    return result;
}

void deleteChannel(const char *chTable, const char *name) {
    PyObject *chTableDict, *channelTable, *o;
    PyGILState_STATE state;
    
    state = PyGILState_Ensure();
    chTableDict = PyDict_GetItemString(saltoDict, "channelTables");  // borrowed
    channelTable = PyDict_GetItemString(chTableDict, chTable);  // borrowed
    if (channelTable) {
        o = PyObject_CallMethod(channelTable, "remove", "(s)", name);  // new
        Py_XDECREF(o);
    }
    PyGILState_Release(state);
}

const char *getUniqueName(const char *chTable, const char *name) {
    PyObject *chTableDict, *channelTable, *unique;
    char *buffer = NULL, *s = NULL;
    size_t length;
    PyGILState_STATE state;

    state = PyGILState_Ensure();
    chTableDict = PyDict_GetItemString(saltoDict, "channelTables");  // borrowed
    channelTable = PyDict_GetItemString(chTableDict, chTable);  // borrowed
    if (channelTable) {
        unique = PyObject_CallMethod(channelTable, "getUnique", "(s)", name);  // new
        if (unique)
            buffer = PyUnicode_AsUTF8AndSize(unique, (Py_ssize_t *)&length);
        if (buffer) {
            s = malloc(++length);
            strcpy(s, buffer);
        }
        Py_XDECREF(unique);
    }
    PyGILState_Release(state);

    return s;
}

const char *newChannelTable(const char *name) {
    PyObject *chTableDict, *chTable, *unique, *makeUniqueKey, *chTableClass;
    char *buffer, *s = NULL;
    size_t length;
    PyGILState_STATE state;

    state = PyGILState_Ensure();
    chTableDict = PyDict_GetItemString(saltoDict, "channelTables");  // borrowed
    makeUniqueKey = PyDict_GetItemString(saltoDict, "makeUniqueKey");  // borrowed
    chTableClass = PyDict_GetItemString(saltoDict, "ChannelTable");  // borrowed
    if (chTableDict && makeUniqueKey && chTableClass) {
        unique = PyObject_CallFunction(makeUniqueKey, "Os", chTableDict, name ? name : "Temporary");  // new
        if (unique) {
            chTable = PyObject_CallFunction(chTableClass, NULL);  // new
            if (chTable) {
                PyDict_SetItem(chTableDict, unique, chTable);
                Py_DECREF(chTable);
            }
            buffer = PyUnicode_AsUTF8AndSize(unique, (Py_ssize_t *)&length);
            if (buffer) {
                s = malloc(++length);
                strcpy(s, buffer);
            }
            Py_DECREF(unique);
        }
    }
    PyGILState_Release(state);

    return s;
}

void deleteChannelTable(const char *name) {
    PyObject *chTableDict;
    PyGILState_STATE state;

    state = PyGILState_Ensure();
    chTableDict = PyDict_GetItemString(saltoDict, "channelTables");  // borrowed
    if (chTableDict) {
        PyDict_DelItemString(chTableDict, name);
    }
    PyGILState_Release(state);
}

const char **getChannelNames(const char *chTable, size_t *size) {
    PyObject *chTableDict, *channelTable, *key, *value, *channels = NULL;
    const char **names = NULL;
    Py_ssize_t pos;
    PyGILState_STATE state;
    size_t i;

    state = PyGILState_Ensure();
    chTableDict = PyDict_GetItemString(saltoDict, "channelTables");  // borrowed
    channelTable = PyDict_GetItemString(chTableDict, chTable);  // borrowed
    if (channelTable)
        channels = PyObject_GetAttrString(channelTable, "channels");  // new
    if (channels) {
        *size = PyDict_Size(channels);
        if (*size)
            names = calloc(*size, sizeof(char *));
        if (names) {
            pos = 0;
            i = 0;
            while (PyDict_Next(channels, &pos, &key, &value)) {  // borrowed
                names[i++] = PyUnicode_AsUTF8(key);
            }
        }
        Py_DECREF(channels);
    }
    PyGILState_Release(state);

    return names;
}

int registerFileFormat(void *obj, const char *format, const char **exts, size_t n_exts) {
    PyObject *pluginClass, *extList, *name, *registerFormat, *o;
    PyGILState_STATE state;
    int result = -1;

    state = PyGILState_Ensure();
    pluginClass = PyDict_GetItemString(saltoDict, "Plugin");  // borrowed
    if (PyObject_IsInstance(obj, pluginClass)) {
        registerFormat = PyUnicode_FromString("registerFormat");  // new
        name = PyUnicode_FromString(format);  // new
        extList = PyList_New(n_exts);  // new
        if (registerFormat && name && extList) {
            for (int i = 0; i < n_exts; i++) {
                PyList_SET_ITEM(extList, i, PyUnicode_FromString(exts[i]));  // stolen
            }
            o = PyObject_CallMethodObjArgs(obj, registerFormat, name, extList, NULL);  // new
            result = (o ? 0 : -1);
            Py_XDECREF(o);
        }
        Py_XDECREF(registerFormat);
        Py_XDECREF(name);
        Py_XDECREF(extList);
    } else {
        PyErr_SetString(PyExc_TypeError, "initPlugin() takes a Plugin object argument that needs to be passed to registerFileFormat()");
    }
    PyGILState_Release(state);
    
    return result;
}

int registerComputation(void *obj, const char *name, const char *funcname,
                        ComputationArgs *inputs, ComputationArgs *outputs) {
    PyObject *pluginClass, *cdll, *func, *o, *defaultValue, *iList, *oList;
    PyGILState_STATE state;
    size_t i;
    int err = 0;

    state = PyGILState_Ensure();
    pluginClass = PyDict_GetItemString(saltoDict, "Plugin");  // borrowed
    if (PyObject_IsInstance(obj, pluginClass)) {
        cdll = PyObject_GetAttrString(obj, "cdll");  // new
        if (cdll) {
            func = PyObject_GetAttrString(cdll, funcname);  // new
            iList = PyList_New(inputs->n_args + 1);  // new
            oList = PyList_New(outputs->n_args + 1);  // new
            if (func && iList && oList) {
                o = Py_BuildValue("ssII", "channelTable", "S",
                                  inputs->min_channels, inputs->max_channels);  // new
                PyList_SET_ITEM(iList, 0, o);  // stolen
                o = Py_BuildValue("ssII", "channelTable", "S",
                                  outputs->min_channels, outputs->max_channels);  // new
                PyList_SET_ITEM(oList, 0, o);  // stolen
                for (i = 0; i < inputs->n_args; i++) {
                    // TODO: Set default values
                    defaultValue = Py_None;
                    o = Py_BuildValue("sssO", inputs->name[i], inputs->format[i],
                                      inputs->description[i], defaultValue);  // new
                    PyList_SET_ITEM(iList, i + 1, o);  // stolen
                }
                for (i = 0; i < outputs->n_args; i++) {
                    o = Py_BuildValue("sss", outputs->name[i], outputs->format[i],
                                      outputs->description[i]);  // new
                    PyList_SET_ITEM(oList, i + 1, o);  // stolen
                }
                o = PyObject_CallMethod(obj, "registerComputation", "(sOOO)",
                                        name, func, iList, oList);  // new
                err = (o ? 0 : -1);
                Py_XDECREF(o);
            } else {
                err = -1;
            }
            Py_XDECREF(func);
            Py_XDECREF(iList);
            Py_XDECREF(oList);
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "initPlugin() takes a Plugin object argument that needs to be passed to registerComputation()");
    }
    PyGILState_Release(state);

    return err;
}

int setCallback(void *obj, const char *type, const char *format, const char *funcname) {
    PyObject *o, *pluginClass, *cdll, *func, *c_char_p, *method, *name, *argtypes = NULL;
    PyGILState_STATE state;
    int result = -1;

    state = PyGILState_Ensure();
    pluginClass = PyDict_GetItemString(saltoDict, "Plugin");  // borrowed
    if (PyObject_IsInstance(obj, pluginClass)) {
        cdll = PyObject_GetAttrString(obj, "cdll");  // new
        if (cdll) {
            func = PyObject_GetAttrString(cdll, funcname);  // new
            if (func) {
                result = 0;
                if (strcmp(type, "Import") == 0 || strcmp(type, "Export") == 0) {
                    c_char_p = PyObject_GetAttrString(PyDict_GetItemString(mainDict, "c"), "c_char_p");  // new
                    argtypes = PyTuple_Pack(2, c_char_p, c_char_p);  // new
                    Py_XDECREF(c_char_p);
                    PyObject_SetAttrString(func, "argtypes", argtypes);
                    Py_XDECREF(argtypes);
                    method = PyUnicode_FromFormat("set%sFunc", type);  // new
                    name = PyUnicode_FromString(format);  // new
                    o = PyObject_CallMethodObjArgs(obj, method, name, func, NULL);  // new
                    result = (o ? 0 : -1);
                    Py_XDECREF(o);
                    Py_XDECREF(method);
                    Py_XDECREF(name);
                } else {
                    PyErr_SetString(PyExc_TypeError, "Unknown callback type in setCallback()");
                    result = -1;
                }
                Py_DECREF(func);
            }
            Py_DECREF(cdll);
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "initPlugin() takes a Plugin object argument that needs to be passed to setCallback()");
    }
    PyGILState_Release(state);

    return result;
}

int addEvent(const char *chTable, const char *name, Event *event) {
    Channel *ch;
    int result = -1;
    PyGILState_STATE state;

    state = PyGILState_Ensure();
    ch = getChannel(chTable, name);
    if (ch && ch->events) {
        result = PySet_Add((PyObject *)ch->events, (PyObject *)event);
        if (PyDict_GetItemString(saltoDict, "gui")) {  // borrowed
            // TODO: Update GUI
        }
    }
    PyGILState_Release(state);

    return result;
}

int removeEvent(const char *chTable, const char *name, Event *event) {
    Channel *ch;
    int result = -1;
    PyGILState_STATE state;

    state = PyGILState_Ensure();
    ch = getChannel(chTable, name);
    if (ch && ch->events) {
        result = PySet_Discard((PyObject *)ch->events, (PyObject *)event);
        if (PyDict_GetItemString(saltoDict, "gui")) {  // borrowed
            // TODO: Update GUI
        }
    }
    PyGILState_Release(state);

    return result;
}

Event **getEvents(const char *chTable, const char *name, size_t *size) {
    PyObject *iterator, *e;
    Channel *ch;
    Py_ssize_t length, i = 0;
    Event **array = NULL;
    PyGILState_STATE state;

    state = PyGILState_Ensure();
    ch = getChannel(chTable, name);
    if (ch && ch->events) {
        length = PySet_GET_SIZE(ch->events);
        if (length > 0) {
            array = calloc(length, sizeof(Event *));
            iterator = PyObject_GetIter((PyObject *)ch->events);  // new
            while ((e = PyIter_Next(iterator))) {  // new
                array[i++] = (Event *)e;
            }
            Py_DECREF(iterator);
        }
    }
    PyGILState_Release(state);

    return array;
}

void clearEvents(const char *chTable, const char *name) {
    Channel *ch;
    PyGILState_STATE state;

    state = PyGILState_Ensure();
    ch = getChannel(chTable, name);
    if (ch && ch->events) {
        PySet_Clear((PyObject *)ch->events);
        if (PyDict_GetItemString(saltoDict, "gui")) {  // borrowed
            // TODO: Update GUI
        }
    }
    PyGILState_Release(state);
}


Event *newEvent(EventVariety type, const char *subtype, struct timespec start,
                struct timespec end, const char *description) {
    Event *event = NULL;
    PyGILState_STATE state;

    state = PyGILState_Ensure();
    event = (Event *)PyObject_CallFunction((PyObject *)&EventType, "isLlLls", type, subtype,
                                               start.tv_sec, start.tv_nsec, end.tv_sec, end.tv_nsec,
                                               description);  // new
    PyGILState_Release(state);

    return event;
}

PyObject *unitRegistry(void) {
    return PyDict_GetItemString(saltoDict, "units");  // borrowed;
}


// Define salto module functions.

PyObject *datetimeFromTimespec(PyObject *self, PyObject *args) {
    // Convert timespec to a Python datetime object.
    long long epoch;
    long nsec = 0;
    PyObject *timestamp, *timedelta, *datetime, *result = NULL;

    if (PyArg_ParseTuple(args, "L|l:datetimeFromTimespec", &epoch, &nsec)) {
        timestamp = Py_BuildValue("(L)", epoch);  // new
        if (timestamp) {
            datetime = PyDateTime_FromTimestamp(timestamp);  // new
            if (datetime) {
                if (nsec) {
                    timedelta = PyDelta_FromDSU(0, 0, round(nsec / 1e3));  // new
                    if (timedelta) {
                        result = PyNumber_Add(datetime, timedelta);  // new
                        Py_DECREF(timedelta);
                    }
                    Py_DECREF(datetime);
                } else {
                    result = datetime;
                }
            }
            Py_DECREF(timestamp);
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "datetimeFromTimespec() takes POSIX time and an optional ns fraction as arguments");
    }
    
    return result;
}

PyObject *timedeltaFromFloat(PyObject *self, PyObject *args) {
    // Converts seconds as Python float to a datetime.timedelta object.
    PyObject *delta = NULL;
    int seconds, useconds;
    double t, s, days;
    
    if (PyArg_ParseTuple(args, "d:timedeltaFromFloat", &t)) {
        useconds = round(modf(t, &s) / 1e6);
        seconds = modf(s / 86400, &days) * 86400;
        if (days <= INT_MAX) {
            delta = PyDelta_FromDSU((int)days, seconds, useconds);  // new
        } else {
            PyErr_SetString(PyExc_OverflowError, "Time argument too big for timedelta");
        }
    }
    
    return delta;
}

static PyMethodDef saltoMethods[] = {
    {"datetimeFromTimespec", datetimeFromTimespec, METH_VARARGS, "Convert timespec to a Python datetime object"},
    {"timedeltaFromFloat", timedeltaFromFloat, METH_VARARGS, "Convert seconds to a Python datetime.timedelta"},
    {NULL, NULL, 0, NULL}  // sentinel
};

static struct PyModuleDef saltoModuleDef = {
    PyModuleDef_HEAD_INIT,
    "salto",
    "OpenSALTO C extension module",
    -1,
    saltoMethods,
    NULL, NULL, NULL, NULL
};

static PyObject *PyInit_salto(void) {
    PyObject *module = NULL;
    PyObject *path;

    EventType.tp_hash = PyBaseObject_Type.tp_hash;
    if (PyType_Ready(&ChannelType) >= 0 && PyType_Ready(&EventType) >= 0) {
        module = PyModule_Create(&saltoModuleDef);
        if (module) {
            // Initialize NumPy.
            import_array();
            // Initialize DateTime object API.
            PyDateTime_IMPORT;
            // Add Channel and Event classes.
            Py_INCREF(&ChannelType);
            PyModule_AddObject(module, "Channel", (PyObject *)&ChannelType);  // stolen
            Py_INCREF(&EventType);
            PyModule_AddObject(module, "Event", (PyObject *)&EventType);  // stolen
            // Add a __path__ attribute, so Python knows this is a package.
            path = PyList_New(1);  // new
            PyList_SetItem(path, 0, PyUnicode_FromString("salto"));  // stolen
            PyModule_AddObject(module, "__path__", path);
            // Get the module dictionary.
            saltoDict = PyModule_GetDict(module);  // borrowed
            Py_XINCREF(saltoDict);
        }
    }

    return module;
}


int saltoInit(const char *saltoPyPath, PyObject* (*guiInitFunc)(void)) {
    PyObject *mainModule, *saltoModule, *saltoGuiModule;
    int result = 0;

    PyImport_AppendInittab("salto", &PyInit_salto);
    if (guiInitFunc)
        PyImport_AppendInittab("salto_gui", guiInitFunc);
    Py_Initialize();
    if (guiInitFunc)
        PyEval_InitThreads();
    
    // Get a reference to the Python global dictionary.
    mainModule = PyImport_AddModule("__main__");  // borrowed
    if (mainModule) {
        mainDict = PyModule_GetDict(mainModule);  // borrowed
        Py_XINCREF(mainDict);
    } else {
        PyErr_Print();
        result = -1;
    }

    // Import the OpenSALTO modules.
    saltoModule = PyImport_ImportModule("salto");
    if (saltoModule) {
        PyModule_AddObject(mainModule, "salto", saltoModule);
    } else {
        PyErr_Print();
    }
    if (guiInitFunc) {
        saltoGuiModule = PyImport_ImportModule("salto_gui");
        if (saltoGuiModule) {
            PyModule_AddObject(saltoModule, "gui", saltoGuiModule);
        } else {
            PyErr_Print();
        }
    }
    
    // Execute salto.py.
    FILE *fp = fopen(saltoPyPath, "r");
    if (fp) {
        PyRun_SimpleFileEx(fp, "salto.py", 1);
    } else {
        perror("fopen()");
        result = -1;
    }
    
    if (guiInitFunc)
        PyEval_ReleaseThread(PyGILState_GetThisThreadState());

    return result;
}

static const char *saltoPyConsoleCode =
"try:\n\
    code.interact(\"OpenSALTO Python console\", None, locals())\n\
except SystemExit:\n\
    pass\n\
print(\"Exiting OpenSALTO\")";

int saltoRun(const char *filename) {
    int result;
    
    if (filename) {
        // Batch mode
        FILE *fp = fopen(filename, "r");
        result = PyRun_SimpleFileEx(fp, filename, 1);
    } else {
        // Interactive mode
        result = PyRun_SimpleString(saltoPyConsoleCode);
    }
    
    return result;
}

PyObject *saltoEval(const char *expr) {
    PyObject *code, *o, *result = NULL;
    PyGILState_STATE state;

    state = PyGILState_Ensure();
    code = Py_CompileString(expr, "<stdin>", Py_eval_input);
    if (code) {
        o = PyEval_EvalCode(code, mainDict, mainDict);
        if (o) {
            result = (o != Py_None) ? PyUnicode_FromFormat("%R\n", o) : PyUnicode_FromString("");  // new
        } else if (PyErr_ExceptionMatches(PyExc_SystemExit)) {
            // PyErr_Print() calls handle_system_exit(), so we need to
            // catch SystemExit before calling PyErr_Print().
            PyErr_Clear();
            // TODO: Show in console.
            fprintf(stderr, "SystemExit raised\n");
        } else {
            PyErr_PrintEx(1);
        }
    } else {
        PyErr_Clear();
        if (PyRun_SimpleString(expr) == 0)
            result = PyUnicode_FromString("");
    }
    PyGILState_Release(state);

    return result;
}

void saltoEnd(void *context) {
    PyGILState_Ensure();
    Py_XDECREF(mainDict);
    Py_XDECREF(saltoDict);
    Py_Finalize();
}
