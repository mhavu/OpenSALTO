//
//  salto.c
//  OpenSALTO
//
//  Created by Marko Havu on 2014-07-09.
//  Released under the terms of GNU General Public License version 3.
//

#include <stdlib.h>
#include <stdio.h>
#include "salto_api.h"
#define PY_ARRAY_UNIQUE_SYMBOL Py_Array_API_OpenSALTO
#include "salto.h"

static PyObject *mainDict = NULL;   // global namespace
static PyObject *saltoDict = NULL;  // salto namespace


// Define API functions.

double duration(const char *chTable, const char *name) {
    double duration;
    PyGILState_STATE state;

    state = PyGILState_Ensure();
    duration = PyFloat_AsDouble(Channel_duration(getChannel(chTable, name)));
    PyGILState_Release(state);

    return duration;
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

void *newIntegerChannel(const char *chTable, const char *name, size_t length, size_t size, int is_signed) {
    int typenum;
    Channel *ch;
    void *ptr = NULL;
    PyObject *ndarray, *channelClass;
    PyGILState_STATE state;

    typenum = numpyType(size, 1, is_signed);
    if (typenum != NPY_NOTYPE) {
        state = PyGILState_Ensure();
        ndarray = PyArray_SimpleNew(1, (npy_intp *)&length, typenum);  // new
        channelClass = PyDict_GetItemString(saltoDict, "Channel");  // borrowed
        if (ndarray && channelClass) {
            ch = (Channel *)PyObject_CallFunctionObjArgs(channelClass, ndarray, NULL);  // new
            if (ch && addChannel(chTable, name, ch) == 0) {
                ptr = PyArray_DATA((PyArrayObject *)ch->data);
                Py_DECREF(ch);
            }
        }
        PyGILState_Release(state);
    } else {
        fprintf(stderr, "Invalid channel data type (%ssigned integer, %d bytes per sample)",
                (is_signed ? "" : "un"), (int)size);
    }

    return ptr;
}

void *newRealChannel(const char *chTable, const char *name, size_t length, size_t size) {
    int typenum;
    Channel *ch;
    void *ptr = NULL;
    PyObject *ndarray, *channelClass;
    PyGILState_STATE state;

    typenum = numpyType(size, 0, 1);
    if (typenum != NPY_NOTYPE) {
        state = PyGILState_Ensure();
        ndarray = PyArray_SimpleNew(1, (npy_intp *)&length, typenum);  // new
        channelClass = PyDict_GetItemString(saltoDict, "Channel");  // borrowed
        if (ndarray && channelClass) {
            ch = (Channel *)PyObject_CallFunction(channelClass, "OdddsLlssis", ndarray, 0.0,
                                                  nan(NULL), nan(NULL), "",
                                                  0, 0, "unknown", "unknown", 0, "{}");  // new
            if (ch && addChannel(chTable, name, ch) == 0) {
                ptr = PyArray_DATA((PyArrayObject *)ch->data);
                Py_DECREF(ch);
            }
        }
        PyGILState_Release(state);
    } else {
        fprintf(stderr, "Invalid channel data type (real, %d bytes per sample)", (int)size);
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
        if (!PyObject_TypeCheck(ch, &ChannelType))
            ch = NULL;
        Py_DECREF(channels);
    }
    PyGILState_Release(state);

    return ch;
}

void *channelData(Channel *ch, size_t *length) {
    Py_ssize_t nParts, i;
    Channel *part;
    void *ptr;
    PyGILState_STATE state;

    if (ch) {
        state = PyGILState_Ensure();
        if (!ch->collection) {
            ptr = PyArray_DATA((PyArrayObject *)ch->data);
            *length = (size_t)PyArray_DIM((PyArrayObject *)ch->data, 0);
        } else {
            ptr = NULL;
            *length = 0;
            nParts = PyList_GET_SIZE(ch->data);
            for (i = 0; i < nParts; i++) {
                part = (Channel *)PyList_GET_ITEM(ch->data, i);  // borrowed
                *length += (size_t)PyArray_DIM((PyArrayObject *)part->data, 0);
            }
        }
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
            if (isChannelObj && !((Channel *)value)->collection &&
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
    PyObject *chTableDict, *channelTable, *channelClass, *o;
    int result = -1;
    PyGILState_STATE state;

    state = PyGILState_Ensure();
    chTableDict = PyDict_GetItemString(saltoDict, "channelTables");  // borrowed
    channelTable = PyDict_GetItemString(chTableDict, chTable);  // borrowed
    if (channelTable) {
        channelClass = PyDict_GetItemString(saltoDict, "Channel");  // borrowed
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
            strlcpy(s, buffer, length);
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
                strlcpy(s, buffer, length);
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
    PyObject *chTableDict, *channelTable, *key, *value;
    const char **names = NULL;
    Py_ssize_t pos;
    PyGILState_STATE state;
    size_t i;

    state = PyGILState_Ensure();
    chTableDict = PyDict_GetItemString(saltoDict, "channelTables");  // borrowed
    channelTable = PyDict_GetItemString(chTableDict, chTable);  // borrowed
    if (channelTable) {
        *size = PyDict_Size(channelTable);
        names = calloc(*size, sizeof(char *));
        if (names) {
            pos = 0;
            i = 0;
            while (PyDict_Next(channelTable, &pos, &key, &value)) {  // borrowed
                names[i++] = PyUnicode_AsUTF8(key);
            }
        }
    }
    PyGILState_Release(state);

    return names;
}

int newCombinationChannel(const char *chTable, const char *name, const char *fromChannelTable, void *fillValues) {
    PyObject *chTableDict, *channelClass, *sourceTable, *sourceDict, *sourceChannels = NULL;
    PyObject *start, *prevEnd, *fill;
    Channel *ch, *part;
    Py_ssize_t i, nParts;
    npy_intp nFillValues;
    PyGILState_STATE state;
    int hasOverlap, isSameType, typenum, err = 0;

    state = PyGILState_Ensure();
    chTableDict = PyDict_GetItemString(saltoDict, "channelTables");  // borrowed
    channelClass = PyDict_GetItemString(saltoDict, "Channel");  // borrowed
    if (chTableDict && channelClass) {
        sourceTable = PyDict_GetItemString(chTableDict, fromChannelTable);  // borrowed
        sourceDict = PyObject_GetAttrString(sourceTable, "channels");  // new
        if (sourceDict) {
            sourceChannels = PyDict_Values(sourceDict);  // new
            Py_DECREF(sourceDict);
        }
        if (sourceChannels && PyList_Sort(sourceChannels) == 0) {
            nParts = PyList_GET_SIZE(sourceChannels);
            switch (nParts) {
                case 0:
                    ch = NULL;
                    break;
                case 1:
                    // In case there is only a single channel in the source table,
                    // return that channel.
                    ch = (Channel *)PyList_GET_ITEM(sourceChannels, 0);  // borrowed
                    Py_INCREF(ch);
                    break;
                default:
                    part = (Channel *)PyList_GET_ITEM(sourceChannels, 0);  // borrowed
                    typenum = PyArray_DTYPE((PyArrayObject *)part->data)->type_num;
                    prevEnd = Channel_end(part);  // new
                    for (i = 1; i < nParts; i++) {
                        part = (Channel *)PyList_GET_ITEM(sourceChannels, i);  // borrowed
                        start = Channel_start(part);  // new
                        hasOverlap = (start < prevEnd);
                        Py_XDECREF(start);
                        Py_XDECREF(prevEnd);
                        isSameType = (PyArray_DTYPE((PyArrayObject *)part->data)->type_num == typenum);
                        if (hasOverlap || !isSameType) {
                            ch = NULL;
                            err = -1;
                            if (hasOverlap) {
                                PyErr_SetString(PyExc_ValueError, "Channels overlap");
                            } else {
                                PyErr_SetString(PyExc_ValueError, "Channel types do not match");
                            }
                            break;
                        }
                        prevEnd = Channel_end(part);  // new
                    }
                    Py_XDECREF(prevEnd);
                    if (!err) {
                        nFillValues = nParts - 1;
                        fill = PyArray_SimpleNewFromData(1, &nFillValues, typenum, fillValues);
                        ch = (Channel *)PyObject_CallFunctionObjArgs(channelClass, sourceChannels,
                                                                     fill, NULL);  // new
                    }
            }
            if (ch) {
                addChannel(chTable, name, ch);
                Py_DECREF(ch);
            }
            Py_DECREF(sourceChannels);
        }
    }
    PyGILState_Release(state);

    return err;
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
        PyErr_SetString(PyExc_TypeError, "initPlugin() takes a Plugin object argument");
    }
    PyGILState_Release(state);
    
    return result;
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
                c_char_p = PyObject_GetAttrString(PyDict_GetItemString(mainDict, "c"), "c_char_p");  // new
                if (strcmp(type, "Import") == 0 || strcmp(type, "Export") == 0) {
                    argtypes = PyTuple_Pack(2, c_char_p, c_char_p);  // new
                    PyObject_SetAttrString(func, "argtypes", argtypes);
                    Py_XDECREF(argtypes);
                } else {
                    PyErr_SetString(PyExc_TypeError, "Unknown callback type in setCallback()");
                    result = -1;
                }
                Py_XDECREF(c_char_p);
                if (result == 0) {
                    method = PyUnicode_FromFormat("set%sFunc", type);  // new
                    name = PyUnicode_FromString(format);  // new
                    o = PyObject_CallMethodObjArgs(obj, method, name, func, NULL);  // new
                    result = (o ? 0 : -1);
                    Py_XDECREF(o);
                    Py_XDECREF(method);
                    Py_XDECREF(name);
                }
                Py_DECREF(func);
            }
            Py_DECREF(cdll);
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "initPlugin() takes a Plugin object argument");
    }
    PyGILState_Release(state);

    return result;
}

int addEvent(const char *chTable, const char *name, Event *event) {
    PyObject *events;
    Channel *ch;
    int result = -1;
    PyGILState_STATE state;

    state = PyGILState_Ensure();
    ch = getChannel(chTable, name);
    if (ch) {
        events = PyObject_GetAttrString((PyObject *)ch, "events");  // new
        if (events) {
            if (!PySequence_Contains(events, (PyObject *)event))
                result = PyList_Append(events, (PyObject *)event);
            Py_DECREF(events);
        }
    }
    PyGILState_Release(state);

    return result;
}

void removeEvent(const char *chTable, const char *name, Event *event) {
    PyObject *events;
    Channel *ch;
    Py_ssize_t i;
    PyGILState_STATE state;

    state = PyGILState_Ensure();
    ch = getChannel(chTable, name);
    if (ch) {
        events = PyObject_GetAttrString((PyObject *)ch, "events");  // new
        if (events) {
            i = PySequence_Index(events, (PyObject *)event);
            if (i >= 0)
                PySequence_DelItem(events, i);
            Py_DECREF(events);
        }
    }
    PyGILState_Release(state);
}

Event **getEvents(const char *chTable, const char *name, size_t *size) {
    PyObject *events;
    Channel *ch;
    Py_ssize_t length, i;
    Event **e = NULL;
    PyGILState_STATE state;

    state = PyGILState_Ensure();
    ch = getChannel(chTable, name);
    if (ch) {
        events = PyObject_GetAttrString((PyObject *)ch, "events");  // new
        if (events) {
            length = PyList_GET_SIZE(events);
            if (length > 0) {
                e = calloc(length, sizeof(Event *));
                for (i = 0; i < length; i++) {
                    e[i] = (Event *)PyList_GET_ITEM(events, i);  // borrowed
                    Py_INCREF(e[i]);
                }
            }
            Py_DECREF(events);
        }
    }
    PyGILState_Release(state);

    return e;
}

void clearEvents(const char *chTable, const char *name) {
    PyObject *events;
    Channel *ch;
    PyGILState_STATE state;

    state = PyGILState_Ensure();
    ch = getChannel(chTable, name);
    if (ch) {
        events = PyObject_GetAttrString((PyObject *)ch, "events");  // new
        if (events) {
            PyList_SetSlice(events, 0, PyList_GET_SIZE(events), NULL);
            Py_DECREF(events);
        }
    }
    PyGILState_Release(state);
}


Event *newEvent(EventVariety type, const char *subtype, struct timespec start,
                struct timespec end, const char *description)
{
    PyObject *eventClass;
    Event *event = NULL;
    PyGILState_STATE state;

    state = PyGILState_Ensure();
    eventClass = PyDict_GetItemString(saltoDict, "Event");  // borrowed
    if (eventClass) {
        event = (Event *)PyObject_CallFunction(eventClass, "siLlLls", type, subtype,
                                               start.tv_sec, start.tv_nsec, end.tv_sec, end.tv_nsec,
                                               description);  // new
    }
    PyGILState_Release(state);

    return event;
}


// Define salto module functions.

PyObject *datetimeFromTimespec(PyObject *self, PyObject *args) {
    // Convert timespec to a Python datetime object.
    long long epoch, nsec = 0;
    PyObject *dtClass, *fromtimestamp, *dt, *empty, *keywords, *micro, *replace, *datetime = NULL;

    if (PyArg_ParseTuple(args, "L|L:datetimeFromTimespec", &epoch, &nsec)) {
        dtClass = PyDict_GetItemString(mainDict, "datetime");  // borrowed
        fromtimestamp = PyObject_GetAttrString(dtClass, "fromtimestamp");  // new
        dt = PyObject_CallFunction(fromtimestamp, "L", epoch);  // new
        replace = PyObject_GetAttrString(dt, "replace");  // new
        empty = PyTuple_New(0);  // new
        keywords = PyDict_New();  // new
        micro = Py_BuildValue("l", nsec / 1000);  // new
        if (PyDict_SetItemString(keywords, "microsecond", micro) == 0) {
            datetime = PyObject_Call(replace, empty, keywords);  // new
        } else {
            Py_INCREF(Py_None);
            datetime = Py_None;
        }
        Py_XDECREF(fromtimestamp);
        Py_XDECREF(dt);
        Py_XDECREF(replace);
        Py_XDECREF(empty);
        Py_XDECREF(keywords);
        Py_XDECREF(micro);
    } else {
        PyErr_SetString(PyExc_TypeError, "datetimeFromTimespec() takes POSIX time and an optional ns fraction as arguments");
    }

    return datetime;
}

static PyMethodDef saltoMethods[] = {
    {"datetimeFromTimespec", datetimeFromTimespec, METH_VARARGS, "Convert timespec to a Python datetime object"},
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

    if (PyType_Ready(&ChannelType) >= 0) {
        module = PyModule_Create(&saltoModuleDef);
        if (module) {
            // Initialize NumPy.
            import_array();
            // Add Channel class and get the module dictionary.
            Py_INCREF(&ChannelType);
            PyModule_AddObject(module, "Channel", (PyObject *)&ChannelType);
            saltoDict = PyModule_GetDict(module);  // borrowed
            Py_XINCREF(saltoDict);
        }
    }

    return module;
}



int main(int argc, const char *argv[]) {
    PyObject *mainModule;

    Py_SetProgramName(L"OpenSALTO");
    PyImport_AppendInittab("salto", &PyInit_salto);
    Py_Initialize();

    // Get a reference to the Python global dictionary.
    mainModule = PyImport_AddModule("__main__");  // borrowed
    if (mainModule) {
        mainDict = PyModule_GetDict(mainModule);  // borrowed
        Py_XINCREF(mainDict);
    } else {
        fprintf(stderr, "Python module __main__ not found\n");
        exit(EXIT_FAILURE);
    }

    // Execute salto.py and run the Python interpreter.
    FILE *fp = fopen("salto.py", "r");
    if (fp) {
        PyRun_SimpleFileEx(fp, "salto.py", 1);
    } else {
        perror("fopen()");
        exit(EXIT_FAILURE);
    }
    Py_Main(0, (wchar_t **)argv);

    // Release the Python objects.
    Py_XDECREF(mainDict);
    Py_XDECREF(saltoDict);

    Py_Finalize();

    return EXIT_SUCCESS;
}
