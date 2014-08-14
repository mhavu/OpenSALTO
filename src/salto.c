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
#include "salto.h"
#include <structmember.h>

static PyObject *mainDict = NULL;   // global namespace
static PyObject *saltoDict = NULL;  // salto namespace


// Define salto module functions.

static PyObject *datetimeFromTimespec(PyObject *self, PyObject *args) {
    // Convert timespec to a Python datetime object.
    long long epoch, nsec = 0;
    PyObject *dtClass, *fromtimestamp, *dt, *empty, *keywords, *micro, *replace, *datetime = NULL;
    PyGILState_STATE state;

    state = PyGILState_Ensure();
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
    PyGILState_Release(state);

    return datetime;
}

static PyMethodDef saltoMethods[] = {
    {"datetimeFromTimespec", datetimeFromTimespec, METH_VARARGS, "Convert timespec to a Python datetime object"},
    {NULL, NULL, 0, NULL}  // sentinel
};


// Define Channel class.

static PyTypeObject ChannelType;

static void Channel_dealloc(Channel* self) {
    free(self->device);
    free(self->serial_no);
    free(self->unit);
    free(self->type);
    free(self->json);
    Py_XDECREF(self->data);
    Py_XDECREF(self->dict);
    Py_XDECREF(self->fill_values);
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
    PyObject *data = NULL;
    int result;
    static char *kwlist[] = {"data", "fill_values", "samplerate", "scale", "offset", "unit", "type",
        "start_sec", "start_nsec", "device", "serial_no", "resolution", "json", NULL};

    self->device = NULL;
    self->serial_no = NULL;
    self->unit = NULL;
    self->type = NULL;
    self->scale = 1.0;
    self->json = NULL;
    self->fill_values = NULL;
    result = !PyArg_ParseTupleAndKeywords(args, kwds, "O|OdddssLlssis", kwlist, &data, &(self->fill_values),
                                          &(self->samplerate), &(self->scale), &(self->offset),
                                          &(self->unit), &(self->type), &(self->start_sec), &(self->start_nsec),
                                          &(self->device), &(self->serial_no), &(self->resolution), &(self->json));
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

    return result;
}

static PyObject *Channel_start(Channel *self) {
    PyObject *timespec = Py_BuildValue("(Ll)", self->start_sec, self->start_nsec);
    return datetimeFromTimespec((PyObject *)self, timespec);
}

static PyObject *Channel_duration(Channel *self) {
    npy_intp length;
    double duration;


    if (self->collection) {
        // TODO: Return total duration for collection channels
        duration = nan(NULL);
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

static PyObject *Channel_end(Channel *self) {
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

static PyMethodDef Channel_methods[] = {
    {"start", (PyCFunction)Channel_start, METH_NOARGS, "channel start time as a datetime object"},
    {"duration", (PyCFunction)Channel_duration, METH_NOARGS, "channel duration in seconds"},
    {"end", (PyCFunction)Channel_end, METH_NOARGS, "channel end time as a datetime object"},
    {NULL}  // sentinel
};

static PyMemberDef Channel_members[] = {
    {"__dict__", T_OBJECT, offsetof(Channel, dict), READONLY, "dictionary for instance variables"},
    {"data", T_OBJECT_EX, offsetof(Channel, data), READONLY, "Channel data as NumPy array or collection of Channel objects"},
    {"samplerate", T_DOUBLE, offsetof(Channel, samplerate), 0, "sample rate in Hz"},
    {"scale", T_DOUBLE, offsetof(Channel, scale), 0, "scale for integer channels"},
    {"offset", T_DOUBLE, offsetof(Channel, offset), 0, "offset for integer channels"},
    {"unit", T_STRING, offsetof(Channel, unit), 0, "channel units"},
    {"start_sec", T_LONGLONG, offsetof(Channel, start_sec), 0, "start time (POSIX time)"},
    {"start_nsec", T_LONG, offsetof(Channel, start_nsec), 0, "nanoseconds to add to the start time"},
    {"device", T_STRING, offsetof(Channel, device), 0, "device make and model"},
    {"serial_no", T_STRING, offsetof(Channel, serial_no), 0, "device serial number"},
    {"resolution", T_INT, offsetof(Channel, resolution), 0, "sampling resolution in bits"},
    {"json", T_STRING, offsetof(Channel, json), 0, "additional metadata in JSON format"},
    {NULL}  // sentinel
};

static PyTypeObject ChannelType = {
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
    0,		                     // tp_richcompare
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


// Define API functions.

double duration(const char *chTable, const char *name) {
    return PyFloat_AsDouble(Channel_duration(getChannel(chTable, name)));
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
    void *ptr;
    PyGILState_STATE state;

    if (ch) {
        if (ch->collection) {
            state = PyGILState_Ensure();
            ptr = PyArray_DATA((PyArrayObject *)ch->data);
            *length = (size_t)PyArray_DIM((PyArrayObject *)ch->data, 0);
            PyGILState_Release(state);
        } else {
            ptr = NULL;
            *length = 0; // TODO: Return total number of samples
        }
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

void deleteChannelTable(const char *chTable) {
    PyObject *chTableDict;
    PyGILState_STATE state;

    state = PyGILState_Ensure();
    chTableDict = PyDict_GetItemString(saltoDict, "channelTables");  // borrowed
    if (chTableDict) {
        PyDict_DelItemString(chTableDict, chTable);
    }
    PyGILState_Release(state);
}

int newCombinationChannel(const char *chTable, const char *name, const char *fromChannelTable, void *fillValues) {
    PyObject *chTableDict, *channelClass, *sourceTable, *sourceChannels;
    Channel *ch;
    PyGILState_STATE state;
    int result = 0;

    state = PyGILState_Ensure();
    chTableDict = PyDict_GetItemString(saltoDict, "channelTables");  // borrowed
    channelClass = PyDict_GetItemString(saltoDict, "Channel");  // borrowed
    if (chTableDict && channelClass) {
        sourceTable = PyDict_GetItemString(chTableDict, fromChannelTable);  // borrowed
        sourceChannels = PyObject_GetAttrString(sourceTable, "channels");  // new
        if (sourceChannels) {
            // TODO: Order by time and check that the channels don't overlap
            // TODO: Check that all channels have same dtype
            // TODO: Create ndarray from fillValues
            ch = (Channel *)PyObject_CallFunctionObjArgs(channelClass, sourceChannels, NULL);  // new
            if (ch) {
                addChannel(chTable, name, ch);
                Py_DECREF(ch);
            }
            Py_DECREF(sourceChannels);
        }
    }
    PyGILState_Release(state);

    return result;
}

int registerFileFormat(void *obj, const char *format, const char **exts, size_t n_exts)
{
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

static struct PyModuleDef saltoModuleDef = {
    PyModuleDef_HEAD_INIT,
    "salto",
    "OpenSALTO C extension module",
    -1,
    saltoMethods,
    NULL, NULL, NULL, NULL
};

static PyObject *PyInit_salto(void)
{
    PyObject *module = NULL;

    if (PyType_Ready(&ChannelType) >= 0) {
        module = PyModule_Create(&saltoModuleDef);
        if (module) {
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

    // Initialize NumPy.
    if (_import_array() < 0) {
        fprintf(stderr, "Module numpy.core.multiarray failed to import\n");
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
