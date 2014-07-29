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
#ifdef __APPLE__
#include <Python/Python.h>
#else
#include <Python.h>
#endif

static PyObject *mainDict = NULL;   // global namespace
static PyObject *saltoDict = NULL;  // salto namespace

Channel *getChannel(const char *chTable, const char *name) {
    PyObject *chTableDict, *channelTable, *channels, *capsule;
    Channel *ch = NULL;
    PyGILState_STATE state;

    state = PyGILState_Ensure();
    chTableDict = PyDict_GetItemString(saltoDict, "channelTables");  // borrowed
    channelTable = PyDict_GetItemString(chTableDict, chTable);  // borrowed
    if (channelTable) {
        channels = PyObject_GetAttrString(channelTable, "channels");  // new
        capsule = PyDict_GetItemString(channels, name);  // borrowed
        ch = PyCapsule_GetPointer(capsule, "Channel");
        Py_DECREF(channels);
    }
    PyGILState_Release(state);

    return ch;
}

const char *getNameForData(const char *chTable, void *ptr) {
    PyObject *chTableDict, *channelTable, *capsule, *name;
    char *s = NULL;
    PyGILState_STATE state;

    state = PyGILState_Ensure();
    chTableDict = PyDict_GetItemString(saltoDict, "channelTables");  // borrowed
    channelTable = PyDict_GetItemString(chTableDict, chTable);  // borrowed
    if (channelTable) {
        capsule = PyCapsule_New(ptr, "ChannelData", NULL);  // new
        name = PyObject_CallMethod(channelTable, "findKeyForPointer", "(O)", capsule);  // new
        s = PyString_AsString(name);
        Py_XDECREF(name);
        Py_XDECREF(capsule);
    }
    PyGILState_Release(state);

    return s;
}

int addChannel(const char *chTable, const char *name, Channel *ch) {
    PyObject *chTableDict, *channelTable, *capsule, *o = NULL;
    Channel *channel;
    int result = -1;
    PyGILState_STATE state;

    channel = (Channel *)malloc(sizeof(Channel));
    *channel = *ch;
    state = PyGILState_Ensure();
    chTableDict = PyDict_GetItemString(saltoDict, "channelTables");  // borrowed
    channelTable = PyDict_GetItemString(chTableDict, chTable);  // borrowed
    if (channelTable) {
        capsule = PyCapsule_New(channel, "Channel", NULL);  // new
        o = PyObject_CallMethod(channelTable, "add", "(sO)", name, capsule);  // new
        result = (o ? 0 : -1);
        Py_XDECREF(o);
        Py_XDECREF(capsule);
    } else {
        free(channel);
    }
    PyGILState_Release(state);

    return result;
}

void removeChannel(const char *chTable, const char *name) {
    PyObject *chTableDict, *channelTable, *o;
    Channel *ch;
    PyGILState_STATE state;
    
    state = PyGILState_Ensure();
    chTableDict = PyDict_GetItemString(saltoDict, "channelTables");  // borrowed
    channelTable = PyDict_GetItemString(chTableDict, chTable);  // borrowed
    if (channelTable) {
        ch = getChannel(chTable, name);
        o = PyObject_CallMethod(channelTable, "remove", "(s)", name);  // new
        Py_XDECREF(o);
        free(ch);
    }
    PyGILState_Release(state);
}

const char *getUniqueName(const char *chTable, const char *name) {
    PyObject *chTableDict, *channelTable, *unique;
    char *buffer, *s = NULL;
    size_t length;
    PyGILState_STATE state;

    state = PyGILState_Ensure();
    chTableDict = PyDict_GetItemString(saltoDict, "channelTables");  // borrowed
    channelTable = PyDict_GetItemString(chTableDict, chTable);  // borrowed
    if (channelTable) {
        unique = PyObject_CallMethod(channelTable, "getUnique", "(s)", name);  // new
        PyString_AsStringAndSize(unique, &buffer, (Py_ssize_t *)&length);
        s = malloc(++length);
        strlcpy(s, buffer, length);
        Py_XDECREF(unique);
    }
    PyGILState_Release(state);

    return s;
}

int setCallback(void *obj, const char *method, const char *format,
                const char *funcname, const char **exts, size_t n_exts) {
    PyObject *pluginClass, *cdll, *readFunc, *argtypes, *c_char_p, *extList;
    PyGILState_STATE state;
    int result = 0;

    state = PyGILState_Ensure();
    pluginClass = PyDict_GetItemString(saltoDict, "CPlugin");  // borrowed
    if (PyObject_IsInstance(obj, pluginClass)) {
        extList = PyList_New(n_exts);  // new
        if (extList) {
            for (int i = 0; i < n_exts; i++) {
                PyList_SET_ITEM(extList, i, PyUnicode_FromString(exts[i]));  // stolen
            }
            PyObject_CallMethod(obj, "registerFormat", "(sO)", format, extList);
            Py_DECREF(extList);
            
            cdll = PyObject_GetAttrString(obj, "cdll");  // new
            if (cdll) {
                readFunc = PyObject_GetAttrString(cdll, funcname);  // new
                if (readFunc) {
                    c_char_p = PyDict_GetItemString(mainDict, "c_char_p");  // borrowed
                    argtypes = PyTuple_Pack(2, c_char_p, c_char_p);  // new
                    PyObject_SetAttrString(readFunc, "argtypes", argtypes);
                    Py_XDECREF(argtypes);
                    PyObject_CallMethod(obj, (char *)method, "(sO)", format, readFunc);
                    Py_DECREF(readFunc);
                } else {
                    result = -1;
                }
                Py_DECREF(cdll);
            } else {
                result = -1;
            }
        } else {
            result = -1;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "initPlugin() takes a Plugin object argument");
        result = -1;
    }
    PyGILState_Release(state);

    return result;
}


static PyObject*
datetimeFromDouble(double t) {
    // Convert timespec to a Python datetime object.
    PyObject *dtClass, *datetime, *utcfromtimestamp, *dt, *empty, *keywords, *micro, *replace;
    PyGILState_STATE state;

    state = PyGILState_Ensure();
    dtClass = PyDict_GetItemString(mainDict, "datetime");  // borrowed
    utcfromtimestamp = PyObject_GetAttrString(dtClass, "utcfromtimestamp");  // new
    dt = PyObject_CallFunction(utcfromtimestamp, "(L)", (long long)t);  // new
    replace = PyObject_GetAttrString(dt, "replace");  // new
    empty = PyTuple_New(0);  // new
    keywords = PyDict_New();  // new
    micro = Py_BuildValue("l", (long)(fmod(t, 1.0) * 1000000));  // new
    if (PyDict_SetItemString(keywords, "microsecond", micro) == 0) {
        datetime = PyObject_Call(replace, empty, keywords);  // new
    } else {
        Py_INCREF(Py_None);
        datetime = Py_None;
    }
    Py_XDECREF(utcfromtimestamp);
    Py_XDECREF(dt);
    Py_XDECREF(replace);
    Py_XDECREF(empty);
    Py_XDECREF(keywords);
    Py_XDECREF(micro);
    PyGILState_Release(state);

    return datetime;
}

static PyObject*
metadata(PyObject *self, PyObject *args)
{
    PyObject *capsule, *dict = NULL;
    PyObject *start, *end, *duration;
    PyObject *length, *wordsize, *samplerate, *scale, *offset, *unit, *device, *serial, *resolution;
    PyObject *jsonClass, *loads, *json;
    Channel *ch;
    int err;
    double t, dt;
    PyGILState_STATE state;

    state = PyGILState_Ensure();
    if (PyArg_ParseTuple(args, "O:metadata", &capsule)) {
        ch = PyCapsule_GetPointer(capsule, "Channel");
        dict = PyDict_New();  // new
        if (dict) {
            length = Py_BuildValue("L", ch->length);  // new
            wordsize = Py_BuildValue("L", ch->bytes_per_sample);  // new
            samplerate = Py_BuildValue("d", ch->samplerate);  // new
            scale = Py_BuildValue("d", ch->scale);  // new
            offset = Py_BuildValue("d", ch->offset);  // new
            unit = PyUnicode_FromString(ch->unit);  // new
            t = ch->start_sec + ch->start_nsec / 1000000000.0;
            start = datetimeFromDouble(t);  // new
            dt = (ch->length - 1) / ch->samplerate;
            duration = Py_BuildValue("d", dt);  // new
            end = datetimeFromDouble(t + dt);  // new
            
            device = PyUnicode_FromString(ch->device);  // new
            serial = PyUnicode_FromString(ch->serial_no);  // new
            resolution = Py_BuildValue("i", ch->resolution);  // new

            // Parse JSON
            if (ch->json) {
                jsonClass = PyDict_GetItemString(mainDict, "json");  // borrowed
                loads = PyObject_GetAttrString(jsonClass, "loads");  // new
                json = PyObject_CallFunction(loads, "(s)", ch->json);  // new
                Py_XDECREF(jsonClass);
                Py_XDECREF(loads);
            } else {
                json = PyDict_New();  // new
            }
            
            err = (PyDict_SetItemString(dict, "length", length) != 0 ||
                   PyDict_SetItemString(dict, "wordsize", wordsize) != 0 ||
                   PyDict_SetItemString(dict, "samplerate", samplerate) != 0 ||
                   PyDict_SetItemString(dict, "scale", scale) != 0 ||
                   PyDict_SetItemString(dict, "offset", offset) != 0 ||
                   PyDict_SetItemString(dict, "unit", unit) != 0 ||
                   PyDict_SetItemString(dict, "start", start) != 0 ||
                   PyDict_SetItemString(dict, "end", end) != 0 ||
                   PyDict_SetItemString(dict, "duration", duration) != 0 ||
                   PyDict_SetItemString(dict, "device", device) != 0 ||
                   PyDict_SetItemString(dict, "serial", serial) != 0 ||
                   PyDict_SetItemString(dict, "resolution", resolution) != 0 ||
                   PyDict_SetItemString(dict, "signed", ch->is_signed ? Py_True : Py_False) != 0 ||
                   PyDict_Merge(dict, json, 0) != 0);
            if (err) {
                Py_DECREF(dict);
                dict = NULL;
            }

            Py_XDECREF(length);
            Py_XDECREF(wordsize);
            Py_XDECREF(samplerate);
            Py_XDECREF(scale);
            Py_XDECREF(offset);
            Py_XDECREF(unit);
            Py_XDECREF(start);
            Py_XDECREF(end);
            Py_XDECREF(duration);
            Py_XDECREF(device);
            Py_XDECREF(serial);
            Py_XDECREF(resolution);
            Py_XDECREF(json);
        }
    }
    PyGILState_Release(state);

    return dict;
}

static PyObject*
getDataPtr(PyObject *self, PyObject *args)
{
    PyObject *capsule = NULL;
    char *name, *chTable;
    size_t length;
    void *ptr;
    PyGILState_STATE state;

    state = PyGILState_Ensure();
    if (PyArg_ParseTuple(args, "ss:getDataPtr", &name, &chTable)) {
        ptr = getChannelData(chTable, name, &length);
        capsule = PyCapsule_New(ptr, "ChannelData", NULL);  // new
    }
    PyGILState_Release(state);

    return capsule;
}

static PyMethodDef SaltoMethods[] = {
    {"metadata", metadata, METH_VARARGS, "Return the metadata for a channel"},
    {"getDataPtr", getDataPtr, METH_VARARGS, "Get pointer to channel data"},
    {NULL, NULL, 0, NULL}  // sentinel
};


int main(int argc, const char * argv[]) {
    PyObject *saltoModule, *mainModule;

    Py_SetProgramName((char *)argv[0]);
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

    // Initialize Salto and get a reference to local dictionary.
    saltoModule = Py_InitModule("salto", SaltoMethods);
    if (saltoModule) {
        saltoDict = PyModule_GetDict(saltoModule);  // borrowed
        Py_XINCREF(saltoDict);
    } else {
        fprintf(stderr, "Python module salto not found\n");
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
    Py_Main(argc, (char **)argv);

    // Release the Python objects.
    Py_XDECREF(mainDict);
    Py_XDECREF(saltoDict);

    Py_Finalize();

    return EXIT_SUCCESS;
}
