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

    chTableDict = PyDict_GetItemString(saltoDict, "channelTables");  // borrowed
    channelTable = PyDict_GetItemString(chTableDict, chTable);  // borrowed
    if (channelTable) {
        channels = PyObject_GetAttrString(channelTable, "channels");  // new
        capsule = PyDict_GetItemString(channels, name);  // borrowed
        ch = PyCapsule_GetPointer(capsule, "Channel");
        Py_DECREF(channels);
    }

    return ch;
}

const char *getNameForData(const char *chTable, void *ptr) {
    PyObject *chTableDict, *channelTable, *capsule, *name;
    char *s = NULL;
    
    chTableDict = PyDict_GetItemString(saltoDict, "channelTables");  // borrowed
    channelTable = PyDict_GetItemString(chTableDict, chTable);  // borrowed
    if (channelTable) {
        capsule = PyCapsule_New(ptr, "ChannelData", NULL);  // new
        name = PyObject_CallMethod(channelTable, "findKeyForPointer", "(O)", capsule);  // new
        s = PyString_AsString(name);
        Py_XDECREF(name);
        Py_XDECREF(capsule);
    }

    return s;
}

int addChannel(const char *chTable, const char *name, Channel *ch) {
    PyObject *chTableDict, *channelTable, *capsule, *o = NULL;
    Channel *channel;
    int result = -1;

    channel = (Channel *)malloc(sizeof(Channel));
    *channel = *ch;
    chTableDict = PyDict_GetItemString(saltoDict, "channelTables");  // borrowed
    channelTable = PyDict_GetItemString(chTableDict, chTable);  // borrowed
    if (channelTable) {
        capsule = PyCapsule_New(channel, "Channel", NULL);  // new
        o = PyObject_CallMethod(channelTable, "add", "(sO)", name, capsule);  // new
        result = (o ? 0 : -1);
        Py_XDECREF(o);
        Py_XDECREF(capsule);
    }

    return result;
}

void removeChannel(const char *chTable, const char *name) {
    PyObject *chTableDict, *channelTable, *o;
    Channel *ch;
    
    chTableDict = PyDict_GetItemString(saltoDict, "channelTables");  // borrowed
    channelTable = PyDict_GetItemString(chTableDict, chTable);  // borrowed
    if (channelTable) {
        ch = getChannel(chTable, name);
        o = PyObject_CallMethod(channelTable, "remove", "(s)", name);  // new
        Py_XDECREF(o);
        free(ch);
    }
}

const char *getUniqueName(const char *chTable, const char *name) {
    PyObject *chTableDict, *channelTable, *unique;
    char *buffer, *s = NULL;
    size_t length;

    chTableDict = PyDict_GetItemString(saltoDict, "channelTables");  // borrowed
    channelTable = PyDict_GetItemString(chTableDict, chTable);  // borrowed
    if (channelTable) {
        unique = PyObject_CallMethod(channelTable, "getUnique", "(s)", name);  // new
        PyString_AsStringAndSize(unique, &buffer, (Py_ssize_t *)&length);
        s = malloc(++length);
        strlcpy(s, buffer, length);
        Py_XDECREF(unique);
    }

    return s;
}

static PyObject*
metadata(PyObject *self, PyObject *args)
{
    PyObject *capsule, *dict = NULL;
    PyObject *dtClass, *datetime, *utcfromtimestamp, *dt, *empty, *keywords, *micro, *replace;
    PyObject *length, *wordsize, *samplerate, *scale, *offset, *unit, *device, *serial, *resolution;
    PyObject *jsonClass, *loads, *json;
    Channel *ch;
    int err;

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

            // Convert start time to Python datetime object
            dtClass = PyDict_GetItemString(mainDict, "datetime");  // borrowed
            utcfromtimestamp = PyObject_GetAttrString(dtClass, "utcfromtimestamp");  // new
            dt = PyObject_CallFunction(utcfromtimestamp, "(L)", ch->start_sec);  // new
            replace = PyObject_GetAttrString(dt, "replace");  // new
            empty = PyTuple_New(0);  // new
            keywords = PyDict_New();  // new
            micro = Py_BuildValue("l", ch->start_nsec / 1000);  // new
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
                   PyDict_SetItemString(dict, "datetime", datetime) != 0 ||
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
            Py_XDECREF(datetime);
            Py_XDECREF(device);
            Py_XDECREF(serial);
            Py_XDECREF(resolution);
            Py_XDECREF(json);
        }
    }

    return dict;
}

static PyObject*
getDataPtr(PyObject *self, PyObject *args)
{
    PyObject *capsule = NULL;
    char *name, *chTable;
    size_t length;
    void *ptr;
    
    if (PyArg_ParseTuple(args, "ss:getDataPtr", &name, &chTable)) {
        ptr = getChannelData(chTable, name, &length);
        capsule = PyCapsule_New(ptr, "ChannelData", NULL);  // new
    }
    
    return capsule;
}

static PyObject*
readATSF(PyObject *self, PyObject *args)
{
    char *filename, *chTable;
    int errCode = -1;
    PyObject *result;

    if (PyArg_ParseTuple(args, "ss:readATSF", &filename, &chTable)) {
        errCode = readFile(filename, chTable);
        if (errCode) {
            result = PyErr_Format(PyExc_IOError, "readATSF(): %s", describeError(errCode));
        } else {
            Py_INCREF(Py_None);
            result = Py_None;
        }
    } else {
        result = PyErr_Format(PyExc_SyntaxError, "readATSF expects two string arguments");
    }

    return result;
}

static PyMethodDef SaltoMethods[] = {
    {"metadata", metadata, METH_VARARGS, "Return the metadata for a channel"},
    {"getDataPtr", getDataPtr, METH_VARARGS, "Get pointer to channel data"},
    {"readFile", readATSF, METH_VARARGS, "Open ATSF data file"},
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
