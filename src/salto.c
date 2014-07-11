//
//  salto.c
//  OpenSALTO
//
//  Created by Marko Havu on 2014-07-09.
//  Released under the terms of GNU General Public License version 3.
//

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "salto_api.h"
#include "salto.h"
#ifdef __APPLE__
#include <Python/Python.h>
#else
#include <Python.h>
#endif

static PyObject *channelTable = NULL;

Channel *getChannel(const char *name) {
    PyObject *channels, *capsule;

    channels = PyObject_GetAttrString(channelTable, "channels");
    capsule = PyDict_GetItemString(channels, name);

    return PyCapsule_GetPointer(capsule, "Channel");
}

const char *getNameForData(void *ptr) {
    PyObject *capsule, *name;
    
    capsule = PyCapsule_New(ptr, "ChannelData", NULL);
    name = PyObject_CallMethod(channelTable, "findKeyForPointer", "(O)", capsule);

    return PyString_AsString(name);
}

int addChannel(const char *name, Channel *ch) {
    PyObject *success, *capsule;
    Channel *channel;

    channel = (Channel *)malloc(sizeof(Channel));
    *channel = *ch;
    capsule = PyCapsule_New(channel, "Channel", NULL);
    success = PyObject_CallMethod(channelTable, "add", "(sO)", name, capsule);

    return (success ? 0 : -1);
}

void removeChannel(const char *name) {
    Channel *ch = getChannel(name);
    PyObject_CallMethod(channelTable, "remove", "(s)", name);
    free(ch);
}

const char *getUniqueName(const char *name) {
    PyObject *unique = PyObject_CallMethod(channelTable, "getUnique", "(s)", name);

    return PyString_AsString(unique);
}

static PyObject*
getData(PyObject *self, PyObject *args)
{
    PyObject *capsule = NULL;
    char *name;
    
    if (PyArg_ParseTuple(args, "s:getData", &name)) {
        capsule = PyCapsule_New(getChannelData(name), "ChannelData", NULL);
    }
    
    return capsule;
}

static PyObject*
readATSF(PyObject *self, PyObject *args)
{
    char *filename;
    int errCode = -1;

    if (PyArg_ParseTuple(args, "s:readATSF", &filename)) {
        errCode = readFile(filename);
    }

    return Py_BuildValue("i", errCode);
}

static PyMethodDef SaltoMethods[] = {
    {"getData", getData, METH_VARARGS, "Get pointer to channel data."},
    {"readFile", readATSF, METH_VARARGS, "Open ATSF data file."},
    {NULL, NULL, 0, NULL}  // sentinel
};


int main(int argc, const char * argv[]) {
    PyObject *mainDict = NULL, *tuple = NULL;

    Py_SetProgramName((char *)argv[0]);
    Py_Initialize();
    Py_InitModule("salto", SaltoMethods);
    FILE *fp = fopen("salto.py", "r");
    if (fp) {
        PyRun_SimpleFileEx(fp, "salto.py", 1);
    } else {
        perror("fopen()");
        exit(EXIT_FAILURE);
    }

    // Get a reference to the Python global dictionary.
    PyObject *mainModule = PyImport_AddModule("__main__");  // borrowed
    if (mainModule) {
        mainDict = PyModule_GetDict(mainModule);  // borrowed
        Py_XINCREF(mainDict);
    } else {
        fprintf(stderr, "Python module __main__ not found\n");
        exit(EXIT_FAILURE);
    }

    // Run the python interpreter.
    channelTable = PyDict_GetItemString(mainDict, "channelTable");   // borrowed
    if (channelTable) {
        Py_XINCREF(channelTable);
        Py_Main(argc, (char **)argv);
    } else {
        fprintf(stderr, "Channel table not found\n");
        exit(EXIT_FAILURE);
    }

    // Release the Python objects.
    Py_XDECREF(channelTable);
    Py_XDECREF(tuple);
    Py_XDECREF(mainDict);

    Py_Finalize();

    return EXIT_SUCCESS;
}
