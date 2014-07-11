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

static PyObject *saltoDict = NULL;

Channel *getChannel(const char *chTable, const char *name) {
    PyObject *channelTable, *channels, *capsule;
    void *ptr = NULL;

    channelTable = PyDict_GetItemString(saltoDict, chTable);
    if (channelTable) {
        channels = PyObject_GetAttrString(channelTable, "channels");
        capsule = PyDict_GetItemString(channels, name);
        ptr = PyCapsule_GetPointer(capsule, "Channel");
    }

    return ptr;
}

const char *getNameForData(const char *chTable, void *ptr) {
    PyObject *channelTable, *capsule, *name;
    char *s = NULL;
    
    capsule = PyCapsule_New(ptr, "ChannelData", NULL);
    channelTable = PyDict_GetItemString(saltoDict, chTable);
    if (channelTable) {
        name = PyObject_CallMethod(channelTable, "findKeyForPointer", "(O)", capsule);
        s = PyString_AsString(name);
    }

    return s;
}

int addChannel(const char *chTable, const char *name, Channel *ch) {
    PyObject *channelTable, *capsule, *success = NULL;
    Channel *channel;

    channel = (Channel *)malloc(sizeof(Channel));
    *channel = *ch;
    capsule = PyCapsule_New(channel, "Channel", NULL);
    channelTable = PyDict_GetItemString(saltoDict, chTable);
    if (channelTable) {
        success = PyObject_CallMethod(channelTable, "add", "(sO)", name, capsule);
    }

    return (success ? 0 : -1);
}

void removeChannel(const char *chTable, const char *name) {
    PyObject *channelTable;
    Channel *ch;
    
    channelTable = PyDict_GetItemString(saltoDict, chTable);
    if (channelTable) {
        ch = getChannel(chTable, name);
        PyObject_CallMethod(channelTable, "remove", "(s)", name);
        free(ch);
    }
}

const char *getUniqueName(const char *chTable, const char *name) {
    PyObject *channelTable, *unique;
    char *s = NULL;

    channelTable = PyDict_GetItemString(saltoDict, chTable);
    if (channelTable) {
        unique = PyObject_CallMethod(channelTable, "getUnique", "(s)", name);
        s = PyString_AsString(unique);
    }

    return s;
}

static PyObject*
getData(PyObject *self, PyObject *args)
{
    PyObject *capsule = NULL;
    char *name, *chTable;
    size_t length;
    void *ptr;
    
    if (PyArg_ParseTuple(args, "ss:getData", &name, &chTable)) {
        ptr = getChannelData(chTable, name, &length);
        capsule = PyCapsule_New(ptr, "ChannelData", NULL);
    }
    
    return capsule;
}

static PyObject*
readATSF(PyObject *self, PyObject *args)
{
    char *filename, *chTable;
    int errCode = -1;

    if (PyArg_ParseTuple(args, "ss:readATSF", &filename, &chTable)) {
        errCode = readFile(filename, chTable);
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
        saltoDict = PyModule_GetDict(mainModule);  // borrowed
        Py_XINCREF(saltoDict);
    } else {
        fprintf(stderr, "Python module __main__ not found\n");
        exit(EXIT_FAILURE);
    }

    // Run the python interpreter.
    Py_Main(argc, (char **)argv);

    // Release the Python objects.
    Py_XDECREF(tuple);
    Py_XDECREF(mainDict);

    Py_Finalize();

    return EXIT_SUCCESS;
}
