//
//  plugintest.c
//  OpenSALTO
//
//  Tests the OpenSALTO API functions.
//
//  Copyright 2014 Marko Havu. Released under the terms of
//  GNU General Public License version 3 or later.
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "salto_api.h"

typedef enum {
    SUCCESS = 0,
    R_FILEFORMAT_FAILED,
    R_IMPORTFUNC_FAILED,
    R_EXPORTFUNC_FAILED,
    R_COMPUTATION_FAILED
} Error;

typedef struct {
    const char *chTable;
    unsigned short arg1;
    const char *arg2;
    double arg3;
} ComputeInputs;

typedef struct {
    const char *chTable;
    float arg1;
    const char *arg2;
} ComputeOutputs;

int readFile(const char *filename, const char *chTable) {
    // TODO: Add calls to API functions.
    
    return 0;
}

int writeFile(const char *filename, const char *chTable) {
    // TODO: Add calls to API functions.

    return 0;
}

int compute(void *inputs, void *outputs) {
    const char *chTable;
    const char **chNames;
    size_t size, i;
    ComputeInputs inputsCopy;

    inputsCopy = *(ComputeInputs *)inputs;
    chTable = newChannelTable("computed");
    chNames = getChannelNames(((ComputeInputs *)inputs)->chTable, &size);
    for (i = 0; i < size; i++) {
        copyChannel(((ComputeInputs *)inputs)->chTable, chNames[i], chTable);
    }
    ((ComputeOutputs *)outputs)->chTable = chTable;
    ((ComputeOutputs *)outputs)->arg1 = ((ComputeInputs *)inputs)->arg3;
    ((ComputeOutputs *)outputs)->arg2 = ((ComputeInputs *)inputs)->arg2;

    return 0;
}

size_t nOutputChannels(const char* computation, size_t nInputChannels) {
    size_t result = 0;

    if (strcmp(computation, "compute") == 0) {
        result = nInputChannels;
    }

    return result;
}

const char *describeError(int err) {
    const char *str;

    switch (err) {
        case R_FILEFORMAT_FAILED:
            str = "Plugin test: registerFileFormat() failed";
            break;
        case R_IMPORTFUNC_FAILED:
            str = "Plugin test: registerImportFunc() failed";
            break;
        case R_EXPORTFUNC_FAILED:
            str = "Plugin test: registerExportFunc() failed";
            break;
        case R_COMPUTATION_FAILED:
            str = "Plugin test: registerComputation() failed";
            break;
        default:
            str = "Plugin test: unknown error code";
            break;
    }

    return str;
}

int initPlugin(void *handle) {
    const char *name = "Plugin test";
    const char *exts[2] = {".none", ".extra"};
    int err = SUCCESS;
    const char *iNames[3] = {"iarg1", "iarg2", "iarg3"};
    const char *oNames[2] = {"oarg1", "oarg2"};
    const char *iFormats[3] = {"u2", "S", "f8"};
    const char *oFormats[2] = {"f4", "S"};
    const char *iDescr[3] = {"unsigned short", "C string", "double"};
    const char *oDescr[2] = {"float", "C string"};
    void *iDefaults[3] = {NULL, NULL, NULL};
    void *oDefaults[2] = {NULL, NULL};
    ComputationArgs inputs = {
        .n_args = 3,
        .min_channels = 1,
        .max_channels = 3,
        .name = iNames,
        .format = iFormats,
        .description = iDescr,
        .default_value = iDefaults
    };
    ComputationArgs outputs = {
        .n_args = 2,
        .min_channels = 1,
        .max_channels = 3,
        .name = oNames,
        .format = oFormats,
        .description = oDescr,
        .default_value = oDefaults
    };

    if (registerFileFormat(handle, name, exts, 1) != 0)
        err = R_FILEFORMAT_FAILED;
    if (registerImportFunc(handle, name, "readFile") != 0)
        err = R_IMPORTFUNC_FAILED;
    if (registerExportFunc(handle, name, "writeFile") != 0)
        err = R_EXPORTFUNC_FAILED;
    if (registerComputation(handle, name, "compute", &inputs, &outputs) != 0)
        err = R_COMPUTATION_FAILED;
    
    return err;
}