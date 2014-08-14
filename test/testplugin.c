//
//  plugintest.c
//  OpenSALTO
//
//  Tests the OpenSALTO API functions.
//
//  Created by Marko Havu on 2014-08-13.
//  Released under the terms of GNU General Public License version 3.
//

#include <stdio.h>
#include <stdlib.h>
#include "salto_api.h"


int readFile(const char *filename, const char *chTable) {
    // TODO: Add calls to API functions.
    
    return 0;
}

int writeFile(const char *filename, const char *chTable) {
    // TODO: Add calls to API functions.

    return 0;
}

const char *describeError(int err) {
    static char str[12];

    snprintf(str, 12, "error %5d", err);

    return str;
}

int initPlugin(void *handle) {
    const char *name = "Plugin test";
    const char *exts[2] = {".none", ".extra"};
    int err;

    err = registerFileFormat(handle, name, exts, 1);
    if (!err)
        err = registerImportFunc(handle, name, "readFile");
    if (!err)
        err = registerExportFunc(handle, name, "writeFile");
   
    return err;
}