//
//  main.c
//  OpenSalto
//
//  Copyright 2014 Marko Havu. Released under the terms of
//  GNU General Public License version 3 or later.
//

#include <stdio.h>
#include "salto.h"

int main(int argc, char *argv[]) {
    int result;

    result = saltoInit("salto.py", NULL);
    if (result == 0) {
        result = saltoRun();
        saltoEnd(NULL);
    }

    return result;
}
