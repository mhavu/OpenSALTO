//
//  saltoGui.h
//  OpenSALTO GUI
//
//  Copyright 2014 Marko Havu. Released under the terms of
//  GNU General Public License version 3 or later.
//

#include "salto.h"

PyMODINIT_FUNC PyInit_saltoGui(void);
PyObject *saltoGuiAddChannel(PyObject *pyself, PyObject *args);
PyObject *saltoGuiRemoveChannel(PyObject *pyself, PyObject *args);
PyObject *saltoGuiTerminate(PyObject *pyself);
void saltoGuiInterrupt(void);

