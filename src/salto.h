//
//  salto.h
//  OpenSALTO
//
//  Copyright 2014 Marko Havu. Released under the terms of
//  GNU General Public License version 3 or later.
//

#ifndef _salto_h
#define _salto_h

#include <Python.h>
#include "Channel.h"
#include "Event.h"

PyObject *datetimeFromTimespec(PyObject *self, PyObject *args);
void *newIntegerChannel(const char *chTable, const char *name, size_t length, size_t size, int is_signed);
void *newRealChannel(const char *chTable, const char *name, size_t length, size_t size);
Channel *getChannel(const char *chTable, const char *name);
struct timespec channelEndTime(Channel *ch);
void *channelData(Channel *ch, size_t *length);
int addChannel(const char *chTable, const char *name, Channel *ch);
const char *getUniqueName(const char *chTable, const char *name);
int setCallback(void *obj, const char *type, const char *format, const char *funcname);
int saltoInit(const char *saltoPyPath, PyObject* (*guiInitFunc)(void));
int saltoRun(void);
PyObject *saltoEval(const char *expr);
void saltoEnd(void *context);

#endif
