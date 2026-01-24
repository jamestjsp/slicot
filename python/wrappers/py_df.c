/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#define PY_SSIZE_T_CLEAN
#define PY_ARRAY_UNIQUE_SYMBOL SLICOT_ARRAY_API
#define NO_IMPORT_ARRAY

#include "py_wrappers.h"
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <ctype.h>


/* Python wrapper for df01md */
PyObject* py_df01md(PyObject* self, PyObject* args) {
    const char *sico_str;
    double dt;
    PyObject *a_obj;
    PyArrayObject *a_array;

    if (!PyArg_ParseTuple(args, "sdO", &sico_str, &dt, &a_obj)) {
        return NULL;
    }

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                                                NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (a_array == NULL) {
        return NULL;
    }

    i32 n = (i32)PyArray_SIZE(a_array);
    f64 *a_data = (f64*)PyArray_DATA(a_array);
    i32 info;

    df01md(sico_str, n, dt, a_data, &info);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyObject *result = Py_BuildValue("Oi", a_array, info);
    Py_DECREF(a_array);
    return result;
}
