/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#define PY_SSIZE_T_CLEAN
/* Define unique symbol for NumPy Array API to avoid multiple static definitions */
#define PY_ARRAY_UNIQUE_SYMBOL SLICOT_ARRAY_API
#define NO_IMPORT_ARRAY

#include "py_wrappers.h"
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <ctype.h>



/* Python wrapper for de01od */
PyObject* py_de01od(PyObject* self, PyObject* args) {
    const char *conv_str;
    PyObject *a_obj, *b_obj;
    PyArrayObject *a_array, *b_array;

    if (!PyArg_ParseTuple(args, "sOO", &conv_str, &a_obj, &b_obj)) {
        return NULL;
    }

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                                                NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (a_array == NULL) {
        return NULL;
    }

    b_array = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_DOUBLE,
                                                NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (b_array == NULL) {
        Py_DECREF(a_array);
        return NULL;
    }

    i32 n = (i32)PyArray_SIZE(a_array);
    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);
    i32 info;

    de01od(conv_str, n, a_data, b_data, &info);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    PyObject *result = Py_BuildValue("Oi", a_array, info);
    Py_DECREF(a_array);
    Py_DECREF(b_array);
    return result;
}


/* Python wrapper for de01pd */
PyObject* py_de01pd(PyObject* self, PyObject* args) {
    const char *conv_str, *wght_str;
    PyObject *a_obj, *b_obj, *w_obj;
    PyArrayObject *a_array, *b_array, *w_array;

    if (!PyArg_ParseTuple(args, "ssOOO", &conv_str, &wght_str, &a_obj, &b_obj, &w_obj)) {
        return NULL;
    }

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                                                NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (a_array == NULL) {
        return NULL;
    }

    b_array = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_DOUBLE,
                                                NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (b_array == NULL) {
        Py_DECREF(a_array);
        return NULL;
    }

    w_array = (PyArrayObject*)PyArray_FROM_OTF(w_obj, NPY_DOUBLE,
                                                NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (w_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        return NULL;
    }

    i32 n = (i32)PyArray_SIZE(a_array);
    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);
    f64 *w_data = (f64*)PyArray_DATA(w_array);
    i32 info;

    de01pd(conv_str, wght_str, n, a_data, b_data, w_data, &info);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(b_array);
    PyArray_ResolveWritebackIfCopy(w_array);
    PyObject *result = Py_BuildValue("OOi", a_array, w_array, info);
    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(w_array);
    return result;
}
