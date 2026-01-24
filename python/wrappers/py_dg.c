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



/* Python wrapper for dg01md */
PyObject* py_dg01md(PyObject* self, PyObject* args) {
    const char *indi_str;
    PyObject *xr_obj, *xi_obj;
    PyArrayObject *xr_array, *xi_array;

    if (!PyArg_ParseTuple(args, "sOO", &indi_str, &xr_obj, &xi_obj)) {
        return NULL;
    }

    xr_array = (PyArrayObject*)PyArray_FROM_OTF(xr_obj, NPY_DOUBLE,
                                                NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (xr_array == NULL) {
        return NULL;
    }

    xi_array = (PyArrayObject*)PyArray_FROM_OTF(xi_obj, NPY_DOUBLE,
                                                NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (xi_array == NULL) {
        Py_DECREF(xr_array);
        return NULL;
    }

    i32 n = (i32)PyArray_SIZE(xr_array);
    f64 *xr_data = (f64*)PyArray_DATA(xr_array);
    f64 *xi_data = (f64*)PyArray_DATA(xi_array);
    i32 info;

    dg01md(indi_str, n, xr_data, xi_data, &info);

    PyArray_ResolveWritebackIfCopy(xr_array);
    PyArray_ResolveWritebackIfCopy(xi_array);
    PyObject *result = Py_BuildValue("OOi", xr_array, xi_array, info);
    Py_DECREF(xr_array);
    Py_DECREF(xi_array);
    return result;
}



/* Python wrapper for dg01nd */
PyObject* py_dg01nd(PyObject* self, PyObject* args) {
    const char *indi_str;
    PyObject *xr_obj, *xi_obj;
    PyArrayObject *xr_array, *xi_array;

    if (!PyArg_ParseTuple(args, "sOO", &indi_str, &xr_obj, &xi_obj)) {
        return NULL;
    }

    xr_array = (PyArrayObject*)PyArray_FROM_OTF(xr_obj, NPY_DOUBLE,
                                                NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (xr_array == NULL) {
        return NULL;
    }

    xi_array = (PyArrayObject*)PyArray_FROM_OTF(xi_obj, NPY_DOUBLE,
                                                NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (xi_array == NULL) {
        Py_DECREF(xr_array);
        return NULL;
    }

    char ind = (char)toupper((unsigned char)indi_str[0]);
    i32 array_size = (i32)PyArray_SIZE(xr_array);
    i32 n = (ind == 'D') ? array_size : array_size - 1;
    i32 info;

    f64 *xr_out = (f64*)malloc((n + 1) * sizeof(f64));
    f64 *xi_out = (f64*)malloc((n + 1) * sizeof(f64));
    if (xr_out == NULL || xi_out == NULL) {
        free(xr_out);
        free(xi_out);
        Py_DECREF(xr_array);
        Py_DECREF(xi_array);
        return PyErr_NoMemory();
    }

    f64 *xr_data = (f64*)PyArray_DATA(xr_array);
    f64 *xi_data = (f64*)PyArray_DATA(xi_array);
    for (i32 i = 0; i < array_size; i++) {
        xr_out[i] = xr_data[i];
        xi_out[i] = xi_data[i];
    }
    if (ind == 'D') {
        xr_out[n] = 0.0;
        xi_out[n] = 0.0;
    }

    dg01nd(indi_str, n, xr_out, xi_out, &info);

    npy_intp out_size = (ind == 'D') ? n + 1 : n;

    npy_intp dims[1] = {out_size};
    PyObject *xr_out_array = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    PyObject *xi_out_array = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    if (xr_out_array == NULL || xi_out_array == NULL) {
        free(xr_out);
        free(xi_out);
        Py_XDECREF(xr_out_array);
        Py_XDECREF(xi_out_array);
        Py_DECREF(xr_array);
        Py_DECREF(xi_array);
        return PyErr_NoMemory();
    }

    f64 *xr_out_data = (f64*)PyArray_DATA((PyArrayObject*)xr_out_array);
    f64 *xi_out_data = (f64*)PyArray_DATA((PyArrayObject*)xi_out_array);
    for (npy_intp i = 0; i < out_size; i++) {
        xr_out_data[i] = xr_out[i];
        xi_out_data[i] = xi_out[i];
    }

    free(xr_out);
    free(xi_out);
    Py_DECREF(xr_array);
    Py_DECREF(xi_array);
    return Py_BuildValue("OOi", xr_out_array, xi_out_array, info);
}


/* Python wrapper for dg01od */
PyObject* py_dg01od(PyObject* self, PyObject* args) {
    const char *scr_str, *wght_str;
    PyObject *a_obj, *w_obj;
    PyArrayObject *a_array, *w_array;

    if (!PyArg_ParseTuple(args, "ssOO", &scr_str, &wght_str, &a_obj, &w_obj)) {
        return NULL;
    }

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                                                NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (a_array == NULL) {
        return NULL;
    }

    w_array = (PyArrayObject*)PyArray_FROM_OTF(w_obj, NPY_DOUBLE,
                                                NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (w_array == NULL) {
        Py_DECREF(a_array);
        return NULL;
    }

    i32 n = (i32)PyArray_SIZE(a_array);
    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *w_data = (f64*)PyArray_DATA(w_array);
    i32 info;

    dg01od(scr_str, wght_str, n, a_data, w_data, &info);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(w_array);
    PyObject *result = Py_BuildValue("OOi", a_array, w_array, info);
    Py_DECREF(a_array);
    Py_DECREF(w_array);
    return result;
}
