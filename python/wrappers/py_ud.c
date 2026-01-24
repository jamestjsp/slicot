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


PyObject* py_ud01bd(PyObject* self, PyObject* args) {
    i32 mp, np_dim, dp;
    PyObject *data_obj;
    PyArrayObject *data_array = NULL;
    PyArrayObject *p_array = NULL;

    if (!PyArg_ParseTuple(args, "iiiO", &mp, &np_dim, &dp, &data_obj)) {
        return NULL;
    }

    data_array = (PyArrayObject*)PyArray_FROM_OTF(data_obj, NPY_DOUBLE,
                                                   NPY_ARRAY_IN_FARRAY);
    if (data_array == NULL) {
        return NULL;
    }

    i32 info = 0;

    if (mp < 1) {
        info = -1;
        Py_DECREF(data_array);
        Py_INCREF(Py_None);
        return Py_BuildValue("Oi", Py_None, info);
    }
    if (np_dim < 1) {
        info = -2;
        Py_DECREF(data_array);
        Py_INCREF(Py_None);
        return Py_BuildValue("Oi", Py_None, info);
    }
    if (dp < 0) {
        info = -3;
        Py_DECREF(data_array);
        Py_INCREF(Py_None);
        return Py_BuildValue("Oi", Py_None, info);
    }

    i32 ldp1 = mp;
    i32 ldp2 = np_dim;

    npy_intp dims[3] = {mp, np_dim, dp + 1};
    npy_intp strides[3] = {sizeof(f64), mp * sizeof(f64), mp * np_dim * sizeof(f64)};
    p_array = (PyArrayObject*)PyArray_New(&PyArray_Type, 3, dims, NPY_DOUBLE,
                                          strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (p_array == NULL) {
        Py_DECREF(data_array);
        return NULL;
    }

    f64 *data = (f64*)PyArray_DATA(data_array);
    f64 *p = (f64*)PyArray_DATA(p_array);

    ud01bd(mp, np_dim, dp, data, p, ldp1, ldp2, &info);

    Py_DECREF(data_array);

    if (info != 0) {
        Py_DECREF(p_array);
        Py_INCREF(Py_None);
        return Py_BuildValue("Oi", Py_None, info);
    }

    PyObject *result = Py_BuildValue("Oi", p_array, info);
    Py_DECREF(p_array);
    return result;
}


PyObject* py_ud01dd(PyObject* self, PyObject* args) {
    i32 m, n;
    PyObject *rows_obj, *cols_obj, *vals_obj;
    PyArrayObject *rows_array = NULL;
    PyArrayObject *cols_array = NULL;
    PyArrayObject *vals_array = NULL;
    PyArrayObject *a_array = NULL;

    if (!PyArg_ParseTuple(args, "iiOOO", &m, &n, &rows_obj, &cols_obj, &vals_obj)) {
        return NULL;
    }

    rows_array = (PyArrayObject*)PyArray_FROM_OTF(rows_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    if (rows_array == NULL) {
        return NULL;
    }

    cols_array = (PyArrayObject*)PyArray_FROM_OTF(cols_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    if (cols_array == NULL) {
        Py_DECREF(rows_array);
        return NULL;
    }

    vals_array = (PyArrayObject*)PyArray_FROM_OTF(vals_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (vals_array == NULL) {
        Py_DECREF(rows_array);
        Py_DECREF(cols_array);
        return NULL;
    }

    i32 info = 0;
    npy_intp nnz = PyArray_SIZE(rows_array);

    if (m < 0) {
        info = -1;
        Py_DECREF(rows_array);
        Py_DECREF(cols_array);
        Py_DECREF(vals_array);
        Py_INCREF(Py_None);
        return Py_BuildValue("Oi", Py_None, info);
    }
    if (n < 0) {
        info = -2;
        Py_DECREF(rows_array);
        Py_DECREF(cols_array);
        Py_DECREF(vals_array);
        Py_INCREF(Py_None);
        return Py_BuildValue("Oi", Py_None, info);
    }

    i32 lda = m > 1 ? m : 1;

    npy_intp dims[2] = {m, n};
    npy_intp strides[2] = {sizeof(f64), lda * sizeof(f64)};
    a_array = (PyArrayObject*)PyArray_New(&PyArray_Type, 2, dims, NPY_DOUBLE,
                                          strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (a_array == NULL) {
        Py_DECREF(rows_array);
        Py_DECREF(cols_array);
        Py_DECREF(vals_array);
        return NULL;
    }

    i32 *rows = (i32*)PyArray_DATA(rows_array);
    i32 *cols = (i32*)PyArray_DATA(cols_array);
    f64 *vals = (f64*)PyArray_DATA(vals_array);
    f64 *a = (f64*)PyArray_DATA(a_array);

    ud01dd(m, n, (i32)nnz, rows, cols, vals, a, lda, &info);

    Py_DECREF(rows_array);
    Py_DECREF(cols_array);
    Py_DECREF(vals_array);

    PyObject *result = Py_BuildValue("Oi", a_array, info);
    Py_DECREF(a_array);
    return result;
}


PyObject* py_ud01md(PyObject* self, PyObject* args) {
    i32 m, n, l;
    PyObject *a_obj;
    const char *text;
    PyArrayObject *a_array = NULL;

    if (!PyArg_ParseTuple(args, "iiiOs", &m, &n, &l, &a_obj, &text)) {
        return NULL;
    }

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                                                NPY_ARRAY_IN_FARRAY);
    if (a_array == NULL) {
        return NULL;
    }

    i32 info = 0;

    if (m < 1) {
        info = -1;
        Py_DECREF(a_array);
        return Py_BuildValue("si", "", info);
    }
    if (n < 1) {
        info = -2;
        Py_DECREF(a_array);
        return Py_BuildValue("si", "", info);
    }
    if (l < 1 || l > 5) {
        info = -3;
        Py_DECREF(a_array);
        return Py_BuildValue("si", "", info);
    }

    npy_intp *dims = PyArray_DIMS(a_array);
    i32 lda = (i32)dims[0];

    if (lda < m) {
        info = -6;
        Py_DECREF(a_array);
        return Py_BuildValue("si", "", info);
    }

    i32 output_size = 1024 + m * n * 20 + m * (n / l + 1) * 100;
    char *output = (char *)PyMem_Malloc((size_t)output_size);
    if (output == NULL) {
        Py_DECREF(a_array);
        return PyErr_NoMemory();
    }
    output[0] = '\0';

    f64 *a = (f64*)PyArray_DATA(a_array);

    info = ud01md(m, n, l, a, lda, text, output, output_size);

    Py_DECREF(a_array);

    PyObject *result = Py_BuildValue("si", output, info);
    PyMem_Free(output);
    return result;
}


PyObject* py_ud01cd(PyObject* self, PyObject* args) {
    i32 mp, np_dim, dp;
    PyObject *rows_obj, *cols_obj, *degrees_obj, *coeffs_obj;
    PyArrayObject *rows_array = NULL;
    PyArrayObject *cols_array = NULL;
    PyArrayObject *degrees_array = NULL;
    PyArrayObject *coeffs_array = NULL;
    PyArrayObject *p_array = NULL;

    if (!PyArg_ParseTuple(args, "iiiOOOO", &mp, &np_dim, &dp,
                          &rows_obj, &cols_obj, &degrees_obj, &coeffs_obj)) {
        return NULL;
    }

    rows_array = (PyArrayObject*)PyArray_FROM_OTF(rows_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    if (rows_array == NULL) {
        return NULL;
    }

    cols_array = (PyArrayObject*)PyArray_FROM_OTF(cols_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    if (cols_array == NULL) {
        Py_DECREF(rows_array);
        return NULL;
    }

    degrees_array = (PyArrayObject*)PyArray_FROM_OTF(degrees_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    if (degrees_array == NULL) {
        Py_DECREF(rows_array);
        Py_DECREF(cols_array);
        return NULL;
    }

    coeffs_array = (PyArrayObject*)PyArray_FROM_OTF(coeffs_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (coeffs_array == NULL) {
        Py_DECREF(rows_array);
        Py_DECREF(cols_array);
        Py_DECREF(degrees_array);
        return NULL;
    }

    i32 info = 0;
    npy_intp nelem = PyArray_SIZE(rows_array);

    if (mp < 1) {
        info = -1;
        Py_DECREF(rows_array);
        Py_DECREF(cols_array);
        Py_DECREF(degrees_array);
        Py_DECREF(coeffs_array);
        Py_INCREF(Py_None);
        return Py_BuildValue("Oi", Py_None, info);
    }
    if (np_dim < 1) {
        info = -2;
        Py_DECREF(rows_array);
        Py_DECREF(cols_array);
        Py_DECREF(degrees_array);
        Py_DECREF(coeffs_array);
        Py_INCREF(Py_None);
        return Py_BuildValue("Oi", Py_None, info);
    }
    if (dp < 0) {
        info = -3;
        Py_DECREF(rows_array);
        Py_DECREF(cols_array);
        Py_DECREF(degrees_array);
        Py_DECREF(coeffs_array);
        Py_INCREF(Py_None);
        return Py_BuildValue("Oi", Py_None, info);
    }

    i32 ldp1 = mp;
    i32 ldp2 = np_dim;

    npy_intp dims[3] = {mp, np_dim, dp + 1};
    npy_intp strides[3] = {sizeof(f64), mp * sizeof(f64), mp * np_dim * sizeof(f64)};
    p_array = (PyArrayObject*)PyArray_New(&PyArray_Type, 3, dims, NPY_DOUBLE,
                                          strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (p_array == NULL) {
        Py_DECREF(rows_array);
        Py_DECREF(cols_array);
        Py_DECREF(degrees_array);
        Py_DECREF(coeffs_array);
        return NULL;
    }

    i32 *rows = (i32*)PyArray_DATA(rows_array);
    i32 *cols = (i32*)PyArray_DATA(cols_array);
    i32 *degrees = (i32*)PyArray_DATA(degrees_array);
    f64 *coeffs = (f64*)PyArray_DATA(coeffs_array);
    f64 *p = (f64*)PyArray_DATA(p_array);

    ud01cd(mp, np_dim, dp, (i32)nelem, rows, cols, degrees, coeffs, p, ldp1, ldp2, &info);

    Py_DECREF(rows_array);
    Py_DECREF(cols_array);
    Py_DECREF(degrees_array);
    Py_DECREF(coeffs_array);

    PyObject *result = Py_BuildValue("Oi", p_array, info);
    Py_DECREF(p_array);
    return result;
}


PyObject* py_ud01mz(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyObject *a_obj;
    const char *text;
    i32 l = 3;

    static char *kwlist[] = {"a", "text", "l", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Os|i", kwlist,
                                     &a_obj, &text, &l)) {
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(
        a_obj, NPY_COMPLEX128, NPY_ARRAY_IN_FARRAY);
    if (a_array == NULL) {
        return NULL;
    }

    i32 ndim = PyArray_NDIM(a_array);
    if (ndim != 2) {
        PyErr_SetString(PyExc_ValueError, "a must be a 2D array");
        Py_DECREF(a_array);
        return NULL;
    }

    i32 m = (i32)PyArray_DIM(a_array, 0);
    i32 n = (i32)PyArray_DIM(a_array, 1);
    i32 lda = m;

    if (m < 1) {
        PyErr_SetString(PyExc_ValueError, "m must be >= 1 (matrix has no rows)");
        Py_DECREF(a_array);
        return NULL;
    }
    if (n < 1) {
        PyErr_SetString(PyExc_ValueError, "n must be >= 1 (matrix has no columns)");
        Py_DECREF(a_array);
        return NULL;
    }
    if (l < 1 || l > 3) {
        PyErr_SetString(PyExc_ValueError, "l must be between 1 and 3");
        Py_DECREF(a_array);
        return NULL;
    }

    c128 *a = (c128*)PyArray_DATA(a_array);

    i32 output_size = 256 + m * n * 80 + ((n + l - 1) / l) * (m + 2) * 100;
    char *output = (char*)PyMem_Malloc((size_t)output_size);
    if (output == NULL) {
        Py_DECREF(a_array);
        return PyErr_NoMemory();
    }

    i32 info = ud01mz(m, n, l, a, lda, text, output, output_size);

    Py_DECREF(a_array);

    if (info != 0) {
        PyMem_Free(output);
        return Py_BuildValue("si", "", info);
    }

    PyObject *result = Py_BuildValue("si", output, info);
    PyMem_Free(output);
    return result;
}


PyObject* py_ud01nd(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyObject *p_obj;
    const char *text;
    i32 mp, np_dim, dp, l;

    static char *kwlist[] = {"mp", "np", "dp", "l", "p", "text", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "iiiiOs", kwlist,
                                     &mp, &np_dim, &dp, &l, &p_obj, &text)) {
        return NULL;
    }

    PyArrayObject *p_array = (PyArrayObject*)PyArray_FROM_OTF(
        p_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (p_array == NULL) {
        return NULL;
    }

    i32 ndim = PyArray_NDIM(p_array);
    if (ndim != 3) {
        PyErr_SetString(PyExc_ValueError, "p must be a 3D array");
        Py_DECREF(p_array);
        return NULL;
    }

    i32 ldp1 = (i32)PyArray_DIM(p_array, 0);
    i32 ldp2 = (i32)PyArray_DIM(p_array, 1);

    if (mp < 1) {
        Py_DECREF(p_array);
        return Py_BuildValue("si", "", -1);
    }
    if (np_dim < 1) {
        Py_DECREF(p_array);
        return Py_BuildValue("si", "", -2);
    }
    if (dp < 0) {
        Py_DECREF(p_array);
        return Py_BuildValue("si", "", -3);
    }
    if (l < 1 || l > 5) {
        Py_DECREF(p_array);
        return Py_BuildValue("si", "", -4);
    }
    if (ldp1 < mp) {
        Py_DECREF(p_array);
        return Py_BuildValue("si", "", -6);
    }
    if (ldp2 < np_dim) {
        Py_DECREF(p_array);
        return Py_BuildValue("si", "", -7);
    }

    f64 *p = (f64*)PyArray_DATA(p_array);

    i32 output_size = 256 + (dp + 1) * (mp * np_dim * 80 + ((np_dim + l - 1) / l) * (mp + 2) * 100);
    char *output = (char*)PyMem_Malloc((size_t)output_size);
    if (output == NULL) {
        Py_DECREF(p_array);
        return PyErr_NoMemory();
    }

    i32 info = ud01nd(mp, np_dim, dp, l, p, ldp1, ldp2, text, output, output_size);

    Py_DECREF(p_array);

    if (info != 0 && info != -100) {
        PyMem_Free(output);
        return Py_BuildValue("si", "", info);
    }

    PyObject *result = Py_BuildValue("si", output, info);
    PyMem_Free(output);
    return result;
}
