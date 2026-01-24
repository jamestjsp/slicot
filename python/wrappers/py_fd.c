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


extern void fd01ad(const char* jp, i32 l, f64 lambda, f64 xin, f64 yin,
                   f64* efor, f64* xf, f64* epsbck, f64* cteta, f64* steta,
                   f64* yq, f64* epos, f64* eout, f64* salph,
                   i32* iwarn, i32* info);


PyObject* py_fd01ad(PyObject* self, PyObject* args) {
    const char *jp_str;
    i32 l;
    f64 lambda, xin, yin, efor_in;
    PyObject *xf_obj, *epsbck_obj, *cteta_obj, *steta_obj, *yq_obj;

    if (!PyArg_ParseTuple(args, "siddddOOOOO",
            &jp_str, &l, &lambda, &xin, &yin, &efor_in,
            &xf_obj, &epsbck_obj, &cteta_obj, &steta_obj, &yq_obj)) {
        return NULL;
    }

    char jp = (char)toupper((unsigned char)jp_str[0]);

    if (jp != 'B' && jp != 'P') {
        PyErr_SetString(PyExc_ValueError, "jp must be 'B' or 'P'");
        return NULL;
    }

    if (l < 1) {
        PyErr_SetString(PyExc_ValueError, "l must be >= 1");
        return NULL;
    }

    if (lambda <= 0.0 || lambda > 1.0) {
        PyErr_SetString(PyExc_ValueError, "lambda must be in (0, 1]");
        return NULL;
    }

    PyArrayObject *xf_array = (PyArrayObject*)PyArray_FROM_OTF(
        xf_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!xf_array) return NULL;

    PyArrayObject *epsbck_array = (PyArrayObject*)PyArray_FROM_OTF(
        epsbck_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!epsbck_array) {
        Py_DECREF(xf_array);
        return NULL;
    }

    PyArrayObject *cteta_array = (PyArrayObject*)PyArray_FROM_OTF(
        cteta_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!cteta_array) {
        Py_DECREF(xf_array);
        Py_DECREF(epsbck_array);
        return NULL;
    }

    PyArrayObject *steta_array = (PyArrayObject*)PyArray_FROM_OTF(
        steta_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!steta_array) {
        Py_DECREF(xf_array);
        Py_DECREF(epsbck_array);
        Py_DECREF(cteta_array);
        return NULL;
    }

    PyArrayObject *yq_array = (PyArrayObject*)PyArray_FROM_OTF(
        yq_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!yq_array) {
        Py_DECREF(xf_array);
        Py_DECREF(epsbck_array);
        Py_DECREF(cteta_array);
        Py_DECREF(steta_array);
        return NULL;
    }

    if (PyArray_DIM(xf_array, 0) != l) {
        PyErr_SetString(PyExc_ValueError, "xf must have dimension l");
        Py_DECREF(xf_array);
        Py_DECREF(epsbck_array);
        Py_DECREF(cteta_array);
        Py_DECREF(steta_array);
        Py_DECREF(yq_array);
        return NULL;
    }

    if (PyArray_DIM(epsbck_array, 0) != l + 1) {
        PyErr_SetString(PyExc_ValueError, "epsbck must have dimension l+1");
        Py_DECREF(xf_array);
        Py_DECREF(epsbck_array);
        Py_DECREF(cteta_array);
        Py_DECREF(steta_array);
        Py_DECREF(yq_array);
        return NULL;
    }

    f64 *xf_data = (f64*)PyArray_DATA(xf_array);
    f64 *epsbck_data = (f64*)PyArray_DATA(epsbck_array);
    f64 *cteta_data = (f64*)PyArray_DATA(cteta_array);
    f64 *steta_data = (f64*)PyArray_DATA(steta_array);
    f64 *yq_data = (f64*)PyArray_DATA(yq_array);

    f64 *salph = (f64*)malloc(l * sizeof(f64));
    if (!salph) {
        Py_DECREF(xf_array);
        Py_DECREF(epsbck_array);
        Py_DECREF(cteta_array);
        Py_DECREF(steta_array);
        Py_DECREF(yq_array);
        PyErr_NoMemory();
        return NULL;
    }

    f64 efor = efor_in;
    f64 epos = 0.0;
    f64 eout = 0.0;
    i32 iwarn = 0;
    i32 info = 0;
    char jp_c[2] = {jp, '\0'};

    fd01ad(jp_c, l, lambda, xin, yin,
           &efor, xf_data, epsbck_data, cteta_data, steta_data,
           yq_data, &epos, &eout, salph,
           &iwarn, &info);

    PyArray_ResolveWritebackIfCopy(xf_array);
    PyArray_ResolveWritebackIfCopy(epsbck_array);
    PyArray_ResolveWritebackIfCopy(cteta_array);
    PyArray_ResolveWritebackIfCopy(steta_array);
    PyArray_ResolveWritebackIfCopy(yq_array);

    npy_intp salph_dims[1] = {l};
    PyObject *salph_array = PyArray_SimpleNew(1, salph_dims, NPY_DOUBLE);
    if (!salph_array) {
        free(salph);
        Py_DECREF(xf_array);
        Py_DECREF(epsbck_array);
        Py_DECREF(cteta_array);
        Py_DECREF(steta_array);
        Py_DECREF(yq_array);
        return NULL;
    }
    memcpy(PyArray_DATA((PyArrayObject*)salph_array), salph, l * sizeof(f64));
    free(salph);

    PyObject *result = Py_BuildValue("OOOOOdddOii",
        xf_array, epsbck_array, cteta_array, steta_array, yq_array,
        efor, epos, eout, salph_array,
        (int)iwarn, (int)info);

    Py_DECREF(xf_array);
    Py_DECREF(epsbck_array);
    Py_DECREF(cteta_array);
    Py_DECREF(steta_array);
    Py_DECREF(yq_array);
    Py_DECREF(salph_array);

    return result;
}
