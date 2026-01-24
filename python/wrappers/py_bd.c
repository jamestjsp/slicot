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

PyObject* py_bd01ad(PyObject* self, PyObject* args) {
    char* def;
    PyObject *nr_obj, *dpar_obj, *ipar_obj;
    PyArrayObject *nr_array, *dpar_array, *ipar_array;

    if (!PyArg_ParseTuple(args, "sOOO", &def, &nr_obj, &dpar_obj, &ipar_obj)) {
        return NULL;
    }

    nr_array = (PyArrayObject*)PyArray_FROM_OTF(nr_obj, NPY_INT32, NPY_ARRAY_IN_FARRAY);
    if (nr_array == NULL) return NULL;

    dpar_array = (PyArrayObject*)PyArray_FROM_OTF(dpar_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (dpar_array == NULL) {
        Py_DECREF(nr_array);
        return NULL;
    }

    ipar_array = (PyArrayObject*)PyArray_FROM_OTF(ipar_obj, NPY_INT32, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (ipar_array == NULL) {
        Py_DECREF(nr_array);
        Py_DECREF(dpar_array);
        return NULL;
    }

    i32* nr = (i32*)PyArray_DATA(nr_array);
    f64* dpar = (f64*)PyArray_DATA(dpar_array);
    i32* ipar = (i32*)PyArray_DATA(ipar_array);

    i32 nmax = 256;
    if (nr[0] == 3 || nr[0] == 4) {
        if (ipar[0] > 0) nmax = ipar[0] > 256 ? ipar[0] : 256;
    }
    if (nr[0] == 3 && nr[1] == 1 && ipar[0] > 0) {
        nmax = 2 * ipar[0] > nmax ? 2 * ipar[0] : nmax;
    }
    if (nr[0] == 4 && nr[1] == 2 && ipar[0] > 0) {
        nmax = 2 * ipar[0] > nmax ? 2 * ipar[0] : nmax;
    }

    i32 mmax = nmax;
    i32 pmax = nmax;

    i32 lde = nmax;
    i32 lda = nmax;
    i32 ldb = nmax;
    i32 ldc = pmax;
    i32 ldd = pmax;
    i32 ldwork = 4 * nmax;

    npy_intp e_dims[2] = {lde, nmax};
    npy_intp e_strides[2] = {sizeof(f64), lde * sizeof(f64)};
    PyObject* e_array = PyArray_New(&PyArray_Type, 2, e_dims, NPY_DOUBLE, e_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (e_array == NULL) {
        Py_DECREF(nr_array);
        Py_DECREF(dpar_array);
        Py_DECREF(ipar_array);
        return NULL;
    }
    f64* e_data = (f64*)PyArray_DATA((PyArrayObject*)e_array);
    memset(e_data, 0, (size_t)lde * nmax * sizeof(f64));

    npy_intp a_dims[2] = {lda, nmax};
    npy_intp a_strides[2] = {sizeof(f64), lda * sizeof(f64)};
    PyObject* a_array = PyArray_New(&PyArray_Type, 2, a_dims, NPY_DOUBLE, a_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (a_array == NULL) {
        Py_DECREF(e_array);
        Py_DECREF(nr_array);
        Py_DECREF(dpar_array);
        Py_DECREF(ipar_array);
        return NULL;
    }
    f64* a_data = (f64*)PyArray_DATA((PyArrayObject*)a_array);
    memset(a_data, 0, (size_t)lda * nmax * sizeof(f64));

    npy_intp b_dims[2] = {ldb, mmax};
    npy_intp b_strides[2] = {sizeof(f64), ldb * sizeof(f64)};
    PyObject* b_array = PyArray_New(&PyArray_Type, 2, b_dims, NPY_DOUBLE, b_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (b_array == NULL) {
        Py_DECREF(e_array);
        Py_DECREF(a_array);
        Py_DECREF(nr_array);
        Py_DECREF(dpar_array);
        Py_DECREF(ipar_array);
        return NULL;
    }
    f64* b_data = (f64*)PyArray_DATA((PyArrayObject*)b_array);
    memset(b_data, 0, (size_t)ldb * mmax * sizeof(f64));

    npy_intp c_dims[2] = {ldc, nmax};
    npy_intp c_strides[2] = {sizeof(f64), ldc * sizeof(f64)};
    PyObject* c_array = PyArray_New(&PyArray_Type, 2, c_dims, NPY_DOUBLE, c_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (c_array == NULL) {
        Py_DECREF(e_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(nr_array);
        Py_DECREF(dpar_array);
        Py_DECREF(ipar_array);
        return NULL;
    }
    f64* c_data = (f64*)PyArray_DATA((PyArrayObject*)c_array);
    memset(c_data, 0, (size_t)ldc * nmax * sizeof(f64));

    npy_intp d_dims[2] = {ldd, mmax};
    npy_intp d_strides[2] = {sizeof(f64), ldd * sizeof(f64)};
    PyObject* d_array = PyArray_New(&PyArray_Type, 2, d_dims, NPY_DOUBLE, d_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (d_array == NULL) {
        Py_DECREF(e_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(nr_array);
        Py_DECREF(dpar_array);
        Py_DECREF(ipar_array);
        return NULL;
    }
    f64* d_data = (f64*)PyArray_DATA((PyArrayObject*)d_array);
    memset(d_data, 0, (size_t)ldd * mmax * sizeof(f64));

    f64* dwork = (f64*)malloc(ldwork * sizeof(f64));
    char* note = (char*)calloc(80, sizeof(char));

    npy_intp vec_dims[1] = {8};
    PyObject* vec_array = PyArray_SimpleNew(1, vec_dims, NPY_BOOL);
    if (vec_array == NULL) {
        free(dwork); free(note);
        Py_DECREF(e_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        Py_DECREF(nr_array);
        Py_DECREF(dpar_array);
        Py_DECREF(ipar_array);
        return PyErr_NoMemory();
    }
    bool* vec = (bool*)PyArray_DATA((PyArrayObject*)vec_array);
    memset(vec, 0, 8 * sizeof(bool));

    if (!dwork || !note) {
        free(dwork); free(note);
        Py_DECREF(vec_array);
        Py_DECREF(e_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        Py_DECREF(nr_array);
        Py_DECREF(dpar_array);
        Py_DECREF(ipar_array);
        return PyErr_NoMemory();
    }

    i32 n, m, p;
    i32 info;

    bd01ad(def, nr, dpar, ipar, vec, &n, &m, &p,
           e_data, lde, a_data, lda, b_data, ldb, c_data, ldc, d_data, ldd,
           note, dwork, ldwork, &info);

    free(dwork);

    PyArray_ResolveWritebackIfCopy(dpar_array);
    PyArray_ResolveWritebackIfCopy(ipar_array);

    PyObject* note_str = PyUnicode_FromString(note);
    free(note);

    PyObject* result = Py_BuildValue("OiiiOOOOOOi",
                                     vec_array, n, m, p,
                                     e_array, a_array, b_array, c_array, d_array,
                                     note_str, info);

    Py_DECREF(vec_array);
    Py_DECREF(e_array);
    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(d_array);
    Py_DECREF(note_str);
    Py_DECREF(nr_array);
    Py_DECREF(dpar_array);
    Py_DECREF(ipar_array);

    return result;
}

PyObject* py_bd02ad(PyObject* self, PyObject* args) {
    char* def;
    PyObject *nr_obj, *dpar_obj, *ipar_obj;
    PyArrayObject *nr_array, *dpar_array, *ipar_array;

    if (!PyArg_ParseTuple(args, "sOOO", &def, &nr_obj, &dpar_obj, &ipar_obj)) {
        return NULL;
    }

    nr_array = (PyArrayObject*)PyArray_FROM_OTF(nr_obj, NPY_INT32, NPY_ARRAY_IN_FARRAY);
    if (nr_array == NULL) return NULL;

    dpar_array = (PyArrayObject*)PyArray_FROM_OTF(dpar_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (dpar_array == NULL) {
        Py_DECREF(nr_array);
        return NULL;
    }

    ipar_array = (PyArrayObject*)PyArray_FROM_OTF(ipar_obj, NPY_INT32, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (ipar_array == NULL) {
        Py_DECREF(nr_array);
        Py_DECREF(dpar_array);
        return NULL;
    }

    i32* nr = (i32*)PyArray_DATA(nr_array);
    f64* dpar = (f64*)PyArray_DATA(dpar_array);
    i32* ipar = (i32*)PyArray_DATA(ipar_array);

    i32 nmax = 256;
    if (nr[0] == 3 || nr[0] == 4) {
        if (ipar[0] > 0) nmax = ipar[0] > 256 ? ipar[0] : 256;
    }

    i32 mmax = nmax;
    i32 pmax = nmax;

    i32 lde = nmax;
    i32 lda = nmax;
    i32 ldb = nmax;
    i32 ldc = pmax;
    i32 ldd = pmax;
    i32 ldwork = 1;

    npy_intp e_dims[2] = {lde, nmax};
    npy_intp e_strides[2] = {sizeof(f64), lde * sizeof(f64)};
    PyObject* e_array = PyArray_New(&PyArray_Type, 2, e_dims, NPY_DOUBLE, e_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (e_array == NULL) {
        Py_DECREF(nr_array);
        Py_DECREF(dpar_array);
        Py_DECREF(ipar_array);
        return NULL;
    }
    f64* e_data = (f64*)PyArray_DATA((PyArrayObject*)e_array);
    memset(e_data, 0, (size_t)lde * nmax * sizeof(f64));

    npy_intp a_dims[2] = {lda, nmax};
    npy_intp a_strides[2] = {sizeof(f64), lda * sizeof(f64)};
    PyObject* a_array = PyArray_New(&PyArray_Type, 2, a_dims, NPY_DOUBLE, a_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (a_array == NULL) {
        Py_DECREF(e_array);
        Py_DECREF(nr_array);
        Py_DECREF(dpar_array);
        Py_DECREF(ipar_array);
        return NULL;
    }
    f64* a_data = (f64*)PyArray_DATA((PyArrayObject*)a_array);
    memset(a_data, 0, (size_t)lda * nmax * sizeof(f64));

    npy_intp b_dims[2] = {ldb, mmax};
    npy_intp b_strides[2] = {sizeof(f64), ldb * sizeof(f64)};
    PyObject* b_array = PyArray_New(&PyArray_Type, 2, b_dims, NPY_DOUBLE, b_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (b_array == NULL) {
        Py_DECREF(e_array);
        Py_DECREF(a_array);
        Py_DECREF(nr_array);
        Py_DECREF(dpar_array);
        Py_DECREF(ipar_array);
        return NULL;
    }
    f64* b_data = (f64*)PyArray_DATA((PyArrayObject*)b_array);
    memset(b_data, 0, (size_t)ldb * mmax * sizeof(f64));

    npy_intp c_dims[2] = {ldc, nmax};
    npy_intp c_strides[2] = {sizeof(f64), ldc * sizeof(f64)};
    PyObject* c_array = PyArray_New(&PyArray_Type, 2, c_dims, NPY_DOUBLE, c_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (c_array == NULL) {
        Py_DECREF(e_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(nr_array);
        Py_DECREF(dpar_array);
        Py_DECREF(ipar_array);
        return NULL;
    }
    f64* c_data = (f64*)PyArray_DATA((PyArrayObject*)c_array);
    memset(c_data, 0, (size_t)ldc * nmax * sizeof(f64));

    npy_intp d_dims[2] = {ldd, mmax};
    npy_intp d_strides[2] = {sizeof(f64), ldd * sizeof(f64)};
    PyObject* d_array = PyArray_New(&PyArray_Type, 2, d_dims, NPY_DOUBLE, d_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (d_array == NULL) {
        Py_DECREF(e_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(nr_array);
        Py_DECREF(dpar_array);
        Py_DECREF(ipar_array);
        return NULL;
    }
    f64* d_data = (f64*)PyArray_DATA((PyArrayObject*)d_array);
    memset(d_data, 0, (size_t)ldd * mmax * sizeof(f64));

    f64* dwork = (f64*)malloc(ldwork * sizeof(f64));
    char* note = (char*)calloc(80, sizeof(char));

    npy_intp vec_dims[1] = {8};
    PyObject* vec_array = PyArray_SimpleNew(1, vec_dims, NPY_BOOL);
    if (vec_array == NULL) {
        free(dwork); free(note);
        Py_DECREF(e_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        Py_DECREF(nr_array);
        Py_DECREF(dpar_array);
        Py_DECREF(ipar_array);
        return PyErr_NoMemory();
    }
    bool* vec = (bool*)PyArray_DATA((PyArrayObject*)vec_array);
    memset(vec, 0, 8 * sizeof(bool));

    if (!dwork || !note) {
        free(dwork); free(note);
        Py_DECREF(vec_array);
        Py_DECREF(e_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        Py_DECREF(nr_array);
        Py_DECREF(dpar_array);
        Py_DECREF(ipar_array);
        return PyErr_NoMemory();
    }

    i32 n, m, p;
    i32 info;

    bd02ad(def, nr, dpar, ipar, vec, &n, &m, &p,
           e_data, lde, a_data, lda, b_data, ldb, c_data, ldc, d_data, ldd,
           note, dwork, ldwork, &info);

    free(dwork);

    PyArray_ResolveWritebackIfCopy(dpar_array);
    PyArray_ResolveWritebackIfCopy(ipar_array);

    PyObject* note_str = PyUnicode_FromString(note);
    free(note);

    PyObject* result = Py_BuildValue("OiiiOOOOOOi",
                                     vec_array, n, m, p,
                                     e_array, a_array, b_array, c_array, d_array,
                                     note_str, info);

    Py_DECREF(vec_array);
    Py_DECREF(e_array);
    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(d_array);
    Py_DECREF(note_str);
    Py_DECREF(nr_array);
    Py_DECREF(dpar_array);
    Py_DECREF(ipar_array);

    return result;
}
