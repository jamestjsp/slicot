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

PyObject* py_bb01ad(PyObject* self, PyObject* args) {
    char* def;
    PyObject *nr_obj, *dpar_obj, *ipar_obj, *bpar_obj;
    PyArrayObject *nr_array, *dpar_array, *ipar_array, *bpar_array;

    if (!PyArg_ParseTuple(args, "sOOOO", &def, &nr_obj, &dpar_obj, &ipar_obj, &bpar_obj)) {
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

    bpar_array = (PyArrayObject*)PyArray_FROM_OTF(bpar_obj, NPY_BOOL, NPY_ARRAY_IN_FARRAY);
    if (bpar_array == NULL) {
        Py_DECREF(nr_array);
        Py_DECREF(dpar_array);
        Py_DECREF(ipar_array);
        return NULL;
    }

    i32* nr = (i32*)PyArray_DATA(nr_array);
    f64* dpar = (f64*)PyArray_DATA(dpar_array);
    i32* ipar = (i32*)PyArray_DATA(ipar_array);
    bool* bpar = (bool*)PyArray_DATA(bpar_array);

    i32 nmax = ipar[0] > 0 ? ipar[0] : 256;
    i32 mmax = (ipar[1] > 0 ? ipar[1] : nmax);
    i32 pmax = (ipar[2] > 0 ? ipar[2] : nmax);

    if (nmax < 2) nmax = 256;
    if (mmax < 1) mmax = 256;
    if (pmax < 1) pmax = 256;

    i32 lda = nmax;
    i32 ldb = nmax;
    i32 ldc = pmax;
    i32 ldg = nmax;
    i32 ldq = nmax;
    i32 ldx = nmax;
    i32 ldwork = nmax * (nmax > 4 ? nmax : 4);

    npy_intp a_dims[2] = {lda, nmax};
    npy_intp a_strides[2] = {sizeof(f64), lda * sizeof(f64)};
    PyObject* a_array = PyArray_New(&PyArray_Type, 2, a_dims, NPY_DOUBLE, a_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (a_array == NULL) {
        Py_DECREF(nr_array);
        Py_DECREF(dpar_array);
        Py_DECREF(ipar_array);
        Py_DECREF(bpar_array);
        return NULL;
    }
    f64* a = (f64*)PyArray_DATA((PyArrayObject*)a_array);
    memset(a, 0, (size_t)lda * nmax * sizeof(f64));

    npy_intp b_dims[2] = {ldb, mmax};
    npy_intp b_strides[2] = {sizeof(f64), ldb * sizeof(f64)};
    PyObject* b_array = PyArray_New(&PyArray_Type, 2, b_dims, NPY_DOUBLE, b_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (b_array == NULL) {
        Py_XDECREF(a_array);
        Py_DECREF(nr_array);
        Py_DECREF(dpar_array);
        Py_DECREF(ipar_array);
        Py_DECREF(bpar_array);
        return NULL;
    }
    f64* b = (f64*)PyArray_DATA((PyArrayObject*)b_array);
    memset(b, 0, (size_t)ldb * mmax * sizeof(f64));

    npy_intp c_dims[2] = {ldc, nmax};
    npy_intp c_strides[2] = {sizeof(f64), ldc * sizeof(f64)};
    PyObject* c_array = PyArray_New(&PyArray_Type, 2, c_dims, NPY_DOUBLE, c_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (c_array == NULL) {
        Py_XDECREF(a_array);
        Py_XDECREF(b_array);
        Py_DECREF(nr_array);
        Py_DECREF(dpar_array);
        Py_DECREF(ipar_array);
        Py_DECREF(bpar_array);
        return NULL;
    }
    f64* c = (f64*)PyArray_DATA((PyArrayObject*)c_array);
    memset(c, 0, (size_t)ldc * nmax * sizeof(f64));

    npy_intp g_dims[2] = {ldg, nmax};
    npy_intp g_strides[2] = {sizeof(f64), ldg * sizeof(f64)};
    PyObject* g_array = PyArray_New(&PyArray_Type, 2, g_dims, NPY_DOUBLE, g_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (g_array == NULL) {
        Py_XDECREF(a_array);
        Py_XDECREF(b_array);
        Py_XDECREF(c_array);
        Py_DECREF(nr_array);
        Py_DECREF(dpar_array);
        Py_DECREF(ipar_array);
        Py_DECREF(bpar_array);
        return NULL;
    }
    f64* g = (f64*)PyArray_DATA((PyArrayObject*)g_array);
    memset(g, 0, (size_t)ldg * nmax * sizeof(f64));

    npy_intp q_dims[2] = {ldq, nmax};
    npy_intp q_strides[2] = {sizeof(f64), ldq * sizeof(f64)};
    PyObject* q_array = PyArray_New(&PyArray_Type, 2, q_dims, NPY_DOUBLE, q_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (q_array == NULL) {
        Py_XDECREF(a_array);
        Py_XDECREF(b_array);
        Py_XDECREF(c_array);
        Py_XDECREF(g_array);
        Py_DECREF(nr_array);
        Py_DECREF(dpar_array);
        Py_DECREF(ipar_array);
        Py_DECREF(bpar_array);
        return NULL;
    }
    f64* q = (f64*)PyArray_DATA((PyArrayObject*)q_array);
    memset(q, 0, (size_t)ldq * nmax * sizeof(f64));

    npy_intp x_dims[2] = {ldx, nmax};
    npy_intp x_strides[2] = {sizeof(f64), ldx * sizeof(f64)};
    PyObject* x_array = PyArray_New(&PyArray_Type, 2, x_dims, NPY_DOUBLE, x_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (x_array == NULL) {
        Py_XDECREF(a_array);
        Py_XDECREF(b_array);
        Py_XDECREF(c_array);
        Py_XDECREF(g_array);
        Py_XDECREF(q_array);
        Py_DECREF(nr_array);
        Py_DECREF(dpar_array);
        Py_DECREF(ipar_array);
        Py_DECREF(bpar_array);
        return NULL;
    }
    f64* x = (f64*)PyArray_DATA((PyArrayObject*)x_array);
    memset(x, 0, (size_t)ldx * nmax * sizeof(f64));

    npy_intp vec_dims[1] = {9};
    PyObject* vec_array = PyArray_SimpleNew(1, vec_dims, NPY_BOOL);
    if (vec_array == NULL) {
        Py_XDECREF(a_array);
        Py_XDECREF(b_array);
        Py_XDECREF(c_array);
        Py_XDECREF(g_array);
        Py_XDECREF(q_array);
        Py_XDECREF(x_array);
        Py_DECREF(nr_array);
        Py_DECREF(dpar_array);
        Py_DECREF(ipar_array);
        Py_DECREF(bpar_array);
        return NULL;
    }
    bool* vec = (bool*)PyArray_DATA((PyArrayObject*)vec_array);
    memset(vec, 0, 9 * sizeof(bool));

    f64* dwork = (f64*)malloc(ldwork * sizeof(f64));
    char* chpar = (char*)calloc(256, sizeof(char));

    if (!dwork || !chpar) {
        Py_XDECREF(a_array);
        Py_XDECREF(b_array);
        Py_XDECREF(c_array);
        Py_XDECREF(g_array);
        Py_XDECREF(q_array);
        Py_XDECREF(x_array);
        Py_XDECREF(vec_array);
        free(dwork); free(chpar);
        Py_DECREF(nr_array);
        Py_DECREF(dpar_array);
        Py_DECREF(ipar_array);
        Py_DECREF(bpar_array);
        return PyErr_NoMemory();
    }

    i32 n, m, p;
    i32 info;

    bb01ad(def, nr, dpar, ipar, bpar, chpar, vec, &n, &m, &p,
           a, lda, b, ldb, c, ldc, g, ldg, q, ldq, x, ldx,
           dwork, ldwork, &info);

    free(dwork);

    PyArray_ResolveWritebackIfCopy(dpar_array);
    PyArray_ResolveWritebackIfCopy(ipar_array);


    PyObject* result = Py_BuildValue("OiiiOOOOOOi",
                                     vec_array, n, m, p,
                                     a_array, b_array, c_array, g_array, q_array, x_array,
                                     info);

    Py_DECREF(vec_array);
    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(g_array);
    Py_DECREF(q_array);
    Py_DECREF(x_array);
    Py_DECREF(nr_array);
    Py_DECREF(dpar_array);
    Py_DECREF(ipar_array);
    Py_DECREF(bpar_array);
    free(chpar);

    return result;
}

PyObject* py_bb02ad(PyObject* self, PyObject* args) {
    char* def;
    PyObject *nr_obj, *dpar_obj, *ipar_obj, *bpar_obj;
    PyArrayObject *nr_array, *dpar_array, *ipar_array, *bpar_array;

    if (!PyArg_ParseTuple(args, "sOOOO", &def, &nr_obj, &dpar_obj, &ipar_obj, &bpar_obj)) {
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

    bpar_array = (PyArrayObject*)PyArray_FROM_OTF(bpar_obj, NPY_BOOL, NPY_ARRAY_IN_FARRAY);
    if (bpar_array == NULL) {
        Py_DECREF(nr_array);
        Py_DECREF(dpar_array);
        Py_DECREF(ipar_array);
        return NULL;
    }

    i32* nr = (i32*)PyArray_DATA(nr_array);
    f64* dpar = (f64*)PyArray_DATA(dpar_array);
    i32* ipar = (i32*)PyArray_DATA(ipar_array);
    bool* bpar = (bool*)PyArray_DATA(bpar_array);

    i32 nmax = ipar[0] > 0 ? ipar[0] : 256;
    i32 mmax = (ipar[1] > 0 ? ipar[1] : nmax);
    i32 pmax = (ipar[2] > 0 ? ipar[2] : nmax);

    if (nmax < 2) nmax = 256;
    if (mmax < 1) mmax = 256;
    if (pmax < 1) pmax = 256;

    i32 lda = nmax;
    i32 ldb = nmax;
    i32 ldc = pmax;
    i32 ldq = nmax;
    i32 ldr = nmax;
    i32 lds = nmax;
    i32 ldx = nmax;
    i32 ldwork = nmax * nmax;

    f64* dwork = (f64*)malloc(ldwork * sizeof(f64));
    char* chpar = (char*)calloc(256, sizeof(char));

    npy_intp vec_dims[1] = {10};
    PyObject* vec_array = PyArray_SimpleNew(1, vec_dims, NPY_BOOL);
    if (vec_array == NULL) {
        free(dwork); free(chpar);
        Py_DECREF(nr_array);
        Py_DECREF(dpar_array);
        Py_DECREF(ipar_array);
        Py_DECREF(bpar_array);
        return NULL;
    }
    bool* vec = (bool*)PyArray_DATA((PyArrayObject*)vec_array);
    memset(vec, 0, 10 * sizeof(bool));

    if (!dwork || !chpar) {
        Py_DECREF(vec_array);
        free(dwork); free(chpar);
        Py_DECREF(nr_array);
        Py_DECREF(dpar_array);
        Py_DECREF(ipar_array);
        Py_DECREF(bpar_array);
        return PyErr_NoMemory();
    }

    npy_intp a_dims[2] = {lda, nmax};
    npy_intp a_strides[2] = {sizeof(f64), lda * sizeof(f64)};
    PyObject* a_array = PyArray_New(&PyArray_Type, 2, a_dims, NPY_DOUBLE, a_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (a_array == NULL) {
        Py_DECREF(vec_array);
        free(dwork); free(chpar);
        Py_DECREF(nr_array);
        Py_DECREF(dpar_array);
        Py_DECREF(ipar_array);
        Py_DECREF(bpar_array);
        return NULL;
    }
    f64* a = (f64*)PyArray_DATA((PyArrayObject*)a_array);
    memset(a, 0, (size_t)lda * nmax * sizeof(f64));

    npy_intp b_dims[2] = {ldb, mmax};
    npy_intp b_strides[2] = {sizeof(f64), ldb * sizeof(f64)};
    PyObject* b_array = PyArray_New(&PyArray_Type, 2, b_dims, NPY_DOUBLE, b_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (b_array == NULL) {
        Py_DECREF(vec_array); Py_DECREF(a_array);
        free(dwork); free(chpar);
        Py_DECREF(nr_array);
        Py_DECREF(dpar_array);
        Py_DECREF(ipar_array);
        Py_DECREF(bpar_array);
        return NULL;
    }
    f64* b = (f64*)PyArray_DATA((PyArrayObject*)b_array);
    memset(b, 0, (size_t)ldb * mmax * sizeof(f64));

    npy_intp c_dims[2] = {ldc, nmax};
    npy_intp c_strides[2] = {sizeof(f64), ldc * sizeof(f64)};
    PyObject* c_array = PyArray_New(&PyArray_Type, 2, c_dims, NPY_DOUBLE, c_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (c_array == NULL) {
        Py_DECREF(vec_array); Py_DECREF(a_array); Py_DECREF(b_array);
        free(dwork); free(chpar);
        Py_DECREF(nr_array);
        Py_DECREF(dpar_array);
        Py_DECREF(ipar_array);
        Py_DECREF(bpar_array);
        return NULL;
    }
    f64* c = (f64*)PyArray_DATA((PyArrayObject*)c_array);
    memset(c, 0, (size_t)ldc * nmax * sizeof(f64));

    npy_intp q_dims[2] = {ldq, nmax};
    npy_intp q_strides[2] = {sizeof(f64), ldq * sizeof(f64)};
    PyObject* q_array = PyArray_New(&PyArray_Type, 2, q_dims, NPY_DOUBLE, q_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (q_array == NULL) {
        Py_DECREF(vec_array); Py_DECREF(a_array); Py_DECREF(b_array); Py_DECREF(c_array);
        free(dwork); free(chpar);
        Py_DECREF(nr_array);
        Py_DECREF(dpar_array);
        Py_DECREF(ipar_array);
        Py_DECREF(bpar_array);
        return NULL;
    }
    f64* q = (f64*)PyArray_DATA((PyArrayObject*)q_array);
    memset(q, 0, (size_t)ldq * nmax * sizeof(f64));

    npy_intp r_dims[2] = {ldr, nmax};
    npy_intp r_strides[2] = {sizeof(f64), ldr * sizeof(f64)};
    PyObject* r_array = PyArray_New(&PyArray_Type, 2, r_dims, NPY_DOUBLE, r_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (r_array == NULL) {
        Py_DECREF(vec_array); Py_DECREF(a_array); Py_DECREF(b_array); Py_DECREF(c_array); Py_DECREF(q_array);
        free(dwork); free(chpar);
        Py_DECREF(nr_array);
        Py_DECREF(dpar_array);
        Py_DECREF(ipar_array);
        Py_DECREF(bpar_array);
        return NULL;
    }
    f64* r = (f64*)PyArray_DATA((PyArrayObject*)r_array);
    memset(r, 0, (size_t)ldr * nmax * sizeof(f64));

    npy_intp s_dims[2] = {lds, mmax};
    npy_intp s_strides[2] = {sizeof(f64), lds * sizeof(f64)};
    PyObject* s_array = PyArray_New(&PyArray_Type, 2, s_dims, NPY_DOUBLE, s_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (s_array == NULL) {
        Py_DECREF(vec_array); Py_DECREF(a_array); Py_DECREF(b_array); Py_DECREF(c_array); Py_DECREF(q_array); Py_DECREF(r_array);
        free(dwork); free(chpar);
        Py_DECREF(nr_array);
        Py_DECREF(dpar_array);
        Py_DECREF(ipar_array);
        Py_DECREF(bpar_array);
        return NULL;
    }
    f64* s = (f64*)PyArray_DATA((PyArrayObject*)s_array);
    memset(s, 0, (size_t)lds * mmax * sizeof(f64));

    npy_intp x_dims[2] = {ldx, nmax};
    npy_intp x_strides[2] = {sizeof(f64), ldx * sizeof(f64)};
    PyObject* x_array = PyArray_New(&PyArray_Type, 2, x_dims, NPY_DOUBLE, x_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (x_array == NULL) {
        Py_DECREF(vec_array); Py_DECREF(a_array); Py_DECREF(b_array); Py_DECREF(c_array); Py_DECREF(q_array); Py_DECREF(r_array); Py_DECREF(s_array);
        free(dwork); free(chpar);
        Py_DECREF(nr_array);
        Py_DECREF(dpar_array);
        Py_DECREF(ipar_array);
        Py_DECREF(bpar_array);
        return NULL;
    }
    f64* x = (f64*)PyArray_DATA((PyArrayObject*)x_array);
    memset(x, 0, (size_t)ldx * nmax * sizeof(f64));

    i32 n, m, p;
    i32 info;

    bb02ad(def, nr, dpar, ipar, bpar, chpar, vec, &n, &m, &p,
           a, lda, b, ldb, c, ldc, q, ldq, r, ldr, s, lds, x, ldx,
           dwork, ldwork, &info);

    free(dwork);

    PyArray_ResolveWritebackIfCopy(dpar_array);
    PyArray_ResolveWritebackIfCopy(ipar_array);

    // Array creation moved to top, removing this block completely
    // No changes needed here because we used `PyArray_New` earlier and didn't wrap raw pointers.
    // But we need to remove the old PyArray_New logic that wrapped `a`, `b`, etc.
    // Wait, I replaced lines 288-302.
    // I need to clear lines 321-367.

    PyObject* result = Py_BuildValue("OiiiOOOOOOOi",
                                     vec_array, n, m, p,
                                     a_array, b_array, c_array, q_array, r_array, s_array, x_array,
                                     info);

    Py_DECREF(vec_array);
    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(q_array);
    Py_DECREF(r_array);
    Py_DECREF(s_array);
    Py_DECREF(x_array);
    Py_DECREF(nr_array);
    Py_DECREF(dpar_array);
    Py_DECREF(ipar_array);
    Py_DECREF(bpar_array);
    free(chpar);

    return result;
}

PyObject* py_bb03ad(PyObject* self, PyObject* args) {
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

    i32 nmax = ipar[0] > 0 ? ipar[0] : 100;
    if (nr[0] == 4 && nr[1] == 4) {
        nmax = nmax * 3;
    }
    if (nmax < 2) nmax = 100;

    i32 mmax = 1;

    i32 lde = nmax;
    i32 lda = nmax;
    i32 ldy = nmax;
    i32 ldb = mmax;
    i32 ldx = nmax;
    i32 ldu = nmax;
    i32 ldwork = nmax * 2;

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
        Py_XDECREF(e_array);
        Py_DECREF(nr_array);
        Py_DECREF(dpar_array);
        Py_DECREF(ipar_array);
        return NULL;
    }
    f64* a_data = (f64*)PyArray_DATA((PyArrayObject*)a_array);
    memset(a_data, 0, (size_t)lda * nmax * sizeof(f64));

    npy_intp y_dims[2] = {ldy, nmax};
    npy_intp y_strides[2] = {sizeof(f64), ldy * sizeof(f64)};
    PyObject* y_array = PyArray_New(&PyArray_Type, 2, y_dims, NPY_DOUBLE, y_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (y_array == NULL) {
        Py_XDECREF(e_array);
        Py_XDECREF(a_array);
        Py_DECREF(nr_array);
        Py_DECREF(dpar_array);
        Py_DECREF(ipar_array);
        return NULL;
    }
    f64* y_data = (f64*)PyArray_DATA((PyArrayObject*)y_array);
    memset(y_data, 0, (size_t)ldy * nmax * sizeof(f64));

    npy_intp b_dims[2] = {ldb, nmax};
    npy_intp b_strides[2] = {sizeof(f64), ldb * sizeof(f64)};
    PyObject* b_array = PyArray_New(&PyArray_Type, 2, b_dims, NPY_DOUBLE, b_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (b_array == NULL) {
        Py_XDECREF(e_array);
        Py_XDECREF(a_array);
        Py_XDECREF(y_array);
        Py_DECREF(nr_array);
        Py_DECREF(dpar_array);
        Py_DECREF(ipar_array);
        return NULL;
    }
    f64* b_data = (f64*)PyArray_DATA((PyArrayObject*)b_array);
    memset(b_data, 0, (size_t)ldb * nmax * sizeof(f64));

    npy_intp x_dims[2] = {ldx, nmax};
    npy_intp x_strides[2] = {sizeof(f64), ldx * sizeof(f64)};
    PyObject* x_array = PyArray_New(&PyArray_Type, 2, x_dims, NPY_DOUBLE, x_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (x_array == NULL) {
        Py_XDECREF(e_array);
        Py_XDECREF(a_array);
        Py_XDECREF(y_array);
        Py_XDECREF(b_array);
        Py_DECREF(nr_array);
        Py_DECREF(dpar_array);
        Py_DECREF(ipar_array);
        return NULL;
    }
    f64* x_data = (f64*)PyArray_DATA((PyArrayObject*)x_array);
    memset(x_data, 0, (size_t)ldx * nmax * sizeof(f64));

    npy_intp u_dims[2] = {ldu, nmax};
    npy_intp u_strides[2] = {sizeof(f64), ldu * sizeof(f64)};
    PyObject* u_array = PyArray_New(&PyArray_Type, 2, u_dims, NPY_DOUBLE, u_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (u_array == NULL) {
        Py_XDECREF(e_array);
        Py_XDECREF(a_array);
        Py_XDECREF(y_array);
        Py_XDECREF(b_array);
        Py_XDECREF(x_array);
        Py_DECREF(nr_array);
        Py_DECREF(dpar_array);
        Py_DECREF(ipar_array);
        return NULL;
    }
    f64* u_data = (f64*)PyArray_DATA((PyArrayObject*)u_array);
    memset(u_data, 0, (size_t)ldu * nmax * sizeof(f64));

    npy_intp vec_dims[1] = {8};
    PyObject* vec_array = PyArray_SimpleNew(1, vec_dims, NPY_BOOL);
    if (vec_array == NULL) {
        Py_XDECREF(e_array);
        Py_XDECREF(a_array);
        Py_XDECREF(y_array);
        Py_XDECREF(b_array);
        Py_XDECREF(x_array);
        Py_XDECREF(u_array);
        Py_DECREF(nr_array);
        Py_DECREF(dpar_array);
        Py_DECREF(ipar_array);
        return NULL;
    }
    bool* vec = (bool*)PyArray_DATA((PyArrayObject*)vec_array);
    memset(vec, 0, 8 * sizeof(bool));

    f64* dwork = (f64*)malloc(ldwork * sizeof(f64));
    char* note = (char*)calloc(80, sizeof(char));

    if (!dwork || !note) {
        Py_XDECREF(e_array);
        Py_XDECREF(a_array);
        Py_XDECREF(y_array);
        Py_XDECREF(b_array);
        Py_XDECREF(x_array);
        Py_XDECREF(u_array);
        Py_XDECREF(vec_array);
        free(dwork); free(note);
        Py_DECREF(nr_array);
        Py_DECREF(dpar_array);
        Py_DECREF(ipar_array);
        return PyErr_NoMemory();
    }

    i32 n, m;
    i32 info;

    bb03ad(def, nr, dpar, ipar, vec, &n, &m,
           e_data, lde, a_data, lda, y_data, ldy, b_data, ldb, x_data, ldx, u_data, ldu,
           note, dwork, ldwork, &info);

    free(dwork);

    PyArray_ResolveWritebackIfCopy(dpar_array);
    PyArray_ResolveWritebackIfCopy(ipar_array);

    PyObject* note_str = PyUnicode_FromString(note);
    free(note);

    PyObject* result = Py_BuildValue("OiiOOOOOOOi",
                                     vec_array, n, m,
                                     e_array, a_array, y_array, b_array, x_array, u_array,
                                     note_str, info);

    Py_DECREF(vec_array);
    Py_DECREF(e_array);
    Py_DECREF(a_array);
    Py_DECREF(y_array);
    Py_DECREF(b_array);
    Py_DECREF(x_array);
    Py_DECREF(u_array);
    Py_DECREF(note_str);
    Py_DECREF(nr_array);
    Py_DECREF(dpar_array);
    Py_DECREF(ipar_array);

    return result;
}

PyObject* py_bb04ad(PyObject* self, PyObject* args) {
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

    i32 nmax = ipar[0] > 0 ? ipar[0] : 100;
    if (nr[0] == 4 && nr[1] == 4) {
        nmax = nmax * 3;
    }
    if (nmax < 2) nmax = 100;

    i32 mmax = 1;

    i32 lde = nmax;
    i32 lda = nmax;
    i32 ldy = nmax;
    i32 ldb = mmax;
    i32 ldx = nmax;
    i32 ldu = nmax;
    i32 ldwork = nmax * 2;

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
        Py_XDECREF(e_array);
        Py_DECREF(nr_array);
        Py_DECREF(dpar_array);
        Py_DECREF(ipar_array);
        return NULL;
    }
    f64* a_data = (f64*)PyArray_DATA((PyArrayObject*)a_array);
    memset(a_data, 0, (size_t)lda * nmax * sizeof(f64));

    npy_intp y_dims[2] = {ldy, nmax};
    npy_intp y_strides[2] = {sizeof(f64), ldy * sizeof(f64)};
    PyObject* y_array = PyArray_New(&PyArray_Type, 2, y_dims, NPY_DOUBLE, y_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (y_array == NULL) {
        Py_XDECREF(e_array);
        Py_XDECREF(a_array);
        Py_DECREF(nr_array);
        Py_DECREF(dpar_array);
        Py_DECREF(ipar_array);
        return NULL;
    }
    f64* y_data = (f64*)PyArray_DATA((PyArrayObject*)y_array);
    memset(y_data, 0, (size_t)ldy * nmax * sizeof(f64));

    npy_intp b_dims[2] = {ldb, nmax};
    npy_intp b_strides[2] = {sizeof(f64), ldb * sizeof(f64)};
    PyObject* b_array = PyArray_New(&PyArray_Type, 2, b_dims, NPY_DOUBLE, b_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (b_array == NULL) {
        Py_XDECREF(e_array);
        Py_XDECREF(a_array);
        Py_XDECREF(y_array);
        Py_DECREF(nr_array);
        Py_DECREF(dpar_array);
        Py_DECREF(ipar_array);
        return NULL;
    }
    f64* b_data = (f64*)PyArray_DATA((PyArrayObject*)b_array);
    memset(b_data, 0, (size_t)ldb * nmax * sizeof(f64));

    npy_intp x_dims[2] = {ldx, nmax};
    npy_intp x_strides[2] = {sizeof(f64), ldx * sizeof(f64)};
    PyObject* x_array = PyArray_New(&PyArray_Type, 2, x_dims, NPY_DOUBLE, x_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (x_array == NULL) {
        Py_XDECREF(e_array);
        Py_XDECREF(a_array);
        Py_XDECREF(y_array);
        Py_XDECREF(b_array);
        Py_DECREF(nr_array);
        Py_DECREF(dpar_array);
        Py_DECREF(ipar_array);
        return NULL;
    }
    f64* x_data = (f64*)PyArray_DATA((PyArrayObject*)x_array);
    memset(x_data, 0, (size_t)ldx * nmax * sizeof(f64));

    npy_intp u_dims[2] = {ldu, nmax};
    npy_intp u_strides[2] = {sizeof(f64), ldu * sizeof(f64)};
    PyObject* u_array = PyArray_New(&PyArray_Type, 2, u_dims, NPY_DOUBLE, u_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (u_array == NULL) {
        Py_XDECREF(e_array);
        Py_XDECREF(a_array);
        Py_XDECREF(y_array);
        Py_XDECREF(b_array);
        Py_XDECREF(x_array);
        Py_DECREF(nr_array);
        Py_DECREF(dpar_array);
        Py_DECREF(ipar_array);
        return NULL;
    }
    f64* u_data = (f64*)PyArray_DATA((PyArrayObject*)u_array);
    memset(u_data, 0, (size_t)ldu * nmax * sizeof(f64));

    npy_intp vec_dims[1] = {8};
    PyObject* vec_array = PyArray_SimpleNew(1, vec_dims, NPY_BOOL);
    if (vec_array == NULL) {
        Py_XDECREF(e_array);
        Py_XDECREF(a_array);
        Py_XDECREF(y_array);
        Py_XDECREF(b_array);
        Py_XDECREF(x_array);
        Py_XDECREF(u_array);
        Py_DECREF(nr_array);
        Py_DECREF(dpar_array);
        Py_DECREF(ipar_array);
        return NULL;
    }
    bool* vec = (bool*)PyArray_DATA((PyArrayObject*)vec_array);
    memset(vec, 0, 8 * sizeof(bool));

    f64* dwork = (f64*)malloc(ldwork * sizeof(f64));
    char* note = (char*)calloc(80, sizeof(char));

    if (!dwork || !note) {
        Py_XDECREF(e_array);
        Py_XDECREF(a_array);
        Py_XDECREF(y_array);
        Py_XDECREF(b_array);
        Py_XDECREF(x_array);
        Py_XDECREF(u_array);
        Py_XDECREF(vec_array);
        free(dwork); free(note);
        Py_DECREF(nr_array);
        Py_DECREF(dpar_array);
        Py_DECREF(ipar_array);
        return PyErr_NoMemory();
    }

    i32 n, m;
    i32 info;

    bb04ad(def, nr, dpar, ipar, vec, &n, &m,
           e_data, lde, a_data, lda, y_data, ldy, b_data, ldb, x_data, ldx, u_data, ldu,
           note, dwork, ldwork, &info);

    free(dwork);

    PyArray_ResolveWritebackIfCopy(dpar_array);
    PyArray_ResolveWritebackIfCopy(ipar_array);

    PyObject* note_str = PyUnicode_FromString(note);
    free(note);

    PyObject* result = Py_BuildValue("OiiOOOOOOOi",
                                     vec_array, n, m,
                                     e_array, a_array, y_array, b_array, x_array, u_array,
                                     note_str, info);

    Py_DECREF(vec_array);
    Py_DECREF(e_array);
    Py_DECREF(a_array);
    Py_DECREF(y_array);
    Py_DECREF(b_array);
    Py_DECREF(x_array);
    Py_DECREF(u_array);
    Py_DECREF(note_str);
    Py_DECREF(nr_array);
    Py_DECREF(dpar_array);
    Py_DECREF(ipar_array);

    return result;
}
