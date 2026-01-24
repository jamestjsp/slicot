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



/* Python wrapper for mb05nd */
PyObject* py_mb05nd(PyObject* self, PyObject* args, PyObject* kwargs) {
    static char *kwlist[] = {"n", "delta", "a", "tol", NULL};

    i32 n;
    f64 delta, tol;
    PyObject *a_obj;
    PyArrayObject *a_array;
    i32 info;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "idOd", kwlist,
                                     &n, &delta, &a_obj, &tol)) {
        return NULL;
    }

    if (n < 0) {
        info = -1;
        npy_intp dims0[2] = {0, 0};
        npy_intp strides0[2] = {sizeof(f64), sizeof(f64)};
        PyObject *ex_array = PyArray_New(&PyArray_Type, 2, dims0, NPY_DOUBLE,
                                         strides0, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        PyObject *exint_array = PyArray_New(&PyArray_Type, 2, dims0, NPY_DOUBLE,
                                            strides0, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        return Py_BuildValue("OOi", ex_array, exint_array, info);
    }

    if (n == 0) {
        info = 0;
        npy_intp dims0[2] = {0, 0};
        npy_intp strides0[2] = {sizeof(f64), sizeof(f64)};
        PyObject *ex_array = PyArray_New(&PyArray_Type, 2, dims0, NPY_DOUBLE,
                                         strides0, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        PyObject *exint_array = PyArray_New(&PyArray_Type, 2, dims0, NPY_DOUBLE,
                                            strides0, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        return Py_BuildValue("OOi", ex_array, exint_array, info);
    }

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY_RO);
    if (a_array == NULL) {
        return NULL;
    }

    if (PyArray_NDIM(a_array) != 2 || PyArray_DIM(a_array, 0) != n || PyArray_DIM(a_array, 1) != n) {
        PyErr_Format(PyExc_ValueError, "a must be %d x %d array", n, n);
        Py_DECREF(a_array);
        return NULL;
    }

    i32 lda = (i32)n;
    i32 ldex = (i32)n;
    i32 ldexin = (i32)n;
    i32 nn = n * n;
    i32 ldwork = 2 * nn;

    f64 *a_data = (f64*)PyArray_DATA(a_array);

    i32 *iwork = (i32*)malloc(n * sizeof(i32));
    f64 *dwork = (f64*)malloc(ldwork * sizeof(f64));

    npy_intp dims[2] = {n, n};
    npy_intp strides[2] = {sizeof(f64), n * sizeof(f64)};

    PyObject *ex_array = PyArray_New(&PyArray_Type, 2, dims, NPY_DOUBLE,
                                     strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    PyObject *exint_array = PyArray_New(&PyArray_Type, 2, dims, NPY_DOUBLE,
                                        strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);

    if (!iwork || !dwork || !ex_array || !exint_array) {
        free(iwork);
        free(dwork);
        Py_XDECREF(ex_array);
        Py_XDECREF(exint_array);
        Py_DECREF(a_array);
        PyErr_NoMemory();
        return NULL;
    }

    f64 *ex = (f64*)PyArray_DATA((PyArrayObject*)ex_array);
    f64 *exint = (f64*)PyArray_DATA((PyArrayObject*)exint_array);

    mb05nd(n, delta, a_data, lda, ex, ldex, exint, ldexin, tol,
           iwork, dwork, ldwork, &info);

    Py_DECREF(a_array);
    free(iwork);
    free(dwork);

    return Py_BuildValue("OOi", ex_array, exint_array, info);
}



/* Python wrapper for mb05md */
PyObject* py_mb05md(PyObject* self, PyObject* args, PyObject* kwargs) {
    static char *kwlist[] = {"balanc", "n", "delta", "a", NULL};

    const char *balanc;
    i32 n;
    f64 delta;
    PyObject *a_obj;
    PyArrayObject *a_array;
    i32 info;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "sidO", kwlist,
                                     &balanc, &n, &delta, &a_obj)) {
        return NULL;
    }

    if (n < 0) {
        info = -2;
        npy_intp dims0[2] = {0, 0};
        npy_intp strides0[2] = {sizeof(f64), sizeof(f64)};
        npy_intp dims1[1] = {0};
        PyObject *exp_array = PyArray_New(&PyArray_Type, 2, dims0, NPY_DOUBLE,
                                          strides0, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        PyObject *v_array = PyArray_New(&PyArray_Type, 2, dims0, NPY_DOUBLE,
                                        strides0, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        PyObject *y_array = PyArray_New(&PyArray_Type, 2, dims0, NPY_DOUBLE,
                                        strides0, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        PyObject *valr_array = PyArray_New(&PyArray_Type, 1, dims1, NPY_DOUBLE,
                                           NULL, NULL, 0, 0, NULL);
        PyObject *vali_array = PyArray_New(&PyArray_Type, 1, dims1, NPY_DOUBLE,
                                           NULL, NULL, 0, 0, NULL);
        return Py_BuildValue("OOOOOi", exp_array, v_array, y_array, valr_array, vali_array, info);
    }

    if (n == 0) {
        info = 0;
        npy_intp dims0[2] = {0, 0};
        npy_intp strides0[2] = {sizeof(f64), sizeof(f64)};
        npy_intp dims1[1] = {0};
        PyObject *exp_array = PyArray_New(&PyArray_Type, 2, dims0, NPY_DOUBLE,
                                          strides0, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        PyObject *v_array = PyArray_New(&PyArray_Type, 2, dims0, NPY_DOUBLE,
                                        strides0, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        PyObject *y_array = PyArray_New(&PyArray_Type, 2, dims0, NPY_DOUBLE,
                                        strides0, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        PyObject *valr_array = PyArray_New(&PyArray_Type, 1, dims1, NPY_DOUBLE,
                                           NULL, NULL, 0, 0, NULL);
        PyObject *vali_array = PyArray_New(&PyArray_Type, 1, dims1, NPY_DOUBLE,
                                           NULL, NULL, 0, 0, NULL);
        return Py_BuildValue("OOOOOi", exp_array, v_array, y_array, valr_array, vali_array, info);
    }

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                                                NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (a_array == NULL) {
        return NULL;
    }

    if (PyArray_NDIM(a_array) != 2 || PyArray_DIM(a_array, 0) != n || PyArray_DIM(a_array, 1) != n) {
        PyErr_Format(PyExc_ValueError, "a must be %d x %d array", n, n);
        Py_DECREF(a_array);
        return NULL;
    }

    i32 lda = (i32)n;
    i32 ldv = (i32)n;
    i32 ldy = (i32)n;
    i32 ldwork = 4 * n;

    f64 *a_data = (f64*)PyArray_DATA(a_array);

    i32 *iwork = (i32*)malloc(n * sizeof(i32));
    f64 *dwork = (f64*)malloc(ldwork * sizeof(f64));

    npy_intp dims[2] = {n, n};
    npy_intp strides[2] = {sizeof(f64), n * sizeof(f64)};
    npy_intp dims1[1] = {n};

    PyObject *v_array = PyArray_New(&PyArray_Type, 2, dims, NPY_DOUBLE,
                                    strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    PyObject *y_array = PyArray_New(&PyArray_Type, 2, dims, NPY_DOUBLE,
                                    strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    PyObject *valr_array = PyArray_New(&PyArray_Type, 1, dims1, NPY_DOUBLE,
                                       NULL, NULL, 0, 0, NULL);
    PyObject *vali_array = PyArray_New(&PyArray_Type, 1, dims1, NPY_DOUBLE,
                                       NULL, NULL, 0, 0, NULL);

    if (!iwork || !dwork || !v_array || !y_array || !valr_array || !vali_array) {
        free(iwork);
        free(dwork);
        Py_XDECREF(v_array);
        Py_XDECREF(y_array);
        Py_XDECREF(valr_array);
        Py_XDECREF(vali_array);
        Py_DECREF(a_array);
        PyErr_NoMemory();
        return NULL;
    }

    f64 *v = (f64*)PyArray_DATA((PyArrayObject*)v_array);
    f64 *y = (f64*)PyArray_DATA((PyArrayObject*)y_array);
    f64 *valr = (f64*)PyArray_DATA((PyArrayObject*)valr_array);
    f64 *vali = (f64*)PyArray_DATA((PyArrayObject*)vali_array);

    mb05md(balanc, n, delta, a_data, lda, v, ldv, y, ldy, valr, vali,
           iwork, dwork, ldwork, &info);

    PyArray_ResolveWritebackIfCopy(a_array);

    free(iwork);
    free(dwork);

    PyObject *exp_array = (PyObject*)a_array;

    PyObject *result = Py_BuildValue("OOOOOi", exp_array, v_array, y_array, valr_array, vali_array, info);
    Py_DECREF(a_array);
    return result;
}



/* Python wrapper for mb05my */
PyObject* py_mb05my(PyObject* self, PyObject* args) {
    const char *balanc_str;
    char balanc;
    PyObject *a_obj;
    PyArrayObject *a_array = NULL;
    f64 *wr = NULL, *wi = NULL, *r = NULL, *q = NULL;
    f64 *dwork = NULL;
    i32 info;

    if (!PyArg_ParseTuple(args, "sO", &balanc_str, &a_obj)) {
        return NULL;
    }

    balanc = balanc_str[0];

    if (balanc != 'N' && balanc != 'n' && balanc != 'S' && balanc != 's') {
        info = -1;
        npy_intp zero_dim[1] = {0};
        npy_intp zero_dim2[2] = {0, 0};
        PyArrayObject *wr_arr = (PyArrayObject*)PyArray_ZEROS(1, zero_dim, NPY_DOUBLE, 0);
        PyArrayObject *wi_arr = (PyArrayObject*)PyArray_ZEROS(1, zero_dim, NPY_DOUBLE, 0);
        PyArrayObject *r_arr = (PyArrayObject*)PyArray_ZEROS(2, zero_dim2, NPY_DOUBLE, 1);
        PyArrayObject *q_arr = (PyArrayObject*)PyArray_ZEROS(2, zero_dim2, NPY_DOUBLE, 1);
        PyArrayObject *t_arr = (PyArrayObject*)PyArray_ZEROS(2, zero_dim2, NPY_DOUBLE, 1);
        PyObject *result = Py_BuildValue("OOOOOi", wr_arr, wi_arr, r_arr, q_arr, t_arr, info);
        Py_DECREF(wr_arr);
        Py_DECREF(wi_arr);
        Py_DECREF(r_arr);
        Py_DECREF(q_arr);
        Py_DECREF(t_arr);
        return result;
    }

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (a_array == NULL) {
        return NULL;
    }

    npy_intp *a_dims = PyArray_DIMS(a_array);
    i32 n = (i32)a_dims[0];
    i32 lda = (n > 1) ? n : 1;
    i32 ldr = lda;
    i32 ldq = lda;

    npy_intp dims_n[1] = {n};
    npy_intp dims_nn[2] = {n, n};
    npy_intp strides[2] = {sizeof(f64), n * sizeof(f64)};
    if (n == 0) {
        strides[0] = sizeof(f64);
        strides[1] = sizeof(f64);
    }

    i32 ldwork = (n > 0) ? 4 * n : 1;
    dwork = (f64*)malloc(ldwork * sizeof(f64));

    PyArrayObject *r_array = (PyArrayObject*)PyArray_New(&PyArray_Type, 2, dims_nn, NPY_DOUBLE, strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    PyArrayObject *q_array = (PyArrayObject*)PyArray_New(&PyArray_Type, 2, dims_nn, NPY_DOUBLE, strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    PyArrayObject *wr_array = (PyArrayObject*)PyArray_New(&PyArray_Type, 1, dims_n, NPY_DOUBLE, NULL, NULL, 0, 0, NULL);
    PyArrayObject *wi_array = (PyArrayObject*)PyArray_New(&PyArray_Type, 1, dims_n, NPY_DOUBLE, NULL, NULL, 0, 0, NULL);

    if (dwork == NULL || r_array == NULL || q_array == NULL || wr_array == NULL || wi_array == NULL) {
        free(dwork);
        Py_XDECREF(r_array);
        Py_XDECREF(q_array);
        Py_XDECREF(wr_array);
        Py_XDECREF(wi_array);
        Py_DECREF(a_array);
        PyErr_NoMemory();
        return NULL;
    }

    r = (f64*)PyArray_DATA(r_array);
    q = (f64*)PyArray_DATA(q_array);
    wr = (f64*)PyArray_DATA(wr_array);
    wi = (f64*)PyArray_DATA(wi_array);
    f64 *a_data = (f64*)PyArray_DATA(a_array);

    mb05my(&balanc, n, a_data, lda, wr, wi, r, ldr, q, ldq, dwork, ldwork, &info);

    PyArray_ResolveWritebackIfCopy(a_array);

    PyArrayObject *t_array = (PyArrayObject*)PyArray_New(&PyArray_Type, 2, dims_nn, NPY_DOUBLE, strides, a_data, 0, NPY_ARRAY_FARRAY, NULL);

    free(dwork);

    PyObject *result = Py_BuildValue("OOOOOi", wr_array, wi_array, r_array, q_array, t_array, info);
    Py_DECREF(wr_array);
    Py_DECREF(wi_array);
    Py_DECREF(r_array);
    Py_DECREF(q_array);
    Py_DECREF(t_array);
    Py_DECREF(a_array);

    return result;
}

PyObject* py_mb05od(PyObject* self, PyObject* args)
{
    (void)self;

    char *balanc_str;
    i32 n_in, ndiag;
    f64 delta;
    PyObject *a_obj;

    if (!PyArg_ParseTuple(args, "siidO", &balanc_str, &n_in, &ndiag, &delta, &a_obj)) {
        return NULL;
    }

    i32 n = n_in;
    char balanc = balanc_str[0];

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(
        a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (a_array == NULL) {
        return NULL;
    }

    i32 lda = (n > 0) ? n : 1;
    i32 ldwork;
    if (n <= 1) {
        ldwork = 1;
    } else {
        ldwork = n * (2*n + ndiag + 1) + ndiag;
    }

    i32 *iwork = (i32*)calloc(n > 0 ? n : 1, sizeof(i32));
    f64 *dwork = (f64*)calloc(ldwork, sizeof(f64));
    if (iwork == NULL || dwork == NULL) {
        free(iwork);
        free(dwork);
        Py_DECREF(a_array);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate workspace");
        return NULL;
    }

    i32 mdig = 0, idig = 0, iwarn = 0, info = 0;

    f64 *a_data = (f64*)PyArray_DATA(a_array);

    mb05od(&balanc, n, ndiag, delta, a_data, lda, &mdig, &idig, iwork, dwork, ldwork, &iwarn, &info);

    free(iwork);
    free(dwork);

    PyArray_ResolveWritebackIfCopy(a_array);

    PyObject *result = Py_BuildValue("Oiiii", a_array, mdig, idig, iwarn, info);
    Py_DECREF(a_array);

    return result;
}

PyObject* py_mb05oy(PyObject* self, PyObject* args)
{
    (void)self;

    char *job_str;
    i32 n, low, igh;
    PyObject *a_obj, *scale_obj;

    if (!PyArg_ParseTuple(args, "siiiOO", &job_str, &n, &low, &igh, &a_obj, &scale_obj)) {
        return NULL;
    }

    char job = job_str[0];

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(
        a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (a_array == NULL) {
        return NULL;
    }

    PyArrayObject *scale_array = (PyArrayObject*)PyArray_FROM_OTF(
        scale_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (scale_array == NULL) {
        Py_DECREF(a_array);
        return NULL;
    }

    i32 lda = (n > 0) ? n : 1;
    i32 info;

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *scale_data = (f64*)PyArray_DATA(scale_array);

    mb05oy(&job, n, low, igh, a_data, lda, scale_data, &info);

    PyArray_ResolveWritebackIfCopy(a_array);
    Py_DECREF(scale_array);

    PyObject *result = Py_BuildValue("Oi", a_array, info);
    Py_DECREF(a_array);

    return result;
}

