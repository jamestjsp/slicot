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



PyObject* py_ma02ed(PyObject* self, PyObject* args) {
    const char *uplo_str;
    PyObject *a_obj;
    PyArrayObject *a_array;

    if (!PyArg_ParseTuple(args, "sO", &uplo_str, &a_obj)) {
        return NULL;
    }

    if (uplo_str == NULL || uplo_str[0] == '\0') {
        PyErr_SetString(PyExc_ValueError, "uplo must be a non-empty string");
        return NULL;
    }
    char uplo = uplo_str[0];

    /* Convert to NumPy array - preserve Fortran-order, in-place modification */
    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (a_array == NULL) {
        return NULL;
    }

    /* Get dimensions */
    npy_intp *a_dims = PyArray_DIMS(a_array);
    i32 n = (i32)a_dims[0];
    i32 lda = n > 0 ? n : 1;

    /* Call C function - modifies a in place */
    f64 *a_data = (f64*)PyArray_DATA(a_array);
    ma02ed(uplo, n, a_data, lda);

    /* Resolve writebackifcopy before returning */
    PyArray_ResolveWritebackIfCopy(a_array);

    /* Return modified array */
    PyObject *result = Py_BuildValue("O", a_array);
    Py_DECREF(a_array);
    return result;
}



PyObject* py_ma02es(PyObject* self, PyObject* args) {
    const char *uplo_str;
    PyObject *a_obj;
    PyArrayObject *a_array;

    if (!PyArg_ParseTuple(args, "sO", &uplo_str, &a_obj)) {
        return NULL;
    }

    if (uplo_str == NULL || uplo_str[0] == '\0') {
        PyErr_SetString(PyExc_ValueError, "uplo must be a non-empty string");
        return NULL;
    }
    char uplo = uplo_str[0];

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (a_array == NULL) {
        return NULL;
    }

    npy_intp *a_dims = PyArray_DIMS(a_array);
    i32 n = (i32)a_dims[0];
    i32 lda = n > 0 ? n : 1;

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    ma02es(uplo, n, a_data, lda);

    PyArray_ResolveWritebackIfCopy(a_array);

    PyObject *result = Py_BuildValue("O", a_array);
    Py_DECREF(a_array);
    return result;
}



PyObject* py_ma01ad(PyObject* self, PyObject* args) {
    f64 xr, xi;
    f64 yr, yi;

    if (!PyArg_ParseTuple(args, "dd", &xr, &xi)) {
        return NULL;
    }

    ma01ad(xr, xi, &yr, &yi);

    return Py_BuildValue("dd", yr, yi);
}



PyObject* py_ma01bd(PyObject* self, PyObject* args) {
    f64 base, lgbas;
    i32 k, inca;
    PyObject *s_obj, *a_obj;

    if (!PyArg_ParseTuple(args, "ddiOOi", &base, &lgbas, &k, &s_obj, &a_obj, &inca)) {
        return NULL;
    }

    PyArrayObject *s_array = (PyArrayObject*)PyArray_FROM_OTF(s_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    if (s_array == NULL) {
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (a_array == NULL) {
        Py_DECREF(s_array);
        return NULL;
    }

    i32 *s_data = (i32*)PyArray_DATA(s_array);
    f64 *a_data = (f64*)PyArray_DATA(a_array);

    f64 alpha, beta;
    i32 scal;

    ma01bd(base, lgbas, k, s_data, a_data, inca, &alpha, &beta, &scal);

    Py_DECREF(s_array);
    Py_DECREF(a_array);

    return Py_BuildValue("ddi", alpha, beta, scal);
}



PyObject* py_ma01bz(PyObject* self, PyObject* args) {
    f64 base;
    i32 k, inca;
    PyObject *s_obj, *a_obj;

    if (!PyArg_ParseTuple(args, "diOOi", &base, &k, &s_obj, &a_obj, &inca)) {
        return NULL;
    }

    PyArrayObject *s_array = (PyArrayObject*)PyArray_FROM_OTF(s_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    if (s_array == NULL) {
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
    if (a_array == NULL) {
        Py_DECREF(s_array);
        return NULL;
    }

    i32 *s_data = (i32*)PyArray_DATA(s_array);
    c128 *a_data = (c128*)PyArray_DATA(a_array);

    c128 alpha, beta;
    i32 scal;

    ma01bz(base, k, s_data, a_data, inca, &alpha, &beta, &scal);

    Py_DECREF(s_array);
    Py_DECREF(a_array);

    return Py_BuildValue("DDi", &alpha, &beta, scal);
}



PyObject* py_ma02ad(PyObject* self, PyObject* args) {
    const char* job;
    PyObject *a_obj;
    PyArrayObject *a_array;

    if (!PyArg_ParseTuple(args, "sO", &job, &a_obj)) {
        return NULL;
    }

    /* Convert to NumPy array - preserve Fortran-order */
    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY);
    if (a_array == NULL) {
        return NULL;
    }

    /* Get dimensions */
    npy_intp *a_dims = PyArray_DIMS(a_array);
    i32 m = (i32)a_dims[0];
    i32 n = (i32)a_dims[1];
    i32 lda = m > 0 ? m : 1;
    i32 ldb = n > 0 ? n : 1;

    /* Allocate output array B with shape (n, m) */
    npy_intp b_dims[2] = {n, m};
    npy_intp b_strides[2] = {sizeof(f64), ldb * sizeof(f64)};
    PyObject *b_array = PyArray_New(&PyArray_Type, 2, b_dims, NPY_DOUBLE,
                                    b_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (b_array == NULL) {
        Py_DECREF(a_array);
        return NULL;
    }
    f64 *b_data = (f64*)PyArray_DATA((PyArrayObject*)b_array);
    if (n > 0 && m > 0) {
        memset(b_data, 0, ldb * m * sizeof(f64));
    }

    /* Call C function */
    f64 *a_data = (f64*)PyArray_DATA(a_array);
    ma02ad(job, m, n, a_data, lda, b_data, ldb);

    Py_DECREF(a_array);
    return b_array;
}



PyObject* py_ma02cd(PyObject* self, PyObject* args) {
    PyObject *a_obj;
    i32 kl, ku;

    if (!PyArg_ParseTuple(args, "Oii", &a_obj, &kl, &ku)) {
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(
        a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (a_array == NULL) {
        return NULL;
    }

    if (PyArray_NDIM(a_array) != 2) {
        Py_DECREF(a_array);
        PyErr_SetString(PyExc_ValueError, "Matrix A must be 2D");
        return NULL;
    }

    npy_intp *a_dims = PyArray_DIMS(a_array);
    i32 n = (i32)a_dims[0];
    if (a_dims[0] != a_dims[1]) {
        Py_DECREF(a_array);
        PyErr_SetString(PyExc_ValueError, "Matrix A must be square");
        return NULL;
    }

    i32 lda = n > 0 ? n : 1;
    f64 *a_data = (f64*)PyArray_DATA(a_array);

    ma02cd(n, kl, ku, a_data, lda);

    PyArray_ResolveWritebackIfCopy(a_array);
    Py_INCREF(a_array);
    Py_DECREF(a_array);

    return (PyObject*)a_array;
}



PyObject* py_ma02dd(PyObject* self, PyObject* args) {
    const char *job, *uplo;
    PyObject *input_obj;
    i32 n_dim = -1;

    if (!PyArg_ParseTuple(args, "ssO|i", &job, &uplo, &input_obj, &n_dim)) {
        return NULL;
    }

    bool pack = (job[0] == 'P' || job[0] == 'p');

    if (pack) {
        /* Pack: input is 2D matrix A, output is 1D packed array AP */
        PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(
            input_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
        if (a_array == NULL) {
            return NULL;
        }

        if (PyArray_NDIM(a_array) != 2) {
            Py_DECREF(a_array);
            PyErr_SetString(PyExc_ValueError, "Matrix A must be 2D for packing");
            return NULL;
        }

        npy_intp *a_dims = PyArray_DIMS(a_array);
        i32 n = (i32)a_dims[0];
        if (a_dims[0] != a_dims[1]) {
            Py_DECREF(a_array);
            PyErr_SetString(PyExc_ValueError, "Matrix A must be square");
            return NULL;
        }

        i32 lda = n > 0 ? n : 1;
        i32 ap_len = n * (n + 1) / 2;
        f64 *a_data = (f64*)PyArray_DATA(a_array);

        npy_intp ap_dims[1] = {ap_len};
        PyObject *ap_array = PyArray_SimpleNew(1, ap_dims, NPY_DOUBLE);
        if (ap_array == NULL) {
            Py_DECREF(a_array);
            return NULL;
        }
        f64 *ap_data = (f64*)PyArray_DATA((PyArrayObject*)ap_array);

        ma02dd(job, uplo, n, a_data, lda, ap_data);

        Py_DECREF(a_array);
        return ap_array;
    } else {
        /* Unpack: input is 1D packed array AP, output is 2D matrix A */
        if (n_dim < 0) {
            PyErr_SetString(PyExc_ValueError, "Matrix dimension n required for unpacking");
            return NULL;
        }

        PyArrayObject *ap_array = (PyArrayObject*)PyArray_FROM_OTF(
            input_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
        if (ap_array == NULL) {
            return NULL;
        }

        i32 n = n_dim;
        i32 lda = n > 0 ? n : 1;
        f64 *ap_data = (f64*)PyArray_DATA(ap_array);

        npy_intp a_dims[2] = {n, n};
        npy_intp a_strides[2] = {sizeof(f64), lda * sizeof(f64)};
        PyObject *a_result = PyArray_New(&PyArray_Type, 2, a_dims, NPY_DOUBLE,
                                         a_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        if (a_result == NULL) {
            Py_DECREF(ap_array);
            return NULL;
        }
        f64 *a_data = (f64*)PyArray_DATA((PyArrayObject*)a_result);
        memset(a_data, 0, lda * n * sizeof(f64));

        ma02dd(job, uplo, n, a_data, lda, ap_data);

        Py_DECREF(ap_array);
        return a_result;
    }
}



/* Python wrapper for ma02gd */
PyObject* py_ma02gd(PyObject* self, PyObject* args) {
    i32 n, k1, k2, incx;
    PyObject *a_obj, *ipiv_obj;

    if (!PyArg_ParseTuple(args, "iOiiOi", &n, &a_obj, &k1, &k2, &ipiv_obj, &incx)) {
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (a_array == NULL) return NULL;

    PyArrayObject *ipiv_array = (PyArrayObject*)PyArray_FROM_OTF(ipiv_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    if (ipiv_array == NULL) {
        Py_DECREF(a_array);
        return NULL;
    }

    i32 lda = (i32)PyArray_DIM(a_array, 0);
    f64 *a_data = (f64*)PyArray_DATA(a_array);
    i32 *ipiv_data = (i32*)PyArray_DATA(ipiv_array);

    ma02gd(n, a_data, lda, k1, k2, ipiv_data, incx);

    Py_DECREF(ipiv_array);
    PyArray_ResolveWritebackIfCopy(a_array);

    Py_INCREF(a_array);
    return (PyObject*)a_array;
}



PyObject* py_ma01cd(PyObject* self, PyObject* args) {
    f64 a, b;
    i32 ia, ib;

    if (!PyArg_ParseTuple(args, "didi", &a, &ia, &b, &ib)) {
        return NULL;
    }

    i32 result = ma01cd(a, ia, b, ib);

    return PyLong_FromLong(result);
}



PyObject* py_ma01dd(PyObject* self, PyObject* args) {
    f64 ar1, ai1, ar2, ai2, eps, safemn;
    f64 d;

    if (!PyArg_ParseTuple(args, "dddddd", &ar1, &ai1, &ar2, &ai2, &eps, &safemn)) {
        return NULL;
    }

    ma01dd(ar1, ai1, ar2, ai2, eps, safemn, &d);

    return PyFloat_FromDouble(d);
}



PyObject* py_ma01dz(PyObject* self, PyObject* args) {
    f64 ar1, ai1, b1, ar2, ai2, b2, eps, safemn;
    f64 d1, d2;
    i32 iwarn;

    if (!PyArg_ParseTuple(args, "dddddddd", &ar1, &ai1, &b1, &ar2, &ai2, &b2, &eps, &safemn)) {
        return NULL;
    }

    ma01dz(ar1, ai1, b1, ar2, ai2, b2, eps, safemn, &d1, &d2, &iwarn);

    return Py_BuildValue("ddi", d1, d2, iwarn);
}



PyObject* py_ma02bd(PyObject* self, PyObject* args) {
    const char *side_str;
    PyObject *a_obj;

    if (!PyArg_ParseTuple(args, "sO", &side_str, &a_obj)) {
        return NULL;
    }

    if (side_str == NULL || side_str[0] == '\0') {
        PyErr_SetString(PyExc_ValueError, "side must be a non-empty string");
        return NULL;
    }
    char side = side_str[0];

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (a_array == NULL) return NULL;

    i32 m = (i32)PyArray_DIM(a_array, 0);
    i32 n = (i32)PyArray_DIM(a_array, 1);
    i32 lda = m > 0 ? m : 1;
    f64 *a_data = (f64*)PyArray_DATA(a_array);

    ma02bd(side, m, n, a_data, lda);

    PyArray_ResolveWritebackIfCopy(a_array);
    Py_INCREF(a_array);
    return (PyObject*)a_array;
}



PyObject* py_ma02pd(PyObject* self, PyObject* args) {
    PyObject *a_obj;

    if (!PyArg_ParseTuple(args, "O", &a_obj)) {
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(
        a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (a_array == NULL) {
        return NULL;
    }

    if (PyArray_NDIM(a_array) != 2) {
        Py_DECREF(a_array);
        PyErr_SetString(PyExc_ValueError, "Matrix A must be 2D");
        return NULL;
    }

    npy_intp *a_dims = PyArray_DIMS(a_array);
    i32 m = (i32)a_dims[0];
    i32 n = (i32)a_dims[1];
    i32 lda = m > 0 ? m : 1;

    f64 *a_data = (f64*)PyArray_DATA(a_array);

    i32 nzr, nzc;
    ma02pd(m, n, a_data, lda, &nzr, &nzc);

    Py_DECREF(a_array);

    return Py_BuildValue("ii", nzr, nzc);
}



PyObject* py_ma02az(PyObject* self, PyObject* args) {
    const char *trans_str, *job_str;
    PyObject *a_obj;

    if (!PyArg_ParseTuple(args, "ssO", &trans_str, &job_str, &a_obj)) {
        return NULL;
    }

    if (trans_str == NULL || trans_str[0] == '\0') {
        PyErr_SetString(PyExc_ValueError, "trans must be a non-empty string");
        return NULL;
    }
    if (job_str == NULL || job_str[0] == '\0') {
        PyErr_SetString(PyExc_ValueError, "job must be a non-empty string");
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(
        a_obj, NPY_COMPLEX128, NPY_ARRAY_FARRAY);
    if (a_array == NULL) {
        return NULL;
    }

    if (PyArray_NDIM(a_array) != 2) {
        Py_DECREF(a_array);
        PyErr_SetString(PyExc_ValueError, "Matrix A must be 2D");
        return NULL;
    }

    npy_intp *a_dims = PyArray_DIMS(a_array);
    i32 m = (i32)a_dims[0];
    i32 n = (i32)a_dims[1];
    i32 lda = m > 0 ? m : 1;
    i32 ldb = n > 0 ? n : 1;

    npy_intp b_dims[2] = {n, m};
    npy_intp b_strides[2] = {sizeof(c128), ldb * sizeof(c128)};
    PyObject *b_array = PyArray_New(&PyArray_Type, 2, b_dims, NPY_COMPLEX128,
                                    b_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (b_array == NULL) {
        Py_DECREF(a_array);
        return NULL;
    }
    c128 *b_data = (c128*)PyArray_DATA((PyArrayObject*)b_array);
    memset(b_data, 0, ldb * m * sizeof(c128));

    c128 *a_data = (c128*)PyArray_DATA(a_array);
    ma02az(trans_str, job_str, m, n, a_data, lda, b_data, ldb);

    Py_DECREF(a_array);
    return b_array;
}



PyObject* py_ma02bz(PyObject* self, PyObject* args) {
    const char *side_str;
    PyObject *a_obj;

    if (!PyArg_ParseTuple(args, "sO", &side_str, &a_obj)) {
        return NULL;
    }

    if (side_str == NULL || side_str[0] == '\0') {
        PyErr_SetString(PyExc_ValueError, "side must be a non-empty string");
        return NULL;
    }
    char side = side_str[0];

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_COMPLEX128,
                                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (a_array == NULL) return NULL;

    i32 m = (i32)PyArray_DIM(a_array, 0);
    i32 n = (i32)PyArray_DIM(a_array, 1);
    i32 lda = m > 0 ? m : 1;
    c128 *a_data = (c128*)PyArray_DATA(a_array);

    ma02bz(side, m, n, a_data, lda);

    PyArray_ResolveWritebackIfCopy(a_array);
    Py_INCREF(a_array);
    return (PyObject*)a_array;
}



PyObject* py_ma02cz(PyObject* self, PyObject* args) {
    PyObject *a_obj;
    i32 kl, ku;

    if (!PyArg_ParseTuple(args, "Oii", &a_obj, &kl, &ku)) {
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(
        a_obj, NPY_COMPLEX128, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (a_array == NULL) {
        return NULL;
    }

    if (PyArray_NDIM(a_array) != 2) {
        Py_DECREF(a_array);
        PyErr_SetString(PyExc_ValueError, "Matrix A must be 2D");
        return NULL;
    }

    npy_intp *a_dims = PyArray_DIMS(a_array);
    i32 n = (i32)a_dims[0];
    if (a_dims[0] != a_dims[1]) {
        Py_DECREF(a_array);
        PyErr_SetString(PyExc_ValueError, "Matrix A must be square");
        return NULL;
    }

    i32 lda = n > 0 ? n : 1;
    c128 *a_data = (c128*)PyArray_DATA(a_array);

    ma02cz(n, kl, ku, a_data, lda);

    PyArray_ResolveWritebackIfCopy(a_array);
    Py_INCREF(a_array);
    Py_DECREF(a_array);

    return (PyObject*)a_array;
}



PyObject* py_ma02ez(PyObject* self, PyObject* args) {
    const char *uplo_str, *trans_str, *skew_str;
    PyObject *a_obj;

    if (!PyArg_ParseTuple(args, "sssO", &uplo_str, &trans_str, &skew_str, &a_obj)) {
        return NULL;
    }

    if (uplo_str == NULL || uplo_str[0] == '\0') {
        PyErr_SetString(PyExc_ValueError, "uplo must be a non-empty string");
        return NULL;
    }
    if (trans_str == NULL || trans_str[0] == '\0') {
        PyErr_SetString(PyExc_ValueError, "trans must be a non-empty string");
        return NULL;
    }
    if (skew_str == NULL || skew_str[0] == '\0') {
        PyErr_SetString(PyExc_ValueError, "skew must be a non-empty string");
        return NULL;
    }

    char uplo = uplo_str[0];
    char trans = trans_str[0];
    char skew = skew_str[0];

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_COMPLEX128,
                                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (a_array == NULL) return NULL;

    if (PyArray_NDIM(a_array) != 2) {
        Py_DECREF(a_array);
        PyErr_SetString(PyExc_ValueError, "Matrix A must be 2D");
        return NULL;
    }

    npy_intp *a_dims = PyArray_DIMS(a_array);
    i32 n = (i32)a_dims[0];
    if (a_dims[0] != a_dims[1]) {
        Py_DECREF(a_array);
        PyErr_SetString(PyExc_ValueError, "Matrix A must be square");
        return NULL;
    }

    i32 lda = n > 0 ? n : 1;
    c128 *a_data = (c128*)PyArray_DATA(a_array);

    ma02ez(uplo, trans, skew, n, a_data, lda);

    PyArray_ResolveWritebackIfCopy(a_array);
    Py_INCREF(a_array);
    Py_DECREF(a_array);

    return (PyObject*)a_array;
}



PyObject* py_ma02gz(PyObject* self, PyObject* args) {
    i32 n, k1, k2, incx;
    PyObject *a_obj, *ipiv_obj;

    if (!PyArg_ParseTuple(args, "iOiiOi", &n, &a_obj, &k1, &k2, &ipiv_obj, &incx)) {
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_COMPLEX128,
                                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (a_array == NULL) return NULL;

    PyArrayObject *ipiv_array = (PyArrayObject*)PyArray_FROM_OTF(ipiv_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    if (ipiv_array == NULL) {
        Py_DECREF(a_array);
        return NULL;
    }

    i32 lda = (i32)PyArray_DIM(a_array, 0);
    c128 *a_data = (c128*)PyArray_DATA(a_array);
    i32 *ipiv_data = (i32*)PyArray_DATA(ipiv_array);

    ma02gz(n, a_data, lda, k1, k2, ipiv_data, incx);

    Py_DECREF(ipiv_array);
    PyArray_ResolveWritebackIfCopy(a_array);

    Py_INCREF(a_array);
    return (PyObject*)a_array;
}



PyObject* py_ma02hd(PyObject* self, PyObject* args) {
    const char *job_str;
    PyObject *a_obj;
    f64 diag;

    if (!PyArg_ParseTuple(args, "sOd", &job_str, &a_obj, &diag)) {
        return NULL;
    }

    if (job_str == NULL || job_str[0] == '\0') {
        PyErr_SetString(PyExc_ValueError, "job must be a non-empty string");
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(
        a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (a_array == NULL) {
        return NULL;
    }

    i32 ndim = PyArray_NDIM(a_array);
    i32 m, n;

    if (ndim == 0) {
        m = 0;
        n = 0;
    } else if (ndim == 1) {
        m = (i32)PyArray_DIM(a_array, 0);
        n = 1;
    } else {
        m = (i32)PyArray_DIM(a_array, 0);
        n = (i32)PyArray_DIM(a_array, 1);
    }

    i32 lda = m > 0 ? m : 1;
    f64 *a_data = (f64*)PyArray_DATA(a_array);

    bool result = ma02hd(job_str, m, n, diag, a_data, lda);

    Py_DECREF(a_array);

    if (result) {
        Py_RETURN_TRUE;
    } else {
        Py_RETURN_FALSE;
    }
}



PyObject* py_ma02hz(PyObject* self, PyObject* args) {
    const char *job_str;
    PyObject *a_obj;
    Py_complex diag_py;

    if (!PyArg_ParseTuple(args, "sOD", &job_str, &a_obj, &diag_py)) {
        return NULL;
    }

    if (job_str == NULL || job_str[0] == '\0') {
        PyErr_SetString(PyExc_ValueError, "job must be a non-empty string");
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(
        a_obj, NPY_COMPLEX128, NPY_ARRAY_FARRAY);
    if (a_array == NULL) {
        return NULL;
    }

    i32 ndim = PyArray_NDIM(a_array);
    i32 m, n;

    if (ndim == 0) {
        m = 0;
        n = 0;
    } else if (ndim == 1) {
        m = (i32)PyArray_DIM(a_array, 0);
        n = 1;
    } else {
        m = (i32)PyArray_DIM(a_array, 0);
        n = (i32)PyArray_DIM(a_array, 1);
    }

    i32 lda = m > 0 ? m : 1;
    c128 *a_data = (c128*)PyArray_DATA(a_array);
    c128 diag = diag_py.real + diag_py.imag * I;

    bool result = ma02hz(job_str, m, n, diag, a_data, lda);

    Py_DECREF(a_array);

    if (result) {
        Py_RETURN_TRUE;
    } else {
        Py_RETURN_FALSE;
    }
}



PyObject* py_ma02iz(PyObject* self, PyObject* args) {
    const char *typ_str, *norm_str;
    PyObject *a_obj, *qg_obj;

    if (!PyArg_ParseTuple(args, "ssOO", &typ_str, &norm_str, &a_obj, &qg_obj)) {
        return NULL;
    }

    if (typ_str == NULL || typ_str[0] == '\0') {
        PyErr_SetString(PyExc_ValueError, "typ must be a non-empty string");
        return NULL;
    }
    if (norm_str == NULL || norm_str[0] == '\0') {
        PyErr_SetString(PyExc_ValueError, "norm must be a non-empty string");
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(
        a_obj, NPY_COMPLEX128, NPY_ARRAY_FARRAY);
    if (a_array == NULL) {
        return NULL;
    }

    PyArrayObject *qg_array = (PyArrayObject*)PyArray_FROM_OTF(
        qg_obj, NPY_COMPLEX128, NPY_ARRAY_FARRAY);
    if (qg_array == NULL) {
        Py_DECREF(a_array);
        return NULL;
    }

    i32 n = (i32)PyArray_DIM(a_array, 0);
    i32 lda = n > 0 ? n : 1;
    i32 ldqg = n > 0 ? n : 1;

    c128 *a_data = (c128*)PyArray_DATA(a_array);
    c128 *qg_data = (c128*)PyArray_DATA(qg_array);

    f64 *dwork = NULL;
    char norm_c = norm_str[0];
    bool need_dwork = (norm_c == '1' || norm_c == 'O' || norm_c == 'o' ||
                       norm_c == 'I' || norm_c == 'i');
    if (need_dwork && n > 0) {
        dwork = (f64*)malloc(2 * n * sizeof(f64));
        if (dwork == NULL) {
            Py_DECREF(a_array);
            Py_DECREF(qg_array);
            PyErr_SetString(PyExc_MemoryError, "Failed to allocate workspace");
            return NULL;
        }
    }

    f64 result = ma02iz(typ_str, norm_str, n, a_data, lda, qg_data, ldqg, dwork);

    free(dwork);
    Py_DECREF(a_array);
    Py_DECREF(qg_array);

    return PyFloat_FromDouble(result);
}



PyObject* py_ma02jd(PyObject* self, PyObject* args) {
    int ltran1, ltran2;
    PyObject *q1_obj, *q2_obj;

    if (!PyArg_ParseTuple(args, "ppOO", &ltran1, &ltran2, &q1_obj, &q2_obj)) {
        return NULL;
    }

    PyArrayObject *q1_array = (PyArrayObject*)PyArray_FROM_OTF(
        q1_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (q1_array == NULL) {
        return NULL;
    }

    PyArrayObject *q2_array = (PyArrayObject*)PyArray_FROM_OTF(
        q2_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (q2_array == NULL) {
        Py_DECREF(q1_array);
        return NULL;
    }

    i32 n;
    if (PyArray_NDIM(q1_array) == 2) {
        n = (i32)PyArray_DIM(q1_array, 0);
    } else if (PyArray_NDIM(q1_array) == 0) {
        n = 0;
    } else {
        n = (i32)PyArray_DIM(q1_array, 0);
        if (n == 0) n = 0;
    }

    i32 ldq1 = n > 0 ? n : 1;
    i32 ldq2 = n > 0 ? n : 1;
    i32 ldres = n > 0 ? n : 1;

    f64 *q1_data = (f64*)PyArray_DATA(q1_array);
    f64 *q2_data = (f64*)PyArray_DATA(q2_array);

    f64 *res = NULL;
    if (n > 0) {
        res = (f64*)malloc(ldres * n * sizeof(f64));
        if (res == NULL) {
            Py_DECREF(q1_array);
            Py_DECREF(q2_array);
            PyErr_SetString(PyExc_MemoryError, "Failed to allocate workspace");
            return NULL;
        }
    }

    f64 result = ma02jd((bool)ltran1, (bool)ltran2, n, q1_data, ldq1,
                        q2_data, ldq2, res, ldres);

    free(res);
    Py_DECREF(q1_array);
    Py_DECREF(q2_array);

    return PyFloat_FromDouble(result);
}



PyObject* py_ma02jz(PyObject* self, PyObject* args) {
    int ltran1, ltran2;
    PyObject *q1_obj, *q2_obj;

    if (!PyArg_ParseTuple(args, "ppOO", &ltran1, &ltran2, &q1_obj, &q2_obj)) {
        return NULL;
    }

    PyArrayObject *q1_array = (PyArrayObject*)PyArray_FROM_OTF(
        q1_obj, NPY_COMPLEX128, NPY_ARRAY_FARRAY);
    if (q1_array == NULL) {
        return NULL;
    }

    PyArrayObject *q2_array = (PyArrayObject*)PyArray_FROM_OTF(
        q2_obj, NPY_COMPLEX128, NPY_ARRAY_FARRAY);
    if (q2_array == NULL) {
        Py_DECREF(q1_array);
        return NULL;
    }

    i32 n;
    if (PyArray_NDIM(q1_array) == 2) {
        n = (i32)PyArray_DIM(q1_array, 0);
    } else if (PyArray_NDIM(q1_array) == 0) {
        n = 0;
    } else {
        n = (i32)PyArray_DIM(q1_array, 0);
        if (n == 0) n = 0;
    }

    i32 ldq1 = n > 0 ? n : 1;
    i32 ldq2 = n > 0 ? n : 1;
    i32 ldres = n > 0 ? n : 1;

    c128 *q1_data = (c128*)PyArray_DATA(q1_array);
    c128 *q2_data = (c128*)PyArray_DATA(q2_array);

    c128 *res = NULL;
    if (n > 0) {
        res = (c128*)malloc(ldres * n * sizeof(c128));
        if (res == NULL) {
            Py_DECREF(q1_array);
            Py_DECREF(q2_array);
            PyErr_SetString(PyExc_MemoryError, "Failed to allocate workspace");
            return NULL;
        }
    }

    f64 result = ma02jz((bool)ltran1, (bool)ltran2, n, q1_data, ldq1,
                        q2_data, ldq2, res, ldres);

    free(res);
    Py_DECREF(q1_array);
    Py_DECREF(q2_array);

    return PyFloat_FromDouble(result);
}



PyObject* py_ma02md(PyObject* self, PyObject* args) {
    const char *norm_str, *uplo_str;
    PyObject *a_obj;

    if (!PyArg_ParseTuple(args, "ssO", &norm_str, &uplo_str, &a_obj)) {
        return NULL;
    }

    if (norm_str == NULL || norm_str[0] == '\0') {
        PyErr_SetString(PyExc_ValueError, "norm must be a non-empty string");
        return NULL;
    }
    if (uplo_str == NULL || uplo_str[0] == '\0') {
        PyErr_SetString(PyExc_ValueError, "uplo must be a non-empty string");
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(
        a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (a_array == NULL) {
        return NULL;
    }

    i32 ndim = PyArray_NDIM(a_array);
    i32 n;

    if (ndim == 0) {
        n = 0;
    } else if (ndim == 2) {
        n = (i32)PyArray_DIM(a_array, 0);
        if (PyArray_DIM(a_array, 0) != PyArray_DIM(a_array, 1)) {
            Py_DECREF(a_array);
            PyErr_SetString(PyExc_ValueError, "Matrix A must be square");
            return NULL;
        }
    } else {
        Py_DECREF(a_array);
        PyErr_SetString(PyExc_ValueError, "Matrix A must be 2D");
        return NULL;
    }

    i32 lda = n > 0 ? n : 1;
    f64 *a_data = (f64*)PyArray_DATA(a_array);

    f64 *dwork = NULL;
    char norm_c = norm_str[0];
    bool need_dwork = (norm_c == '1' || norm_c == 'O' || norm_c == 'o' ||
                       norm_c == 'I' || norm_c == 'i');
    if (need_dwork && n > 0) {
        dwork = (f64*)malloc(n * sizeof(f64));
        if (dwork == NULL) {
            Py_DECREF(a_array);
            PyErr_SetString(PyExc_MemoryError, "Failed to allocate workspace");
            return NULL;
        }
    }

    f64 result = ma02md(norm_str, uplo_str, n, a_data, lda, dwork);

    free(dwork);
    Py_DECREF(a_array);

    return PyFloat_FromDouble(result);
}



PyObject* py_ma02mz(PyObject* self, PyObject* args) {
    const char *norm_str, *uplo_str;
    PyObject *a_obj;

    if (!PyArg_ParseTuple(args, "ssO", &norm_str, &uplo_str, &a_obj)) {
        return NULL;
    }

    if (norm_str == NULL || norm_str[0] == '\0') {
        PyErr_SetString(PyExc_ValueError, "norm must be a non-empty string");
        return NULL;
    }
    if (uplo_str == NULL || uplo_str[0] == '\0') {
        PyErr_SetString(PyExc_ValueError, "uplo must be a non-empty string");
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(
        a_obj, NPY_COMPLEX128, NPY_ARRAY_FARRAY);
    if (a_array == NULL) {
        return NULL;
    }

    i32 ndim = PyArray_NDIM(a_array);
    i32 n;

    if (ndim == 0) {
        n = 0;
    } else if (ndim == 2) {
        n = (i32)PyArray_DIM(a_array, 0);
        if (PyArray_DIM(a_array, 0) != PyArray_DIM(a_array, 1)) {
            Py_DECREF(a_array);
            PyErr_SetString(PyExc_ValueError, "Matrix A must be square");
            return NULL;
        }
    } else {
        Py_DECREF(a_array);
        PyErr_SetString(PyExc_ValueError, "Matrix A must be 2D");
        return NULL;
    }

    i32 lda = n > 0 ? n : 1;
    c128 *a_data = (c128*)PyArray_DATA(a_array);

    f64 *dwork = NULL;
    char norm_c = norm_str[0];
    bool need_dwork = (norm_c == '1' || norm_c == 'O' || norm_c == 'o' ||
                       norm_c == 'I' || norm_c == 'i');
    if (need_dwork && n > 0) {
        dwork = (f64*)malloc(n * sizeof(f64));
        if (dwork == NULL) {
            Py_DECREF(a_array);
            PyErr_SetString(PyExc_MemoryError, "Failed to allocate workspace");
            return NULL;
        }
    }

    f64 result = ma02mz(norm_str, uplo_str, n, a_data, lda, dwork);

    free(dwork);
    Py_DECREF(a_array);

    return PyFloat_FromDouble(result);
}



PyObject* py_ma02nz(PyObject* self, PyObject* args) {
    const char *uplo_str, *trans_str, *skew_str;
    i32 k, l;
    PyObject *a_obj;

    if (!PyArg_ParseTuple(args, "sssiiO", &uplo_str, &trans_str, &skew_str, &k, &l, &a_obj)) {
        return NULL;
    }

    if (uplo_str == NULL || uplo_str[0] == '\0') {
        PyErr_SetString(PyExc_ValueError, "uplo must be a non-empty string");
        return NULL;
    }
    if (trans_str == NULL || trans_str[0] == '\0') {
        PyErr_SetString(PyExc_ValueError, "trans must be a non-empty string");
        return NULL;
    }
    if (skew_str == NULL || skew_str[0] == '\0') {
        PyErr_SetString(PyExc_ValueError, "skew must be a non-empty string");
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_COMPLEX128,
                                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (a_array == NULL) return NULL;

    if (PyArray_NDIM(a_array) != 2) {
        Py_DECREF(a_array);
        PyErr_SetString(PyExc_ValueError, "Matrix A must be 2D");
        return NULL;
    }

    npy_intp *a_dims = PyArray_DIMS(a_array);
    i32 n = (i32)a_dims[0];
    if (a_dims[0] != a_dims[1]) {
        Py_DECREF(a_array);
        PyErr_SetString(PyExc_ValueError, "Matrix A must be square");
        return NULL;
    }

    i32 lda = n > 0 ? n : 1;
    c128 *a_data = (c128*)PyArray_DATA(a_array);

    ma02nz(uplo_str, trans_str, skew_str, n, k, l, a_data, lda);

    PyArray_ResolveWritebackIfCopy(a_array);
    Py_INCREF(a_array);
    Py_DECREF(a_array);

    return (PyObject*)a_array;
}



PyObject* py_ma02od(PyObject* self, PyObject* args) {
    const char *skew_str;
    PyObject *a_obj, *de_obj;

    if (!PyArg_ParseTuple(args, "sOO", &skew_str, &a_obj, &de_obj)) {
        return NULL;
    }

    if (skew_str == NULL || skew_str[0] == '\0') {
        PyErr_SetString(PyExc_ValueError, "skew must be a non-empty string");
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(
        a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (a_array == NULL) {
        return NULL;
    }

    PyArrayObject *de_array = (PyArrayObject*)PyArray_FROM_OTF(
        de_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (de_array == NULL) {
        Py_DECREF(a_array);
        return NULL;
    }

    i32 m = 0;
    if (PyArray_NDIM(a_array) == 2) {
        m = (i32)PyArray_DIM(a_array, 0);
    }

    i32 lda = m > 0 ? m : 1;
    i32 ldde = m > 0 ? m : 1;

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *de_data = (f64*)PyArray_DATA(de_array);

    i32 nz = ma02od(skew_str, m, a_data, lda, de_data, ldde);

    Py_DECREF(a_array);
    Py_DECREF(de_array);

    return PyLong_FromLong(nz);
}



PyObject* py_ma02oz(PyObject* self, PyObject* args) {
    const char *skew_str;
    PyObject *a_obj, *de_obj;

    if (!PyArg_ParseTuple(args, "sOO", &skew_str, &a_obj, &de_obj)) {
        return NULL;
    }

    if (skew_str == NULL || skew_str[0] == '\0') {
        PyErr_SetString(PyExc_ValueError, "skew must be a non-empty string");
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(
        a_obj, NPY_COMPLEX128, NPY_ARRAY_FARRAY);
    if (a_array == NULL) {
        return NULL;
    }

    PyArrayObject *de_array = (PyArrayObject*)PyArray_FROM_OTF(
        de_obj, NPY_COMPLEX128, NPY_ARRAY_FARRAY);
    if (de_array == NULL) {
        Py_DECREF(a_array);
        return NULL;
    }

    i32 m = 0;
    if (PyArray_NDIM(a_array) == 2) {
        m = (i32)PyArray_DIM(a_array, 0);
    }

    i32 lda = m > 0 ? m : 1;
    i32 ldde = m > 0 ? m : 1;

    c128 *a_data = (c128*)PyArray_DATA(a_array);
    c128 *de_data = (c128*)PyArray_DATA(de_array);

    i32 nz = ma02oz(skew_str, m, a_data, lda, de_data, ldde);

    Py_DECREF(a_array);
    Py_DECREF(de_array);

    return PyLong_FromLong(nz);
}



PyObject* py_ma02pz(PyObject* self, PyObject* args) {
    PyObject *a_obj;

    if (!PyArg_ParseTuple(args, "O", &a_obj)) {
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(
        a_obj, NPY_COMPLEX128, NPY_ARRAY_FARRAY);
    if (a_array == NULL) {
        return NULL;
    }

    if (PyArray_NDIM(a_array) != 2) {
        Py_DECREF(a_array);
        PyErr_SetString(PyExc_ValueError, "Matrix A must be 2D");
        return NULL;
    }

    npy_intp *a_dims = PyArray_DIMS(a_array);
    i32 m = (i32)a_dims[0];
    i32 n = (i32)a_dims[1];
    i32 lda = m > 0 ? m : 1;

    c128 *a_data = (c128*)PyArray_DATA(a_array);

    i32 nzr, nzc;
    ma02pz(m, n, a_data, lda, &nzr, &nzc);

    Py_DECREF(a_array);

    return Py_BuildValue("ii", nzr, nzc);
}



PyObject* py_ma02rd(PyObject* self, PyObject* args) {
    const char *id_str;
    PyObject *d_obj, *e_obj;

    if (!PyArg_ParseTuple(args, "sOO", &id_str, &d_obj, &e_obj)) {
        return NULL;
    }

    if (id_str == NULL || id_str[0] == '\0') {
        PyErr_SetString(PyExc_ValueError, "id must be a non-empty string");
        return NULL;
    }
    char id = id_str[0];

    PyArrayObject *d_array = (PyArrayObject*)PyArray_FROM_OTF(d_obj, NPY_DOUBLE,
                                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (d_array == NULL) return NULL;

    PyArrayObject *e_array = (PyArrayObject*)PyArray_FROM_OTF(e_obj, NPY_DOUBLE,
                                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (e_array == NULL) {
        Py_DECREF(d_array);
        return NULL;
    }

    i32 n = (i32)PyArray_SIZE(d_array);
    f64 *d_data = (f64*)PyArray_DATA(d_array);
    f64 *e_data = (f64*)PyArray_DATA(e_array);

    i32 info = ma02rd(id, n, d_data, e_data);

    PyArray_ResolveWritebackIfCopy(d_array);
    PyArray_ResolveWritebackIfCopy(e_array);

    if (info < 0) {
        Py_DECREF(d_array);
        Py_DECREF(e_array);
        PyErr_Format(PyExc_ValueError, "MA02RD: illegal value in argument %d", -info);
        return NULL;
    }

    PyObject *result = Py_BuildValue("OOi", d_array, e_array, info);
    Py_DECREF(d_array);
    Py_DECREF(e_array);
    return result;
}



PyObject* py_ma02sd(PyObject* self, PyObject* args) {
    PyObject *a_obj;

    if (!PyArg_ParseTuple(args, "O", &a_obj)) {
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(
        a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (a_array == NULL) {
        return NULL;
    }

    i32 ndim = PyArray_NDIM(a_array);
    i32 m, n;

    if (ndim == 0) {
        m = 0;
        n = 0;
    } else if (ndim == 1) {
        m = (i32)PyArray_DIM(a_array, 0);
        n = 1;
    } else {
        m = (i32)PyArray_DIM(a_array, 0);
        n = (i32)PyArray_DIM(a_array, 1);
    }

    i32 lda = m > 0 ? m : 1;
    f64 *a_data = (f64*)PyArray_DATA(a_array);

    f64 result = ma02sd(m, n, a_data, lda);

    Py_DECREF(a_array);

    return PyFloat_FromDouble(result);
}
