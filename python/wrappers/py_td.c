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



/* Python wrapper for td03ay */
PyObject* py_td03ay(PyObject* self, PyObject* args) {
    i32 mwork, pwork, n;
    PyObject *index_obj, *dcoeff_obj, *ucoeff_obj;
    PyArrayObject *index_array, *dcoeff_array, *ucoeff_array;
    i32 info;

    if (!PyArg_ParseTuple(args, "iiOOOi", &mwork, &pwork, &index_obj,
                          &dcoeff_obj, &ucoeff_obj, &n)) {
        return NULL;
    }

    index_array = (PyArrayObject*)PyArray_FROM_OTF(index_obj, NPY_INT32,
                                                    NPY_ARRAY_CARRAY_RO);
    if (index_array == NULL) {
        return NULL;
    }

    dcoeff_array = (PyArrayObject*)PyArray_FROM_OTF(dcoeff_obj, NPY_DOUBLE,
                                                     NPY_ARRAY_FARRAY);
    if (dcoeff_array == NULL) {
        Py_DECREF(index_array);
        return NULL;
    }

    ucoeff_array = (PyArrayObject*)PyArray_FROM_OTF(ucoeff_obj, NPY_DOUBLE,
                                                     NPY_ARRAY_FARRAY);
    if (ucoeff_array == NULL) {
        Py_DECREF(index_array);
        Py_DECREF(dcoeff_array);
        return NULL;
    }

    i32 *index_data = (i32*)PyArray_DATA(index_array);
    npy_intp *dcoeff_dims = PyArray_DIMS(dcoeff_array);
    npy_intp *ucoeff_dims = PyArray_DIMS(ucoeff_array);

    i32 lddcoe = (i32)dcoeff_dims[0];
    i32 lduco1 = (i32)ucoeff_dims[0];
    i32 lduco2 = (i32)ucoeff_dims[1];

    i32 lda = (n > 0) ? n : 1;
    i32 ldb = (n > 0) ? n : 1;
    i32 ldc = (pwork > 0) ? pwork : 1;
    i32 ldd = (pwork > 0) ? pwork : 1;

    npy_intp a_dims[2] = {lda, (n > 0) ? n : 0};
    npy_intp b_dims[2] = {ldb, mwork};
    npy_intp c_dims[2] = {ldc, (n > 0) ? n : 0};
    npy_intp d_dims[2] = {ldd, mwork};

    npy_intp a_strides[2] = {sizeof(f64), lda * (npy_intp)sizeof(f64)};
    npy_intp b_strides[2] = {sizeof(f64), ldb * (npy_intp)sizeof(f64)};
    npy_intp c_strides[2] = {sizeof(f64), ldc * (npy_intp)sizeof(f64)};
    npy_intp d_strides[2] = {sizeof(f64), ldd * (npy_intp)sizeof(f64)};

    PyObject *a_array = PyArray_New(&PyArray_Type, 2, a_dims, NPY_DOUBLE, a_strides,
                                    NULL, 0, NPY_ARRAY_FARRAY, NULL);
    PyObject *b_array = PyArray_New(&PyArray_Type, 2, b_dims, NPY_DOUBLE, b_strides,
                                    NULL, 0, NPY_ARRAY_FARRAY, NULL);
    PyObject *c_array = PyArray_New(&PyArray_Type, 2, c_dims, NPY_DOUBLE, c_strides,
                                    NULL, 0, NPY_ARRAY_FARRAY, NULL);
    PyObject *d_array = PyArray_New(&PyArray_Type, 2, d_dims, NPY_DOUBLE, d_strides,
                                    NULL, 0, NPY_ARRAY_FARRAY, NULL);

    if (!a_array || !b_array || !c_array || !d_array) {
        Py_XDECREF(a_array);
        Py_XDECREF(b_array);
        Py_XDECREF(c_array);
        Py_XDECREF(d_array);
        Py_DECREF(index_array);
        Py_DECREF(dcoeff_array);
        Py_DECREF(ucoeff_array);
        PyErr_NoMemory();
        return NULL;
    }

    f64 *a = (f64*)PyArray_DATA((PyArrayObject*)a_array);
    f64 *b = (f64*)PyArray_DATA((PyArrayObject*)b_array);
    f64 *c = (f64*)PyArray_DATA((PyArrayObject*)c_array);
    f64 *d = (f64*)PyArray_DATA((PyArrayObject*)d_array);

    memset(a, 0, lda * ((n > 0) ? n : 1) * sizeof(f64));
    memset(b, 0, ldb * mwork * sizeof(f64));
    memset(c, 0, ldc * ((n > 0) ? n : 1) * sizeof(f64));
    memset(d, 0, ldd * mwork * sizeof(f64));

    f64 *dcoeff_data = (f64*)PyArray_DATA(dcoeff_array);
    f64 *ucoeff_data = (f64*)PyArray_DATA(ucoeff_array);

    td03ay(mwork, pwork, index_data, dcoeff_data, lddcoe,
           ucoeff_data, lduco1, lduco2, n,
           a, lda, b, ldb, c, ldc, d, ldd, &info);

    Py_DECREF(index_array);
    Py_DECREF(dcoeff_array);
    Py_DECREF(ucoeff_array);

    PyObject *result = Py_BuildValue("OOOOi", a_array, b_array, c_array, d_array, info);
    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(d_array);

    return result;
}



/* Python wrapper for td03ad */
PyObject* py_td03ad(PyObject* self, PyObject* args) {
    const char *rowcol_str, *leri_str, *equil_str;
    i32 m, p;
    PyObject *indexd_obj, *dcoeff_obj, *ucoeff_obj;
    f64 tol;

    if (!PyArg_ParseTuple(args, "sssiiOOOd", &rowcol_str, &leri_str, &equil_str,
                          &m, &p, &indexd_obj, &dcoeff_obj, &ucoeff_obj, &tol)) {
        return NULL;
    }

    PyArrayObject *indexd_array = (PyArrayObject*)PyArray_FROM_OTF(
        indexd_obj, NPY_INT32, NPY_ARRAY_CARRAY_RO);
    if (indexd_array == NULL) {
        return NULL;
    }

    PyArrayObject *dcoeff_array = (PyArrayObject*)PyArray_FROM_OTF(
        dcoeff_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (dcoeff_array == NULL) {
        Py_DECREF(indexd_array);
        return NULL;
    }

    PyArrayObject *ucoeff_array = (PyArrayObject*)PyArray_FROM_OTF(
        ucoeff_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (ucoeff_array == NULL) {
        Py_DECREF(indexd_array);
        Py_DECREF(dcoeff_array);
        return NULL;
    }

    char rc = rowcol_str[0];
    char lr = leri_str[0];
    bool lrowco = (rc == 'R' || rc == 'r');
    bool lleri = (lr == 'L' || lr == 'l');

    i32 pwork = lrowco ? p : m;
    i32 maxmp = (m > p) ? m : p;
    i32 mplim = (maxmp > 1) ? maxmp : 1;
    i32 pormp = lleri ? (lrowco ? p : m) : (lrowco ? m : p);

    i32 *indexd_data = (i32*)PyArray_DATA(indexd_array);

    i32 n = 0;
    i32 kdcoef = 0;
    for (i32 i = 0; i < pwork; i++) {
        if (indexd_data[i] > kdcoef) kdcoef = indexd_data[i];
        n += indexd_data[i];
    }
    kdcoef = kdcoef + 1;

    npy_intp *dcoeff_dims = PyArray_DIMS(dcoeff_array);
    npy_intp *ucoeff_dims = PyArray_DIMS(ucoeff_array);

    i32 lddcoe = (i32)dcoeff_dims[0];
    i32 lduco1 = (i32)ucoeff_dims[0];
    i32 lduco2 = (i32)ucoeff_dims[1];

    i32 n1 = (n > 0) ? n : 1;
    i32 lda = n1;
    i32 ldb = n1;
    i32 ldc = mplim;
    i32 ldd = mplim;

    i32 ldpco1 = (pwork > 1) ? pwork : 1;
    i32 ldpco2 = (pwork > 1) ? pwork : 1;

    i32 pm = lrowco ? p : m;
    i32 mp = lrowco ? m : p;
    i32 ldqco1 = lleri ? ((pm > 1) ? pm : 1) : mplim;
    i32 ldqco2 = lleri ? ((mp > 1) ? mp : 1) : mplim;

    i32 ldvco1 = (pwork > 1) ? pwork : 1;
    i32 ldvco2 = n1;

    i32 max_n_3mp = n;
    if (3 * maxmp > max_n_3mp) max_n_3mp = 3 * maxmp;
    i32 ldwork_min = n + max_n_3mp;
    i32 pm2 = pwork * (pwork + 2);
    if (pm2 > ldwork_min) ldwork_min = pm2;
    if (ldwork_min < 1) ldwork_min = 1;
    i32 ldwork = ldwork_min * 2;

    i32 liwork = n + mplim;
    if (liwork < 1) liwork = 1;

    i32 a_cols = n1;
    i32 b_cols = mplim;
    i32 c_cols = n1;
    i32 d_cols = mplim;
    i32 kpcoef = n + 1;

    f64 *dwork = (f64*)calloc(ldwork + 1, sizeof(f64));
    i32 *iwork = (i32*)calloc(liwork + 1, sizeof(i32));

    if (!dwork || !iwork) {
        free(dwork); free(iwork);
        PyArray_DiscardWritebackIfCopy(ucoeff_array);
        Py_DECREF(indexd_array);
        Py_DECREF(dcoeff_array);
        Py_DECREF(ucoeff_array);
        PyErr_NoMemory();
        return NULL;
    }

    npy_intp a_out_dims[2] = {lda, a_cols};
    npy_intp b_out_dims[2] = {ldb, b_cols};
    npy_intp c_out_dims[2] = {ldc, c_cols};
    npy_intp d_out_dims[2] = {ldd, d_cols};
    npy_intp indexp_dims[1] = {pormp};
    npy_intp pcoeff_dims[3] = {ldpco1, ldpco2, kpcoef};
    npy_intp qcoeff_dims[3] = {ldqco1, ldqco2, kpcoef};
    npy_intp vcoeff_dims[3] = {ldvco1, ldvco2, kpcoef};
    npy_intp iwork_out_dims[1] = {n1};

    npy_intp a_strides[2] = {sizeof(f64), lda * (npy_intp)sizeof(f64)};
    npy_intp b_strides[2] = {sizeof(f64), ldb * (npy_intp)sizeof(f64)};
    npy_intp c_strides[2] = {sizeof(f64), ldc * (npy_intp)sizeof(f64)};
    npy_intp d_strides[2] = {sizeof(f64), ldd * (npy_intp)sizeof(f64)};

    PyObject *a_array = PyArray_New(&PyArray_Type, 2, a_out_dims, NPY_DOUBLE, a_strides,
                                    NULL, 0, NPY_ARRAY_FARRAY, NULL);
    PyObject *b_array = PyArray_New(&PyArray_Type, 2, b_out_dims, NPY_DOUBLE, b_strides,
                                    NULL, 0, NPY_ARRAY_FARRAY, NULL);
    PyObject *c_array = PyArray_New(&PyArray_Type, 2, c_out_dims, NPY_DOUBLE, c_strides,
                                    NULL, 0, NPY_ARRAY_FARRAY, NULL);
    PyObject *d_array = PyArray_New(&PyArray_Type, 2, d_out_dims, NPY_DOUBLE, d_strides,
                                    NULL, 0, NPY_ARRAY_FARRAY, NULL);

    PyObject *indexp_array = PyArray_ZEROS(1, indexp_dims, NPY_INT32, 0);
    PyObject *pcoeff_array = PyArray_ZEROS(3, pcoeff_dims, NPY_DOUBLE, 1);
    PyObject *qcoeff_array = PyArray_ZEROS(3, qcoeff_dims, NPY_DOUBLE, 1);
    PyObject *vcoeff_array = PyArray_ZEROS(3, vcoeff_dims, NPY_DOUBLE, 1);
    PyObject *iwork_out_array = PyArray_ZEROS(1, iwork_out_dims, NPY_INT32, 0);

    if (!a_array || !b_array || !c_array || !d_array ||
        !indexp_array || !pcoeff_array || !qcoeff_array || !vcoeff_array ||
        !iwork_out_array) {
        Py_XDECREF(a_array);
        Py_XDECREF(b_array);
        Py_XDECREF(c_array);
        Py_XDECREF(d_array);
        Py_XDECREF(indexp_array);
        Py_XDECREF(pcoeff_array);
        Py_XDECREF(qcoeff_array);
        Py_XDECREF(vcoeff_array);
        Py_XDECREF(iwork_out_array);
        free(dwork); free(iwork);
        PyArray_DiscardWritebackIfCopy(ucoeff_array);
        Py_DECREF(indexd_array);
        Py_DECREF(dcoeff_array);
        Py_DECREF(ucoeff_array);
        PyErr_NoMemory();
        return NULL;
    }

    f64 *a = (f64*)PyArray_DATA((PyArrayObject*)a_array);
    f64 *b = (f64*)PyArray_DATA((PyArrayObject*)b_array);
    f64 *c = (f64*)PyArray_DATA((PyArrayObject*)c_array);
    f64 *d = (f64*)PyArray_DATA((PyArrayObject*)d_array);
    i32 *indexp = (i32*)PyArray_DATA((PyArrayObject*)indexp_array);
    f64 *pcoeff = (f64*)PyArray_DATA((PyArrayObject*)pcoeff_array);
    f64 *qcoeff = (f64*)PyArray_DATA((PyArrayObject*)qcoeff_array);
    f64 *vcoeff = (f64*)PyArray_DATA((PyArrayObject*)vcoeff_array);
    i32 *iwork_out = (i32*)PyArray_DATA((PyArrayObject*)iwork_out_array);

    memset(a, 0, lda * a_cols * sizeof(f64));
    memset(b, 0, ldb * b_cols * sizeof(f64));
    memset(c, 0, ldc * c_cols * sizeof(f64));
    memset(d, 0, ldd * d_cols * sizeof(f64));

    f64 *dcoeff_data = (f64*)PyArray_DATA(dcoeff_array);
    f64 *ucoeff_data = (f64*)PyArray_DATA(ucoeff_array);

    i32 nr, info;

    td03ad(rowcol_str, leri_str, equil_str, m, p, indexd_data, dcoeff_data, lddcoe,
           ucoeff_data, lduco1, lduco2, &nr, a, lda, b, ldb, c, ldc, d, ldd,
           indexp, pcoeff, ldpco1, ldpco2, qcoeff, ldqco1, ldqco2,
           vcoeff, ldvco1, ldvco2, tol, iwork, dwork, ldwork, &info);

    for (i32 i = 0; i < n1 && i < liwork; i++) {
        iwork_out[i] = iwork[i];
    }

    PyArray_ResolveWritebackIfCopy(ucoeff_array);
    Py_DECREF(indexd_array);
    Py_DECREF(dcoeff_array);
    Py_DECREF(ucoeff_array);

    free(dwork);
    free(iwork);

    PyObject *result = Py_BuildValue("iOOOOOOOOOi", nr, a_array, b_array, c_array, d_array,
                                     indexp_array, pcoeff_array, qcoeff_array, vcoeff_array,
                                     iwork_out_array, info);
    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(d_array);
    Py_DECREF(indexp_array);
    Py_DECREF(pcoeff_array);
    Py_DECREF(qcoeff_array);
    Py_DECREF(vcoeff_array);
    Py_DECREF(iwork_out_array);

    return result;
}


/* Python wrapper for td04ad */
PyObject* py_td04ad(PyObject* self, PyObject* args) {
    const char *rowcol_str;
    i32 m, p;
    PyObject *index_obj, *dcoeff_obj, *ucoeff_obj;
    f64 tol;

    if (!PyArg_ParseTuple(args, "siiOOOd", &rowcol_str, &m, &p, &index_obj,
                          &dcoeff_obj, &ucoeff_obj, &tol)) {
        return NULL;
    }

    PyArrayObject *index_array = (PyArrayObject*)PyArray_FROM_OTF(
        index_obj, NPY_INT32, NPY_ARRAY_CARRAY_RO);
    if (index_array == NULL) {
        return NULL;
    }

    PyArrayObject *dcoeff_array = (PyArrayObject*)PyArray_FROM_OTF(
        dcoeff_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (dcoeff_array == NULL) {
        Py_DECREF(index_array);
        return NULL;
    }

    PyArrayObject *ucoeff_array = (PyArrayObject*)PyArray_FROM_OTF(
        ucoeff_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (ucoeff_array == NULL) {
        Py_DECREF(index_array);
        Py_DECREF(dcoeff_array);
        return NULL;
    }

    char rc = rowcol_str[0];
    bool lrocor = (rc == 'R' || rc == 'r');

    i32 pwork = lrocor ? p : m;
    i32 mplim = (m > p) ? m : p;
    if (mplim < 1) mplim = 1;

    i32 *index_data = (i32*)PyArray_DATA(index_array);

    i32 n = 0;
    i32 kdcoef = 0;
    for (i32 i = 0; i < pwork; i++) {
        if (index_data[i] > kdcoef) kdcoef = index_data[i];
        n += index_data[i];
    }
    kdcoef = kdcoef + 1;

    npy_intp *dcoeff_dims = PyArray_DIMS(dcoeff_array);
    npy_intp *ucoeff_dims = PyArray_DIMS(ucoeff_array);

    i32 lddcoe = (i32)dcoeff_dims[0];
    i32 lduco1 = (i32)ucoeff_dims[0];
    i32 lduco2 = (i32)ucoeff_dims[1];

    i32 lda = (n > 0) ? n : 1;
    i32 ldb = (n > 0) ? n : 1;
    i32 ldc = mplim;
    i32 ldd = lrocor ? ((p > 0) ? p : 1) : mplim;

    i32 max_n_3mp = n;
    if (3 * mplim > max_n_3mp) max_n_3mp = 3 * mplim;
    i32 ldwork = n + max_n_3mp;
    if (ldwork < 1) ldwork = 1;

    i32 liwork = n + mplim;
    if (liwork < 1) liwork = 1;

    i32 a_cols = (n > 0) ? n : 1;
    i32 b_cols = mplim;
    i32 c_cols = (n > 0) ? n : 1;
    i32 d_cols = lrocor ? m : mplim;
    if (d_cols < 1) d_cols = 1;

    f64 *dwork = (f64*)calloc(ldwork + 1, sizeof(f64));
    i32 *iwork = (i32*)calloc(liwork + 1, sizeof(i32));

    if (!dwork || !iwork) {
        free(dwork); free(iwork);
        Py_DECREF(index_array);
        Py_DECREF(dcoeff_array);
        Py_DECREF(ucoeff_array);
        PyErr_NoMemory();
        return NULL;
    }

    npy_intp a_out_dims[2] = {lda, a_cols};
    npy_intp b_out_dims[2] = {ldb, b_cols};
    npy_intp c_out_dims[2] = {ldc, c_cols};
    npy_intp d_out_dims[2] = {ldd, d_cols};

    npy_intp a_strides[2] = {sizeof(f64), lda * (npy_intp)sizeof(f64)};
    npy_intp b_strides[2] = {sizeof(f64), ldb * (npy_intp)sizeof(f64)};
    npy_intp c_strides[2] = {sizeof(f64), ldc * (npy_intp)sizeof(f64)};
    npy_intp d_strides[2] = {sizeof(f64), ldd * (npy_intp)sizeof(f64)};

    PyObject *a_array, *b_array, *c_array, *d_array;

    if (n > 0) {
        a_array = PyArray_New(&PyArray_Type, 2, a_out_dims, NPY_DOUBLE, a_strides,
                              NULL, 0, NPY_ARRAY_FARRAY, NULL);
        b_array = PyArray_New(&PyArray_Type, 2, b_out_dims, NPY_DOUBLE, b_strides,
                              NULL, 0, NPY_ARRAY_FARRAY, NULL);
        c_array = PyArray_New(&PyArray_Type, 2, c_out_dims, NPY_DOUBLE, c_strides,
                              NULL, 0, NPY_ARRAY_FARRAY, NULL);
    } else {
        npy_intp zero_a[2] = {0, 0};
        npy_intp zero_b[2] = {0, m > 0 ? m : 1};
        npy_intp zero_c[2] = {p > 0 ? p : 1, 0};
        a_array = PyArray_ZEROS(2, zero_a, NPY_DOUBLE, 1);
        b_array = PyArray_ZEROS(2, zero_b, NPY_DOUBLE, 1);
        c_array = PyArray_ZEROS(2, zero_c, NPY_DOUBLE, 1);
    }

    d_array = PyArray_New(&PyArray_Type, 2, d_out_dims, NPY_DOUBLE, d_strides,
                          NULL, 0, NPY_ARRAY_FARRAY, NULL);

    if (!a_array || !b_array || !c_array || !d_array) {
        Py_XDECREF(a_array);
        Py_XDECREF(b_array);
        Py_XDECREF(c_array);
        Py_XDECREF(d_array);
        free(dwork); free(iwork);
        Py_DECREF(index_array);
        Py_DECREF(dcoeff_array);
        Py_DECREF(ucoeff_array);
        PyErr_NoMemory();
        return NULL;
    }

    f64 *a = NULL, *b = NULL, *c = NULL;
    if (n > 0) {
        a = (f64*)PyArray_DATA((PyArrayObject*)a_array);
        b = (f64*)PyArray_DATA((PyArrayObject*)b_array);
        c = (f64*)PyArray_DATA((PyArrayObject*)c_array);
        if (lda * a_cols > 0) memset(a, 0, lda * a_cols * sizeof(f64));
        if (ldb * b_cols > 0) memset(b, 0, ldb * b_cols * sizeof(f64));
        if (ldc * c_cols > 0) memset(c, 0, ldc * c_cols * sizeof(f64));
    }
    f64 *d = (f64*)PyArray_DATA((PyArrayObject*)d_array);
    if (ldd * d_cols > 0) memset(d, 0, ldd * d_cols * sizeof(f64));

    f64 *dcoeff_data = (f64*)PyArray_DATA(dcoeff_array);
    f64 *ucoeff_data = (f64*)PyArray_DATA(ucoeff_array);

    i32 nr, info;

    td04ad(rowcol_str, m, p, index_data, dcoeff_data, lddcoe,
           ucoeff_data, lduco1, lduco2, &nr, a, lda, b, ldb, c, ldc, d, ldd,
           tol, iwork, dwork, ldwork, &info);

    Py_DECREF(index_array);
    Py_DECREF(dcoeff_array);
    Py_DECREF(ucoeff_array);

    free(dwork);
    free(iwork);

    PyObject *result = Py_BuildValue("iOOOOi", nr, a_array, b_array, c_array, d_array, info);
    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(d_array);

    return result;
}


/* Python wrapper for td05ad */
PyObject* py_td05ad(PyObject* self, PyObject* args) {
    const char *unitf_str, *output_str;
    i32 np1, mp1;
    f64 w;
    PyObject *a_obj, *b_obj;

    if (!PyArg_ParseTuple(args, "ssiidOO", &unitf_str, &output_str, &np1, &mp1,
                          &w, &a_obj, &b_obj)) {
        return NULL;
    }

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(
        a_obj, NPY_DOUBLE, NPY_ARRAY_CARRAY_RO);
    if (a_array == NULL) {
        return NULL;
    }

    PyArrayObject *b_array = (PyArrayObject*)PyArray_FROM_OTF(
        b_obj, NPY_DOUBLE, NPY_ARRAY_CARRAY_RO);
    if (b_array == NULL) {
        Py_DECREF(a_array);
        return NULL;
    }

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);

    f64 valr, vali;
    i32 info;

    td05ad(unitf_str, output_str, np1, mp1, w, a_data, b_data, &valr, &vali, &info);

    Py_DECREF(a_array);
    Py_DECREF(b_array);

    return Py_BuildValue("ddi", valr, vali, info);
}

