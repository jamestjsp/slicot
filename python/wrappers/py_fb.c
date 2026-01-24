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


extern void fb01sd(const char* jobx, const char* multab, const char* multrc,
                   i32 n, i32 m, i32 p,
                   f64* sinv, i32 ldsinv,
                   const f64* ainv, i32 ldainv,
                   const f64* b, i32 ldb,
                   const f64* rinv, i32 ldrinv,
                   const f64* c, i32 ldc,
                   f64* qinv, i32 ldqinv,
                   f64* x, const f64* rinvy, const f64* z, f64* e,
                   f64 tol,
                   i32* iwork,
                   f64* dwork, i32 ldwork,
                   i32* info);

extern void fb01td(const char* jobx, const char* multrc,
                   i32 n, i32 m, i32 p,
                   f64* sinv, i32 ldsinv,
                   const f64* ainv, i32 ldainv,
                   const f64* ainvb, i32 ldainb,
                   const f64* rinv, i32 ldrinv,
                   const f64* c, i32 ldc,
                   f64* qinv, i32 ldqinv,
                   f64* x, const f64* rinvy, const f64* z, f64* e,
                   f64 tol,
                   i32* iwork,
                   f64* dwork, i32 ldwork,
                   i32* info);

extern void fb01rd(const char* jobk, const char* multbq,
                   i32 n, i32 m, i32 p,
                   f64* s, i32 lds,
                   const f64* a, i32 lda,
                   const f64* b, i32 ldb,
                   const f64* q, i32 ldq,
                   const f64* c, i32 ldc,
                   f64* r, i32 ldr,
                   f64* k, i32 ldk,
                   f64 tol,
                   i32* iwork,
                   f64* dwork, i32 ldwork,
                   i32* info);


PyObject* py_fb01sd(PyObject* self, PyObject* args) {
    const char *jobx_str, *multab_str, *multrc_str;
    PyObject *sinv_obj, *ainv_obj, *b_obj, *rinv_obj, *c_obj, *qinv_obj;
    PyObject *x_obj, *rinvy_obj, *z_obj, *e_obj;
    f64 tol = 0.0;

    if (!PyArg_ParseTuple(args, "sssOOOOOOOOOO|d",
            &jobx_str, &multab_str, &multrc_str,
            &sinv_obj, &ainv_obj, &b_obj, &rinv_obj, &c_obj, &qinv_obj,
            &x_obj, &rinvy_obj, &z_obj, &e_obj, &tol)) {
        return NULL;
    }

    char jobx = (char)toupper((unsigned char)jobx_str[0]);
    char multab = (char)toupper((unsigned char)multab_str[0]);
    char multrc = (char)toupper((unsigned char)multrc_str[0]);

    if (jobx != 'X' && jobx != 'N') {
        PyErr_SetString(PyExc_ValueError, "jobx must be 'X' or 'N'");
        return NULL;
    }
    if (multab != 'P' && multab != 'N') {
        PyErr_SetString(PyExc_ValueError, "multab must be 'P' or 'N'");
        return NULL;
    }
    if (multrc != 'P' && multrc != 'N') {
        PyErr_SetString(PyExc_ValueError, "multrc must be 'P' or 'N'");
        return NULL;
    }

    PyArrayObject *sinv_array = (PyArrayObject*)PyArray_FROM_OTF(
        sinv_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!sinv_array) return NULL;

    PyArrayObject *ainv_array = (PyArrayObject*)PyArray_FROM_OTF(
        ainv_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (!ainv_array) {
        Py_DECREF(sinv_array);
        return NULL;
    }

    PyArrayObject *b_array = (PyArrayObject*)PyArray_FROM_OTF(
        b_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (!b_array) {
        Py_DECREF(sinv_array);
        Py_DECREF(ainv_array);
        return NULL;
    }

    PyArrayObject *rinv_array = (PyArrayObject*)PyArray_FROM_OTF(
        rinv_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (!rinv_array) {
        Py_DECREF(sinv_array);
        Py_DECREF(ainv_array);
        Py_DECREF(b_array);
        return NULL;
    }

    PyArrayObject *c_array = (PyArrayObject*)PyArray_FROM_OTF(
        c_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (!c_array) {
        Py_DECREF(sinv_array);
        Py_DECREF(ainv_array);
        Py_DECREF(b_array);
        Py_DECREF(rinv_array);
        return NULL;
    }

    PyArrayObject *qinv_array = (PyArrayObject*)PyArray_FROM_OTF(
        qinv_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!qinv_array) {
        Py_DECREF(sinv_array);
        Py_DECREF(ainv_array);
        Py_DECREF(b_array);
        Py_DECREF(rinv_array);
        Py_DECREF(c_array);
        return NULL;
    }

    PyArrayObject *x_array = (PyArrayObject*)PyArray_FROM_OTF(
        x_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!x_array) {
        Py_DECREF(sinv_array);
        Py_DECREF(ainv_array);
        Py_DECREF(b_array);
        Py_DECREF(rinv_array);
        Py_DECREF(c_array);
        Py_DECREF(qinv_array);
        return NULL;
    }

    PyArrayObject *rinvy_array = (PyArrayObject*)PyArray_FROM_OTF(
        rinvy_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (!rinvy_array) {
        Py_DECREF(sinv_array);
        Py_DECREF(ainv_array);
        Py_DECREF(b_array);
        Py_DECREF(rinv_array);
        Py_DECREF(c_array);
        Py_DECREF(qinv_array);
        Py_DECREF(x_array);
        return NULL;
    }

    PyArrayObject *z_array = (PyArrayObject*)PyArray_FROM_OTF(
        z_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (!z_array) {
        Py_DECREF(sinv_array);
        Py_DECREF(ainv_array);
        Py_DECREF(b_array);
        Py_DECREF(rinv_array);
        Py_DECREF(c_array);
        Py_DECREF(qinv_array);
        Py_DECREF(x_array);
        Py_DECREF(rinvy_array);
        return NULL;
    }

    PyArrayObject *e_array = (PyArrayObject*)PyArray_FROM_OTF(
        e_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!e_array) {
        Py_DECREF(sinv_array);
        Py_DECREF(ainv_array);
        Py_DECREF(b_array);
        Py_DECREF(rinv_array);
        Py_DECREF(c_array);
        Py_DECREF(qinv_array);
        Py_DECREF(x_array);
        Py_DECREF(rinvy_array);
        Py_DECREF(z_array);
        return NULL;
    }

    npy_intp *sinv_dims = PyArray_DIMS(sinv_array);
    npy_intp *ainv_dims = PyArray_DIMS(ainv_array);
    npy_intp *b_dims = PyArray_DIMS(b_array);
    npy_intp *rinv_dims = PyArray_DIMS(rinv_array);
    npy_intp *c_dims = PyArray_DIMS(c_array);
    npy_intp *qinv_dims = PyArray_DIMS(qinv_array);
    npy_intp *e_dims = PyArray_DIMS(e_array);

    i32 n = (i32)sinv_dims[0];
    i32 m = (i32)qinv_dims[1];
    i32 p = (i32)e_dims[0];

    i32 ldsinv = (i32)sinv_dims[0];
    i32 ldainv = (i32)ainv_dims[0];
    i32 ldb = (i32)b_dims[0];
    i32 ldrinv = (i32)rinv_dims[0];
    i32 ldc = (i32)c_dims[0];
    i32 ldqinv = (i32)qinv_dims[0];

    i32 liwork = (jobx == 'X') ? n : 1;
    i32 liwork_actual = (liwork > 1) ? liwork : 1;

    i32 ldwork = (n > 0 && m > 0) ? (n*(n + 2*m) + 3*m) : 1;
    i32 ldwork2 = ((n + p) > 0 && (n + 1) > 0) ? ((n+p)*(n + 1) + 2*n) : 1;
    ldwork = (ldwork > ldwork2) ? ldwork : ldwork2;
    if (jobx == 'X') {
        ldwork = (ldwork > 3*n) ? ldwork : 3*n;
        ldwork = (ldwork > 2) ? ldwork : 2;
    } else {
        ldwork = (ldwork > 1) ? ldwork : 1;
    }

    i32 *iwork = (i32*)malloc(liwork_actual * sizeof(i32));
    f64 *dwork = (f64*)malloc(ldwork * sizeof(f64));

    if (!iwork || !dwork) {
        free(iwork);
        free(dwork);
        Py_DECREF(sinv_array);
        Py_DECREF(ainv_array);
        Py_DECREF(b_array);
        Py_DECREF(rinv_array);
        Py_DECREF(c_array);
        Py_DECREF(qinv_array);
        Py_DECREF(x_array);
        Py_DECREF(rinvy_array);
        Py_DECREF(z_array);
        Py_DECREF(e_array);
        PyErr_NoMemory();
        return NULL;
    }

    f64 *sinv_data = (f64*)PyArray_DATA(sinv_array);
    const f64 *ainv_data = (const f64*)PyArray_DATA(ainv_array);
    const f64 *b_data = (const f64*)PyArray_DATA(b_array);
    const f64 *rinv_data = (const f64*)PyArray_DATA(rinv_array);
    const f64 *c_data = (const f64*)PyArray_DATA(c_array);
    f64 *qinv_data = (f64*)PyArray_DATA(qinv_array);
    f64 *x_data = (f64*)PyArray_DATA(x_array);
    const f64 *rinvy_data = (const f64*)PyArray_DATA(rinvy_array);
    const f64 *z_data = (const f64*)PyArray_DATA(z_array);
    f64 *e_data = (f64*)PyArray_DATA(e_array);

    i32 info = 0;
    char jobx_c[2] = {jobx, '\0'};
    char multab_c[2] = {multab, '\0'};
    char multrc_c[2] = {multrc, '\0'};

    fb01sd(jobx_c, multab_c, multrc_c,
           n, m, p,
           sinv_data, ldsinv,
           ainv_data, ldainv,
           b_data, ldb,
           rinv_data, ldrinv,
           c_data, ldc,
           qinv_data, ldqinv,
           x_data, rinvy_data, z_data, e_data,
           tol,
           iwork,
           dwork, ldwork,
           &info);

    free(iwork);
    free(dwork);

    PyArray_ResolveWritebackIfCopy(sinv_array);
    PyArray_ResolveWritebackIfCopy(qinv_array);
    PyArray_ResolveWritebackIfCopy(x_array);
    PyArray_ResolveWritebackIfCopy(e_array);

    PyObject *result = Py_BuildValue("OOOOi", sinv_array, qinv_array, x_array, e_array, (int)info);

    Py_DECREF(sinv_array);
    Py_DECREF(ainv_array);
    Py_DECREF(b_array);
    Py_DECREF(rinv_array);
    Py_DECREF(c_array);
    Py_DECREF(qinv_array);
    Py_DECREF(x_array);
    Py_DECREF(rinvy_array);
    Py_DECREF(z_array);
    Py_DECREF(e_array);

    return result;
}

PyObject* py_fb01td(PyObject* self, PyObject* args) {
    const char *jobx_str, *multrc_str;
    PyObject *sinv_obj, *ainv_obj, *ainvb_obj, *rinv_obj, *c_obj, *qinv_obj;
    PyObject *x_obj, *rinvy_obj, *z_obj, *e_obj;
    f64 tol = 0.0;

    if (!PyArg_ParseTuple(args, "ssOOOOOOOOOO|d",
            &jobx_str, &multrc_str,
            &sinv_obj, &ainv_obj, &ainvb_obj, &rinv_obj, &c_obj, &qinv_obj,
            &x_obj, &rinvy_obj, &z_obj, &e_obj, &tol)) {
        return NULL;
    }

    char jobx = (char)toupper((unsigned char)jobx_str[0]);
    char multrc = (char)toupper((unsigned char)multrc_str[0]);

    if (jobx != 'X' && jobx != 'N') {
        PyErr_SetString(PyExc_ValueError, "jobx must be 'X' or 'N'");
        return NULL;
    }
    if (multrc != 'P' && multrc != 'N') {
        PyErr_SetString(PyExc_ValueError, "multrc must be 'P' or 'N'");
        return NULL;
    }

    PyArrayObject *sinv_array = (PyArrayObject*)PyArray_FROM_OTF(
        sinv_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!sinv_array) return NULL;

    PyArrayObject *ainv_array = (PyArrayObject*)PyArray_FROM_OTF(
        ainv_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (!ainv_array) {
        Py_DECREF(sinv_array);
        return NULL;
    }

    PyArrayObject *ainvb_array = (PyArrayObject*)PyArray_FROM_OTF(
        ainvb_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (!ainvb_array) {
        Py_DECREF(sinv_array);
        Py_DECREF(ainv_array);
        return NULL;
    }

    PyArrayObject *rinv_array = (PyArrayObject*)PyArray_FROM_OTF(
        rinv_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (!rinv_array) {
        Py_DECREF(sinv_array);
        Py_DECREF(ainv_array);
        Py_DECREF(ainvb_array);
        return NULL;
    }

    PyArrayObject *c_array = (PyArrayObject*)PyArray_FROM_OTF(
        c_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (!c_array) {
        Py_DECREF(sinv_array);
        Py_DECREF(ainv_array);
        Py_DECREF(ainvb_array);
        Py_DECREF(rinv_array);
        return NULL;
    }

    PyArrayObject *qinv_array = (PyArrayObject*)PyArray_FROM_OTF(
        qinv_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!qinv_array) {
        Py_DECREF(sinv_array);
        Py_DECREF(ainv_array);
        Py_DECREF(ainvb_array);
        Py_DECREF(rinv_array);
        Py_DECREF(c_array);
        return NULL;
    }

    PyArrayObject *x_array = (PyArrayObject*)PyArray_FROM_OTF(
        x_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!x_array) {
        Py_DECREF(sinv_array);
        Py_DECREF(ainv_array);
        Py_DECREF(ainvb_array);
        Py_DECREF(rinv_array);
        Py_DECREF(c_array);
        Py_DECREF(qinv_array);
        return NULL;
    }

    PyArrayObject *rinvy_array = (PyArrayObject*)PyArray_FROM_OTF(
        rinvy_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS);
    if (!rinvy_array) {
        Py_DECREF(sinv_array);
        Py_DECREF(ainv_array);
        Py_DECREF(ainvb_array);
        Py_DECREF(rinv_array);
        Py_DECREF(c_array);
        Py_DECREF(qinv_array);
        Py_DECREF(x_array);
        return NULL;
    }

    PyArrayObject *z_array = (PyArrayObject*)PyArray_FROM_OTF(
        z_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS);
    if (!z_array) {
        Py_DECREF(sinv_array);
        Py_DECREF(ainv_array);
        Py_DECREF(ainvb_array);
        Py_DECREF(rinv_array);
        Py_DECREF(c_array);
        Py_DECREF(qinv_array);
        Py_DECREF(x_array);
        Py_DECREF(rinvy_array);
        return NULL;
    }

    PyArrayObject *e_array = (PyArrayObject*)PyArray_FROM_OTF(
        e_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!e_array) {
        Py_DECREF(sinv_array);
        Py_DECREF(ainv_array);
        Py_DECREF(ainvb_array);
        Py_DECREF(rinv_array);
        Py_DECREF(c_array);
        Py_DECREF(qinv_array);
        Py_DECREF(x_array);
        Py_DECREF(rinvy_array);
        Py_DECREF(z_array);
        return NULL;
    }

    i32 n = (i32)PyArray_DIM(sinv_array, 0);
    i32 m = (i32)PyArray_DIM(qinv_array, 0);
    i32 p = (i32)PyArray_DIM(c_array, 0);
    i32 ldsinv = (i32)PyArray_DIM(sinv_array, 0);
    i32 ldainv = (i32)PyArray_DIM(ainv_array, 0);
    i32 ldainb = (i32)PyArray_DIM(ainvb_array, 0);
    i32 ldrinv = (i32)PyArray_DIM(rinv_array, 0);
    i32 ldc = (i32)PyArray_DIM(c_array, 0);
    i32 ldqinv = (i32)PyArray_DIM(qinv_array, 0);

    i32 liwork = (jobx == 'X') ? n : 1;
    i32 ldwork = ((n * (n + 2*m) + 3*m) >
                  ((n + p) * (n + 1) + n + ((n-1 > m+1) ? n-1 : m+1))) ?
                  (n * (n + 2*m) + 3*m) :
                  ((n + p) * (n + 1) + n + ((n-1 > m+1) ? n-1 : m+1));
    if (jobx == 'X') {
        ldwork = (ldwork > 3*n) ? ldwork : 3*n;
        ldwork = (ldwork > 2) ? ldwork : 2;
    } else {
        ldwork = (ldwork > 1) ? ldwork : 1;
    }

    i32 *iwork = (i32*)malloc(liwork * sizeof(i32));
    f64 *dwork = (f64*)malloc(ldwork * sizeof(f64));

    if (!iwork || !dwork) {
        free(iwork);
        free(dwork);
        Py_DECREF(sinv_array);
        Py_DECREF(ainv_array);
        Py_DECREF(ainvb_array);
        Py_DECREF(rinv_array);
        Py_DECREF(c_array);
        Py_DECREF(qinv_array);
        Py_DECREF(x_array);
        Py_DECREF(rinvy_array);
        Py_DECREF(z_array);
        Py_DECREF(e_array);
        PyErr_NoMemory();
        return NULL;
    }

    f64 *sinv_data = (f64*)PyArray_DATA(sinv_array);
    const f64 *ainv_data = (const f64*)PyArray_DATA(ainv_array);
    const f64 *ainvb_data = (const f64*)PyArray_DATA(ainvb_array);
    const f64 *rinv_data = (const f64*)PyArray_DATA(rinv_array);
    const f64 *c_data = (const f64*)PyArray_DATA(c_array);
    f64 *qinv_data = (f64*)PyArray_DATA(qinv_array);
    f64 *x_data = (f64*)PyArray_DATA(x_array);
    const f64 *rinvy_data = (const f64*)PyArray_DATA(rinvy_array);
    const f64 *z_data = (const f64*)PyArray_DATA(z_array);
    f64 *e_data = (f64*)PyArray_DATA(e_array);

    i32 info = 0;
    char jobx_c[2] = {jobx, '\0'};
    char multrc_c[2] = {multrc, '\0'};

    fb01td(jobx_c, multrc_c,
           n, m, p,
           sinv_data, ldsinv,
           ainv_data, ldainv,
           ainvb_data, ldainb,
           rinv_data, ldrinv,
           c_data, ldc,
           qinv_data, ldqinv,
           x_data, rinvy_data, z_data, e_data,
           tol,
           iwork,
           dwork, ldwork,
           &info);

    free(iwork);
    free(dwork);

    PyArray_ResolveWritebackIfCopy(sinv_array);
    PyArray_ResolveWritebackIfCopy(qinv_array);
    PyArray_ResolveWritebackIfCopy(x_array);
    PyArray_ResolveWritebackIfCopy(e_array);

    PyObject *result = Py_BuildValue("OOOOi", sinv_array, qinv_array, x_array, e_array, (int)info);

    Py_DECREF(sinv_array);
    Py_DECREF(ainv_array);
    Py_DECREF(ainvb_array);
    Py_DECREF(rinv_array);
    Py_DECREF(c_array);
    Py_DECREF(qinv_array);
    Py_DECREF(x_array);
    Py_DECREF(rinvy_array);
    Py_DECREF(z_array);
    Py_DECREF(e_array);

    return result;
}


extern void fb01vd(i32 n, i32 m, i32 l,
                   f64* p, i32 ldp,
                   const f64* a, i32 lda,
                   const f64* b, i32 ldb,
                   const f64* c, i32 ldc,
                   f64* q, i32 ldq,
                   f64* r, i32 ldr,
                   f64* k, i32 ldk,
                   f64 tol,
                   i32* iwork,
                   f64* dwork, i32 ldwork,
                   i32* info);


PyObject* py_fb01vd(PyObject* self, PyObject* args) {
    i32 n, m, l;
    PyObject *p_obj, *a_obj, *b_obj, *c_obj, *q_obj, *r_obj;
    f64 tol = 0.0;

    if (!PyArg_ParseTuple(args, "iiiOOOOOO|d",
            &n, &m, &l, &p_obj, &a_obj, &b_obj, &c_obj, &q_obj, &r_obj, &tol)) {
        return NULL;
    }

    if (n < 0 || m < 0 || l < 0) {
        PyErr_SetString(PyExc_ValueError, "n, m, l must be non-negative");
        return NULL;
    }

    PyArrayObject *p_array = (PyArrayObject*)PyArray_FROM_OTF(
        p_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!p_array) return NULL;

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(
        a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (!a_array) {
        Py_DECREF(p_array);
        return NULL;
    }

    PyArrayObject *b_array_v = (PyArrayObject*)PyArray_FROM_OTF(
        b_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (!b_array_v) {
        Py_DECREF(p_array);
        Py_DECREF(a_array);
        return NULL;
    }

    PyArrayObject *c_array_v = (PyArrayObject*)PyArray_FROM_OTF(
        c_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (!c_array_v) {
        Py_DECREF(p_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array_v);
        return NULL;
    }

    PyArrayObject *q_array = (PyArrayObject*)PyArray_FROM_OTF(
        q_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!q_array) {
        Py_DECREF(p_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array_v);
        Py_DECREF(c_array_v);
        return NULL;
    }

    PyArrayObject *r_array = (PyArrayObject*)PyArray_FROM_OTF(
        r_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!r_array) {
        Py_DECREF(p_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array_v);
        Py_DECREF(c_array_v);
        Py_DECREF(q_array);
        return NULL;
    }

    i32 ldp = (n > 0) ? n : 1;
    i32 lda = (n > 0) ? n : 1;
    i32 ldb = (n > 0) ? n : 1;
    i32 ldc = (l > 0) ? l : 1;
    i32 ldq = (m > 0) ? m : 1;
    i32 ldr = (l > 0) ? l : 1;
    i32 ldk = (n > 0) ? n : 1;

    i32 ldwork = l*n + 3*l;
    i32 w2 = n*n;
    i32 w3 = n*m;
    if (w2 > ldwork) ldwork = w2;
    if (w3 > ldwork) ldwork = w3;
    if (ldwork < 1) ldwork = 1;

    i32 liwork = (l > 0) ? l : 1;
    i32 *iwork = (i32*)malloc(liwork * sizeof(i32));
    f64 *dwork = (f64*)malloc(ldwork * sizeof(f64));

    npy_intp k_dims[2] = {n, l};
    npy_intp k_strides[2] = {sizeof(f64), n * sizeof(f64)};
    
    PyObject *k_array = NULL;
    f64 *k_data = NULL;
    
    if (n > 0 && l > 0) {
        k_array = PyArray_New(&PyArray_Type, 2, k_dims, NPY_DOUBLE, k_strides,
                                NULL, 0, NPY_ARRAY_FARRAY, NULL);
        if (!k_array) {
           free(iwork);
           free(dwork);
           Py_DECREF(p_array);
           Py_DECREF(a_array);
           Py_DECREF(b_array_v);
           Py_DECREF(c_array_v);
           Py_DECREF(q_array);
           Py_DECREF(r_array);
           return NULL;
        }
        k_data = (f64*)PyArray_DATA((PyArrayObject*)k_array);
        memset(k_data, 0, n * l * sizeof(f64));
    } else {
         npy_intp empty_dims[2] = {n, l};
         k_array = PyArray_ZEROS(2, empty_dims, NPY_DOUBLE, 1);
         if (!k_array) {
            free(iwork);
            free(dwork);
            Py_DECREF(p_array);
            Py_DECREF(a_array);
            Py_DECREF(b_array_v);
            Py_DECREF(c_array_v);
            Py_DECREF(q_array);
            Py_DECREF(r_array);
            return NULL;
         }
         if (PyArray_SIZE((PyArrayObject*)k_array) > 0) {
             k_data = (f64*)PyArray_DATA((PyArrayObject*)k_array);
         }
    }

    if (!iwork || !dwork) {
        free(iwork);
        free(dwork);
        Py_DECREF(p_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array_v);
        Py_DECREF(c_array_v);
        Py_DECREF(q_array);
        Py_DECREF(r_array);
        Py_XDECREF(k_array);
        PyErr_NoMemory();
        return NULL;
    }

    f64 *p_data = (f64*)PyArray_DATA(p_array);
    const f64 *a_data = (const f64*)PyArray_DATA(a_array);
    const f64 *b_data = (const f64*)PyArray_DATA(b_array_v);
    const f64 *c_data = (const f64*)PyArray_DATA(c_array_v);
    f64 *q_data = (f64*)PyArray_DATA(q_array);
    f64 *r_data = (f64*)PyArray_DATA(r_array);

    i32 info = 0;
    fb01vd(n, m, l, p_data, ldp, a_data, lda, b_data, ldb, c_data, ldc,
           q_data, ldq, r_data, ldr, k_data, ldk, tol, iwork, dwork, ldwork, &info);

    f64 rcond = dwork[0];
    free(iwork);
    free(dwork);

    PyArray_ResolveWritebackIfCopy(p_array);
    PyArray_ResolveWritebackIfCopy(q_array);
    PyArray_ResolveWritebackIfCopy(r_array);

    // k_array is already ready.

    PyObject *result = Py_BuildValue("OOOdi", p_array, k_array, r_array, rcond, (int)info);

    Py_DECREF(p_array);
    Py_DECREF(a_array);
    Py_DECREF(b_array_v);
    Py_DECREF(c_array_v);
    Py_DECREF(q_array);
    Py_DECREF(r_array);
    Py_DECREF(k_array);

    return result;
}


extern void fb01qd(const char* jobk, const char* multbq,
                   i32 n, i32 m, i32 p,
                   f64* s, i32 lds,
                   const f64* a, i32 lda,
                   const f64* b, i32 ldb,
                   const f64* q, i32 ldq,
                   const f64* c, i32 ldc,
                   f64* r, i32 ldr,
                   f64* k, i32 ldk,
                   f64 tol,
                   i32* iwork,
                   f64* dwork, i32 ldwork,
                   i32* info);


PyObject* py_fb01qd(PyObject* self, PyObject* args) {
    const char *jobk_str, *multbq_str;
    PyObject *s_obj, *a_obj, *b_obj, *q_obj, *c_obj, *r_obj;
    f64 tol = 0.0;

    if (!PyArg_ParseTuple(args, "ssOOOOOO|d",
            &jobk_str, &multbq_str, &s_obj, &a_obj, &b_obj, &q_obj,
            &c_obj, &r_obj, &tol)) {
        return NULL;
    }

    char jobk = (char)toupper((unsigned char)jobk_str[0]);
    char multbq = (char)toupper((unsigned char)multbq_str[0]);

    if (jobk != 'K' && jobk != 'N') {
        PyErr_SetString(PyExc_ValueError, "JOBK must be 'K' or 'N'");
        return NULL;
    }
    if (multbq != 'P' && multbq != 'N') {
        PyErr_SetString(PyExc_ValueError, "MULTBQ must be 'P' or 'N'");
        return NULL;
    }

    PyArrayObject *s_array = (PyArrayObject*)PyArray_FROM_OTF(
        s_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!s_array) return NULL;

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(
        a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (!a_array) {
        Py_DECREF(s_array);
        return NULL;
    }

    PyArrayObject *b_array = (PyArrayObject*)PyArray_FROM_OTF(
        b_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (!b_array) {
        Py_DECREF(s_array);
        Py_DECREF(a_array);
        return NULL;
    }

    PyArrayObject *q_array = (PyArrayObject*)PyArray_FROM_OTF(
        q_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (!q_array) {
        Py_DECREF(s_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        return NULL;
    }

    PyArrayObject *c_array = (PyArrayObject*)PyArray_FROM_OTF(
        c_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (!c_array) {
        Py_DECREF(s_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(q_array);
        return NULL;
    }

    PyArrayObject *r_array = (PyArrayObject*)PyArray_FROM_OTF(
        r_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!r_array) {
        Py_DECREF(s_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(q_array);
        Py_DECREF(c_array);
        return NULL;
    }

    i32 n = (i32)PyArray_DIM(s_array, 0);
    i32 m = (i32)PyArray_DIM(b_array, 1);
    i32 p = (i32)PyArray_DIM(c_array, 0);

    i32 lds = (n > 1) ? n : 1;
    i32 lda = (n > 1) ? n : 1;
    i32 ldb = (n > 1) ? n : 1;
    i32 ldq = (multbq == 'N') ? ((m > 1) ? m : 1) : 1;
    i32 ldc = (p > 1) ? p : 1;
    i32 ldr = (p > 1) ? p : 1;
    i32 ldk = (n > 1) ? n : 1;

    i32 pn = p + n;
    i32 val1 = pn * n + 2 * p;
    i32 val2 = n * (n + m + 2);
    i32 val3 = 3 * p;
    i32 ldwork;
    if (jobk == 'K') {
        ldwork = (val1 > val2) ? val1 : val2;
        ldwork = (ldwork > val3) ? ldwork : val3;
        ldwork = (ldwork > 2) ? ldwork : 2;
    } else {
        ldwork = (val1 > val2) ? val1 : val2;
        ldwork = (ldwork > 1) ? ldwork : 1;
    }

    i32 liwork = (jobk == 'K') ? p : 1;
    liwork = (liwork > 1) ? liwork : 1;

    i32* iwork = (i32*)calloc(liwork, sizeof(i32));
    f64* dwork = (f64*)calloc(ldwork, sizeof(f64));

    npy_intp k_dims[2] = {n, p};
    npy_intp k_strides[2] = {sizeof(f64), n * sizeof(f64)};
    
    PyObject *k_array_out = NULL;
    f64* k_data = NULL;

    if (n > 0 && p > 0) {
        k_array_out = PyArray_New(&PyArray_Type, 2, k_dims, NPY_DOUBLE, k_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        if (!k_array_out) {
            free(iwork); free(dwork);
            Py_DECREF(s_array); Py_DECREF(a_array); Py_DECREF(b_array); Py_DECREF(q_array); Py_DECREF(c_array); Py_DECREF(r_array);
            return NULL;
        }
        k_data = (f64*)PyArray_DATA((PyArrayObject*)k_array_out);
        memset(k_data, 0, n * p * sizeof(f64));
    } else {
        npy_intp empty_dims[2] = {n, p};
        k_array_out = PyArray_ZEROS(2, empty_dims, NPY_DOUBLE, 1);
        if (!k_array_out) {
            free(iwork); free(dwork);
            Py_DECREF(s_array); Py_DECREF(a_array); Py_DECREF(b_array); Py_DECREF(q_array); Py_DECREF(c_array); Py_DECREF(r_array);
            return NULL;
        }
        if (PyArray_SIZE((PyArrayObject*)k_array_out) > 0) {
             k_data = (f64*)PyArray_DATA((PyArrayObject*)k_array_out);
        }
    }

    if (!iwork || !dwork) {
        free(iwork);
        free(dwork);
        Py_DECREF(k_array_out);
        Py_DECREF(s_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(q_array);
        Py_DECREF(c_array);
        Py_DECREF(r_array);
        PyErr_NoMemory();
        return NULL;
    }

    f64* s_data = (f64*)PyArray_DATA(s_array);
    const f64* a_data = (const f64*)PyArray_DATA(a_array);
    const f64* b_data = (const f64*)PyArray_DATA(b_array);
    const f64* q_data = (const f64*)PyArray_DATA(q_array);
    const f64* c_data = (const f64*)PyArray_DATA(c_array);
    f64* r_data = (f64*)PyArray_DATA(r_array);

    i32 info = 0;
    char jobk_c[2] = {jobk, '\0'};
    char multbq_c[2] = {multbq, '\0'};

    fb01qd(jobk_c, multbq_c, n, m, p,
           s_data, lds, a_data, lda, b_data, ldb, q_data, ldq,
           c_data, ldc, r_data, ldr, k_data, ldk, tol,
           iwork, dwork, ldwork, &info);

    f64 rcond = (jobk == 'K' && info == 0) ? dwork[1] : 0.0;
    free(iwork);
    free(dwork);

    PyArray_ResolveWritebackIfCopy(s_array);
    PyArray_ResolveWritebackIfCopy(r_array);

    // k_array_out is already ready and populated.

    PyObject *result = Py_BuildValue("OOOdi", s_array, k_array_out, r_array, rcond, (int)info);

    Py_DECREF(s_array);
    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(q_array);
    Py_DECREF(c_array);
    Py_DECREF(r_array);
    Py_DECREF(k_array_out);

    return result;
}


PyObject* py_fb01rd(PyObject* self, PyObject* args) {
    const char *jobk_str, *multbq_str;
    PyObject *s_obj, *a_obj, *b_obj, *q_obj, *c_obj, *r_obj;
    f64 tol = 0.0;
    i32 n_override = -1;

    if (!PyArg_ParseTuple(args, "ssOOOOOO|di",
            &jobk_str, &multbq_str, &s_obj, &a_obj, &b_obj, &q_obj,
            &c_obj, &r_obj, &tol, &n_override)) {
        return NULL;
    }

    char jobk = (char)toupper((unsigned char)jobk_str[0]);
    char multbq = (char)toupper((unsigned char)multbq_str[0]);

    if (jobk != 'K' && jobk != 'N') {
        PyErr_SetString(PyExc_ValueError, "JOBK must be 'K' or 'N'");
        return NULL;
    }
    if (multbq != 'P' && multbq != 'N') {
        PyErr_SetString(PyExc_ValueError, "MULTBQ must be 'P' or 'N'");
        return NULL;
    }

    PyArrayObject *s_array = (PyArrayObject*)PyArray_FROM_OTF(
        s_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!s_array) return NULL;

    PyArrayObject *a_array = (PyArrayObject*)PyArray_FROM_OTF(
        a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (!a_array) {
        Py_DECREF(s_array);
        return NULL;
    }

    PyArrayObject *b_array = (PyArrayObject*)PyArray_FROM_OTF(
        b_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (!b_array) {
        Py_DECREF(s_array);
        Py_DECREF(a_array);
        return NULL;
    }

    PyArrayObject *q_array = (PyArrayObject*)PyArray_FROM_OTF(
        q_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (!q_array) {
        Py_DECREF(s_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        return NULL;
    }

    PyArrayObject *c_array = (PyArrayObject*)PyArray_FROM_OTF(
        c_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (!c_array) {
        Py_DECREF(s_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(q_array);
        return NULL;
    }

    PyArrayObject *r_array = (PyArrayObject*)PyArray_FROM_OTF(
        r_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!r_array) {
        Py_DECREF(s_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(q_array);
        Py_DECREF(c_array);
        return NULL;
    }

    i32 n = (n_override >= 0) ? n_override : (i32)PyArray_DIM(s_array, 0);
    i32 m = (i32)PyArray_DIM(b_array, 1);
    i32 p = (i32)PyArray_DIM(c_array, 0);

    i32 lds = (n > 1) ? n : 1;
    i32 lda = (n > 1) ? n : 1;
    i32 ldb = (n > 1) ? n : 1;
    i32 ldq = (multbq == 'N') ? ((m > 1) ? m : 1) : 1;
    i32 ldc = (p > 1) ? p : 1;
    i32 ldr = (p > 1) ? p : 1;
    i32 ldk = (n > 1) ? n : 1;

    i32 pn = p + n;
    i32 val1 = pn * n + n;
    i32 val2 = pn * n + 2 * p;
    i32 val3 = n * (n + m + 2);
    i32 val4 = 3 * p;
    i32 ldwork;
    if (jobk == 'K') {
        ldwork = (val1 > val2) ? val1 : val2;
        ldwork = (ldwork > val3) ? ldwork : val3;
        ldwork = (ldwork > val4) ? ldwork : val4;
        ldwork = (ldwork > 2) ? ldwork : 2;
    } else {
        ldwork = (val1 > val2) ? val1 : val2;
        ldwork = (ldwork > val3) ? ldwork : val3;
        ldwork = (ldwork > 1) ? ldwork : 1;
    }

    i32 liwork = (jobk == 'K') ? p : 1;
    liwork = (liwork > 1) ? liwork : 1;

    i32* iwork = (i32*)calloc(liwork, sizeof(i32));
    f64* dwork = (f64*)calloc(ldwork, sizeof(f64));

    npy_intp k_dims[2] = {n, p};
    npy_intp k_strides[2] = {sizeof(f64), n * sizeof(f64)};
    f64* k_data = NULL;
    PyObject *k_array_out = NULL;

    if (n > 0 && p > 0) {
        k_array_out = PyArray_New(&PyArray_Type, 2, k_dims, NPY_DOUBLE, k_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        if (!k_array_out) {
            free(iwork); free(dwork);
            Py_DECREF(s_array); Py_DECREF(a_array); Py_DECREF(b_array); Py_DECREF(q_array); Py_DECREF(c_array); Py_DECREF(r_array);
            return NULL;
        }
        k_data = (f64*)PyArray_DATA((PyArrayObject*)k_array_out);
        memset(k_data, 0, n * p * sizeof(f64));
    } else {
        npy_intp empty_dims[2] = {n, p};
        k_array_out = PyArray_ZEROS(2, empty_dims, NPY_DOUBLE, 1);
         if (!k_array_out) {
             free(iwork); free(dwork);
            Py_DECREF(s_array); Py_DECREF(a_array); Py_DECREF(b_array); Py_DECREF(q_array); Py_DECREF(c_array); Py_DECREF(r_array);
            return NULL;
        }
        if (PyArray_SIZE((PyArrayObject*)k_array_out) > 0) {
            k_data = (f64*)PyArray_DATA((PyArrayObject*)k_array_out);
        }
    }

    if (!iwork || !dwork) {
        free(iwork);
        free(dwork);
        Py_DECREF(k_array_out);
        Py_DECREF(s_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(q_array);
        Py_DECREF(c_array);
        Py_DECREF(r_array);
        PyErr_NoMemory();
        return NULL;
    }

    f64* s_data = (f64*)PyArray_DATA(s_array);
    const f64* a_data = (const f64*)PyArray_DATA(a_array);
    const f64* b_data = (const f64*)PyArray_DATA(b_array);
    const f64* q_data = (const f64*)PyArray_DATA(q_array);
    const f64* c_data = (const f64*)PyArray_DATA(c_array);
    f64* r_data = (f64*)PyArray_DATA(r_array);

    i32 info = 0;
    char jobk_c[2] = {jobk, '\0'};
    char multbq_c[2] = {multbq, '\0'};

    fb01rd(jobk_c, multbq_c, n, m, p,
           s_data, lds, a_data, lda, b_data, ldb, q_data, ldq,
           c_data, ldc, r_data, ldr, k_data, ldk, tol,
           iwork, dwork, ldwork, &info);

    free(iwork);
    free(dwork);

    PyArray_ResolveWritebackIfCopy(s_array);
    PyArray_ResolveWritebackIfCopy(r_array);

    // k_array_out is already ready.

    PyObject *result = Py_BuildValue("OOOi", s_array, r_array, k_array_out, (int)info);

    Py_DECREF(s_array);
    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(q_array);
    Py_DECREF(c_array);
    Py_DECREF(r_array);
    Py_DECREF(k_array_out);

    return result;
}
