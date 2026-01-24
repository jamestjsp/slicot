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



/* Python wrapper for nf01br */
PyObject* py_nf01br(PyObject* self, PyObject* args) {
    const char *cond, *uplo, *trans;
    i32 n;
    PyObject *ipar_obj, *r_obj, *sdiag_obj, *s_obj, *b_obj, *ranks_obj;
    f64 tol;
    PyArrayObject *ipar_array, *r_array, *sdiag_array, *s_array, *b_array, *ranks_array;
    i32 info;

    if (!PyArg_ParseTuple(args, "sssiOOOOOOd", &cond, &uplo, &trans, &n, 
                          &ipar_obj, &r_obj, &sdiag_obj, &s_obj, &b_obj, &ranks_obj, &tol)) {
        return NULL;
    }
    
    ipar_array = (PyArrayObject*)PyArray_FROM_OTF(ipar_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    r_array = (PyArrayObject*)PyArray_FROM_OTF(r_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    sdiag_array = (PyArrayObject*)PyArray_FROM_OTF(sdiag_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    s_array = (PyArrayObject*)PyArray_FROM_OTF(s_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    b_array = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    ranks_array = (PyArrayObject*)PyArray_FROM_OTF(ranks_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    
    if (!ipar_array || !r_array || !sdiag_array || !s_array || !b_array || !ranks_array) {
         Py_XDECREF(ipar_array); Py_XDECREF(r_array); Py_XDECREF(sdiag_array);
         Py_XDECREF(s_array); Py_XDECREF(b_array); Py_XDECREF(ranks_array);
         return NULL;
    }
    
    i32 lipar = (i32)PyArray_SIZE(ipar_array);
    if (lipar < 4) {
         Py_DECREF(ipar_array); Py_DECREF(r_array); Py_DECREF(sdiag_array);
         Py_DECREF(s_array); Py_DECREF(b_array); Py_DECREF(ranks_array);
         PyErr_SetString(PyExc_ValueError, "ipar must have length >= 4");
         return NULL;
    }
    
    i32 *ipar_data = (i32*)PyArray_DATA(ipar_array);
    i32 st = ipar_data[0];
    i32 bn = ipar_data[1];
    i32 bsn = ipar_data[3];
    bool full = (bn <= 1 || bsn == 0);
    bool econd = (cond[0] == 'E' || cond[0] == 'e');
    
    i32 lwork_size;
    if (econd) {
        if (full) lwork_size = 2 * n;
        else lwork_size = 2 * ((bsn > st) ? bsn : st);
    } else {
        lwork_size = 1;
    }
    
    f64 *dwork = (f64*)malloc((lwork_size > 0 ? lwork_size : 1) * sizeof(f64));
    if (!dwork) {
         Py_DECREF(ipar_array); Py_DECREF(r_array); Py_DECREF(sdiag_array);
         Py_DECREF(s_array); Py_DECREF(b_array); Py_DECREF(ranks_array);
         return PyErr_NoMemory();
    }
    
    i32 ldr = (i32)PyArray_DIM(r_array, 0);
    i32 lds = (i32)PyArray_DIM(s_array, 0);
    if (lds < 1) lds = 1; /* Avoid zero dim */
    
    f64 *r_data = (f64*)PyArray_DATA(r_array);
    f64 *sdiag_data = (f64*)PyArray_DATA(sdiag_array);
    f64 *s_data = (f64*)PyArray_DATA(s_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);
    i32 *ranks_data = (i32*)PyArray_DATA(ranks_array);
    
    nf01br(cond, uplo, trans, n, ipar_data, lipar, r_data, ldr, sdiag_data, 
           s_data, lds, b_data, ranks_data, tol, dwork, lwork_size, &info);
           
    free(dwork);
    
    PyArray_ResolveWritebackIfCopy(b_array);
    PyArray_ResolveWritebackIfCopy(ranks_array);
    PyArray_ResolveWritebackIfCopy(r_array); // Potentially modified if UPLO=L
    PyArray_ResolveWritebackIfCopy(sdiag_array);
    PyArray_ResolveWritebackIfCopy(s_array);
    
    Py_DECREF(ipar_array); 
    Py_DECREF(r_array); 
    Py_DECREF(sdiag_array); 
    Py_DECREF(s_array); 
    
    /* Return modified b and ranks */
    PyObject *result = Py_BuildValue("OOi", b_array, ranks_array, info);
    Py_DECREF(b_array); Py_DECREF(ranks_array);
    
    return result;
}



/* Python wrapper for nf01bs */
PyObject* py_nf01bs(PyObject* self, PyObject* args) {
    i32 n;
    PyObject *ipar_obj, *j_obj, *e_obj;
    f64 fnorm;
    PyArrayObject *ipar_array, *j_array, *e_array;
    i32 info;

    if (!PyArg_ParseTuple(args, "iOdOO", &n, &ipar_obj, &fnorm, &j_obj, &e_obj)) {
        return NULL;
    }
    
    if (n < 0 || fnorm < 0) {
        PyErr_Format(PyExc_ValueError, "Invalid arguments");
        return NULL;
    }

    ipar_array = (PyArrayObject*)PyArray_FROM_OTF(ipar_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    j_array = (PyArrayObject*)PyArray_FROM_OTF(j_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    e_array = (PyArrayObject*)PyArray_FROM_OTF(e_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    
    if (!ipar_array || !j_array || !e_array) {
         Py_XDECREF(ipar_array); Py_XDECREF(j_array); Py_XDECREF(e_array);
         return NULL;
    }
    
    i32 lipar = (i32)PyArray_SIZE(ipar_array);
    if (lipar < 4) {
         Py_DECREF(ipar_array); Py_DECREF(j_array); Py_DECREF(e_array);
         PyErr_SetString(PyExc_ValueError, "ipar must have length >= 4");
         return NULL;
    }
    
    i32 *ipar_data = (i32*)PyArray_DATA(ipar_array);
    i32 bn = ipar_data[1];
    i32 bsn = ipar_data[3];
    
    i32 ldj = (i32)PyArray_DIM(j_array, 0);
    f64 *j_data = (f64*)PyArray_DATA(j_array);
    f64 *e_data = (f64*)PyArray_DATA(e_array);
    
    npy_intp jnorms_dims[1] = {n};
    PyObject *jnorms_array = PyArray_SimpleNew(1, jnorms_dims, NPY_DOUBLE);
    
    npy_intp ipvt_dims[1] = {n};
    PyObject *ipvt_array = PyArray_SimpleNew(1, ipvt_dims, NPY_INT32);
    
    f64 gnorm;
    
    if (!jnorms_array || !ipvt_array) {
         Py_DECREF(ipar_array); Py_DECREF(j_array); Py_DECREF(e_array);
         Py_XDECREF(jnorms_array); Py_XDECREF(ipvt_array);
         return NULL;
    }
    
    f64 *jnorms_data = (f64*)PyArray_DATA((PyArrayObject*)jnorms_array);
    i32 *ipvt_data = (i32*)PyArray_DATA((PyArrayObject*)ipvt_array);
    
    i32 lwork_size;
    if (n == 0) lwork_size = 1;
    else if (bn <= 1 || bsn == 0) lwork_size = 4*n + 1;
    else {
        /* Conservative estimate for general case */
        i32 st = ipar_data[0];
        i32 bsm = ipar_data[2];
        i32 jwork = bsn + (3*bsn + 1 > st ? 3*bsn + 1 : st);
        if (bsm > bsn) jwork = (jwork > 4*st + 1) ? jwork : 4*st + 1;
        lwork_size = jwork;
    }
    
    f64 *dwork = (f64*)malloc((lwork_size > 0 ? lwork_size : 1) * sizeof(f64));
    if (!dwork) {
         Py_DECREF(ipar_array); Py_DECREF(j_array); Py_DECREF(e_array);
         Py_DECREF(jnorms_array); Py_DECREF(ipvt_array);
         return PyErr_NoMemory();
    }
    
    nf01bs(n, ipar_data, lipar, fnorm, j_data, &ldj, e_data, 
           jnorms_data, &gnorm, ipvt_data, dwork, lwork_size, &info);
           
    free(dwork);
    
    PyArray_ResolveWritebackIfCopy(j_array);
    PyArray_ResolveWritebackIfCopy(e_array);
    Py_DECREF(ipar_array);
    
    PyObject *result = Py_BuildValue("OOOdOi", j_array, e_array, jnorms_array, gnorm, ipvt_array, info);
    
    Py_DECREF(j_array);
    Py_DECREF(e_array);
    Py_DECREF(jnorms_array);
    Py_DECREF(ipvt_array);
    
    return result;
}



/* Python wrapper for nf01ay */
PyObject* py_nf01ay(PyObject* self, PyObject* args) {
    i32 nsmp, nz, l;
    PyObject *ipar_obj, *wb_obj, *z_obj;
    PyArrayObject *ipar_array, *wb_array, *z_array;
    i32 info;

    if (!PyArg_ParseTuple(args, "iiiOOO", &nsmp, &nz, &l, &ipar_obj, &wb_obj, &z_obj)) {
        return NULL;
    }
    
    if (nsmp < 0 || nz < 0 || l < 0) {
        PyErr_Format(PyExc_ValueError, "Dimensions must be non-negative");
        return NULL;
    }

    ipar_array = (PyArrayObject*)PyArray_FROM_OTF(ipar_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    wb_array = (PyArrayObject*)PyArray_FROM_OTF(wb_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    z_array = (PyArrayObject*)PyArray_FROM_OTF(z_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    
    if (!ipar_array || !wb_array || !z_array) {
         Py_XDECREF(ipar_array); Py_XDECREF(wb_array); Py_XDECREF(z_array);
         return NULL;
    }
    
    i32 lipar = (i32)PyArray_SIZE(ipar_array);
    i32 lwb = (i32)PyArray_SIZE(wb_array);
    i32 ldz = (i32)PyArray_DIM(z_array, 0);
    
    if (ldz < 1) ldz = 1;
    
    i32 *ipar_data = (i32*)PyArray_DATA(ipar_array);
    if (lipar < 1) {
         Py_DECREF(ipar_array); Py_DECREF(wb_array); Py_DECREF(z_array);
         PyErr_SetString(PyExc_ValueError, "ipar must have length >= 1");
         return NULL;
    }
    i32 nn = ipar_data[0];
    
    i32 block_size = 64;
    if (block_size > nsmp) block_size = nsmp;
    if (block_size < 2) block_size = 2;
    
    i32 ldwork_target = nn * (block_size + 1);
    if (ldwork_target < 2*nn) ldwork_target = 2*nn;
    
    f64 *dwork = (f64*)malloc(ldwork_target * sizeof(f64));
    if (!dwork) {
         Py_DECREF(ipar_array); Py_DECREF(wb_array); Py_DECREF(z_array);
         return PyErr_NoMemory();
    }
    
    npy_intp y_dims[2] = {nsmp, l};
    npy_intp y_strides[2] = {sizeof(f64), (nsmp > 0 ? nsmp : 1) * sizeof(f64)};
    PyObject *y_array = PyArray_New(&PyArray_Type, 2, y_dims, NPY_DOUBLE, y_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    
    if (!y_array) {
         free(dwork);
         Py_DECREF(ipar_array); Py_DECREF(wb_array); Py_DECREF(z_array);
         return NULL;
    }
    
    i32 ldy = (nsmp > 0) ? nsmp : 1;
    f64 *y_data = (f64*)PyArray_DATA((PyArrayObject*)y_array);
    f64 *wb_data = (f64*)PyArray_DATA(wb_array);
    f64 *z_data = (f64*)PyArray_DATA(z_array);
    
    nf01ay(nsmp, nz, l, ipar_data, lipar, wb_data, lwb, z_data, ldz, y_data, ldy, dwork, ldwork_target, &info);
    
    free(dwork);
    Py_DECREF(ipar_array); 
    Py_DECREF(wb_array); 
    Py_DECREF(z_array);
    
    PyObject *result = Py_BuildValue("Oi", y_array, info);
    Py_DECREF(y_array);
    
    return result;
}



/* Python wrapper for nf01by */
PyObject* py_nf01by(PyObject* self, PyObject* args) {
    const char *cjte;
    i32 nsmp, nz, l;
    PyObject *ipar_obj, *wb_obj, *z_obj, *e_obj;
    PyArrayObject *ipar_array, *wb_array, *z_array, *e_array;
    i32 info;

    if (!PyArg_ParseTuple(args, "siiiOOOO", &cjte, &nsmp, &nz, &l, &ipar_obj, &wb_obj, &z_obj, &e_obj)) {
        return NULL;
    }
    
    if (nsmp < 0 || nz < 0 || l != 1) {
        PyErr_Format(PyExc_ValueError, "Dimensions invalid");
        return NULL;
    }

    ipar_array = (PyArrayObject*)PyArray_FROM_OTF(ipar_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    wb_array = (PyArrayObject*)PyArray_FROM_OTF(wb_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    z_array = (PyArrayObject*)PyArray_FROM_OTF(z_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    e_array = (PyArrayObject*)PyArray_FROM_OTF(e_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    
    if (!ipar_array || !wb_array || !z_array || !e_array) {
         Py_XDECREF(ipar_array); Py_XDECREF(wb_array); Py_XDECREF(z_array); Py_XDECREF(e_array);
         return NULL;
    }
    
    i32 lipar = (i32)PyArray_SIZE(ipar_array);
    i32 lwb = (i32)PyArray_SIZE(wb_array);
    i32 ldz = (i32)PyArray_DIM(z_array, 0);
    
    if (ldz < 1) ldz = 1;
    
    i32 *ipar_data = (i32*)PyArray_DATA(ipar_array);
    if (lipar < 1) {
         Py_DECREF(ipar_array); Py_DECREF(wb_array); Py_DECREF(z_array); Py_DECREF(e_array);
         PyErr_SetString(PyExc_ValueError, "ipar must have length >= 1");
         return NULL;
    }
    i32 nn = ipar_data[0];
    i32 nwb = nn * (nz + 2) + 1;
    
    i32 ldj = (nsmp > 0) ? nsmp : 1;
    
    npy_intp j_dims[2] = {nsmp, nwb};
    npy_intp j_strides[2] = {sizeof(f64), ldj * sizeof(f64)};
    PyObject *j_array = PyArray_New(&PyArray_Type, 2, j_dims, NPY_DOUBLE, j_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    
    npy_intp jte_dims[1] = {nwb};
    PyObject *jte_array = PyArray_SimpleNew(1, jte_dims, NPY_DOUBLE);
    
    if (!j_array || !jte_array) {
         Py_XDECREF(j_array); Py_XDECREF(jte_array);
         Py_DECREF(ipar_array); Py_DECREF(wb_array); Py_DECREF(z_array); Py_DECREF(e_array);
         return NULL;
    }
    
    i32 ldwork = 2 * nn;
    if (ldwork < 1) ldwork = 1;
    
    f64 *dwork = (f64*)malloc(ldwork * sizeof(f64));
    if (!dwork) {
         Py_DECREF(j_array); Py_DECREF(jte_array);
         Py_DECREF(ipar_array); Py_DECREF(wb_array); Py_DECREF(z_array); Py_DECREF(e_array);
         return PyErr_NoMemory();
    }
    
    f64 *wb_data = (f64*)PyArray_DATA(wb_array);
    f64 *z_data = (f64*)PyArray_DATA(z_array);
    f64 *e_data = (f64*)PyArray_DATA(e_array);
    f64 *j_data = (f64*)PyArray_DATA((PyArrayObject*)j_array);
    f64 *jte_data = (f64*)PyArray_DATA((PyArrayObject*)jte_array);
    
    nf01by(cjte, nsmp, nz, l, ipar_data, lipar, wb_data, lwb, z_data, ldz, e_data, 
           j_data, ldj, jte_data, dwork, ldwork, &info);
           
    free(dwork);
    Py_DECREF(ipar_array); Py_DECREF(wb_array); Py_DECREF(z_array); Py_DECREF(e_array);
    
    PyObject *result = Py_BuildValue("OOi", j_array, jte_array, info);
    Py_DECREF(j_array); Py_DECREF(jte_array);
    
    return result;
}

