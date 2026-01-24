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



typedef struct {
    PyObject *fcn_callable;
    PyObject *jac_callable;
    i32 m;
    i32 n;
} md03bd_callback_data;

static md03bd_callback_data *g_cb_data = NULL;

static void md03bd_fcn_wrapper(
    i32* iflag, i32 m, i32 n, i32* ipar, i32 lipar,
    const f64* dpar1, i32 ldpar1, const f64* dpar2, i32 ldpar2,
    const f64* x, i32* nfevl, f64* e, f64* j, i32* ldj,
    f64* dwork, i32 ldwork, i32* info
)
{
    *info = 0;
    *nfevl = 0;

    if (*iflag == 3) {
        *ldj = m;
        ipar[0] = m * n;
        ipar[1] = 0;
        ipar[2] = 0;
        ipar[3] = 4*n + 1;
        ipar[4] = 4*n;
        return;
    }

    if (*iflag == 0 || g_cb_data == NULL) {
        return;
    }

    npy_intp x_dims[1] = {n};
    PyObject *x_array = PyArray_SimpleNewFromData(1, x_dims, NPY_DOUBLE, (void*)x);
    if (x_array == NULL) {
        *info = -1;
        return;
    }
    PyArray_CLEARFLAGS((PyArrayObject*)x_array, NPY_ARRAY_WRITEABLE);

    if (*iflag == 1) {
        PyObject *result = PyObject_CallFunctionObjArgs(g_cb_data->fcn_callable, x_array, NULL);
        Py_DECREF(x_array);

        if (result == NULL) {
            PyErr_Print();
            *info = -1;
            return;
        }

        PyArrayObject *e_result = (PyArrayObject*)PyArray_FROM_OTF(result, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
        Py_DECREF(result);

        if (e_result == NULL || PyArray_SIZE(e_result) != m) {
            Py_XDECREF(e_result);
            *info = -1;
            return;
        }

        f64 *e_data = (f64*)PyArray_DATA(e_result);
        for (i32 i = 0; i < m; i++) {
            e[i] = e_data[i];
        }

        Py_DECREF(e_result);

    } else if (*iflag == 2) {
        PyObject *result = PyObject_CallFunctionObjArgs(g_cb_data->jac_callable, x_array, NULL);
        Py_DECREF(x_array);

        if (result == NULL) {
            PyErr_Print();
            *info = -1;
            return;
        }

        PyArrayObject *j_result = (PyArrayObject*)PyArray_FROM_OTF(result, NPY_DOUBLE, NPY_ARRAY_FARRAY);
        Py_DECREF(result);

        if (j_result == NULL || PyArray_DIM(j_result, 0) != m || PyArray_DIM(j_result, 1) != n) {
            Py_XDECREF(j_result);
            *info = -1;
            return;
        }

        f64 *j_data = (f64*)PyArray_DATA(j_result);
        for (i32 col = 0; col < n; col++) {
            for (i32 row = 0; row < m; row++) {
                j[row + col * m] = j_data[row + col * m];
            }
        }

        Py_DECREF(j_result);
    }
}

static void md03bd_qrfact_wrapper(
    i32 n, const i32* ipar, i32 lipar, f64 fnorm,
    f64* j, i32* ldj, f64* e, f64* jnorms, f64* gnorm,
    i32* ipvt, f64* dwork, i32 ldwork, i32* info
)
{
    i32 m = g_cb_data->m;
    md03bx(m, n, fnorm, j, ldj, e, jnorms, gnorm, ipvt, dwork, ldwork, info);
}

static void md03bd_lmparm_wrapper(
    const char* cond, i32 n, const i32* ipar, i32 lipar,
    f64* r, i32 ldr, const i32* ipvt, const f64* diag,
    const f64* qtb, f64 delta, f64* par, i32* rank,
    f64* x, f64* rx, f64 tol, f64* dwork, i32 ldwork, i32* info
)
{
    md03by(cond, n, r, ldr, ipvt, diag, qtb, delta, par, rank, x, rx, tol, dwork, ldwork, info);
}

PyObject* py_md03bd(PyObject* self, PyObject* args, PyObject* kwargs) {
    i32 m, n, itmax = 100;
    f64 ftol = -1.0, xtol = -1.0, gtol = -1.0;
    PyObject *x_obj, *fcn_obj, *jac_obj;

    static char *kwlist[] = {"m", "n", "x", "fcn", "jac", "itmax", "ftol", "xtol", "gtol", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "iiOOO|iddd", kwlist,
                                     &m, &n, &x_obj, &fcn_obj, &jac_obj,
                                     &itmax, &ftol, &xtol, &gtol)) {
        return NULL;
    }

    if (m < 0 || n < 0 || n > m) {
        PyErr_SetString(PyExc_ValueError, "Invalid dimensions: m >= n >= 0 required");
        return NULL;
    }

    if (!PyCallable_Check(fcn_obj) || !PyCallable_Check(jac_obj)) {
        PyErr_SetString(PyExc_TypeError, "fcn and jac must be callable");
        return NULL;
    }

    PyArrayObject *x_array = (PyArrayObject*)PyArray_FROM_OTF(x_obj, NPY_DOUBLE,
                                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (x_array == NULL) {
        return NULL;
    }

    if (PyArray_SIZE(x_array) != n) {
        PyErr_SetString(PyExc_ValueError, "x must have length n");
        Py_DECREF(x_array);
        return NULL;
    }

    f64 *x_data = (f64*)PyArray_DATA(x_array);
    f64 *diag = (f64*)calloc(n, sizeof(f64));
    i32 *iwork = (i32*)calloc(n + 1, sizeof(i32));

    i32 sizej = m * n;
    i32 lfcn1 = 0;
    i32 lfcn2 = 0;
    i32 lqrf = 4*n + 1;
    i32 llmp = 4*n;

    i32 max1 = (lfcn1 > lfcn2) ? lfcn1 : lfcn2;
    max1 = (max1 > (n + lqrf)) ? max1 : (n + lqrf);
    i32 max2 = (m + lfcn1 > n + llmp) ? (m + lfcn1) : (n + llmp);
    i32 max3 = (n*n + n + max2 > sizej + max1) ? (n*n + n + max2) : (sizej + max1);
    i32 ldwork = (m + max3 > 4) ? (m + max3) : 4;
    f64 *dwork = (f64*)malloc(ldwork * sizeof(f64));

    if (diag == NULL || iwork == NULL || dwork == NULL) {
        free(diag);
        free(iwork);
        free(dwork);
        Py_DECREF(x_array);
        return PyErr_NoMemory();
    }

    md03bd_callback_data cb_data;
    cb_data.fcn_callable = fcn_obj;
    cb_data.jac_callable = jac_obj;
    cb_data.m = m;
    cb_data.n = n;

    g_cb_data = &cb_data;

    i32 lipar = 5;
    i32 ipar_storage[5];
    for (i32 k = 0; k < 5; k++) ipar_storage[k] = 0;

    i32 nfev, njev, iwarn, info;
    const char xinit = 'G';
    const char scale = 'I';
    const char cond = 'N';
    f64 factor = 100.0;
    i32 nprint = 0;
    f64 tol_rank = -1.0;

    md03bd(&xinit, &scale, &cond,
           md03bd_fcn_wrapper, md03bd_qrfact_wrapper, md03bd_lmparm_wrapper,
           m, n, itmax, factor, nprint,
           ipar_storage, lipar,
           NULL, 0,
           NULL, 0,
           x_data, diag, &nfev, &njev,
           ftol, xtol, gtol, tol_rank,
           iwork, dwork, ldwork, &iwarn, &info);

    g_cb_data = NULL;
    f64 fnorm = dwork[1];

    npy_intp x_dims[1] = {n};
    PyObject *x_out = PyArray_EMPTY(1, x_dims, NPY_DOUBLE, 0);
    if (x_out == NULL) {
        free(diag);
        free(iwork);
        free(dwork);
        Py_DECREF(x_array);
        return NULL;
    }

    f64 *x_out_data = (f64*)PyArray_DATA((PyArrayObject*)x_out);
    for (i32 i = 0; i < n; i++) {
        x_out_data[i] = x_data[i];
    }

    free(diag);
    free(iwork);
    free(dwork);

    PyArray_ResolveWritebackIfCopy(x_array);
    PyObject *result = Py_BuildValue("(Oiidii)", x_out, nfev, njev, fnorm, iwarn, info);
    Py_DECREF(x_array);
    Py_DECREF(x_out);

    return result;
}



PyObject* py_md03by(PyObject* self, PyObject* args, PyObject* kwargs) {
    static char* kwlist[] = {"cond", "n", "r", "ipvt", "diag", "qtb", "delta", "par", "rank", "tol", NULL};

    char* cond;
    i32 n, rank_in;
    f64 delta, par, tol;
    PyObject *r_obj, *ipvt_obj, *diag_obj, *qtb_obj;
    PyArrayObject *r_array, *ipvt_array, *diag_array, *qtb_array;
    i32 info;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "siOOOOddid", kwlist,
                                     &cond, &n, &r_obj, &ipvt_obj, &diag_obj, &qtb_obj,
                                     &delta, &par, &rank_in, &tol)) {
        return NULL;
    }

    if (n < 0) {
        PyErr_SetString(PyExc_ValueError, "n must be non-negative");
        return NULL;
    }

    if (delta <= 0.0) {
        PyErr_SetString(PyExc_ValueError, "delta must be positive");
        return NULL;
    }

    if (par < 0.0) {
        PyErr_SetString(PyExc_ValueError, "par must be non-negative");
        return NULL;
    }

    r_array = (PyArrayObject*)PyArray_FROM_OTF(r_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (r_array == NULL) return NULL;

    ipvt_array = (PyArrayObject*)PyArray_FROM_OTF(ipvt_obj, NPY_INT32, NPY_ARRAY_IN_FARRAY);
    if (ipvt_array == NULL) {
        Py_DECREF(r_array);
        return NULL;
    }

    diag_array = (PyArrayObject*)PyArray_FROM_OTF(diag_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (diag_array == NULL) {
        Py_DECREF(r_array);
        Py_DECREF(ipvt_array);
        return NULL;
    }

    qtb_array = (PyArrayObject*)PyArray_FROM_OTF(qtb_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (qtb_array == NULL) {
        Py_DECREF(r_array);
        Py_DECREF(ipvt_array);
        Py_DECREF(diag_array);
        return NULL;
    }

    i32 ldr = (i32)PyArray_DIM(r_array, 0);
    f64* r_data = (f64*)PyArray_DATA(r_array);
    i32* ipvt_data = (i32*)PyArray_DATA(ipvt_array);
    f64* diag_data = (f64*)PyArray_DATA(diag_array);
    f64* qtb_data = (f64*)PyArray_DATA(qtb_array);

    bool econd = (*cond == 'E' || *cond == 'e');
    i32 ldwork = econd ? 4*n : 2*n;
    if (ldwork < 1) ldwork = 1;
    f64* dwork = (f64*)malloc(ldwork * sizeof(f64));
    if (dwork == NULL) {
        Py_DECREF(r_array);
        Py_DECREF(ipvt_array);
        Py_DECREF(diag_array);
        Py_DECREF(qtb_array);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate workspace");
        return NULL;
    }

    npy_intp x_dims[1] = {n > 0 ? n : 0};
    PyObject* x_array = PyArray_ZEROS(1, x_dims, NPY_DOUBLE, 0);
    if (x_array == NULL) {
        free(dwork);
        Py_DECREF(r_array);
        Py_DECREF(ipvt_array);
        Py_DECREF(diag_array);
        Py_DECREF(qtb_array);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate x array");
        return NULL;
    }
    f64* x_data = (n > 0) ? (f64*)PyArray_DATA((PyArrayObject*)x_array) : NULL;

    PyObject* rx_array = PyArray_ZEROS(1, x_dims, NPY_DOUBLE, 0);
    if (rx_array == NULL) {
        free(dwork);
        Py_DECREF(x_array);
        Py_DECREF(r_array);
        Py_DECREF(ipvt_array);
        Py_DECREF(diag_array);
        Py_DECREF(qtb_array);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate rx array");
        return NULL;
    }
    f64* rx_data = (n > 0) ? (f64*)PyArray_DATA((PyArrayObject*)rx_array) : NULL;

    i32 rank = rank_in;

    md03by(cond, n, r_data, ldr, ipvt_data, diag_data, qtb_data, delta,
           &par, &rank, x_data, rx_data, tol, dwork, ldwork, &info);

    free(dwork);

    PyArray_ResolveWritebackIfCopy(r_array);

    if (info < 0) {
        Py_DECREF(r_array);
        Py_DECREF(ipvt_array);
        Py_DECREF(diag_array);
        Py_DECREF(qtb_array);
        Py_DECREF(x_array);
        Py_DECREF(rx_array);
        PyErr_Format(PyExc_ValueError, "md03by: parameter %d is invalid", -info);
        return NULL;
    }

    PyObject* result = Py_BuildValue("OdiOOi", r_array, par, rank, x_array, rx_array, info);

    Py_DECREF(r_array);
    Py_DECREF(ipvt_array);
    Py_DECREF(diag_array);
    Py_DECREF(qtb_array);
    Py_DECREF(x_array);
    Py_DECREF(rx_array);

    return result;
}



/* Python wrapper for md03bb */
PyObject* py_md03bb(PyObject* self, PyObject* args, PyObject* kwargs) {
    static char* kwlist[] = {"cond", "n", "ipar", "r", "ipvt", "diag", "qtb", "delta", "par", "ranks", "tol", NULL};

    char* cond;
    i32 n;
    f64 delta, par, tol;
    PyObject *ipar_obj, *r_obj, *ipvt_obj, *diag_obj, *qtb_obj, *ranks_obj;
    PyArrayObject *ipar_array, *r_array, *ipvt_array, *diag_array, *qtb_array, *ranks_array;
    i32 info;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "siOOOOOddOd", kwlist,
                                     &cond, &n, &ipar_obj, &r_obj, &ipvt_obj, &diag_obj, &qtb_obj,
                                     &delta, &par, &ranks_obj, &tol)) {
        return NULL;
    }

    if (n < 0) {
        PyErr_SetString(PyExc_ValueError, "n must be non-negative");
        return NULL;
    }

    if (delta <= 0.0) {
        PyErr_SetString(PyExc_ValueError, "delta must be positive");
        return NULL;
    }

    if (par < 0.0) {
        PyErr_SetString(PyExc_ValueError, "par must be non-negative");
        return NULL;
    }
    
    ipar_array = (PyArrayObject*)PyArray_FROM_OTF(ipar_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    if (ipar_array == NULL) return NULL;

    r_array = (PyArrayObject*)PyArray_FROM_OTF(r_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (r_array == NULL) {
        Py_DECREF(ipar_array);
        return NULL;
    }

    ipvt_array = (PyArrayObject*)PyArray_FROM_OTF(ipvt_obj, NPY_INT32, NPY_ARRAY_IN_FARRAY);
    if (ipvt_array == NULL) {
        Py_DECREF(ipar_array);
        Py_DECREF(r_array);
        return NULL;
    }

    diag_array = (PyArrayObject*)PyArray_FROM_OTF(diag_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (diag_array == NULL) {
        Py_DECREF(ipar_array);
        Py_DECREF(r_array);
        Py_DECREF(ipvt_array);
        return NULL;
    }

    qtb_array = (PyArrayObject*)PyArray_FROM_OTF(qtb_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (qtb_array == NULL) {
        Py_DECREF(ipar_array);
        Py_DECREF(r_array);
        Py_DECREF(ipvt_array);
        Py_DECREF(diag_array);
        return NULL;
    }

    ranks_array = (PyArrayObject*)PyArray_FROM_OTF(ranks_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (ranks_array == NULL) {
        Py_DECREF(ipar_array);
        Py_DECREF(r_array);
        Py_DECREF(ipvt_array);
        Py_DECREF(diag_array);
        Py_DECREF(qtb_array);
        return NULL;
    }

    i32 lipar = (i32)PyArray_SIZE(ipar_array);
    i32 *ipar_data = (i32*)PyArray_DATA(ipar_array);
    
    i32 ldr = (i32)PyArray_DIM(r_array, 0);
    f64* r_data = (f64*)PyArray_DATA(r_array);
    i32* ipvt_data = (i32*)PyArray_DATA(ipvt_array);
    f64* diag_data = (f64*)PyArray_DATA(diag_array);
    f64* qtb_data = (f64*)PyArray_DATA(qtb_array);
    i32* ranks_data = (i32*)PyArray_DATA(ranks_array);

    bool econd = (*cond == 'E' || *cond == 'e');
    i32 ldwork = econd ? 4*n : 2*n;
    if (ldwork < 1) ldwork = 1;
    f64* dwork = (f64*)malloc(ldwork * sizeof(f64));
    if (dwork == NULL) {
        Py_DECREF(ipar_array);
        Py_DECREF(r_array);
        Py_DECREF(ipvt_array);
        Py_DECREF(diag_array);
        Py_DECREF(qtb_array);
        Py_DECREF(ranks_array);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate workspace");
        return NULL;
    }

    npy_intp x_dims[1] = {n > 0 ? n : 0};
    PyObject* x_array = PyArray_EMPTY(1, x_dims, NPY_DOUBLE, 0);
    if (x_array == NULL) {
        free(dwork);
        Py_DECREF(ipar_array);
        Py_DECREF(r_array);
        Py_DECREF(ipvt_array);
        Py_DECREF(diag_array);
        Py_DECREF(qtb_array);
        Py_DECREF(ranks_array);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate x array");
        return NULL;
    }
    f64* x_data = (f64*)PyArray_DATA((PyArrayObject*)x_array);
    if (n > 0) memset(x_data, 0, n * sizeof(f64));

    PyObject* rx_array = PyArray_EMPTY(1, x_dims, NPY_DOUBLE, 0);
    if (rx_array == NULL) {
        free(dwork);
        Py_DECREF(x_array);
        Py_DECREF(ipar_array);
        Py_DECREF(r_array);
        Py_DECREF(ipvt_array);
        Py_DECREF(diag_array);
        Py_DECREF(qtb_array);
        Py_DECREF(ranks_array);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate rx array");
        return NULL;
    }
    f64* rx_data = (f64*)PyArray_DATA((PyArrayObject*)rx_array);
    if (n > 0) memset(rx_data, 0, n * sizeof(f64));

    md03bb(cond, n, ipar_data, lipar, r_data, ldr, ipvt_data, diag_data, qtb_data, delta,
           &par, ranks_data, x_data, rx_data, tol, dwork, ldwork, &info);

    free(dwork);

    PyArray_ResolveWritebackIfCopy(r_array);
    PyArray_ResolveWritebackIfCopy(ranks_array);
    Py_DECREF(ipar_array);

    if (info < 0) {
        Py_DECREF(r_array);
        Py_DECREF(ipvt_array);
        Py_DECREF(diag_array);
        Py_DECREF(qtb_array);
        Py_DECREF(x_array);
        Py_DECREF(rx_array);
        Py_DECREF(ranks_array);
        PyErr_Format(PyExc_ValueError, "md03bb: parameter %d is invalid", -info);
        return NULL;
    }

    PyObject* result = Py_BuildValue("OdOOOi", r_array, par, ranks_array, x_array, rx_array, info);

    Py_DECREF(r_array);
    Py_DECREF(ipvt_array);
    Py_DECREF(diag_array);
    Py_DECREF(qtb_array);
    Py_DECREF(x_array);
    Py_DECREF(rx_array);
    Py_DECREF(ranks_array);

    return result;
}



/* Python wrapper for md03ba */
PyObject* py_md03ba(PyObject* self, PyObject* args) {
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
    i32 *ipar_data = (i32*)PyArray_DATA(ipar_array);
    
    if (lipar < 1) {
         Py_DECREF(ipar_array); Py_DECREF(j_array); Py_DECREF(e_array);
         return NULL;
    }
    
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
    
    i32 ldwork = (n > 1) ? (4*n + 1) : 1;
    f64 *dwork = (f64*)malloc(ldwork * sizeof(f64));
    if (!dwork) {
         Py_DECREF(ipar_array); Py_DECREF(j_array); Py_DECREF(e_array);
         Py_DECREF(jnorms_array); Py_DECREF(ipvt_array);
         return PyErr_NoMemory();
    }
    
    md03ba(n, ipar_data, lipar, fnorm, j_data, &ldj, e_data, 
           jnorms_data, &gnorm, ipvt_data, dwork, ldwork, &info);
           
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


PyObject* py_md03bf(PyObject* self, PyObject* args) {
    (void)self;

    i32 iflag;
    PyObject *x_obj;

    if (!PyArg_ParseTuple(args, "iO", &iflag, &x_obj)) {
        return NULL;
    }

    if (iflag < 1 || iflag > 3) {
        PyErr_SetString(PyExc_ValueError, "iflag must be 1, 2, or 3");
        return NULL;
    }

    PyArrayObject *x_array = (PyArrayObject*)PyArray_FROM_OTF(x_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (x_array == NULL) return NULL;

    i32 m = 15;
    i32 n = 3;

    if (iflag != 3 && PyArray_SIZE(x_array) < n) {
        Py_DECREF(x_array);
        PyErr_SetString(PyExc_ValueError, "x must have at least 3 elements");
        return NULL;
    }

    f64 *x_data = (f64*)PyArray_DATA(x_array);

    i32 ipar[5] = {0};
    i32 lipar = 5;
    i32 ldj = m;
    i32 nfevl = 0;
    i32 info = 0;

    npy_intp e_dims[1] = {m};
    npy_intp j_dims[2] = {m, n};

    PyObject *e_array = PyArray_ZEROS(1, e_dims, NPY_DOUBLE, 0);
    PyObject *j_array = PyArray_ZEROS(2, j_dims, NPY_DOUBLE, 1);

    if (!e_array || !j_array) {
        Py_XDECREF(e_array);
        Py_XDECREF(j_array);
        Py_DECREF(x_array);
        return PyErr_NoMemory();
    }

    f64 *e_data = (f64*)PyArray_DATA((PyArrayObject*)e_array);
    f64 *j_data = (f64*)PyArray_DATA((PyArrayObject*)j_array);

    md03bf(&iflag, m, n, ipar, lipar, NULL, 0, NULL, 0,
           x_data, &nfevl, e_data, j_data, &ldj, NULL, 0, &info);

    Py_DECREF(x_array);

    if (iflag == 1) {
        Py_DECREF(j_array);
        return Py_BuildValue("Oi", e_array, info);
    } else if (iflag == 2) {
        PyObject *result = Py_BuildValue("OOii", j_array, e_array, nfevl, info);
        Py_DECREF(e_array);
        Py_DECREF(j_array);
        return result;
    } else {
        Py_DECREF(e_array);
        Py_DECREF(j_array);
        return Py_BuildValue("(iiiii)i", ipar[0], ipar[1], ipar[2], ipar[3], ipar[4], info);
    }
}

