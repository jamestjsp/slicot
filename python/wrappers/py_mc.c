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



/* Python wrapper for mc01pd */
PyObject* py_mc01pd(PyObject* self, PyObject* args) {
    PyObject *rez_obj, *imz_obj;
    PyArrayObject *rez_array, *imz_array;

    if (!PyArg_ParseTuple(args, "OO", &rez_obj, &imz_obj)) {
        return NULL;
    }

    rez_array = (PyArrayObject*)PyArray_FROM_OTF(rez_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (rez_array == NULL) {
        return NULL;
    }

    imz_array = (PyArrayObject*)PyArray_FROM_OTF(imz_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (imz_array == NULL) {
        Py_DECREF(rez_array);
        return NULL;
    }

    i32 k = (i32)PyArray_SIZE(rez_array);
    if (k != (i32)PyArray_SIZE(imz_array)) {
        PyErr_SetString(PyExc_ValueError, "REZ and IMZ must have same length");
        Py_DECREF(rez_array);
        Py_DECREF(imz_array);
        return NULL;
    }

    f64 *rez_data = (f64*)PyArray_DATA(rez_array);
    f64 *imz_data = (f64*)PyArray_DATA(imz_array);

    npy_intp dims[1] = {k + 1};
    PyObject *p_array = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    if (p_array == NULL) {
        Py_DECREF(rez_array);
        Py_DECREF(imz_array);
        return NULL;
    }
    f64 *p = (f64*)PyArray_DATA((PyArrayObject*)p_array);

    f64 *dwork = (f64*)malloc((size_t)(k + 1) * sizeof(f64));
    if (!dwork) {
        Py_DECREF(p_array);
        Py_DECREF(rez_array);
        Py_DECREF(imz_array);
        PyErr_NoMemory();
        return NULL;
    }

    i32 info;
    mc01pd(k, rez_data, imz_data, p, dwork, &info);

    Py_DECREF(rez_array);
    Py_DECREF(imz_array);
    free(dwork);

    return Py_BuildValue("Oi", p_array, info);
}



/* Python wrapper for mc01td */
PyObject* py_mc01td(PyObject* self, PyObject* args) {
    const char *dico_str;
    PyObject *p_obj;
    PyArrayObject *p_array;

    if (!PyArg_ParseTuple(args, "sO", &dico_str, &p_obj)) {
        return NULL;
    }

    char d = toupper((unsigned char)dico_str[0]);
    if (d != 'C' && d != 'D') {
        PyErr_SetString(PyExc_ValueError, "DICO must be 'C' or 'D'");
        return NULL;
    }

    p_array = (PyArrayObject*)PyArray_FROM_OTF(p_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (p_array == NULL) {
        return NULL;
    }

    i32 n = (i32)PyArray_SIZE(p_array);
    if (n < 1) {
        PyErr_SetString(PyExc_ValueError, "Polynomial must have at least 1 coefficient");
        Py_DECREF(p_array);
        return NULL;
    }

    i32 dp = n - 1;
    f64 *p_data = (f64*)PyArray_DATA(p_array);

    bool stable;
    i32 nz, dp_out, iwarn, info;

    mc01td(dico_str, dp, p_data, &stable, &nz, &dp_out, &iwarn, &info);

    Py_DECREF(p_array);

    PyObject *stable_obj = stable ? Py_True : Py_False;
    Py_INCREF(stable_obj);

    return Py_BuildValue("Oiiii", stable_obj, nz, dp_out, iwarn, info);
}



/* Python wrapper for mc01sx */
PyObject* py_mc01sx(PyObject* self, PyObject* args) {
    PyObject *e_obj, *mant_obj;
    PyArrayObject *e_array, *mant_array;

    if (!PyArg_ParseTuple(args, "OO", &e_obj, &mant_obj)) {
        return NULL;
    }

    e_array = (PyArrayObject*)PyArray_FROM_OTF(e_obj, NPY_INT32, NPY_ARRAY_FARRAY);
    if (e_array == NULL) {
        return NULL;
    }

    mant_array = (PyArrayObject*)PyArray_FROM_OTF(mant_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (mant_array == NULL) {
        Py_DECREF(e_array);
        return NULL;
    }

    i32 n = (i32)PyArray_SIZE(e_array);
    if (n != (i32)PyArray_SIZE(mant_array)) {
        PyErr_SetString(PyExc_ValueError, "E and MANT must have same length");
        Py_DECREF(e_array);
        Py_DECREF(mant_array);
        return NULL;
    }

    if (n < 1) {
        PyErr_SetString(PyExc_ValueError, "Arrays must have at least 1 element");
        Py_DECREF(e_array);
        Py_DECREF(mant_array);
        return NULL;
    }

    i32 *e_data = (i32*)PyArray_DATA(e_array);
    f64 *mant_data = (f64*)PyArray_DATA(mant_array);

    i32 variation = mc01sx(1, n, e_data, mant_data);

    Py_DECREF(e_array);
    Py_DECREF(mant_array);

    return PyLong_FromLong(variation);
}



/* Python wrapper for mc01md */
PyObject* py_mc01md(PyObject* self, PyObject* args) {
    double alpha;
    int k;
    PyObject *p_obj;
    PyArrayObject *p_array;

    if (!PyArg_ParseTuple(args, "diO", &alpha, &k, &p_obj)) {
        return NULL;
    }

    p_array = (PyArrayObject*)PyArray_FROM_OTF(p_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (p_array == NULL) {
        return NULL;
    }

    i32 n = (i32)PyArray_SIZE(p_array);
    i32 dp = n - 1;

    f64 *p_data = (f64*)PyArray_DATA(p_array);

    npy_intp dims[1] = {n};
    PyObject *q_array = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    if (q_array == NULL) {
        Py_DECREF(p_array);
        return NULL;
    }
    f64 *q = (f64*)PyArray_DATA((PyArrayObject*)q_array);

    i32 info;
    mc01md(dp, alpha, (i32)k, p_data, q, &info);

    Py_DECREF(p_array);

    return Py_BuildValue("Oi", q_array, info);
}


/* Python wrapper for mc01nd */
PyObject* py_mc01nd(PyObject* self, PyObject* args) {
    double xr, xi;
    PyObject *p_obj;
    PyArrayObject *p_array;

    if (!PyArg_ParseTuple(args, "ddO", &xr, &xi, &p_obj)) {
        return NULL;
    }

    p_array = (PyArrayObject*)PyArray_FROM_OTF(p_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (p_array == NULL) {
        return NULL;
    }

    i32 n = (i32)PyArray_SIZE(p_array);
    i32 dp = n - 1;

    f64 *p_data = (f64*)PyArray_DATA(p_array);

    f64 vr, vi;
    i32 info;
    mc01nd(dp, xr, xi, p_data, &vr, &vi, &info);

    Py_DECREF(p_array);

    return Py_BuildValue("ddi", vr, vi, info);
}


/* Python wrapper for mc01od */
PyObject* py_mc01od(PyObject* self, PyObject* args) {
    PyObject *rez_obj, *imz_obj;
    PyArrayObject *rez_array, *imz_array;

    if (!PyArg_ParseTuple(args, "OO", &rez_obj, &imz_obj)) {
        return NULL;
    }

    rez_array = (PyArrayObject*)PyArray_FROM_OTF(rez_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (rez_array == NULL) {
        return NULL;
    }

    imz_array = (PyArrayObject*)PyArray_FROM_OTF(imz_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (imz_array == NULL) {
        Py_DECREF(rez_array);
        return NULL;
    }

    i32 k = (i32)PyArray_SIZE(rez_array);
    if (k != (i32)PyArray_SIZE(imz_array)) {
        PyErr_SetString(PyExc_ValueError, "REZ and IMZ must have same length");
        Py_DECREF(rez_array);
        Py_DECREF(imz_array);
        return NULL;
    }

    f64 *rez_data = (f64*)PyArray_DATA(rez_array);
    f64 *imz_data = (f64*)PyArray_DATA(imz_array);

    npy_intp dims[1] = {k + 1};
    PyObject *rep_array = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    if (rep_array == NULL) {
        Py_DECREF(rez_array);
        Py_DECREF(imz_array);
        return NULL;
    }
    f64 *rep = (f64*)PyArray_DATA((PyArrayObject*)rep_array);

    PyObject *imp_array = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    if (imp_array == NULL) {
        Py_DECREF(rep_array);
        Py_DECREF(rez_array);
        Py_DECREF(imz_array);
        return NULL;
    }
    f64 *imp = (f64*)PyArray_DATA((PyArrayObject*)imp_array);

    f64 *dwork = (f64*)malloc((size_t)(2 * k + 2) * sizeof(f64));
    if (!dwork) {
        Py_DECREF(rep_array);
        Py_DECREF(imp_array);
        Py_DECREF(rez_array);
        Py_DECREF(imz_array);
        PyErr_NoMemory();
        return NULL;
    }

    i32 info;
    mc01od(k, rez_data, imz_data, rep, imp, dwork, &info);

    Py_DECREF(rez_array);
    Py_DECREF(imz_array);
    free(dwork);

    return Py_BuildValue("OOi", rep_array, imp_array, info);
}

/* Python wrapper for mc01py */
PyObject* py_mc01py(PyObject* self, PyObject* args) {
    PyObject *rez_obj, *imz_obj;
    PyArrayObject *rez_array, *imz_array;

    if (!PyArg_ParseTuple(args, "OO", &rez_obj, &imz_obj)) {
        return NULL;
    }

    rez_array = (PyArrayObject*)PyArray_FROM_OTF(rez_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (rez_array == NULL) {
        return NULL;
    }

    imz_array = (PyArrayObject*)PyArray_FROM_OTF(imz_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (imz_array == NULL) {
        Py_DECREF(rez_array);
        return NULL;
    }

    i32 k = (i32)PyArray_SIZE(rez_array);
    if (k != (i32)PyArray_SIZE(imz_array)) {
        PyErr_SetString(PyExc_ValueError, "REZ and IMZ must have same length");
        Py_DECREF(rez_array);
        Py_DECREF(imz_array);
        return NULL;
    }

    f64 *rez_data = (f64*)PyArray_DATA(rez_array);
    f64 *imz_data = (f64*)PyArray_DATA(imz_array);

    npy_intp dims[1] = {k + 1};
    PyObject *p_array = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    if (p_array == NULL) {
        Py_DECREF(rez_array);
        Py_DECREF(imz_array);
        return NULL;
    }
    f64 *p = (f64*)PyArray_DATA((PyArrayObject*)p_array);

    f64 *dwork = (f64*)malloc((size_t)(k + 1) * sizeof(f64));
    if (!dwork) {
        Py_DECREF(p_array);
        Py_DECREF(rez_array);
        Py_DECREF(imz_array);
        PyErr_NoMemory();
        return NULL;
    }

    i32 info;
    mc01py(k, rez_data, imz_data, p, dwork, &info);

    Py_DECREF(rez_array);
    Py_DECREF(imz_array);
    free(dwork);

    return Py_BuildValue("Oi", p_array, info);
}

/* Python wrapper for mc01qd - polynomial division */
PyObject* py_mc01qd(PyObject* self, PyObject* args) {
    PyObject *a_obj, *b_obj;
    PyArrayObject *a_array, *b_array;

    if (!PyArg_ParseTuple(args, "OO", &a_obj, &b_obj)) {
        return NULL;
    }

    a_array = (PyArrayObject *)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                NPY_ARRAY_IN_FARRAY);
    if (a_array == NULL) {
        return NULL;
    }

    b_array = (PyArrayObject *)PyArray_FROM_OTF(b_obj, NPY_DOUBLE,
                NPY_ARRAY_IN_FARRAY);
    if (b_array == NULL) {
        Py_DECREF(a_array);
        return NULL;
    }

    i32 da = (i32)PyArray_SIZE(a_array) - 1;
    i32 db = (i32)PyArray_SIZE(b_array) - 1;

    if (db < 0) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        PyErr_SetString(PyExc_ValueError, "Divisor polynomial b must be non-empty");
        return NULL;
    }

    const f64 *a = (const f64 *)PyArray_DATA(a_array);
    const f64 *b = (const f64 *)PyArray_DATA(b_array);

    npy_intp rq_size = (da >= 0) ? da + 1 : 0;
    npy_intp dims[1] = {rq_size};
    PyObject *rq_array = PyArray_ZEROS(1, dims, NPY_DOUBLE, 0);
    if (rq_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        return NULL;
    }

    f64 *rq = (f64 *)PyArray_DATA((PyArrayObject *)rq_array);
    i32 iwarn, info;

    mc01qd(da, &db, a, b, rq, &iwarn, &info);

    Py_DECREF(a_array);
    Py_DECREF(b_array);

    return Py_BuildValue("Oiii", rq_array, db, iwarn, info);
}

/* Python wrapper for mc01rd - polynomial P(x) = P1(x)*P2(x) + alpha*P3(x) */
PyObject* py_mc01rd(PyObject* self, PyObject* args, PyObject* kwargs) {
    static char *kwlist[] = {"p1", "p2", "p3", "alpha", "dp1", "dp2", "dp3", NULL};
    PyObject *p1_obj, *p2_obj, *p3_obj;
    f64 alpha;
    int dp1_arg = -999, dp2_arg = -999, dp3_arg = -999;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOOd|iii", kwlist,
                                     &p1_obj, &p2_obj, &p3_obj, &alpha,
                                     &dp1_arg, &dp2_arg, &dp3_arg)) {
        return NULL;
    }

    PyArrayObject *p1_array = (PyArrayObject*)PyArray_FROM_OTF(
        p1_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (p1_array == NULL) return NULL;

    PyArrayObject *p2_array = (PyArrayObject*)PyArray_FROM_OTF(
        p2_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (p2_array == NULL) {
        Py_DECREF(p1_array);
        return NULL;
    }

    PyArrayObject *p3_array = (PyArrayObject*)PyArray_FROM_OTF(
        p3_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (p3_array == NULL) {
        Py_DECREF(p1_array);
        Py_DECREF(p2_array);
        return NULL;
    }

    i32 dp1 = (dp1_arg != -999) ? dp1_arg : (i32)PyArray_SIZE(p1_array) - 1;
    i32 dp2 = (dp2_arg != -999) ? dp2_arg : (i32)PyArray_SIZE(p2_array) - 1;
    i32 dp3 = (dp3_arg != -999) ? dp3_arg : (i32)PyArray_SIZE(p3_array) - 1;

    i32 max_dp12 = (dp1 >= 0 && dp2 >= 0) ? dp1 + dp2 : -1;
    i32 max_deg = (max_dp12 > dp3) ? max_dp12 : dp3;
    if (max_deg < 0) max_deg = 0;
    npy_intp out_size = max_deg + 1;

    npy_intp dims[1] = {out_size};
    PyObject *result_array = PyArray_ZEROS(1, dims, NPY_DOUBLE, 0);
    if (result_array == NULL) {
        Py_DECREF(p1_array);
        Py_DECREF(p2_array);
        Py_DECREF(p3_array);
        return NULL;
    }
    f64 *p3_out = (f64*)PyArray_DATA((PyArrayObject*)result_array);

    const f64 *p3_in = (const f64*)PyArray_DATA(p3_array);
    npy_intp p3_len = PyArray_SIZE(p3_array);
    for (npy_intp i = 0; i < p3_len && i < out_size; i++) {
        p3_out[i] = p3_in[i];
    }

    const f64 *p1 = (const f64*)PyArray_DATA(p1_array);
    const f64 *p2 = (const f64*)PyArray_DATA(p2_array);
    i32 info;

    mc01rd(dp1, dp2, &dp3, alpha, p1, p2, p3_out, &info);

    Py_DECREF(p1_array);
    Py_DECREF(p2_array);
    Py_DECREF(p3_array);

    return Py_BuildValue("Oii", result_array, dp3, info);
}

/* Python wrapper for mc01sw */
PyObject* py_mc01sw(PyObject* self, PyObject* args) {
    f64 a;
    i32 b;

    if (!PyArg_ParseTuple(args, "di", &a, &b)) {
        return NULL;
    }

    if (b < 2) {
        PyErr_SetString(PyExc_ValueError, "Base b must be >= 2");
        return NULL;
    }

    f64 m;
    i32 e;
    mc01sw(a, b, &m, &e);

    return Py_BuildValue("di", m, e);
}

/* Python wrapper for mc01sy */
PyObject* py_mc01sy(PyObject* self, PyObject* args) {
    f64 m;
    i32 e, b;

    if (!PyArg_ParseTuple(args, "dii", &m, &e, &b)) {
        return NULL;
    }

    if (b < 2) {
        PyErr_SetString(PyExc_ValueError, "Base b must be >= 2");
        return NULL;
    }

    f64 a;
    bool ovflow;
    mc01sy(m, e, b, &a, &ovflow);

    return Py_BuildValue("dO", a, ovflow ? Py_True : Py_False);
}

/* Python wrapper for mc03nx */
PyObject* py_mc03nx(PyObject* self, PyObject* args) {
    PyObject *p_obj;
    PyArrayObject *p_array;

    if (!PyArg_ParseTuple(args, "O", &p_obj)) {
        return NULL;
    }

    p_array = (PyArrayObject*)PyArray_FROM_OTF(p_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (p_array == NULL) {
        return NULL;
    }

    if (PyArray_NDIM(p_array) != 3) {
        Py_DECREF(p_array);
        PyErr_SetString(PyExc_ValueError, "p must be a 3D array (mp, np, dp+1)");
        return NULL;
    }

    npy_intp *p_dims = PyArray_DIMS(p_array);
    i32 mp = (i32)p_dims[0];
    i32 np = (i32)p_dims[1];
    i32 dp = (i32)p_dims[2] - 1;

    if (dp < 1) {
        Py_DECREF(p_array);
        PyErr_SetString(PyExc_ValueError, "Polynomial degree dp must be >= 1");
        return NULL;
    }

    i32 ldp1 = mp;
    i32 ldp2 = np;

    i32 nrows = dp * mp;
    i32 ncols = (dp - 1) * mp + np;
    i32 lda = nrows;
    i32 lde = nrows;

    npy_intp dims[2] = {nrows, ncols};
    npy_intp strides[2] = {sizeof(f64), lda * sizeof(f64)};

    PyObject *a_array = PyArray_New(&PyArray_Type, 2, dims, NPY_DOUBLE,
                                    strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (a_array == NULL) {
        Py_DECREF(p_array);
        return NULL;
    }
    f64 *a = (f64*)PyArray_DATA((PyArrayObject*)a_array);
    memset(a, 0, (size_t)lda * ncols * sizeof(f64));

    PyObject *e_array = PyArray_New(&PyArray_Type, 2, dims, NPY_DOUBLE,
                                    strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (e_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(p_array);
        return NULL;
    }
    f64 *e = (f64*)PyArray_DATA((PyArrayObject*)e_array);
    memset(e, 0, (size_t)lde * ncols * sizeof(f64));

    f64 *p_data = (f64*)PyArray_DATA(p_array);
    mc03nx(mp, np, dp, p_data, ldp1, ldp2, a, lda, e, lde);

    Py_DECREF(p_array);

    return Py_BuildValue("OO", a_array, e_array);
}

/* Python wrapper for mc01wd */
PyObject* py_mc01wd(PyObject* self, PyObject* args) {
    PyObject *p_obj;
    f64 u1, u2;

    if (!PyArg_ParseTuple(args, "Odd", &p_obj, &u1, &u2)) {
        return NULL;
    }

    PyArrayObject *p_array = (PyArrayObject*)PyArray_FROM_OTF(
        p_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (p_array == NULL) {
        return NULL;
    }

    if (PyArray_NDIM(p_array) != 1) {
        Py_DECREF(p_array);
        PyErr_SetString(PyExc_ValueError, "p must be a 1D array");
        return NULL;
    }

    npy_intp n = PyArray_SIZE(p_array);
    i32 dp = (i32)(n - 1);

    if (dp < 0) {
        Py_DECREF(p_array);
        PyErr_SetString(PyExc_ValueError, "Polynomial degree dp must be >= 0");
        return NULL;
    }

    npy_intp dims[1] = {n};
    PyObject *q_array = PyArray_ZEROS(1, dims, NPY_DOUBLE, 0);
    if (q_array == NULL) {
        Py_DECREF(p_array);
        return NULL;
    }
    f64 *q = (f64*)PyArray_DATA((PyArrayObject*)q_array);

    f64 *p_data = (f64*)PyArray_DATA(p_array);
    i32 info;

    mc01wd(dp, p_data, u1, u2, q, &info);

    Py_DECREF(p_array);

    return Py_BuildValue("Oi", q_array, info);
}

PyObject* py_mc01xd(PyObject* self, PyObject* args) {
    f64 alpha, beta, gamma, delta;

    if (!PyArg_ParseTuple(args, "dddd", &alpha, &beta, &gamma, &delta)) {
        return NULL;
    }

    npy_intp dims[1] = {3};
    PyObject *evr_array = PyArray_ZEROS(1, dims, NPY_DOUBLE, 0);
    if (evr_array == NULL) {
        return NULL;
    }
    f64 *evr = (f64*)PyArray_DATA((PyArrayObject*)evr_array);

    PyObject *evi_array = PyArray_ZEROS(1, dims, NPY_DOUBLE, 0);
    if (evi_array == NULL) {
        Py_DECREF(evr_array);
        return NULL;
    }
    f64 *evi = (f64*)PyArray_DATA((PyArrayObject*)evi_array);

    PyObject *evq_array = PyArray_ZEROS(1, dims, NPY_DOUBLE, 0);
    if (evq_array == NULL) {
        Py_DECREF(evr_array);
        Py_DECREF(evi_array);
        return NULL;
    }
    f64 *evq = (f64*)PyArray_DATA((PyArrayObject*)evq_array);

    i32 ldwork = -1;
    f64 work_query[2];
    i32 info;

    mc01xd(alpha, beta, gamma, delta, evr, evi, evq, work_query, ldwork, &info);

    ldwork = (i32)work_query[0];
    f64 *dwork = (f64*)calloc(ldwork, sizeof(f64));
    if (dwork == NULL) {
        Py_DECREF(evr_array);
        Py_DECREF(evi_array);
        Py_DECREF(evq_array);
        PyErr_NoMemory();
        return NULL;
    }

    mc01xd(alpha, beta, gamma, delta, evr, evi, evq, dwork, ldwork, &info);

    free(dwork);

    return Py_BuildValue("OOOi", evr_array, evi_array, evq_array, info);
}

/* Python wrapper for mc03md - polynomial matrix P(x) = P1(x)*P2(x) + alpha*P3(x) */
PyObject* py_mc03md(PyObject* self, PyObject* args, PyObject* kwargs) {
    static char *kwlist[] = {"p1", "p2", "p3", "alpha", "dp1", "dp2", "dp3", NULL};
    PyObject *p1_obj, *p2_obj, *p3_obj;
    f64 alpha;
    int dp1_arg = -999, dp2_arg = -999, dp3_arg = -999;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOOd|iii", kwlist,
                                     &p1_obj, &p2_obj, &p3_obj, &alpha,
                                     &dp1_arg, &dp2_arg, &dp3_arg)) {
        return NULL;
    }

    PyArrayObject *p1_array = (PyArrayObject*)PyArray_FROM_OTF(
        p1_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (p1_array == NULL) return NULL;

    PyArrayObject *p2_array = (PyArrayObject*)PyArray_FROM_OTF(
        p2_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (p2_array == NULL) {
        Py_DECREF(p1_array);
        return NULL;
    }

    PyArrayObject *p3_array = (PyArrayObject*)PyArray_FROM_OTF(
        p3_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (p3_array == NULL) {
        Py_DECREF(p1_array);
        Py_DECREF(p2_array);
        return NULL;
    }

    int p1_ndim = PyArray_NDIM(p1_array);
    int p2_ndim = PyArray_NDIM(p2_array);
    int p3_ndim = PyArray_NDIM(p3_array);

    npy_intp *p1_dims = PyArray_DIMS(p1_array);
    npy_intp *p2_dims = PyArray_DIMS(p2_array);
    npy_intp *p3_dims = PyArray_DIMS(p3_array);

    i32 rp1, cp1, cp2, dp1, dp2, dp3;

    if (p1_ndim == 3) {
        rp1 = (i32)p1_dims[0];
        cp1 = (i32)p1_dims[1];
        dp1 = (dp1_arg != -999) ? dp1_arg : (i32)p1_dims[2] - 1;
    } else {
        Py_DECREF(p1_array);
        Py_DECREF(p2_array);
        Py_DECREF(p3_array);
        PyErr_SetString(PyExc_ValueError, "p1 must be a 3D array (rp1, cp1, dp1+1)");
        return NULL;
    }

    if (p2_ndim == 3) {
        cp2 = (i32)p2_dims[1];
        dp2 = (dp2_arg != -999) ? dp2_arg : (i32)p2_dims[2] - 1;
    } else {
        Py_DECREF(p1_array);
        Py_DECREF(p2_array);
        Py_DECREF(p3_array);
        PyErr_SetString(PyExc_ValueError, "p2 must be a 3D array (cp1, cp2, dp2+1)");
        return NULL;
    }

    if (p3_ndim == 3) {
        dp3 = (dp3_arg != -999) ? dp3_arg : (i32)p3_dims[2] - 1;
    } else {
        Py_DECREF(p1_array);
        Py_DECREF(p2_array);
        Py_DECREF(p3_array);
        PyErr_SetString(PyExc_ValueError, "p3 must be a 3D array (rp1, cp2, dp3+1)");
        return NULL;
    }

    i32 max_dp12 = (dp1 >= 0 && dp2 >= 0) ? dp1 + dp2 : -1;
    i32 max_deg = (max_dp12 > dp3) ? max_dp12 : dp3;
    if (max_deg < 0) max_deg = 0;

    i32 ldp11 = (rp1 > 1) ? rp1 : 1;
    i32 ldp12 = (cp1 > 1) ? cp1 : 1;
    i32 ldp21 = (cp1 > 1) ? cp1 : 1;
    i32 ldp22 = (cp2 > 1) ? cp2 : 1;
    i32 ldp31 = (rp1 > 1) ? rp1 : 1;
    i32 ldp32 = (cp2 > 1) ? cp2 : 1;

    npy_intp out_size = (npy_intp)ldp31 * ldp32 * (max_deg + 1);

    npy_intp dims[3] = {rp1, cp2, max_deg + 1};
    npy_intp strides[3] = {sizeof(f64), ldp31 * sizeof(f64), ldp31 * ldp32 * sizeof(f64)};
    PyObject *result_array = PyArray_New(&PyArray_Type, 3, dims, NPY_DOUBLE,
                                         strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (result_array == NULL) {
        Py_DECREF(p1_array);
        Py_DECREF(p2_array);
        Py_DECREF(p3_array);
        return NULL;
    }
    f64 *p3_out = (f64*)PyArray_DATA((PyArrayObject*)result_array);
    memset(p3_out, 0, out_size * sizeof(f64));

    f64 *dwork = (f64*)malloc((cp1 > 1 ? cp1 : 1) * sizeof(f64));
    if (dwork == NULL) {
        Py_DECREF(result_array);
        Py_DECREF(p1_array);
        Py_DECREF(p2_array);
        Py_DECREF(p3_array);
        PyErr_NoMemory();
        return NULL;
    }

    const f64 *p3_in = (const f64*)PyArray_DATA(p3_array);
    npy_intp p3_size = PyArray_SIZE(p3_array);
    for (npy_intp i = 0; i < p3_size && i < out_size; i++) {
        p3_out[i] = p3_in[i];
    }

    const f64 *p1 = (const f64*)PyArray_DATA(p1_array);
    const f64 *p2 = (const f64*)PyArray_DATA(p2_array);
    i32 info;

    SLC_MC03MD(rp1, cp1, cp2, dp1, dp2, &dp3, alpha,
               p1, ldp11, ldp12, p2, ldp21, ldp22,
               p3_out, ldp31, ldp32, dwork, &info);

    free(dwork);
    Py_DECREF(p1_array);
    Py_DECREF(p2_array);
    Py_DECREF(p3_array);

    return Py_BuildValue("Oii", result_array, dp3, info);
}

PyObject* py_mc03ny(PyObject* self, PyObject* args) {
    i32 nblcks, nra, nca;
    PyObject *a_obj, *e_obj, *imuk_obj, *inuk_obj;
    PyArrayObject *a_array, *e_array, *imuk_array, *inuk_array;

    if (!PyArg_ParseTuple(args, "iiiOOOO", &nblcks, &nra, &nca,
                          &a_obj, &e_obj, &imuk_obj, &inuk_obj)) {
        return NULL;
    }

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                                                NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (a_array == NULL) {
        return NULL;
    }

    e_array = (PyArrayObject*)PyArray_FROM_OTF(e_obj, NPY_DOUBLE,
                                                NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (e_array == NULL) {
        Py_DECREF(a_array);
        return NULL;
    }

    imuk_array = (PyArrayObject*)PyArray_FROM_OTF(imuk_obj, NPY_INT32,
                                                   NPY_ARRAY_CARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (imuk_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(e_array);
        return NULL;
    }

    inuk_array = (PyArrayObject*)PyArray_FROM_OTF(inuk_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    if (inuk_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(e_array);
        Py_DECREF(imuk_array);
        return NULL;
    }

    i32 lda = nra > 1 ? nra : 1;
    i32 lde = nra > 1 ? nra : 1;
    i32 ldveps = nca > 1 ? nca : 1;

    i32 ncv = 0;
    i32 *imuk_data = (i32*)PyArray_DATA(imuk_array);
    i32 *inuk_data = (i32*)PyArray_DATA(inuk_array);
    if (nblcks > 0 && nca >= 0) {
        for (i32 i = 0; i < nblcks; i++) {
            ncv += (i + 1) * (imuk_data[i] - inuk_data[i]);
        }
    }
    if (ncv < 1) ncv = 1;

    npy_intp veps_dims[2] = {nca, ncv};
    npy_intp veps_strides[2] = {sizeof(f64), ldveps * sizeof(f64)};

    PyObject *veps_array = PyArray_New(&PyArray_Type, 2, veps_dims, NPY_DOUBLE,
                                       veps_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (veps_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(e_array);
        Py_DECREF(imuk_array);
        Py_DECREF(inuk_array);
        return NULL;
    }
    f64 *veps = (f64*)PyArray_DATA((PyArrayObject*)veps_array);
    if (nca > 0 && ncv > 0) {
        memset(veps, 0, (size_t)ldveps * ncv * sizeof(f64));
    }

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *e_data = (f64*)PyArray_DATA(e_array);

    i32 info;
    mc03ny(nblcks, nra, nca, a_data, lda, e_data, lde,
           imuk_data, inuk_data, veps, ldveps, &info);

    PyArray_ResolveWritebackIfCopy(imuk_array);

    Py_DECREF(a_array);
    Py_DECREF(e_array);
    Py_DECREF(inuk_array);

    npy_intp imuk_dims[1] = {nblcks > 0 ? nblcks : 1};
    PyObject *imuk_out_array = PyArray_ZEROS(1, imuk_dims, NPY_INT32, 0);
    if (imuk_out_array == NULL) {
        Py_DECREF(veps_array);
        Py_DECREF(imuk_array);
        return NULL;
    }
    i32 *imuk_out = (i32*)PyArray_DATA((PyArrayObject*)imuk_out_array);
    for (i32 i = 0; i < nblcks; i++) {
        imuk_out[i] = imuk_data[i];
    }

    Py_DECREF(imuk_array);

    return Py_BuildValue("OOi", veps_array, imuk_out_array, info);
}

/* Python wrapper for mc01sd */
PyObject* py_mc01sd(PyObject* self, PyObject* args) {
    PyObject *p_obj;
    PyArrayObject *p_array;

    if (!PyArg_ParseTuple(args, "O", &p_obj)) {
        return NULL;
    }

    p_array = (PyArrayObject *)PyArray_FROM_OTF(p_obj, NPY_DOUBLE,
                                                NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (p_array == NULL) {
        return NULL;
    }

    if (PyArray_NDIM(p_array) != 1) {
        PyErr_SetString(PyExc_ValueError, "p must be 1D array");
        Py_DECREF(p_array);
        return NULL;
    }

    npy_intp len = PyArray_DIM(p_array, 0);
    if (len < 1) {
        PyErr_SetString(PyExc_ValueError, "p must have at least 1 element");
        Py_DECREF(p_array);
        return NULL;
    }

    i32 dp = (i32)(len - 1);
    f64 *p_data = (f64 *)PyArray_DATA(p_array);

    npy_intp dims[1] = {len};

    PyObject *mant_array = PyArray_ZEROS(1, dims, NPY_DOUBLE, 0);
    if (mant_array == NULL) {
        Py_DECREF(p_array);
        return NULL;
    }
    f64 *mant = (f64 *)PyArray_DATA((PyArrayObject *)mant_array);

    PyObject *e_array = PyArray_ZEROS(1, dims, NPY_INT32, 0);
    if (e_array == NULL) {
        Py_DECREF(mant_array);
        Py_DECREF(p_array);
        return NULL;
    }
    i32 *e = (i32 *)PyArray_DATA((PyArrayObject *)e_array);

    i32 *iwork = (i32 *)calloc(len, sizeof(i32));
    if (iwork == NULL) {
        Py_DECREF(e_array);
        Py_DECREF(mant_array);
        Py_DECREF(p_array);
        PyErr_NoMemory();
        return NULL;
    }

    i32 s, t, info;
    mc01sd(dp, p_data, &s, &t, mant, e, iwork, &info);

    free(iwork);

    PyArray_ResolveWritebackIfCopy(p_array);

    PyObject *result = Py_BuildValue("OiiOOi", p_array, s, t, mant_array, e_array, info);

    Py_DECREF(p_array);
    Py_DECREF(mant_array);
    Py_DECREF(e_array);

    return result;
}

PyObject* py_mc01vd(PyObject* self, PyObject* args)
{
    (void)self;
    double a, b, c;

    if (!PyArg_ParseTuple(args, "ddd", &a, &b, &c)) {
        return NULL;
    }

    f64 z1re, z1im, z2re, z2im;
    i32 info;

    mc01vd(a, b, c, &z1re, &z1im, &z2re, &z2im, &info);

    return Py_BuildValue("ddddi", z1re, z1im, z2re, z2im, info);
}

PyObject* py_mc03nd(PyObject* self, PyObject* args)
{
    (void)self;
    int mp, np_dim, dp;
    PyObject *p_obj;
    double tol;

    if (!PyArg_ParseTuple(args, "iiiOd", &mp, &np_dim, &dp, &p_obj, &tol)) {
        return NULL;
    }

    if (dp <= 0) {
        PyErr_SetString(PyExc_ValueError, "dp must be >= 1");
        return NULL;
    }

    PyArrayObject *p_array = (PyArrayObject*)PyArray_FROM_OTF(
        p_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY);
    if (p_array == NULL) {
        return NULL;
    }

    if (PyArray_NDIM(p_array) != 3) {
        PyErr_SetString(PyExc_ValueError, "P must be a 3D array");
        Py_DECREF(p_array);
        return NULL;
    }

    npy_intp *dims = PyArray_DIMS(p_array);
    if (dims[0] != mp || dims[1] != np_dim || dims[2] != dp + 1) {
        PyErr_SetString(PyExc_ValueError, "P dimensions must be (MP, NP, DP+1)");
        Py_DECREF(p_array);
        return NULL;
    }

    i32 m = dp * mp;
    i32 h = m - mp;
    i32 n = h + np_dim;

    i32 ldp1 = (mp > 1) ? mp : 1;
    i32 ldp2 = (np_dim > 1) ? np_dim : 1;
    i32 ldnull = (np_dim > 1) ? np_dim : 1;
    i32 ldker1 = (np_dim > 1) ? np_dim : 1;
    i32 ldker2 = (np_dim > 1) ? np_dim : 1;

    i32 max_n_mp1 = (n > m + 1) ? n : (m + 1);
    i32 liwork = m + 2 * max_n_mp1 + n;
    i32 ldwork = n * (m * n + 2 * (m + n));

    i32 gam_size = m + 1;
    i32 nullsp_cols = gam_size * np_dim;
    i32 ker_depth = gam_size;

    i32 *gam_buf = (i32*)calloc(gam_size, sizeof(i32));
    i32 *iwork = (i32*)calloc(liwork, sizeof(i32));
    f64 *dwork = (f64*)calloc(ldwork, sizeof(f64));
    f64 *nullsp_buf = (f64*)calloc(ldnull * nullsp_cols, sizeof(f64));
    f64 *ker_buf = (f64*)calloc(ldker1 * ldker2 * ker_depth, sizeof(f64));

    if (gam_buf == NULL || iwork == NULL || dwork == NULL || nullsp_buf == NULL || ker_buf == NULL) {
        free(gam_buf);
        free(iwork);
        free(dwork);
        free(nullsp_buf);
        free(ker_buf);
        Py_DECREF(p_array);
        PyErr_NoMemory();
        return NULL;
    }

    f64 *p_data = (f64*)PyArray_DATA(p_array);

    i32 dk, info;
    mc03nd(mp, np_dim, dp, p_data, ldp1, ldp2,
           &dk, gam_buf, nullsp_buf, ldnull,
           ker_buf, ldker1, ldker2, tol,
           iwork, dwork, ldwork, &info);

    free(iwork);
    free(dwork);

    if (info != 0 || dk < 0) {
        free(gam_buf);
        free(nullsp_buf);
        free(ker_buf);
        Py_DECREF(p_array);
        return Py_BuildValue("iOOOi", dk, Py_None, Py_None, Py_None, info);
    }

    i32 nk = 0;
    i32 ncv = 0;
    for (i32 i = 0; i <= dk; i++) {
        nk += gam_buf[i];
        ncv += (i + 1) * gam_buf[i];
    }

    npy_intp gam_dims[1] = {dk + 1};
    PyObject *gam_array = PyArray_SimpleNew(1, gam_dims, NPY_INT32);
    if (gam_array == NULL) {
        free(gam_buf);
        free(nullsp_buf);
        free(ker_buf);
        Py_DECREF(p_array);
        return NULL;
    }
    memcpy(PyArray_DATA((PyArrayObject *)gam_array), gam_buf, (dk + 1) * sizeof(i32));

    npy_intp nullsp_dims[2] = {np_dim, ncv};
    npy_intp nullsp_strides[2] = {sizeof(f64), np_dim * (npy_intp)sizeof(f64)};
    PyObject *nullsp_array = PyArray_New(&PyArray_Type, 2, nullsp_dims, NPY_DOUBLE,
                                         nullsp_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (nullsp_array == NULL) {
        free(nullsp_buf);
        free(ker_buf);
        Py_DECREF(p_array);
        Py_DECREF(gam_array);
        return NULL;
    }
    memcpy(PyArray_DATA((PyArrayObject *)nullsp_array), nullsp_buf,
           (size_t)np_dim * ncv * sizeof(f64));
    free(nullsp_buf);

    npy_intp ker_dims[3] = {np_dim, nk, dk + 1};
    PyObject *ker_array = PyArray_EMPTY(3, ker_dims, NPY_DOUBLE, 1);
    if (ker_array == NULL) {
        free(ker_buf);
        Py_DECREF(p_array);
        Py_DECREF(gam_array);
        Py_DECREF(nullsp_array);
        return NULL;
    }
    f64 *ker_out = (f64*)PyArray_DATA((PyArrayObject*)ker_array);
    for (i32 k = 0; k <= dk; k++) {
        for (i32 j = 0; j < nk; j++) {
            memcpy(ker_out + k * np_dim * nk + j * np_dim,
                   ker_buf + k * ldker1 * ldker2 + j * ldker1,
                   np_dim * sizeof(f64));
        }
    }
    free(gam_buf);
    free(ker_buf);

    PyObject *result = Py_BuildValue("iOOOi", dk, gam_array, nullsp_array, ker_array, info);

    Py_DECREF(p_array);
    Py_DECREF(gam_array);
    Py_DECREF(nullsp_array);
    Py_DECREF(ker_array);

    return result;
}
