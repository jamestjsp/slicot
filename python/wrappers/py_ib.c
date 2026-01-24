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



/* Python wrapper for ib01od */
PyObject* py_ib01od(PyObject* self, PyObject* args) {
    const char *ctrl_str;
    i32 nobr, l;
    f64 tol;
    PyObject *sv_obj;
    PyArrayObject *sv_array;
    i32 n, iwarn, info;

    if (!PyArg_ParseTuple(args, "siiOd", &ctrl_str, &nobr, &l, &sv_obj, &tol)) {
        return NULL;
    }

    char ctrl = ctrl_str[0];

    /* Validate CTRL parameter */
    if (ctrl != 'C' && ctrl != 'c' && ctrl != 'N' && ctrl != 'n') {
        PyErr_SetString(PyExc_ValueError, "CTRL must be 'C' or 'N'");
        return NULL;
    }

    /* Validate NOBR */
    if (nobr <= 0) {
        PyErr_SetString(PyExc_ValueError, "NOBR must be positive");
        return NULL;
    }

    /* Validate L */
    if (l <= 0) {
        PyErr_SetString(PyExc_ValueError, "L must be positive");
        return NULL;
    }

    /* Convert SV array */
    sv_array = (PyArrayObject*)PyArray_FROM_OTF(sv_obj, NPY_DOUBLE,
                                                NPY_ARRAY_IN_FARRAY);
    if (sv_array == NULL) {
        return NULL;
    }

    /* Validate SV size */
    npy_intp sv_size = PyArray_SIZE(sv_array);
    i32 lnobr = l * nobr;
    if (sv_size < lnobr) {
        Py_DECREF(sv_array);
        PyErr_SetString(PyExc_ValueError, "SV must have at least L*NOBR elements");
        return NULL;
    }

    f64 *sv = (f64*)PyArray_DATA(sv_array);

    /* Call C function */
    SLC_IB01OD(ctrl, nobr, l, sv, &n, tol, &iwarn, &info);

    Py_DECREF(sv_array);

    if (info < 0) {
        PyErr_Format(PyExc_ValueError, "Parameter %d had an illegal value", -info);
        return NULL;
    }

    return Py_BuildValue("iii", n, iwarn, info);
}



/* Python wrapper for ib01nd */
PyObject* py_ib01nd(PyObject* self, PyObject* args) {
    const char *meth_str, *jobd_str;
    i32 nobr, m, l;
    f64 tol;
    PyObject *r_obj;
    PyArrayObject *r_array;
    i32 iwarn, info;

    if (!PyArg_ParseTuple(args, "ssiiiOd", &meth_str, &jobd_str, &nobr, &m, &l,
                          &r_obj, &tol)) {
        return NULL;
    }

    char meth = meth_str[0];
    char jobd = jobd_str[0];

    /* Validate METH */
    if (meth != 'M' && meth != 'm' && meth != 'N' && meth != 'n') {
        PyErr_SetString(PyExc_ValueError, "METH must be 'M' or 'N'");
        return NULL;
    }

    /* Validate JOBD for MOESP */
    if ((meth == 'M' || meth == 'm') &&
        jobd != 'M' && jobd != 'm' && jobd != 'N' && jobd != 'n') {
        PyErr_SetString(PyExc_ValueError, "JOBD must be 'M' or 'N'");
        return NULL;
    }

    /* Validate dimensions */
    if (nobr <= 0) {
        PyErr_SetString(PyExc_ValueError, "NOBR must be positive");
        return NULL;
    }
    if (m < 0) {
        PyErr_SetString(PyExc_ValueError, "M must be non-negative");
        return NULL;
    }
    if (l <= 0) {
        PyErr_SetString(PyExc_ValueError, "L must be positive");
        return NULL;
    }

    /* Convert R array */
    r_array = (PyArrayObject*)PyArray_FROM_OTF(r_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (r_array == NULL) {
        return NULL;
    }

    /* Get dimensions */
    int ndim = PyArray_NDIM(r_array);
    if (ndim != 2) {
        Py_DECREF(r_array);
        PyErr_SetString(PyExc_ValueError, "R must be a 2D array");
        return NULL;
    }

    npy_intp *r_dims = PyArray_DIMS(r_array);
    i32 ldr = (i32)r_dims[0];
    i32 nr = 2 * (m + l) * nobr;

    /* Validate R size */
    if (ldr < nr || r_dims[1] < nr) {
        Py_DECREF(r_array);
        PyErr_SetString(PyExc_ValueError, "R must be at least 2*(m+l)*nobr x 2*(m+l)*nobr");
        return NULL;
    }

    f64 *r_data = (f64*)PyArray_DATA(r_array);

    /* Allocate output singular values */
    i32 lnobr = l * nobr;
    npy_intp sv_dims[1] = {lnobr};
    PyArrayObject *sv_array = (PyArrayObject*)PyArray_SimpleNew(1, sv_dims, NPY_DOUBLE);
    if (sv_array == NULL) {
        Py_DECREF(r_array);
        return NULL;
    }
    f64 *sv = (f64*)PyArray_DATA(sv_array);

    /* Allocate workspace */
    i32 lmnobr = lnobr + m * nobr;
    i32 ldwork;
    bool moesp = (meth == 'M' || meth == 'm');
    bool jobdm = (jobd == 'M' || jobd == 'm');

    if (moesp) {
        if (jobdm) {
            i32 t1 = (2 * m - 1) * nobr;
            if (t1 < 0) t1 = 1;
            i32 t2 = lmnobr;
            i32 t3 = 5 * lnobr;
            ldwork = t1 > t2 ? t1 : t2;
            ldwork = ldwork > t3 ? ldwork : t3;
        } else {
            ldwork = 5 * lnobr;
        }
    } else {
        ldwork = 5 * lmnobr + 1;
    }
    ldwork = ldwork > 1 ? ldwork : 1;

    f64 *dwork = (f64*)malloc(ldwork * sizeof(f64));
    i32 *iwork = (i32*)malloc(lmnobr * sizeof(i32));
    if (dwork == NULL || iwork == NULL) {
        free(dwork);
        free(iwork);
        Py_DECREF(r_array);
        Py_DECREF(sv_array);
        PyErr_NoMemory();
        return NULL;
    }

    /* Call C function */
    SLC_IB01ND(meth, jobd, nobr, m, l, r_data, ldr, sv, tol,
               iwork, dwork, ldwork, &iwarn, &info);

    /* Get rcond values for N4SID */
    f64 rcond1 = 0.0, rcond2 = 0.0;
    if (!moesp) {
        rcond1 = dwork[1];
        rcond2 = dwork[2];
    }

    free(dwork);
    free(iwork);

    if (info < 0) {
        Py_DECREF(r_array);
        Py_DECREF(sv_array);
        PyErr_Format(PyExc_ValueError, "Parameter %d had an illegal value", -info);
        return NULL;
    }

    /* Build result: (r, sv, rcond1, rcond2, iwarn, info) */
    PyObject *result = Py_BuildValue("OOddii", r_array, sv_array,
                                      rcond1, rcond2, iwarn, info);
    Py_DECREF(r_array);
    Py_DECREF(sv_array);
    return result;
}



/* Python wrapper for ib01ad - System identification driver */
PyObject* py_ib01ad(PyObject* self, PyObject* args) {
    const char *meth_str, *alg_str, *jobd_str, *batch_str, *conct_str, *ctrl_str;
    i32 nobr, m, l;
    f64 rcond, tol;
    PyObject *u_obj, *y_obj;
    PyArrayObject *u_array = NULL, *y_array = NULL;
    i32 n, iwarn, info;

    if (!PyArg_ParseTuple(args, "ssssssiiiOOdd",
                          &meth_str, &alg_str, &jobd_str, &batch_str, &conct_str,
                          &ctrl_str, &nobr, &m, &l, &u_obj, &y_obj, &rcond, &tol)) {
        return NULL;
    }

    char meth = toupper((unsigned char)meth_str[0]);
    char alg = toupper((unsigned char)alg_str[0]);
    char jobd = toupper((unsigned char)jobd_str[0]);
    char batch = toupper((unsigned char)batch_str[0]);

    if (meth != 'M' && meth != 'N') {
        PyErr_SetString(PyExc_ValueError, "METH must be 'M' or 'N'");
        return NULL;
    }
    if (alg != 'C' && alg != 'F' && alg != 'Q') {
        PyErr_SetString(PyExc_ValueError, "ALG must be 'C', 'F', or 'Q'");
        return NULL;
    }
    if (meth == 'M' && jobd != 'M' && jobd != 'N') {
        PyErr_SetString(PyExc_ValueError, "JOBD must be 'M' or 'N' for MOESP");
        return NULL;
    }
    if (batch != 'F' && batch != 'I' && batch != 'L' && batch != 'O') {
        PyErr_SetString(PyExc_ValueError, "BATCH must be 'F', 'I', 'L', or 'O'");
        return NULL;
    }
    if (nobr <= 0) {
        PyErr_SetString(PyExc_ValueError, "NOBR must be positive");
        return NULL;
    }
    if (m < 0) {
        PyErr_SetString(PyExc_ValueError, "M must be non-negative");
        return NULL;
    }
    if (l <= 0) {
        PyErr_SetString(PyExc_ValueError, "L must be positive");
        return NULL;
    }

    u_array = (PyArrayObject*)PyArray_FROM_OTF(u_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    y_array = (PyArrayObject*)PyArray_FROM_OTF(y_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);

    if (!u_array || !y_array) {
        Py_XDECREF(u_array);
        Py_XDECREF(y_array);
        return NULL;
    }

    npy_intp *y_dims = PyArray_DIMS(y_array);
    i32 nsmp = (i32)y_dims[0];
    i32 ldu = (m > 0) ? nsmp : 1;
    i32 ldy = nsmp;

    i32 nobr2 = 2 * nobr;
    i32 nr = 2 * (m + l) * nobr;
    bool onebch = (batch == 'O');
    i32 min_nsmp = onebch ? (2 * (m + l + 1) * nobr - 1) : nobr2;

    if (nsmp < min_nsmp) {
        Py_DECREF(u_array);
        Py_DECREF(y_array);
        PyErr_SetString(PyExc_ValueError, "NSMP too small for the problem");
        return NULL;
    }

    const f64 *u_data = (const f64*)PyArray_DATA(u_array);
    const f64 *y_data = (const f64*)PyArray_DATA(y_array);

    i32 lnobr = l * nobr;
    i32 lmnobr = lnobr + m * nobr;
    bool jobdm = (jobd == 'M');

    i32 ldr = nr;
    if (meth == 'M' && jobdm && 3 * m * nobr > ldr) {
        ldr = 3 * m * nobr;
    }

    f64 *r_data = (f64*)calloc(ldr * nr, sizeof(f64));
    if (!r_data) {
        Py_DECREF(u_array);
        Py_DECREF(y_array);
        return PyErr_NoMemory();
    }

    npy_intp sv_dims[1] = {lnobr};
    PyArrayObject *sv_array = (PyArrayObject*)PyArray_SimpleNew(1, sv_dims, NPY_DOUBLE);
    if (!sv_array) {
        free(r_data);
        Py_DECREF(u_array);
        Py_DECREF(y_array);
        return PyErr_NoMemory();
    }
    f64 *sv = (f64*)PyArray_DATA(sv_array);

    i32 liwork = (meth == 'N') ? lmnobr : ((m + l > 3) ? (m + l) : 3);
    i32 *iwork = (i32*)calloc(liwork, sizeof(i32));
    if (!iwork) {
        free(r_data);
        Py_DECREF(u_array);
        Py_DECREF(y_array);
        Py_DECREF(sv_array);
        return PyErr_NoMemory();
    }

    i32 ns = nsmp - 2 * nobr + 1;
    i32 ldwork;

    if (alg == 'C') {
        ldwork = 5 * lnobr;
        if (meth == 'M' && jobdm) {
            i32 t1 = 2 * m * nobr - nobr;
            if (t1 < 1) t1 = 1;
            i32 t2 = lmnobr;
            if (t1 > ldwork) ldwork = t1;
            if (t2 > ldwork) ldwork = t2;
        }
        if (meth == 'N') {
            ldwork = 5 * lmnobr + 1;
        }
    } else if (alg == 'F') {
        ldwork = 2 * nr * (m + l + 1) + nr;
    } else {
        ldwork = 6 * (m + l) * nobr;
        if (onebch && ldr >= ns) {
            if (meth == 'M') {
                i32 t = 5 * lnobr;
                if (ldwork < t) ldwork = t;
            } else {
                ldwork = 5 * lmnobr + 1;
            }
        }
    }
    i32 t = (ns + 2) * nr;
    if (ldwork < t) ldwork = t;
    ldwork = (ldwork > 1) ? ldwork : 1;

    f64 *dwork = (f64*)malloc(ldwork * sizeof(f64));
    if (!dwork) {
        free(r_data);
        free(iwork);
        Py_DECREF(u_array);
        Py_DECREF(y_array);
        Py_DECREF(sv_array);
        return PyErr_NoMemory();
    }

    ib01ad(meth_str, alg_str, jobd_str, batch_str, conct_str, ctrl_str,
           nobr, m, l, nsmp, u_data, ldu, y_data, ldy,
           &n, r_data, ldr, sv, rcond, tol,
           iwork, dwork, ldwork, &iwarn, &info);

    free(dwork);
    free(iwork);
    Py_DECREF(u_array);
    Py_DECREF(y_array);

    if (info < 0) {
        free(r_data);
        Py_DECREF(sv_array);
        PyErr_Format(PyExc_ValueError, "Parameter %d had an illegal value", -info);
        return NULL;
    }

    npy_intp r_dims[2] = {nr, nr};
    npy_intp r_strides[2] = {sizeof(f64), nr * sizeof(f64)};
    PyArrayObject *r_array = (PyArrayObject*)PyArray_New(
        &PyArray_Type, 2, r_dims, NPY_DOUBLE, r_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (!r_array) {
        free(r_data);
        Py_DECREF(sv_array);
        return PyErr_NoMemory();
    }
    f64 *r_out = (f64*)PyArray_DATA(r_array);
    for (i32 j = 0; j < nr; j++) {
        for (i32 i = 0; i < nr; i++) {
            r_out[i + j * nr] = r_data[i + j * ldr];
        }
    }
    free(r_data);

    PyObject *result = Py_BuildValue("iOOii", n, r_array, sv_array, iwarn, info);
    Py_DECREF(r_array);
    Py_DECREF(sv_array);
    return result;
}



/* Python wrapper for ib01bd - State-space matrices estimation */
PyObject* py_ib01bd(PyObject* self, PyObject* args) {
    const char *meth_str, *job_str, *jobck_str;
    i32 nobr, n, m, l, nsmpl;
    f64 tol;
    PyObject *r_obj;
    PyArrayObject *r_array = NULL;
    i32 iwarn, info;

    if (!PyArg_ParseTuple(args, "sssiiiiiOd",
                          &meth_str, &job_str, &jobck_str, &nobr, &n, &m, &l,
                          &nsmpl, &r_obj, &tol)) {
        return NULL;
    }

    char meth = toupper((unsigned char)meth_str[0]);
    char job = toupper((unsigned char)job_str[0]);
    char jobck = toupper((unsigned char)jobck_str[0]);

    if (meth != 'M' && meth != 'N' && meth != 'C') {
        PyErr_SetString(PyExc_ValueError, "METH must be 'M', 'N', or 'C'");
        return NULL;
    }
    if (job != 'A' && job != 'C' && job != 'B' && job != 'D') {
        PyErr_SetString(PyExc_ValueError, "JOB must be 'A', 'C', 'B', or 'D'");
        return NULL;
    }
    if (jobck != 'K' && jobck != 'C' && jobck != 'N') {
        PyErr_SetString(PyExc_ValueError, "JOBCK must be 'K', 'C', or 'N'");
        return NULL;
    }
    if (nobr <= 1) {
        PyErr_SetString(PyExc_ValueError, "NOBR must be > 1");
        return NULL;
    }
    if (n <= 0 || n >= nobr) {
        PyErr_SetString(PyExc_ValueError, "N must be in range (0, NOBR)");
        return NULL;
    }
    if (m < 0) {
        PyErr_SetString(PyExc_ValueError, "M must be >= 0");
        return NULL;
    }
    if (l <= 0) {
        PyErr_SetString(PyExc_ValueError, "L must be > 0");
        return NULL;
    }

    r_array = (PyArrayObject*)PyArray_FROM_OTF(r_obj, NPY_DOUBLE,
        NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (r_array == NULL) return NULL;

    i32 ldr = (i32)PyArray_DIM(r_array, 0);
    f64 *r_data = (f64*)PyArray_DATA(r_array);

    i32 lda = n > 1 ? n : 1;
    i32 ldc = l;
    i32 ldb = n > 1 ? n : 1;
    i32 ldd = l;
    i32 ldq = n > 1 ? n : 1;
    i32 ldry = l;
    i32 lds = n > 1 ? n : 1;
    i32 ldk = n > 1 ? n : 1;

    npy_intp a_dims[2] = {n, n};
    npy_intp c_dims[2] = {l, n};
    npy_intp b_dims[2] = {n, m > 0 ? m : 1};
    npy_intp d_dims[2] = {l, m > 0 ? m : 1};
    npy_intp q_dims[2] = {n, n};
    npy_intp ry_dims[2] = {l, l};
    npy_intp s_dims[2] = {n, l};
    npy_intp k_dims[2] = {n, l};

    npy_intp a_strides[2] = {sizeof(f64), lda * sizeof(f64)};
    npy_intp c_strides[2] = {sizeof(f64), ldc * sizeof(f64)};
    npy_intp b_strides[2] = {sizeof(f64), ldb * sizeof(f64)};
    npy_intp d_strides[2] = {sizeof(f64), ldd * sizeof(f64)};
    npy_intp q_strides[2] = {sizeof(f64), ldq * sizeof(f64)};
    npy_intp ry_strides[2] = {sizeof(f64), ldry * sizeof(f64)};
    npy_intp s_strides[2] = {sizeof(f64), lds * sizeof(f64)};
    npy_intp k_strides[2] = {sizeof(f64), ldk * sizeof(f64)};

    PyObject *a_out = PyArray_New(&PyArray_Type, 2, a_dims, NPY_DOUBLE, a_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    PyObject *c_out = PyArray_New(&PyArray_Type, 2, c_dims, NPY_DOUBLE, c_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    PyObject *b_out = PyArray_New(&PyArray_Type, 2, b_dims, NPY_DOUBLE, b_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    PyObject *d_out = PyArray_New(&PyArray_Type, 2, d_dims, NPY_DOUBLE, d_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    PyObject *q_out = PyArray_New(&PyArray_Type, 2, q_dims, NPY_DOUBLE, q_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    PyObject *ry_out = PyArray_New(&PyArray_Type, 2, ry_dims, NPY_DOUBLE, ry_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    PyObject *s_out = PyArray_New(&PyArray_Type, 2, s_dims, NPY_DOUBLE, s_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    PyObject *k_out = PyArray_New(&PyArray_Type, 2, k_dims, NPY_DOUBLE, k_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);

    if (!a_out || !c_out || !b_out || !d_out || !q_out || !ry_out || !s_out || !k_out) {
        Py_XDECREF(a_out); Py_XDECREF(c_out); Py_XDECREF(b_out); Py_XDECREF(d_out);
        Py_XDECREF(q_out); Py_XDECREF(ry_out); Py_XDECREF(s_out); Py_XDECREF(k_out);
        Py_DECREF(r_array);
        return PyErr_NoMemory();
    }

    f64 *a = (f64*)PyArray_DATA((PyArrayObject*)a_out);
    f64 *c = (f64*)PyArray_DATA((PyArrayObject*)c_out);
    f64 *b = (f64*)PyArray_DATA((PyArrayObject*)b_out);
    f64 *d = (f64*)PyArray_DATA((PyArrayObject*)d_out);
    f64 *q = (f64*)PyArray_DATA((PyArrayObject*)q_out);
    f64 *ry = (f64*)PyArray_DATA((PyArrayObject*)ry_out);
    f64 *s = (f64*)PyArray_DATA((PyArrayObject*)s_out);
    f64 *k = (f64*)PyArray_DATA((PyArrayObject*)k_out);

    memset(a, 0, n * n * sizeof(f64));
    memset(c, 0, l * n * sizeof(f64));
    memset(b, 0, (n * m > 1 ? n * m : 1) * sizeof(f64));
    memset(d, 0, (l * m > 1 ? l * m : 1) * sizeof(f64));
    memset(q, 0, n * n * sizeof(f64));
    memset(ry, 0, l * l * sizeof(f64));
    memset(s, 0, n * l * sizeof(f64));
    memset(k, 0, n * l * sizeof(f64));

    i32 ldun2 = l * nobr - l;
    i32 ldunn = ldun2 * n;
    i32 mnobrn = m * nobr + n;
    i32 lmmnol = l * nobr + 2 * m * nobr + l;
    i32 npl = n + l;
    i32 nn = n * n;
    i32 n2 = n + n;

    i32 minwrk = ldunn + 4*n;
    i32 alt1 = 2*ldunn + n2;
    i32 alt2 = ldunn + nn + 7*n;
    if (alt1 > minwrk) minwrk = alt1;
    if (alt2 > minwrk) minwrk = alt2;

    if (m > 0) {
        i32 alt = 2*ldunn + nn + n + 7*n;
        if (alt > minwrk) minwrk = alt;
        alt = ldunn + n + 6*m*nobr;
        if (alt > minwrk) minwrk = alt;
    }

    i32 tmp = 5*n > lmmnol ? 5*n : lmmnol;
    i32 alt = ldunn + n + nn + n2 + tmp;
    if (alt > minwrk) minwrk = alt;
    alt = n + 4*mnobrn + 1;
    if (alt > minwrk) minwrk = alt;

    if (m > 0) {
        i32 npl2 = npl * npl;
        i32 mnpl = m * npl;
        i32 tmp2 = 4*mnpl + 1;
        if (npl2 > tmp2) tmp2 = npl2;
        alt = m*nobr * npl * (mnpl + 1) + tmp2;
        if (alt > minwrk) minwrk = alt;
    }

    if (jobck == 'K') {
        i32 nl = n * l;
        i32 ll = l * l;
        i32 tmp = 3*l > nl ? 3*l : nl;
        alt = 4*nn + 2*nl + ll + tmp;
        if (alt > minwrk) minwrk = alt;
        alt = 14*nn + 12*n + 5;
        if (alt > minwrk) minwrk = alt;
        alt = 2*nn + 2*nl + ll + 3*l;
        if (alt > minwrk) minwrk = alt;
    }

    i32 ldwork = minwrk > 50000 ? minwrk : 50000;

    i32 liwork = m*nobr + n;
    if (l*nobr > liwork) liwork = l*nobr;
    if (n2 > liwork) liwork = n2;
    if (m > liwork) liwork = m;
    liwork += 10;

    i32 *iwork = (i32*)calloc(liwork, sizeof(i32));
    f64 *dwork = (f64*)calloc(ldwork, sizeof(f64));
    i32 lbwork = jobck == 'K' ? 2*n : 1;
    i32 *bwork = (i32*)calloc(lbwork, sizeof(i32));

    if (!iwork || !dwork || !bwork) {
        Py_DECREF(a_out); Py_DECREF(c_out); Py_DECREF(b_out); Py_DECREF(d_out);
        Py_DECREF(q_out); Py_DECREF(ry_out); Py_DECREF(s_out); Py_DECREF(k_out);
        free(iwork); free(dwork); free(bwork);
        Py_DECREF(r_array);
        return PyErr_NoMemory();
    }

    ib01bd(meth_str, job_str, jobck_str, nobr, n, m, l, nsmpl,
           r_data, ldr, a, lda, c, ldc, b, ldb, d, ldd, q, ldq,
           ry, ldry, s, lds, k, ldk, tol, iwork, dwork, ldwork, bwork,
           &iwarn, &info);

    free(iwork);
    free(dwork);
    free(bwork);

    PyArray_ResolveWritebackIfCopy(r_array);
    Py_DECREF(r_array);

    if (info < 0) {
        Py_DECREF(a_out); Py_DECREF(c_out); Py_DECREF(b_out); Py_DECREF(d_out);
        Py_DECREF(q_out); Py_DECREF(ry_out); Py_DECREF(s_out); Py_DECREF(k_out);
        PyErr_Format(PyExc_ValueError, "Parameter %d had an illegal value", -info);
        return NULL;
    }

    PyObject *result = Py_BuildValue("NNNNNNNNii",
        a_out, c_out, b_out, d_out, q_out, ry_out, s_out, k_out, iwarn, info);
    return result;
}



/* Python wrapper for ib03ad - Wiener system identification with algorithm choice */
PyObject* py_ib03ad(PyObject* self, PyObject* args, PyObject* kwargs) {
    const char *init_str, *alg_str, *stor_str;
    i32 nobr, m, l, nsmp, n, nn, itmax1, itmax2, nprint;
    f64 tol1, tol2;
    PyObject *u_obj, *y_obj;
    PyObject *dwork_seed_obj = Py_None;
    PyObject *x_init_obj = Py_None;
    PyArrayObject *u_array = NULL, *y_array = NULL;
    PyArrayObject *dwork_seed_array = NULL, *x_init_array = NULL;
    i32 iwarn, info;

    static char *kwlist[] = {"init", "alg", "stor", "nobr", "m", "l", "nsmp", "n", "nn",
                             "itmax1", "itmax2", "u", "y", "tol1", "tol2",
                             "dwork_seed", "x_init", "nprint", NULL};

    nprint = 0;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "sssiiiiiiiiOOdd|OOi", kwlist,
                                      &init_str, &alg_str, &stor_str,
                                      &nobr, &m, &l, &nsmp, &n, &nn,
                                      &itmax1, &itmax2, &u_obj, &y_obj,
                                      &tol1, &tol2, &dwork_seed_obj, &x_init_obj,
                                      &nprint)) {
        return NULL;
    }

    char init = toupper((unsigned char)init_str[0]);
    char alg = toupper((unsigned char)alg_str[0]);
    char stor = toupper((unsigned char)stor_str[0]);

    if (init != 'L' && init != 'S' && init != 'B' && init != 'N') {
        PyErr_SetString(PyExc_ValueError, "INIT must be 'L', 'S', 'B', or 'N'");
        return NULL;
    }
    if (alg != 'D' && alg != 'I') {
        PyErr_SetString(PyExc_ValueError, "ALG must be 'D' (Cholesky) or 'I' (CG)");
        return NULL;
    }
    if (stor != 'F' && stor != 'P') {
        PyErr_SetString(PyExc_ValueError, "STOR must be 'F' (full) or 'P' (packed)");
        return NULL;
    }

    u_array = (PyArrayObject*)PyArray_FROM_OTF(u_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    y_array = (PyArrayObject*)PyArray_FROM_OTF(y_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);

    if (!u_array || !y_array) {
        Py_XDECREF(u_array);
        Py_XDECREF(y_array);
        return NULL;
    }

    i32 ldu = (i32)PyArray_DIM(u_array, 0);
    i32 ldy = (i32)PyArray_DIM(y_array, 0);
    f64 *u_data = (f64*)PyArray_DATA(u_array);
    f64 *y_data = (f64*)PyArray_DATA(y_array);

    i32 ml = m + l;
    i32 bsn = nn * (l + 2) + 1;
    i32 nths = bsn * l;
    i32 lths = n * (ml + 1) + l * m;
    i32 nx = nths + lths;
    i32 lx = nx;

    f64 *x = (f64*)calloc(nx, sizeof(f64));
    if (!x) {
        Py_DECREF(u_array);
        Py_DECREF(y_array);
        return PyErr_NoMemory();
    }

    if (x_init_obj != Py_None) {
        x_init_array = (PyArrayObject*)PyArray_FROM_OTF(x_init_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
        if (!x_init_array) {
            free(x);
            Py_DECREF(u_array);
            Py_DECREF(y_array);
            return NULL;
        }
        i32 x_init_len = (i32)PyArray_SIZE(x_init_array);
        f64 *x_init_data = (f64*)PyArray_DATA(x_init_array);
        i32 copy_len = x_init_len < nx ? x_init_len : nx;
        for (i32 i = 0; i < copy_len; i++) {
            x[i] = x_init_data[i];
        }
        Py_DECREF(x_init_array);
    }

    i32 nsml = nsmp * l;
    i32 ldac = (n > 0) ? n + l : 1;
    i32 isad = (n > 0) ? ldac * (n + m) : 0;

    i32 ldwork = 50000;

    if (init == 'L' || init == 'B') {
        i32 iw = 2 * ml * nobr * (2 * ml * (nobr + 1) + 3) + l * nobr;
        if (iw > ldwork) ldwork = iw;
        i32 two_ml_nobr_sq = (2 * ml * nobr) * (2 * ml * nobr);
        iw = two_ml_nobr_sq + isad + n * n + 8 * n + 100;
        if (iw > ldwork) ldwork = iw;
        iw = nsml + isad + ldac + 2 * n + m + 100;
        if (iw > ldwork) ldwork = iw;
    }

    if (init == 'S' || init == 'B') {
        i32 iw = nsml + bsn * bsn + bsn + nsmp + 2 * nn + 100;
        if (iw > ldwork) ldwork = iw;
    }

    i32 iw = nsml + nx + nsml * (bsn + lths) + nx * (bsn + lths) + 2 * nx + 100;
    if (iw > ldwork) ldwork = iw;

    ldwork *= 2;

    i32 liwork = nx + l + 10;
    if (m * nobr + n > liwork) liwork = m * nobr + n;
    if (m * (n + l) > liwork) liwork = m * (n + l);
    if (nn * (l + 2) + 2 > liwork) liwork = nn * (l + 2) + 2;

    i32 *iwork = (i32*)calloc(liwork, sizeof(i32));
    f64 *dwork = (f64*)calloc(ldwork, sizeof(f64));

    if (!iwork || !dwork) {
        free(x);
        free(iwork);
        free(dwork);
        Py_DECREF(u_array);
        Py_DECREF(y_array);
        return PyErr_NoMemory();
    }

    if (dwork_seed_obj != Py_None) {
        dwork_seed_array = (PyArrayObject*)PyArray_FROM_OTF(dwork_seed_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
        if (dwork_seed_array && PyArray_SIZE(dwork_seed_array) >= 4) {
            f64 *seed_data = (f64*)PyArray_DATA(dwork_seed_array);
            for (i32 i = 0; i < 4; i++) {
                dwork[i] = seed_data[i];
            }
        }
        Py_XDECREF(dwork_seed_array);
    }

    i32 n_local = n;

    ib03ad(init_str, alg_str, stor_str, nobr, m, l, nsmp, &n_local, nn,
           itmax1, itmax2, nprint,
           u_data, ldu, y_data, ldy, x, &lx, tol1, tol2,
           iwork, dwork, ldwork, &iwarn, &info);

    Py_DECREF(u_array);
    Py_DECREF(y_array);

    if (info < 0) {
        free(x);
        free(iwork);
        free(dwork);
        PyErr_Format(PyExc_ValueError, "IB03AD error: INFO = %d", info);
        return NULL;
    }

    npy_intp x_dims[1] = {nx};
    PyObject *x_out = PyArray_New(&PyArray_Type, 1, x_dims, NPY_DOUBLE, NULL, NULL, 0, 0, NULL);
    if (!x_out) {
        free(x);
        free(iwork);
        free(dwork);
        return PyErr_NoMemory();
    }
    memcpy(PyArray_DATA((PyArrayObject*)x_out), x, nx * sizeof(f64));
    free(x);

    npy_intp dwork_dims[1] = {8};
    PyObject *dwork_out = PyArray_New(&PyArray_Type, 1, dwork_dims, NPY_DOUBLE, NULL, NULL, 0, 0, NULL);
    if (!dwork_out) {
        Py_DECREF(x_out);
        free(iwork);
        free(dwork);
        return PyErr_NoMemory();
    }
    f64 *dwork_out_data = (f64*)PyArray_DATA((PyArrayObject*)dwork_out);
    for (i32 i = 0; i < 8 && i < ldwork; i++) {
        dwork_out_data[i] = dwork[i];
    }

    free(iwork);
    free(dwork);

    PyObject *result = Py_BuildValue("NiiN", x_out, iwarn, info, dwork_out);
    return result;
}



/* Python wrapper for ib03bd - Wiener system identification */
PyObject* py_ib03bd(PyObject* self, PyObject* args, PyObject* kwargs) {
    const char *init_str;
    i32 nobr, m, l, nsmp, n, nn, itmax1, itmax2, nprint;
    f64 tol1, tol2;
    PyObject *u_obj, *y_obj;
    PyObject *dwork_seed_obj = Py_None;
    PyObject *x_init_obj = Py_None;
    PyArrayObject *u_array = NULL, *y_array = NULL;
    PyArrayObject *dwork_seed_array = NULL, *x_init_array = NULL;
    i32 iwarn, info;

    static char *kwlist[] = {"init", "nobr", "m", "l", "nsmp", "n", "nn",
                             "itmax1", "itmax2", "u", "y", "tol1", "tol2",
                             "dwork_seed", "x_init", "nprint", NULL};

    nprint = 0;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "siiiiiiiiOOdd|OOi", kwlist,
                                      &init_str, &nobr, &m, &l, &nsmp, &n, &nn,
                                      &itmax1, &itmax2, &u_obj, &y_obj,
                                      &tol1, &tol2, &dwork_seed_obj, &x_init_obj,
                                      &nprint)) {
        return NULL;
    }

    char init = toupper((unsigned char)init_str[0]);

    if (init != 'L' && init != 'S' && init != 'B' && init != 'N') {
        PyErr_SetString(PyExc_ValueError, "INIT must be 'L', 'S', 'B', or 'N'");
        return NULL;
    }

    u_array = (PyArrayObject*)PyArray_FROM_OTF(u_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    y_array = (PyArrayObject*)PyArray_FROM_OTF(y_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);

    if (!u_array || !y_array) {
        Py_XDECREF(u_array);
        Py_XDECREF(y_array);
        return NULL;
    }

    i32 ldu = (i32)PyArray_DIM(u_array, 0);
    i32 ldy = (i32)PyArray_DIM(y_array, 0);
    f64 *u_data = (f64*)PyArray_DATA(u_array);
    f64 *y_data = (f64*)PyArray_DATA(y_array);

    i32 ml = m + l;
    i32 bsn = nn * (l + 2) + 1;
    i32 nths = bsn * l;
    i32 lths = n * (ml + 1) + l * m;
    i32 nx = nths + lths;
    i32 lx = nx;

    f64 *x = (f64*)calloc(nx, sizeof(f64));
    if (!x) {
        Py_DECREF(u_array);
        Py_DECREF(y_array);
        return PyErr_NoMemory();
    }

    if (x_init_obj != Py_None) {
        x_init_array = (PyArrayObject*)PyArray_FROM_OTF(x_init_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
        if (!x_init_array) {
            free(x);
            Py_DECREF(u_array);
            Py_DECREF(y_array);
            return NULL;
        }
        i32 x_init_len = (i32)PyArray_SIZE(x_init_array);
        f64 *x_init_data = (f64*)PyArray_DATA(x_init_array);
        i32 copy_len = x_init_len < nx ? x_init_len : nx;
        for (i32 i = 0; i < copy_len; i++) {
            x[i] = x_init_data[i];
        }
        Py_DECREF(x_init_array);
    }

    i32 nsml = nsmp * l;
    i32 ldac = (n > 0) ? n + l : 1;
    i32 isad = (n > 0) ? ldac * (n + m) : 0;

    i32 ldwork = 50000;

    if (init == 'L' || init == 'B') {
        i32 iw = 2 * ml * nobr * (2 * ml * (nobr + 1) + 3) + l * nobr;
        if (iw > ldwork) ldwork = iw;
        i32 two_ml_nobr_sq = (2 * ml * nobr) * (2 * ml * nobr);
        iw = two_ml_nobr_sq + isad + n * n + 8 * n + 100;
        if (iw > ldwork) ldwork = iw;
        iw = nsml + isad + ldac + 2 * n + m + 100;
        if (iw > ldwork) ldwork = iw;
    }

    if (init == 'S' || init == 'B') {
        i32 iw = nsml + bsn * bsn + bsn + nsmp + 2 * nn + 100;
        if (iw > ldwork) ldwork = iw;
    }

    i32 iw = nsml + nx + nsml * (bsn + lths) + nx * (bsn + lths) + 2 * nx + 100;
    if (iw > ldwork) ldwork = iw;

    ldwork *= 2;

    i32 liwork = nx + l + 10;
    if (m * nobr + n > liwork) liwork = m * nobr + n;
    if (m * (n + l) > liwork) liwork = m * (n + l);
    if (nn * (l + 2) + 2 > liwork) liwork = nn * (l + 2) + 2;

    i32 *iwork = (i32*)calloc(liwork, sizeof(i32));
    f64 *dwork = (f64*)calloc(ldwork, sizeof(f64));

    if (!iwork || !dwork) {
        free(x);
        free(iwork);
        free(dwork);
        Py_DECREF(u_array);
        Py_DECREF(y_array);
        return PyErr_NoMemory();
    }

    if (dwork_seed_obj != Py_None) {
        dwork_seed_array = (PyArrayObject*)PyArray_FROM_OTF(dwork_seed_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
        if (dwork_seed_array && PyArray_SIZE(dwork_seed_array) >= 4) {
            f64 *seed_data = (f64*)PyArray_DATA(dwork_seed_array);
            for (i32 i = 0; i < 4; i++) {
                dwork[i] = seed_data[i];
            }
        }
        Py_XDECREF(dwork_seed_array);
    }

    i32 n_local = n;

    ib03bd(init_str, nobr, m, l, nsmp, &n_local, nn, itmax1, itmax2, nprint,
           u_data, ldu, y_data, ldy, x, &lx, tol1, tol2,
           iwork, dwork, ldwork, &iwarn, &info);

    Py_DECREF(u_array);
    Py_DECREF(y_array);

    if (info < 0) {
        free(x);
        free(iwork);
        free(dwork);
        PyErr_Format(PyExc_ValueError, "IB03BD error: INFO = %d", info);
        return NULL;
    }

    npy_intp x_dims[1] = {nx};
    PyObject *x_out = PyArray_New(&PyArray_Type, 1, x_dims, NPY_DOUBLE, NULL, NULL, 0, 0, NULL);
    if (!x_out) {
        free(x);
        free(iwork);
        free(dwork);
        return PyErr_NoMemory();
    }
    memcpy(PyArray_DATA((PyArrayObject*)x_out), x, nx * sizeof(f64));
    free(x);

    npy_intp dwork_dims[1] = {8};
    PyObject *dwork_out = PyArray_New(&PyArray_Type, 1, dwork_dims, NPY_DOUBLE, NULL, NULL, 0, 0, NULL);
    if (!dwork_out) {
        Py_DECREF(x_out);
        free(iwork);
        free(dwork);
        return PyErr_NoMemory();
    }
    f64 *dwork_out_data = (f64*)PyArray_DATA((PyArrayObject*)dwork_out);
    for (i32 i = 0; i < 8 && i < ldwork; i++) {
        dwork_out_data[i] = dwork[i];
    }

    free(iwork);
    free(dwork);

    PyObject *result = Py_BuildValue("NiiN", x_out, iwarn, info, dwork_out);
    return result;
}



/* Python wrapper for ib01md */
PyObject* py_ib01md(PyObject* self, PyObject* args, PyObject* kwargs) {
    const char *meth_str, *alg_str, *batch_str, *conct_str;
    i32 nobr, m, l;
    PyObject *u_obj, *y_obj;
    PyObject *r_obj = Py_None;
    PyObject *iwork_obj = Py_None;
    PyArrayObject *u_array = NULL, *y_array = NULL;
    PyArrayObject *r_array = NULL, *iwork_array = NULL;
    i32 iwarn, info;

    static char *kwlist[] = {"meth", "alg", "batch", "conct", "nobr", "m", "l",
                             "u", "y", "r", "iwork", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ssssiiiOO|OO", kwlist,
                                      &meth_str, &alg_str, &batch_str, &conct_str,
                                      &nobr, &m, &l, &u_obj, &y_obj,
                                      &r_obj, &iwork_obj)) {
        return NULL;
    }

    char meth = toupper((unsigned char)meth_str[0]);
    char alg = toupper((unsigned char)alg_str[0]);
    char batch = toupper((unsigned char)batch_str[0]);
    char conct = toupper((unsigned char)conct_str[0]);

    if (meth != 'M' && meth != 'N') {
        PyErr_SetString(PyExc_ValueError, "METH must be 'M' or 'N'");
        return NULL;
    }
    if (alg != 'C' && alg != 'F' && alg != 'Q') {
        PyErr_SetString(PyExc_ValueError, "ALG must be 'C', 'F', or 'Q'");
        return NULL;
    }
    if (batch != 'F' && batch != 'I' && batch != 'L' && batch != 'O') {
        PyErr_SetString(PyExc_ValueError, "BATCH must be 'F', 'I', 'L', or 'O'");
        return NULL;
    }
    if (nobr <= 0) {
        PyErr_SetString(PyExc_ValueError, "NOBR must be positive");
        return NULL;
    }
    if (m < 0) {
        PyErr_SetString(PyExc_ValueError, "M must be non-negative");
        return NULL;
    }
    if (l <= 0) {
        PyErr_SetString(PyExc_ValueError, "L must be positive");
        return NULL;
    }

    u_array = (PyArrayObject*)PyArray_FROM_OTF(u_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    y_array = (PyArrayObject*)PyArray_FROM_OTF(y_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);

    if (!u_array || !y_array) {
        Py_XDECREF(u_array);
        Py_XDECREF(y_array);
        return NULL;
    }

    npy_intp *y_dims = PyArray_DIMS(y_array);
    i32 nsmp = (i32)y_dims[0];
    i32 ldu = (m > 0) ? nsmp : 1;
    i32 ldy = nsmp;

    i32 nobr2 = 2 * nobr;
    i32 nr = 2 * (m + l) * nobr;

    bool onebch = (batch == 'O');
    bool first = (batch == 'F') || onebch;
    bool last = (batch == 'L') || onebch;

    i32 min_nsmp_seq = nobr2;
    i32 min_nsmp_non = 2 * (m + l + 1) * nobr - 1;
    i32 min_nsmp = onebch ? min_nsmp_non : min_nsmp_seq;

    if (nsmp < min_nsmp) {
        Py_DECREF(u_array);
        Py_DECREF(y_array);
        PyErr_SetString(PyExc_ValueError, "NSMP too small for the problem");
        return NULL;
    }

    const f64 *u_data = (const f64*)PyArray_DATA(u_array);
    const f64 *y_data = (const f64*)PyArray_DATA(y_array);

    npy_intp r_dims[2] = {nr, nr};
    if (r_obj != Py_None && batch != 'F' && batch != 'O') {
        r_array = (PyArrayObject*)PyArray_FROM_OTF(r_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
        if (!r_array) {
            Py_DECREF(u_array);
            Py_DECREF(y_array);
            return NULL;
        }
    } else {
        npy_intp r_strides[2] = {sizeof(f64), nr * sizeof(f64)};
        r_array = (PyArrayObject*)PyArray_New(&PyArray_Type, 2, r_dims, NPY_DOUBLE,
                                               r_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        if (!r_array) {
            Py_DECREF(u_array);
            Py_DECREF(y_array);
            return PyErr_NoMemory();
        }
        memset(PyArray_DATA(r_array), 0, nr * nr * sizeof(f64));
    }

    i32 ldr = nr;

    i32 liwork = (alg == 'F') ? ((m + l > 3) ? (m + l) : 3) : 3;
    i32 *iwork_data;
    bool iwork_allocated = false;

    if (iwork_obj != Py_None && batch != 'F' && batch != 'O') {
        iwork_array = (PyArrayObject*)PyArray_FROM_OTF(iwork_obj, NPY_INT32, NPY_ARRAY_CARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
        if (!iwork_array) {
            Py_DECREF(u_array);
            Py_DECREF(y_array);
            Py_DECREF(r_array);
            return NULL;
        }
        iwork_data = (i32*)PyArray_DATA(iwork_array);
    } else {
        iwork_data = (i32*)calloc(liwork, sizeof(i32));
        if (!iwork_data) {
            Py_DECREF(u_array);
            Py_DECREF(y_array);
            Py_DECREF(r_array);
            return PyErr_NoMemory();
        }
        iwork_allocated = true;
    }

    i32 ns = nsmp - 2 * nobr + 1;
    i32 ldwork;
    bool connec = (batch != 'O') && (conct == 'C');

    if (alg == 'C') {
        if (!onebch && connec) {
            ldwork = 2 * (nr - m - l);
        } else {
            ldwork = 1;
        }
    } else if (alg == 'F') {
        if (!onebch && connec) {
            ldwork = nr * (m + l + 3);
        } else if (first || batch == 'I') {
            ldwork = nr * (m + l + 1);
        } else {
            ldwork = 2 * nr * (m + l + 1) + nr;
        }
    } else {
        if (first && ldr >= ns) {
            ldwork = 4 * (m + l) * nobr;
        } else {
            ldwork = 6 * (m + l) * nobr;
        }
        if (!first && connec) {
            ldwork = 4 * (nobr + 1) * (m + l) * nobr;
        }
    }
    ldwork = (ldwork > (ns + 2) * nr) ? ldwork : ((ns + 2) * nr);

    f64 *dwork = (f64*)malloc(ldwork * sizeof(f64));
    if (!dwork) {
        Py_DECREF(u_array);
        Py_DECREF(y_array);
        Py_DECREF(r_array);
        if (iwork_allocated) free(iwork_data);
        else { PyArray_ResolveWritebackIfCopy(iwork_array); Py_DECREF(iwork_array); }
        return PyErr_NoMemory();
    }

    f64 *r_data = (f64*)PyArray_DATA(r_array);

    ib01md(meth_str, alg_str, batch_str, conct_str, nobr, m, l, nsmp,
           u_data, ldu, y_data, ldy, r_data, ldr, iwork_data, dwork, ldwork,
           &iwarn, &info);

    free(dwork);

    PyArray_ResolveWritebackIfCopy(r_array);
    Py_DECREF(u_array);
    Py_DECREF(y_array);

    if (info < 0) {
        Py_DECREF(r_array);
        if (iwork_allocated) free(iwork_data);
        else { PyArray_ResolveWritebackIfCopy(iwork_array); Py_DECREF(iwork_array); }
        PyErr_Format(PyExc_ValueError, "Parameter %d had an illegal value", -info);
        return NULL;
    }

    PyObject *result;
    if (last) {
        if (iwork_allocated) free(iwork_data);
        else { PyArray_ResolveWritebackIfCopy(iwork_array); Py_DECREF(iwork_array); }
        result = Py_BuildValue("Oii", r_array, iwarn, info);
        Py_DECREF(r_array);
    } else {
        npy_intp iwork_dims[1] = {liwork};
        PyArrayObject *iwork_out;
        if (iwork_allocated) {
            iwork_out = (PyArrayObject*)PyArray_SimpleNew(1, iwork_dims, NPY_INT32);
            if (!iwork_out) {
                free(iwork_data);
                Py_DECREF(r_array);
                return PyErr_NoMemory();
            }
            memcpy(PyArray_DATA(iwork_out), iwork_data, liwork * sizeof(i32));
            free(iwork_data);
        } else {
            PyArray_ResolveWritebackIfCopy(iwork_array);
            iwork_out = iwork_array;
            Py_INCREF(iwork_out);
            Py_DECREF(iwork_array);
        }
        result = Py_BuildValue("OOii", r_array, iwork_out, iwarn, info);
        Py_DECREF(r_array);
        Py_DECREF(iwork_out);
    }

    return result;
}



/* Python wrapper for ib01qd */
PyObject* py_ib01qd(PyObject* self, PyObject* args) {
    const char *jobx0_str, *job_str;
    i32 n, m, l;
    f64 tol;
    PyObject *a_obj, *c_obj, *u_obj, *y_obj;
    PyArrayObject *a_array, *c_array, *u_array, *y_array;
    i32 iwarn, info;

    if (!PyArg_ParseTuple(args, "ssiiiOOOOd",
                          &jobx0_str, &job_str, &n, &m, &l,
                          &a_obj, &c_obj, &u_obj, &y_obj, &tol)) {
        return NULL;
    }

    char jobx0 = toupper((unsigned char)jobx0_str[0]);
    char job = toupper((unsigned char)job_str[0]);

    if (jobx0 != 'X' && jobx0 != 'N') {
        PyErr_SetString(PyExc_ValueError, "JOBX0 must be 'X' or 'N'");
        return NULL;
    }
    if (job != 'B' && job != 'D') {
        PyErr_SetString(PyExc_ValueError, "JOB must be 'B' or 'D'");
        return NULL;
    }
    if (n < 0) {
        PyErr_SetString(PyExc_ValueError, "N must be non-negative");
        return NULL;
    }
    if (m < 0) {
        PyErr_SetString(PyExc_ValueError, "M must be non-negative");
        return NULL;
    }
    if (l <= 0) {
        PyErr_SetString(PyExc_ValueError, "L must be positive");
        return NULL;
    }

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    c_array = (PyArrayObject*)PyArray_FROM_OTF(c_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    u_array = (PyArrayObject*)PyArray_FROM_OTF(u_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    y_array = (PyArrayObject*)PyArray_FROM_OTF(y_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);

    if (!a_array || !c_array || !u_array || !y_array) {
        Py_XDECREF(a_array);
        Py_XDECREF(c_array);
        Py_XDECREF(u_array);
        Py_XDECREF(y_array);
        return NULL;
    }

    npy_intp *u_dims = PyArray_DIMS(u_array);
    i32 nsmp = (i32)u_dims[0];
    i32 ldu = nsmp > 0 ? nsmp : 1;
    i32 ldy = nsmp > 0 ? nsmp : 1;
    i32 lda = n > 0 ? n : 1;
    i32 ldc = l;
    i32 ldb = (n > 0 && m > 0) ? n : 1;
    i32 ldd = (m > 0 && job == 'D') ? l : 1;

    bool withx0 = (jobx0 == 'X');
    bool withd = (job == 'D');
    i32 ncol = n * m + (withx0 ? n : 0);
    i32 minsmp = ncol;
    if (withd) {
        minsmp += m;
    } else if (!withx0) {
        minsmp += 1;
    }

    if (nsmp < minsmp) {
        Py_DECREF(a_array);
        Py_DECREF(c_array);
        Py_DECREF(u_array);
        Py_DECREF(y_array);
        PyErr_SetString(PyExc_ValueError, "NSMP too small for the problem dimensions");
        return NULL;
    }

    const f64 *a_data = (const f64*)PyArray_DATA(a_array);
    const f64 *c_data = (const f64*)PyArray_DATA(c_array);
    f64 *u_data = (f64*)PyArray_DATA(u_array);
    const f64 *y_data = (const f64*)PyArray_DATA(y_array);

    npy_intp x0_dims[1] = {n};
    PyArrayObject *x0_array = (PyArrayObject*)PyArray_SimpleNew(1, x0_dims, NPY_DOUBLE);

    npy_intp b_dims[2] = {n > 0 ? n : 1, m > 0 ? m : 1};
    npy_intp b_strides[2] = {sizeof(f64), ldb * sizeof(f64)};
    f64 *b_data = NULL;
    PyArrayObject *b_array = NULL;
    if (n > 0 && m > 0) {
        b_array = (PyArrayObject*)PyArray_New(&PyArray_Type, 2, b_dims, NPY_DOUBLE,
                                               b_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        if (!b_array) {
            Py_DECREF(a_array);
            Py_DECREF(c_array);
            Py_DECREF(u_array);
            Py_DECREF(y_array);
            Py_DECREF(x0_array);
            return PyErr_NoMemory();
        }
        b_data = (f64*)PyArray_DATA(b_array);
        memset(b_data, 0, n * m * sizeof(f64));
    } else {
        b_array = (PyArrayObject*)PyArray_SimpleNew(2, b_dims, NPY_DOUBLE);
        b_data = (f64*)PyArray_DATA(b_array);
    }

    npy_intp d_dims[2] = {l, m > 0 ? m : 1};
    npy_intp d_strides[2] = {sizeof(f64), ldd * sizeof(f64)};
    f64 *d_data = NULL;
    PyArrayObject *d_array = NULL;
    if (m > 0 && withd) {
        d_array = (PyArrayObject*)PyArray_New(&PyArray_Type, 2, d_dims, NPY_DOUBLE,
                                               d_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        if (!d_array) {
            Py_DECREF(a_array);
            Py_DECREF(c_array);
            Py_DECREF(u_array);
            Py_DECREF(y_array);
            Py_DECREF(x0_array);
            Py_DECREF(b_array);
            return PyErr_NoMemory();
        }
        d_data = (f64*)PyArray_DATA(d_array);
        memset(d_data, 0, l * m * sizeof(f64));
    } else {
        d_array = (PyArrayObject*)PyArray_SimpleNew(2, d_dims, NPY_DOUBLE);
        d_data = (f64*)PyArray_DATA(d_array);
    }

    if (!x0_array || !b_array || !d_array) {
        Py_DECREF(a_array);
        Py_DECREF(c_array);
        Py_DECREF(u_array);
        Py_DECREF(y_array);
        Py_XDECREF(x0_array);
        Py_XDECREF(b_array);
        Py_XDECREF(d_array);
        return NULL;
    }

    f64 *x0_data = (f64*)PyArray_DATA(x0_array);

    i32 nsmpl = nsmp * l;
    i32 iq = ncol + (withd ? m : (withx0 ? 0 : 1));
    iq = iq * l;
    i32 ncp1 = ncol + 1;
    i32 isize = nsmpl * ncp1;

    i32 ic = (n > 0 && withx0) ? (2 * n * n + n) : 0;
    i32 minwls = ncol * ncp1;
    if (withd) minwls += l * m * ncp1;

    i32 ia;
    if (m > 0 && withd) {
        ia = m + (2 * ncol > m ? 2 * ncol : m);
    } else {
        ia = 2 * ncol;
    }

    i32 itau = n * n * m + (ic > ia ? ic : ia);
    if (withx0) itau += l * n;

    i32 ldw2 = isize + (n + (ic > ia ? ic : ia));
    i32 t = 6 * ncol;
    if (n + (ic > ia ? ic : ia) < t) ldw2 = isize + t;

    i32 ldw3 = minwls + (iq * ncp1 + itau);
    if (iq * ncp1 + itau < 6 * ncol) ldw3 = minwls + 6 * ncol;

    if (m > 0 && withd) {
        i32 t2 = isize + 2 * m * m + 6 * m;
        if (t2 > ldw2) ldw2 = t2;
        t2 = minwls + 2 * m * m + 6 * m;
        if (t2 > ldw3) ldw3 = t2;
    }

    i32 ldwork = (ldw2 < ldw3) ? ldw2 : ldw3;
    if (ldwork < 2) ldwork = 2;
    if (m > 0 && withd && ldwork < 3) ldwork = 3;
    ldwork = ldw2;

    i32 liwork = n * m + (withx0 ? n : 0);
    if (withd && m > liwork) liwork = m;
    if (liwork < 1) liwork = 1;

    f64 *dwork = (f64*)malloc(ldwork * sizeof(f64));
    i32 *iwork = (i32*)malloc(liwork * sizeof(i32));
    if (!dwork || !iwork) {
        free(dwork);
        free(iwork);
        Py_DECREF(a_array);
        Py_DECREF(c_array);
        Py_DECREF(u_array);
        Py_DECREF(y_array);
        Py_DECREF(x0_array);
        Py_DECREF(b_array);
        Py_DECREF(d_array);
        return PyErr_NoMemory();
    }

    slicot_ib01qd(jobx0_str, job_str, n, m, l, nsmp,
                  a_data, lda, c_data, ldc,
                  u_data, ldu, y_data, ldy,
                  x0_data, b_data, ldb, d_data, ldd,
                  tol, iwork, dwork, ldwork, &iwarn, &info);

    f64 rcond_w2 = dwork[1];
    f64 rcond_u = (m > 0 && withd) ? dwork[2] : 1.0;

    free(dwork);
    free(iwork);

    PyArray_ResolveWritebackIfCopy(u_array);
    Py_DECREF(a_array);
    Py_DECREF(c_array);
    Py_DECREF(u_array);
    Py_DECREF(y_array);

    if (info < 0) {
        Py_DECREF(x0_array);
        Py_DECREF(b_array);
        Py_DECREF(d_array);
        PyErr_Format(PyExc_ValueError, "Parameter %d had an illegal value", -info);
        return NULL;
    }

    PyObject *result = Py_BuildValue("OOOddii", x0_array, b_array, d_array,
                                      rcond_w2, rcond_u, iwarn, info);
    Py_DECREF(x0_array);
    Py_DECREF(b_array);
    Py_DECREF(d_array);
    return result;
}



/* Python wrapper for ib01rd */
PyObject* py_ib01rd(PyObject* self, PyObject* args) {
    const char *job_str;
    i32 n, m, l, nsmp;
    f64 tol;
    PyObject *a_obj, *b_obj, *c_obj, *d_obj, *u_obj, *y_obj;
    PyArrayObject *a_array, *b_array, *c_array, *d_array, *u_array, *y_array;
    i32 iwarn, info;

    if (!PyArg_ParseTuple(args, "siiiiOOOOOOd",
                          &job_str, &n, &m, &l, &nsmp,
                          &a_obj, &b_obj, &c_obj, &d_obj,
                          &u_obj, &y_obj, &tol)) {
        return NULL;
    }

    char job = toupper((unsigned char)job_str[0]);

    if (job != 'Z' && job != 'N') {
        PyErr_SetString(PyExc_ValueError, "JOB must be 'Z' or 'N'");
        return NULL;
    }
    if (n < 0) {
        PyErr_SetString(PyExc_ValueError, "N must be non-negative");
        return NULL;
    }
    if (m < 0) {
        PyErr_SetString(PyExc_ValueError, "M must be non-negative");
        return NULL;
    }
    if (l <= 0) {
        PyErr_SetString(PyExc_ValueError, "L must be positive");
        return NULL;
    }
    if (nsmp < n) {
        PyErr_SetString(PyExc_ValueError, "NSMP must be >= N");
        return NULL;
    }

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    b_array = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    c_array = (PyArrayObject*)PyArray_FROM_OTF(c_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    d_array = (PyArrayObject*)PyArray_FROM_OTF(d_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    u_array = (PyArrayObject*)PyArray_FROM_OTF(u_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    y_array = (PyArrayObject*)PyArray_FROM_OTF(y_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);

    if (!a_array || !b_array || !c_array || !d_array || !u_array || !y_array) {
        Py_XDECREF(a_array);
        Py_XDECREF(b_array);
        Py_XDECREF(c_array);
        Py_XDECREF(d_array);
        Py_XDECREF(u_array);
        Py_XDECREF(y_array);
        return NULL;
    }

    i32 lda = n > 0 ? n : 1;
    i32 ldb = (n > 0 && m > 0) ? n : 1;
    i32 ldc = l;
    bool withd = (job == 'N');
    i32 ldd = (withd && m > 0) ? l : 1;
    i32 ldu = (m > 0) ? nsmp : 1;
    i32 ldy = nsmp > 0 ? nsmp : 1;

    const f64 *a_data = (const f64*)PyArray_DATA(a_array);
    const f64 *b_data = (const f64*)PyArray_DATA(b_array);
    const f64 *c_data = (const f64*)PyArray_DATA(c_array);
    const f64 *d_data = (const f64*)PyArray_DATA(d_array);
    const f64 *u_data = (const f64*)PyArray_DATA(u_array);
    const f64 *y_data = (const f64*)PyArray_DATA(y_array);

    npy_intp x0_dims[1] = {n};
    PyArrayObject *x0_array = (PyArrayObject*)PyArray_SimpleNew(1, x0_dims, NPY_DOUBLE);
    if (!x0_array) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        Py_DECREF(u_array);
        Py_DECREF(y_array);
        return NULL;
    }
    f64 *x0_data = (f64*)PyArray_DATA(x0_array);

    i32 nn = n * n;
    i32 nsmpl = nsmp * l;
    i32 iq = n * l;
    i32 ncp1 = n + 1;
    i32 isize = nsmpl * ncp1;
    i32 ic = 2 * nn;
    i32 minwls = n * ncp1;
    i32 itau_calc = ic + l * n;
    i32 ldw1 = isize + 2 * n + ((ic > 4 * n) ? ic : 4 * n);
    i32 ldw2 = minwls + 2 * n + ((iq * ncp1 + itau_calc > 4 * n) ? (iq * ncp1 + itau_calc) : 4 * n);
    i32 ldwork = (ldw1 > ldw2) ? ldw1 : ldw2;
    if (ldwork < 2) ldwork = 2;

    i32 liwork = (n > 0) ? n : 1;

    f64 *dwork = (f64*)malloc(ldwork * sizeof(f64));
    i32 *iwork = (i32*)malloc(liwork * sizeof(i32));
    if (!dwork || !iwork) {
        free(dwork);
        free(iwork);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        Py_DECREF(u_array);
        Py_DECREF(y_array);
        Py_DECREF(x0_array);
        return PyErr_NoMemory();
    }

    char job_c[2] = {job, '\0'};
    slicot_ib01rd(job_c, n, m, l, nsmp,
                  a_data, lda, b_data, ldb, c_data, ldc, d_data, ldd,
                  u_data, ldu, y_data, ldy,
                  x0_data, tol, iwork, dwork, ldwork, &iwarn, &info);

    f64 rcond = (n > 0) ? dwork[1] : 1.0;

    free(dwork);
    free(iwork);

    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(d_array);
    Py_DECREF(u_array);
    Py_DECREF(y_array);

    if (info < 0) {
        Py_DECREF(x0_array);
        PyErr_Format(PyExc_ValueError, "Parameter %d had an illegal value", -info);
        return NULL;
    }

    PyObject *result = Py_BuildValue("Odii", x0_array, rcond, iwarn, info);
    Py_DECREF(x0_array);
    return result;
}



/* Python wrapper for ib01cd */
PyObject* py_ib01cd(PyObject* self, PyObject* args) {
    const char *jobx0_str, *comuse_str, *job_str;
    i32 n, m, l;
    f64 tol;
    PyObject *a_obj, *b_obj, *c_obj, *d_obj, *u_obj, *y_obj;
    PyArrayObject *a_array, *b_array, *c_array, *d_array, *u_array, *y_array;
    i32 iwarn, info;

    if (!PyArg_ParseTuple(args, "sssiiiOOOOOOd",
                          &jobx0_str, &comuse_str, &job_str, &n, &m, &l,
                          &a_obj, &b_obj, &c_obj, &d_obj,
                          &u_obj, &y_obj, &tol)) {
        return NULL;
    }

    char jobx0 = toupper((unsigned char)jobx0_str[0]);
    char comuse = toupper((unsigned char)comuse_str[0]);
    char job = toupper((unsigned char)job_str[0]);

    if (jobx0 != 'X' && jobx0 != 'N') {
        PyErr_SetString(PyExc_ValueError, "JOBX0 must be 'X' or 'N'");
        return NULL;
    }
    if (comuse != 'C' && comuse != 'U' && comuse != 'N') {
        PyErr_SetString(PyExc_ValueError, "COMUSE must be 'C', 'U', or 'N'");
        return NULL;
    }
    if (job != 'B' && job != 'D') {
        PyErr_SetString(PyExc_ValueError, "JOB must be 'B' or 'D'");
        return NULL;
    }
    if (n < 0) {
        PyErr_SetString(PyExc_ValueError, "N must be non-negative");
        return NULL;
    }
    if (m < 0) {
        PyErr_SetString(PyExc_ValueError, "M must be non-negative");
        return NULL;
    }
    if (l <= 0) {
        PyErr_SetString(PyExc_ValueError, "L must be positive");
        return NULL;
    }

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    c_array = (PyArrayObject*)PyArray_FROM_OTF(c_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    u_array = (PyArrayObject*)PyArray_FROM_OTF(u_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    y_array = (PyArrayObject*)PyArray_FROM_OTF(y_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);

    if (!a_array || !c_array || !u_array || !y_array) {
        Py_XDECREF(a_array);
        Py_XDECREF(c_array);
        Py_XDECREF(u_array);
        Py_XDECREF(y_array);
        return NULL;
    }

    bool withx0 = (jobx0 == 'X');
    bool compbd = (comuse == 'C');
    (void)comuse;
    bool withd = (job == 'D');
    bool maxdia = withx0 || compbd;

    npy_intp *u_dims = PyArray_DIMS(u_array);
    i32 nsmp = (i32)u_dims[0];
    i32 ldu = nsmp > 0 ? nsmp : 1;
    i32 ldy = nsmp > 0 ? nsmp : 1;
    i32 lda = n > 0 ? n : 1;
    i32 ldc = l;
    i32 ldb = (n > 0 && m > 0) ? n : 1;
    i32 ldd = (m > 0 && withd) ? l : 1;
    i32 ldv = n > 0 ? n : 1;

    i32 ncol, minsmp;
    if (compbd) {
        ncol = n * m;
        if (withx0) ncol += n;
        minsmp = ncol;
        if (withd) {
            minsmp += m;
        } else if (!withx0) {
            minsmp += 1;
        }
    } else {
        ncol = n;
        minsmp = withx0 ? n : 0;
    }

    if (nsmp < minsmp) {
        Py_DECREF(a_array);
        Py_DECREF(c_array);
        Py_DECREF(u_array);
        Py_DECREF(y_array);
        PyErr_SetString(PyExc_ValueError, "NSMP too small for the problem dimensions");
        return NULL;
    }

    b_array = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    d_array = (PyArrayObject*)PyArray_FROM_OTF(d_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);

    if (!b_array || !d_array) {
        Py_DECREF(a_array);
        Py_DECREF(c_array);
        Py_DECREF(u_array);
        Py_DECREF(y_array);
        Py_XDECREF(b_array);
        Py_XDECREF(d_array);
        return NULL;
    }

    const f64 *a_data = (const f64*)PyArray_DATA(a_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);
    const f64 *c_data = (const f64*)PyArray_DATA(c_array);
    f64 *d_data = (f64*)PyArray_DATA(d_array);
    f64 *u_data = (f64*)PyArray_DATA(u_array);
    const f64 *y_data = (const f64*)PyArray_DATA(y_array);

    npy_intp x0_dims[1] = {n};
    PyArrayObject *x0_array = (PyArrayObject*)PyArray_SimpleNew(1, x0_dims, NPY_DOUBLE);

    npy_intp v_dims[2] = {n > 0 ? n : 1, n > 0 ? n : 1};
    npy_intp v_strides[2] = {sizeof(f64), ldv * sizeof(f64)};
    f64 *v_data = NULL;
    PyArrayObject *v_array = NULL;
    if (n > 0) {
        v_array = (PyArrayObject*)PyArray_New(&PyArray_Type, 2, v_dims, NPY_DOUBLE,
                                               v_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        if (!v_array) {
            Py_DECREF(a_array);
            Py_DECREF(b_array);
            Py_DECREF(c_array);
            Py_DECREF(d_array);
            Py_DECREF(u_array);
            Py_DECREF(y_array);
            Py_DECREF(x0_array);
            return PyErr_NoMemory();
        }
        v_data = (f64*)PyArray_DATA(v_array);
        memset(v_data, 0, n * n * sizeof(f64));
    } else {
        v_array = (PyArrayObject*)PyArray_SimpleNew(2, v_dims, NPY_DOUBLE);
        v_data = (f64*)PyArray_DATA(v_array);
    }

    if (!x0_array || !v_array) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        Py_DECREF(u_array);
        Py_DECREF(y_array);
        Py_XDECREF(x0_array);
        Py_XDECREF(v_array);
        return NULL;
    }

    f64 *x0_data = (f64*)PyArray_DATA(x0_array);

    i32 nn = n * n;
    i32 nm = n * m;
    i32 ln = l * n;
    i32 lm = l * m;
    i32 n2m = n * nm;
    i32 ia_base = (compbd && m > 0 && withd) ? 3 : 2;

    i32 ldwork;
    if (!maxdia || n == 0) {
        ldwork = 2;
    } else {
        i32 nsmpl = nsmp * l;
        i32 iq_calc = ncol;
        if (compbd && withd) {
            iq_calc += m;
        }
        iq_calc *= l;
        i32 ncp1 = ncol + 1;
        i32 isize = nsmpl * ncp1;

        i32 ic_calc;
        if (compbd) {
            ic_calc = (n > 0 && withx0) ? (2 * nn + n) : 0;
        } else {
            ic_calc = 2 * nn;
        }

        i32 minwls = ncol * ncp1;
        if (compbd && withd) minwls += lm * ncp1;

        i32 ia_calc;
        if (compbd) {
            if (m > 0 && withd) {
                i32 twoncol = 2 * ncol;
                ia_calc = m + ((twoncol > m) ? twoncol : m);
            } else {
                ia_calc = 2 * ncol;
            }
        } else {
            ia_calc = 2 * ncol;
        }

        i32 itau = n2m + ((ic_calc > ia_calc) ? ic_calc : ia_calc);
        if (compbd && withx0) {
            itau += ln;
        } else if (!compbd) {
            itau = ic_calc + ln;
        }

        i32 ldw2, ldw3;
        if (compbd) {
            i32 max_ic_ia = (ic_calc > ia_calc) ? ic_calc : ia_calc;
            i32 t1 = n + max_ic_ia;
            i32 t2 = 6 * ncol;
            ldw2 = isize + ((t1 > t2) ? t1 : t2);
            ldw3 = minwls + ((iq_calc * ncp1 + itau > 6 * ncol) ? (iq_calc * ncp1 + itau) : 6 * ncol);
            if (m > 0 && withd) {
                i32 t3 = isize + 2 * m * m + 6 * m;
                if (t3 > ldw2) ldw2 = t3;
                t3 = minwls + 2 * m * m + 6 * m;
                if (t3 > ldw3) ldw3 = t3;
            }
        } else {
            ldw2 = isize + 2 * n + ((ic_calc > 4 * n) ? ic_calc : (4 * n));
            ldw3 = minwls + 2 * n + ((iq_calc * ncp1 + itau > 4 * n) ? (iq_calc * ncp1 + itau) : (4 * n));
        }

        i32 min_ldw = (ldw2 < ldw3) ? ldw2 : ldw3;
        i32 t_5n = 5 * n;
        i32 max_5n_ia = (t_5n > ia_base) ? t_5n : ia_base;
        i32 max_term = (max_5n_ia > min_ldw) ? max_5n_ia : min_ldw;
        ldwork = ia_base + nn + nm + ln + max_term;
    }

    if (ldwork < 2) ldwork = 2;

    i32 liwork = nm;
    if (withx0) liwork += n;
    if (compbd && withd && m > liwork) liwork = m;
    if (liwork < 1) liwork = 1;

    f64 *dwork = (f64*)malloc(ldwork * sizeof(f64));
    i32 *iwork = (i32*)malloc(liwork * sizeof(i32));
    if (!dwork || !iwork) {
        free(dwork);
        free(iwork);
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        Py_DECREF(u_array);
        Py_DECREF(y_array);
        Py_DECREF(x0_array);
        Py_DECREF(v_array);
        return PyErr_NoMemory();
    }

    char jobx0_c[2] = {jobx0, '\0'};
    char comuse_c[2] = {comuse, '\0'};
    char job_c[2] = {job, '\0'};

    slicot_ib01cd(jobx0_c, comuse_c, job_c, n, m, l, nsmp,
                  a_data, lda, b_data, ldb, c_data, ldc, d_data, ldd,
                  u_data, ldu, y_data, ldy,
                  x0_data, v_data, ldv,
                  tol, iwork, dwork, ldwork, &iwarn, &info);

    f64 rcond = (n > 0 && info >= 0) ? dwork[1] : 1.0;

    free(dwork);
    free(iwork);

    PyArray_ResolveWritebackIfCopy(b_array);
    PyArray_ResolveWritebackIfCopy(d_array);
    PyArray_ResolveWritebackIfCopy(u_array);
    Py_DECREF(a_array);
    Py_DECREF(c_array);
    Py_DECREF(u_array);
    Py_DECREF(y_array);

    if (info < 0) {
        Py_DECREF(x0_array);
        Py_DECREF(b_array);
        Py_DECREF(d_array);
        Py_DECREF(v_array);
        PyErr_Format(PyExc_ValueError, "Parameter %d had an illegal value", -info);
        return NULL;
    }

    PyObject *result = Py_BuildValue("OOOOdii", x0_array, b_array, d_array, v_array,
                                      rcond, iwarn, info);
    Py_DECREF(x0_array);
    Py_DECREF(b_array);
    Py_DECREF(d_array);
    Py_DECREF(v_array);
    return result;
}



/* Python wrapper for ib01pd */
PyObject* py_ib01pd(PyObject* self, PyObject* args, PyObject* kwargs) {
    const char *meth_str, *job_str, *jobcv_str;
    i32 nobr, n, m, l, nsmpl;
    PyObject *r_obj;
    PyObject *a_obj = Py_None, *c_obj = Py_None;
    f64 tol = 0.0;

    static char *kwlist[] = {"meth", "job", "jobcv", "nobr", "n", "m", "l",
                             "nsmpl", "r", "tol", "a", "c", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "sssiiiiiOd|OO", kwlist,
            &meth_str, &job_str, &jobcv_str, &nobr, &n, &m, &l,
            &nsmpl, &r_obj, &tol, &a_obj, &c_obj)) {
        return NULL;
    }

    char meth = meth_str[0];
    char job = job_str[0];
    char jobcv = jobcv_str[0];

    if (meth != 'M' && meth != 'm' && meth != 'N' && meth != 'n') {
        PyErr_SetString(PyExc_ValueError, "METH must be 'M' or 'N'");
        return NULL;
    }

    if (job != 'A' && job != 'a' && job != 'C' && job != 'c' &&
        job != 'B' && job != 'b' && job != 'D' && job != 'd') {
        PyErr_SetString(PyExc_ValueError, "JOB must be 'A', 'C', 'B', or 'D'");
        return NULL;
    }

    if (jobcv != 'C' && jobcv != 'c' && jobcv != 'N' && jobcv != 'n') {
        PyErr_SetString(PyExc_ValueError, "JOBCV must be 'C' or 'N'");
        return NULL;
    }

    if (nobr <= 1) {
        PyErr_SetString(PyExc_ValueError, "NOBR must be > 1");
        return NULL;
    }

    if (n <= 0 || n >= nobr) {
        PyErr_SetString(PyExc_ValueError, "N must be in range (0, NOBR)");
        return NULL;
    }

    if (m < 0) {
        PyErr_SetString(PyExc_ValueError, "M must be >= 0");
        return NULL;
    }

    if (l <= 0) {
        PyErr_SetString(PyExc_ValueError, "L must be > 0");
        return NULL;
    }

    i32 nr = 2 * (m + l) * nobr;
    bool withc = (job == 'A' || job == 'a' || job == 'C' || job == 'c');
    bool withb = (job == 'A' || job == 'a' || job == 'B' || job == 'b' ||
                  job == 'D' || job == 'd');
    bool withco = (jobcv == 'C' || jobcv == 'c');
    bool n4sid = (meth == 'N' || meth == 'n');

    PyArrayObject *r_array = (PyArrayObject*)PyArray_FROM_OTF(
        r_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (r_array == NULL) return NULL;

    i32 ldr = (i32)PyArray_DIM(r_array, 0);
    f64 *r_data = (f64*)PyArray_DATA(r_array);

    i32 lda = n > 1 ? n : 1;
    i32 ldc = l;
    i32 ldb = n > 1 ? n : 1;
    i32 ldd = l;
    i32 ldq = n > 1 ? n : 1;
    i32 ldry = l;
    i32 lds = n > 1 ? n : 1;
    i32 ldo = l * nobr;

    npy_intp a_dims[2] = {n, n};
    npy_intp c_dims[2] = {l, n};
    npy_intp b_dims[2] = {n, m};
    npy_intp d_dims[2] = {l, m};
    npy_intp q_dims[2] = {n, n};
    npy_intp ry_dims[2] = {l, l};
    npy_intp s_dims[2] = {n, l};
    npy_intp o_dims[2] = {l * nobr, n};

    npy_intp strides_nn[2] = {sizeof(f64), lda * sizeof(f64)};
    npy_intp strides_ln[2] = {sizeof(f64), ldc * sizeof(f64)};
    npy_intp strides_nm[2] = {sizeof(f64), ldb * sizeof(f64)};
    npy_intp strides_lm[2] = {sizeof(f64), ldd * sizeof(f64)};
    npy_intp strides_ll[2] = {sizeof(f64), ldry * sizeof(f64)};
    npy_intp strides_nl[2] = {sizeof(f64), lds * sizeof(f64)};
    npy_intp strides_on[2] = {sizeof(f64), ldo * sizeof(f64)};

    PyObject *a_out = PyArray_New(&PyArray_Type, 2, a_dims, NPY_DOUBLE, strides_nn, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    PyObject *c_out = PyArray_New(&PyArray_Type, 2, c_dims, NPY_DOUBLE, strides_ln, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    PyObject *b_out = PyArray_New(&PyArray_Type, 2, b_dims, NPY_DOUBLE, strides_nm, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    PyObject *d_out = PyArray_New(&PyArray_Type, 2, d_dims, NPY_DOUBLE, strides_lm, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    PyObject *q_out = PyArray_New(&PyArray_Type, 2, q_dims, NPY_DOUBLE, strides_nn, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    PyObject *ry_out = PyArray_New(&PyArray_Type, 2, ry_dims, NPY_DOUBLE, strides_ll, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    PyObject *s_out = PyArray_New(&PyArray_Type, 2, s_dims, NPY_DOUBLE, strides_nl, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    PyObject *o_out = PyArray_New(&PyArray_Type, 2, o_dims, NPY_DOUBLE, strides_on, NULL, 0, NPY_ARRAY_FARRAY, NULL);

    if (!a_out || !c_out || !b_out || !d_out || !q_out || !ry_out || !s_out || !o_out) {
        Py_XDECREF(a_out); Py_XDECREF(c_out); Py_XDECREF(b_out); Py_XDECREF(d_out);
        Py_XDECREF(q_out); Py_XDECREF(ry_out); Py_XDECREF(s_out); Py_XDECREF(o_out);
        Py_DECREF(r_array);
        PyErr_NoMemory();
        return NULL;
    }

    f64 *a = (f64*)PyArray_DATA((PyArrayObject*)a_out);
    f64 *c = (f64*)PyArray_DATA((PyArrayObject*)c_out);
    f64 *b = (f64*)PyArray_DATA((PyArrayObject*)b_out);
    f64 *d = (f64*)PyArray_DATA((PyArrayObject*)d_out);
    f64 *q = (f64*)PyArray_DATA((PyArrayObject*)q_out);
    f64 *ry = (f64*)PyArray_DATA((PyArrayObject*)ry_out);
    f64 *s = (f64*)PyArray_DATA((PyArrayObject*)s_out);
    f64 *o = (f64*)PyArray_DATA((PyArrayObject*)o_out);

    memset(a, 0, n * n * sizeof(f64));
    memset(c, 0, l * n * sizeof(f64));
    if (m > 0) {
        memset(b, 0, n * m * sizeof(f64));
        memset(d, 0, l * m * sizeof(f64));
    }
    memset(q, 0, n * n * sizeof(f64));
    memset(ry, 0, l * l * sizeof(f64));
    memset(s, 0, n * l * sizeof(f64));
    memset(o, 0, l * nobr * n * sizeof(f64));

    if (n4sid && (job == 'B' || job == 'b' || job == 'D' || job == 'd')) {
        if (a_obj != Py_None) {
            PyArrayObject *a_in = (PyArrayObject*)PyArray_FROM_OTF(
                a_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
            if (a_in) {
                memcpy(a, PyArray_DATA(a_in), n * n * sizeof(f64));
                Py_DECREF(a_in);
            }
        }
        if (c_obj != Py_None) {
            PyArrayObject *c_in = (PyArrayObject*)PyArray_FROM_OTF(
                c_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
            if (c_in) {
                memcpy(c, PyArray_DATA(c_in), l * n * sizeof(f64));
                Py_DECREF(c_in);
            }
        }
    }

    i32 liwork = n;
    if (m > 0 && withb) {
        i32 alt = m * nobr + n;
        if (l * nobr > liwork) liwork = l * nobr;
        if (alt > liwork) liwork = alt;
    }
    if (n4sid) {
        i32 alt = m * (n + l);
        if (alt > liwork) liwork = alt;
    }

    i32 *iwork = (i32*)calloc(liwork > 1 ? liwork : 1, sizeof(i32));

    i32 ldwork = 2 * (l * nobr - l) * n + n * n + 8 * n + nr * (n + l) + 1000;
    if (n4sid && m > 0 && withb) {
        i32 alt = m * nobr * (n + l) * (m * (n + l) + 1) + (n + l) * (n + l) + 4 * m * (n + l) + 1;
        if (alt > ldwork) ldwork = alt;
    }
    f64 *dwork = (f64*)calloc(ldwork, sizeof(f64));

    if (!iwork || !dwork) {
        Py_DECREF(a_out); Py_DECREF(c_out); Py_DECREF(b_out); Py_DECREF(d_out);
        Py_DECREF(q_out); Py_DECREF(ry_out); Py_DECREF(s_out); Py_DECREF(o_out);
        free(iwork); free(dwork);
        Py_DECREF(r_array);
        PyErr_NoMemory();
        return NULL;
    }

    i32 iwarn, info;
    ib01pd(meth_str, job_str, jobcv_str, nobr, n, m, l, nsmpl,
           r_data, ldr, a, lda, c, ldc, b, ldb, d, ldd, q, ldq,
           ry, ldry, s, lds, o, ldo, tol, iwork, dwork, ldwork,
           &iwarn, &info);

    free(iwork);

    npy_intp rcond_dims[1] = {4};
    PyObject *rcond_array = PyArray_SimpleNew(1, rcond_dims, NPY_DOUBLE);
    f64 *rcond_data = (f64*)PyArray_DATA((PyArrayObject*)rcond_array);
    rcond_data[0] = dwork[1];
    rcond_data[1] = dwork[2];
    rcond_data[2] = dwork[3];
    rcond_data[3] = dwork[4];

    free(dwork);

    PyArray_ResolveWritebackIfCopy(r_array);
    Py_DECREF(r_array);

    PyObject *result;

    if (withc && withb && withco) {
        result = Py_BuildValue("NNNNNNNNNii", a_out, c_out, b_out, d_out,
                               q_out, ry_out, s_out, o_out, rcond_array, iwarn, info);
    } else if (withc && withb && !withco) {
        Py_DECREF(q_out); Py_DECREF(ry_out); Py_DECREF(s_out); Py_DECREF(o_out);
        result = Py_BuildValue("NNNNNii", a_out, c_out, b_out, d_out, rcond_array, iwarn, info);
    } else if (withc && !withb) {
        Py_DECREF(b_out); Py_DECREF(d_out); Py_DECREF(q_out); Py_DECREF(ry_out); Py_DECREF(s_out); Py_DECREF(o_out);
        result = Py_BuildValue("NNNii", a_out, c_out, rcond_array, iwarn, info);
    } else if (!withc && withb) {
        Py_DECREF(a_out); Py_DECREF(c_out); Py_DECREF(q_out); Py_DECREF(ry_out); Py_DECREF(s_out); Py_DECREF(o_out);
        result = Py_BuildValue("NNNii", b_out, d_out, rcond_array, iwarn, info);
    } else {
        Py_DECREF(a_out); Py_DECREF(c_out); Py_DECREF(b_out); Py_DECREF(d_out);
        Py_DECREF(q_out); Py_DECREF(ry_out); Py_DECREF(s_out); Py_DECREF(o_out);
        result = Py_BuildValue("Nii", rcond_array, iwarn, info);
    }

    return result;
}



/* Python wrapper for ib01oy */
PyObject* py_ib01oy(PyObject* self, PyObject* args) {
    i32 ns, nmax, n;
    PyObject *sv_obj;
    PyArrayObject *sv_array;
    i32 info;

    if (!PyArg_ParseTuple(args, "iiiO", &ns, &nmax, &n, &sv_obj)) {
        return NULL;
    }

    /* Validate parameters before array conversion */
    if (ns <= 0) {
        PyErr_SetString(PyExc_ValueError, "NS must be positive");
        return NULL;
    }

    if (nmax < 0 || nmax > ns) {
        PyErr_SetString(PyExc_ValueError, "NMAX must be in range [0, NS]");
        return NULL;
    }

    if (n < 0 || n > ns) {
        PyErr_SetString(PyExc_ValueError, "N must be in range [0, NS]");
        return NULL;
    }

    /* Convert SV array */
    sv_array = (PyArrayObject*)PyArray_FROM_OTF(sv_obj, NPY_DOUBLE,
                                                NPY_ARRAY_IN_FARRAY);
    if (sv_array == NULL) {
        return NULL;
    }

    /* Validate SV size */
    npy_intp sv_size = PyArray_SIZE(sv_array);
    if (sv_size < ns) {
        Py_DECREF(sv_array);
        PyErr_SetString(PyExc_ValueError, "SV must have at least NS elements");
        return NULL;
    }

    f64 *sv = (f64*)PyArray_DATA(sv_array);

    /* Call C function */
    SLC_IB01OY(ns, nmax, &n, sv, &info);

    Py_DECREF(sv_array);

    if (info < 0) {
        PyErr_Format(PyExc_ValueError, "Parameter %d had an illegal value", -info);
        return NULL;
    }

    return Py_BuildValue("ii", n, info);
}

