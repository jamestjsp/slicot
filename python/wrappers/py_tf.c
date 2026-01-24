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



PyObject* py_tf01md(PyObject* self, PyObject* args) {
    PyObject *a_obj, *b_obj, *c_obj, *d_obj, *u_obj, *x_obj;
    PyArrayObject *a_array, *b_array, *c_array, *d_array, *u_array, *x_array;
    i32 info;

    if (!PyArg_ParseTuple(args, "OOOOOO", &a_obj, &b_obj, &c_obj, &d_obj, &u_obj, &x_obj)) {
        return NULL;
    }

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (a_array == NULL) return NULL;

    b_array = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (b_array == NULL) {
        Py_DECREF(a_array);
        return NULL;
    }

    c_array = (PyArrayObject*)PyArray_FROM_OTF(c_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (c_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        return NULL;
    }

    d_array = (PyArrayObject*)PyArray_FROM_OTF(d_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (d_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        return NULL;
    }

    u_array = (PyArrayObject*)PyArray_FROM_OTF(u_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (u_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        return NULL;
    }

    x_array = (PyArrayObject*)PyArray_FROM_OTF(x_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (x_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        Py_DECREF(u_array);
        return NULL;
    }

    i32 n = (i32)PyArray_DIM(a_array, 0);
    i32 m = (PyArray_NDIM(b_array) >= 2) ? (i32)PyArray_DIM(b_array, 1) : ((n > 0) ? 1 : 0);
    i32 p = (i32)PyArray_DIM(c_array, 0);
    i32 ny = (PyArray_NDIM(u_array) >= 2) ? (i32)PyArray_DIM(u_array, 1) : ((m > 0) ? 1 : 0);

    if (PyArray_DIM(a_array, 1) != n) {
        PyErr_SetString(PyExc_ValueError, "A must be square");
        goto fail;
    }
    if (n > 0 && (i32)PyArray_DIM(b_array, 0) != n) {
        PyErr_SetString(PyExc_ValueError, "B row count must match A row count");
        goto fail;
    }
    if ((i32)PyArray_DIM(c_array, 1) != n) {
        PyErr_SetString(PyExc_ValueError, "C column count must match A column count");
        goto fail;
    }
    if ((i32)PyArray_DIM(d_array, 0) != p) {
        PyErr_SetString(PyExc_ValueError, "D row count must match C row count");
        goto fail;
    }
    if ((i32)PyArray_DIM(d_array, 1) != m) {
        PyErr_SetString(PyExc_ValueError, "D column count must match B column count");
        goto fail;
    }
    if ((i32)PyArray_DIM(u_array, 0) != m) {
        PyErr_SetString(PyExc_ValueError, "U row count must match B column count");
        goto fail;
    }
    if ((i32)PyArray_SIZE(x_array) != n) {
        PyErr_SetString(PyExc_ValueError, "x length must match state dimension");
        goto fail;
    }

    i32 lda = (n > 1) ? n : 1;
    i32 ldb = (n > 1) ? n : 1;
    i32 ldc = (p > 1) ? p : 1;
    i32 ldd = (p > 1) ? p : 1;
    i32 ldu = (m > 1) ? m : 1;
    i32 ldy = (p > 1) ? p : 1;

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);
    f64 *c_data = (f64*)PyArray_DATA(c_array);
    f64 *d_data = (f64*)PyArray_DATA(d_array);
    f64 *u_data = (f64*)PyArray_DATA(u_array);
    f64 *x_data = (f64*)PyArray_DATA(x_array);

    npy_intp dims_y[2] = {p, ny};
    npy_intp strides_y[2] = {sizeof(f64), ldy * sizeof(f64)};
    PyObject *y_array = PyArray_New(&PyArray_Type, 2, dims_y, NPY_DOUBLE, strides_y,
                                     NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (y_array == NULL) {
        goto fail;
    }

    f64 *y_data = (f64*)PyArray_DATA((PyArrayObject*)y_array);

    f64 *dwork = NULL;
    if (n > 0) {
        dwork = (f64*)malloc(n * sizeof(f64));
        if (dwork == NULL) {
            PyErr_SetString(PyExc_MemoryError, "Failed to allocate workspace");
            Py_DECREF(y_array);
            goto fail;
        }
    }

    tf01md(n, m, p, ny, a_data, lda, b_data, ldb, c_data, ldc, d_data, ldd,
           u_data, ldu, x_data, y_data, ldy, dwork, &info);

    if (dwork != NULL) {
        free(dwork);
    }

    PyArray_ResolveWritebackIfCopy(x_array);

    PyObject *result = Py_BuildValue("OOi", y_array, x_array, info);

    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(d_array);
    Py_DECREF(u_array);
    Py_DECREF(x_array);
    Py_DECREF(y_array);

    return result;

fail:
    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(d_array);
    Py_DECREF(u_array);
    Py_DECREF(x_array);
    return NULL;
}



PyObject* py_tf01mx(PyObject* self, PyObject* args) {
    i32 n, m, p, ny;
    PyObject *s_obj, *u_obj, *x_obj;
    PyArrayObject *s_array, *u_array, *x_array;
    i32 info;

    if (!PyArg_ParseTuple(args, "iiiiOOO", &n, &m, &p, &ny, &s_obj, &u_obj, &x_obj)) {
        return NULL;
    }

    if (n < 0) {
        PyErr_SetString(PyExc_ValueError, "N must be >= 0");
        return NULL;
    }
    if (m < 0) {
        PyErr_SetString(PyExc_ValueError, "M must be >= 0");
        return NULL;
    }
    if (p < 0) {
        PyErr_SetString(PyExc_ValueError, "P must be >= 0");
        return NULL;
    }
    if (ny < 0) {
        PyErr_SetString(PyExc_ValueError, "NY must be >= 0");
        return NULL;
    }

    s_array = (PyArrayObject*)PyArray_FROM_OTF(s_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (s_array == NULL) return NULL;

    u_array = (PyArrayObject*)PyArray_FROM_OTF(u_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (u_array == NULL) {
        Py_DECREF(s_array);
        return NULL;
    }

    x_array = (PyArrayObject*)PyArray_FROM_OTF(x_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (x_array == NULL) {
        Py_DECREF(s_array);
        Py_DECREF(u_array);
        return NULL;
    }

    i32 lds = (i32)PyArray_DIM(s_array, 0);
    i32 ldu = (ny > 1) ? (i32)PyArray_DIM(u_array, 0) : 1;
    i32 ldy = (ny > 1) ? ny : 1;

    f64 *s_data = (f64*)PyArray_DATA(s_array);
    f64 *u_data = (f64*)PyArray_DATA(u_array);
    f64 *x_data = (f64*)PyArray_DATA(x_array);

    npy_intp dims_y[2] = {ny, p};
    npy_intp strides_y[2] = {sizeof(f64), ldy * sizeof(f64)};
    PyObject *y_array = PyArray_New(&PyArray_Type, 2, dims_y, NPY_DOUBLE, strides_y,
                                     NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (y_array == NULL) {
        Py_DECREF(s_array);
        Py_DECREF(u_array);
        Py_DECREF(x_array);
        return NULL;
    }

    f64 *y_data = (f64*)PyArray_DATA((PyArrayObject*)y_array);

    i32 ldwork;
    if (n == 0 || p == 0 || ny == 0) {
        ldwork = 0;
    } else if (m == 0) {
        ldwork = n + p;
    } else {
        ldwork = 2 * n + m + p;
    }

    f64 *dwork = NULL;
    if (ldwork > 0) {
        dwork = (f64*)malloc(ldwork * sizeof(f64));
        if (dwork == NULL) {
            PyErr_SetString(PyExc_MemoryError, "Failed to allocate workspace");
            Py_DECREF(s_array);
            Py_DECREF(u_array);
            Py_DECREF(x_array);
            Py_DECREF(y_array);
            return NULL;
        }
    }

    tf01mx(n, m, p, ny, s_data, lds, u_data, ldu, x_data, y_data, ldy, dwork, ldwork, &info);

    if (dwork != NULL) {
        free(dwork);
    }

    PyObject *result = Py_BuildValue("OOi", y_array, x_array, info);

    Py_DECREF(s_array);
    Py_DECREF(u_array);
    Py_DECREF(x_array);
    Py_DECREF(y_array);

    return result;
}



PyObject* py_tf01my(PyObject* self, PyObject* args) {
    PyObject *a_obj, *b_obj, *c_obj, *d_obj, *u_obj, *x_obj;
    PyArrayObject *a_array = NULL, *b_array = NULL, *c_array = NULL;
    PyArrayObject *d_array = NULL, *u_array = NULL, *x_array = NULL;
    i32 info;

    if (!PyArg_ParseTuple(args, "OOOOOO", &a_obj, &b_obj, &c_obj, &d_obj, &u_obj, &x_obj)) {
        return NULL;
    }

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (a_array == NULL) return NULL;

    b_array = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (b_array == NULL) goto fail_tf01my;

    c_array = (PyArrayObject*)PyArray_FROM_OTF(c_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (c_array == NULL) goto fail_tf01my;

    d_array = (PyArrayObject*)PyArray_FROM_OTF(d_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (d_array == NULL) goto fail_tf01my;

    u_array = (PyArrayObject*)PyArray_FROM_OTF(u_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (u_array == NULL) goto fail_tf01my;

    x_array = (PyArrayObject*)PyArray_FROM_OTF(x_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (x_array == NULL) goto fail_tf01my;

    i32 n = (i32)PyArray_DIM(a_array, 0);
    i32 m = (PyArray_NDIM(b_array) >= 2) ? (i32)PyArray_DIM(b_array, 1) : ((n > 0) ? 1 : 0);
    i32 p = (i32)PyArray_DIM(c_array, 0);
    i32 ny = (i32)PyArray_DIM(u_array, 0);

    if (PyArray_NDIM(a_array) >= 2 && (i32)PyArray_DIM(a_array, 1) != n) {
        PyErr_SetString(PyExc_ValueError, "A must be square");
        goto fail_tf01my;
    }
    if (n > 0 && (i32)PyArray_DIM(b_array, 0) != n) {
        PyErr_SetString(PyExc_ValueError, "B row count must match A row count");
        goto fail_tf01my;
    }
    if (PyArray_NDIM(c_array) >= 2 && (i32)PyArray_DIM(c_array, 1) != n) {
        PyErr_SetString(PyExc_ValueError, "C column count must match A column count");
        goto fail_tf01my;
    }
    if ((i32)PyArray_DIM(d_array, 0) != p) {
        PyErr_SetString(PyExc_ValueError, "D row count must match C row count");
        goto fail_tf01my;
    }
    if (PyArray_NDIM(d_array) >= 2 && (i32)PyArray_DIM(d_array, 1) != m) {
        PyErr_SetString(PyExc_ValueError, "D column count must match B column count");
        goto fail_tf01my;
    }
    if (PyArray_NDIM(u_array) >= 2 && (i32)PyArray_DIM(u_array, 1) != m) {
        PyErr_SetString(PyExc_ValueError, "U column count must match B column count");
        goto fail_tf01my;
    }
    if ((i32)PyArray_SIZE(x_array) != n) {
        PyErr_SetString(PyExc_ValueError, "x length must match state dimension");
        goto fail_tf01my;
    }

    i32 lda = (n > 1) ? n : 1;
    i32 ldb = (n > 1) ? n : 1;
    i32 ldc = (p > 1) ? p : 1;
    i32 ldd = (p > 1) ? p : 1;
    i32 ldu = (ny > 1) ? ny : 1;
    i32 ldy = (ny > 1) ? ny : 1;

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);
    f64 *c_data = (f64*)PyArray_DATA(c_array);
    f64 *d_data = (f64*)PyArray_DATA(d_array);
    f64 *u_data = (f64*)PyArray_DATA(u_array);
    f64 *x_data = (f64*)PyArray_DATA(x_array);

    npy_intp dims_y[2] = {ny, p};
    npy_intp strides_y[2] = {sizeof(f64), ldy * sizeof(f64)};
    PyObject *y_array = PyArray_New(&PyArray_Type, 2, dims_y, NPY_DOUBLE, strides_y,
                                     NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (y_array == NULL) goto fail_tf01my;

    f64 *y_data = (f64*)PyArray_DATA((PyArrayObject*)y_array);

    i32 ldwork = (n > 1) ? n : 1;
    f64 *dwork = (f64*)malloc(ldwork * sizeof(f64));
    if (dwork == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate workspace");
        Py_DECREF(y_array);
        goto fail_tf01my;
    }

    tf01my(n, m, p, ny, a_data, lda, b_data, ldb, c_data, ldc, d_data, ldd,
           u_data, ldu, x_data, y_data, ldy, dwork, ldwork, &info);

    free(dwork);

    PyArray_ResolveWritebackIfCopy(x_array);

    PyObject *result = Py_BuildValue("OOi", y_array, x_array, info);

    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(d_array);
    Py_DECREF(u_array);
    Py_DECREF(x_array);
    Py_DECREF(y_array);

    return result;

fail_tf01my:
    Py_XDECREF(a_array);
    Py_XDECREF(b_array);
    Py_XDECREF(c_array);
    Py_XDECREF(d_array);
    Py_XDECREF(u_array);
    Py_XDECREF(x_array);
    return NULL;
}


/* Python wrapper for tf01rd */
PyObject* py_tf01rd(PyObject* self, PyObject* args) {
    PyObject *a_obj, *b_obj, *c_obj;
    i32 n_param;
    PyArrayObject *a_array, *b_array, *c_array;
    i32 info;

    if (!PyArg_ParseTuple(args, "OOOi", &a_obj, &b_obj, &c_obj, &n_param)) {
        return NULL;
    }

    if (n_param < 0) {
        PyErr_SetString(PyExc_ValueError, "N must be >= 0");
        return NULL;
    }

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (a_array == NULL) return NULL;

    b_array = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (b_array == NULL) {
        Py_DECREF(a_array);
        return NULL;
    }

    c_array = (PyArrayObject*)PyArray_FROM_OTF(c_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (c_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        return NULL;
    }

    i32 na = (i32)PyArray_DIM(a_array, 0);
    i32 nb = (PyArray_NDIM(b_array) >= 2) ? (i32)PyArray_DIM(b_array, 1) : 1;
    i32 nc = (i32)PyArray_DIM(c_array, 0);

    if ((i32)PyArray_DIM(a_array, 1) != na) {
        PyErr_SetString(PyExc_ValueError, "A must be square");
        goto fail;
    }
    if (na > 0 && (i32)PyArray_DIM(b_array, 0) != na) {
        PyErr_SetString(PyExc_ValueError, "B row count must match A row count");
        goto fail;
    }
    if ((i32)PyArray_DIM(c_array, 1) != na) {
        PyErr_SetString(PyExc_ValueError, "C column count must match A column count");
        goto fail;
    }

    i32 lda = (na > 1) ? na : 1;
    i32 ldb = (na > 1) ? na : 1;
    i32 ldc = (nc > 1) ? nc : 1;
    i32 ldh = (nc > 1) ? nc : 1;

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);
    f64 *c_data = (f64*)PyArray_DATA(c_array);

    npy_intp h_cols = (npy_intp)n_param * (npy_intp)nb;
    npy_intp dims_h[2] = {nc, h_cols};
    npy_intp strides_h[2] = {sizeof(f64), ldh * sizeof(f64)};
    PyObject *h_array = PyArray_New(&PyArray_Type, 2, dims_h, NPY_DOUBLE, strides_h,
                                     NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (h_array == NULL) {
        goto fail;
    }

    f64 *h_data = (f64*)PyArray_DATA((PyArrayObject*)h_array);

    i32 ldwork = (2 * na * nc > 1) ? 2 * na * nc : 1;
    f64 *dwork = (f64*)malloc(ldwork * sizeof(f64));
    if (dwork == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate workspace");
        Py_DECREF(h_array);
        goto fail;
    }

    tf01rd(na, nb, nc, n_param, a_data, lda, b_data, ldb, c_data, ldc,
           h_data, ldh, dwork, ldwork, &info);

    free(dwork);

    PyObject *result = Py_BuildValue("Oi", h_array, info);

    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(h_array);

    return result;

fail:
    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    return NULL;
}



/* Python wrapper for tf01od */
PyObject* py_tf01od(PyObject* self, PyObject* args) {
    PyObject *h_obj;
    i32 nr, nc;
    PyArrayObject *h_array = NULL;
    i32 info;

    if (!PyArg_ParseTuple(args, "Oii", &h_obj, &nr, &nc)) {
        return NULL;
    }

    if (nr < 0) {
        PyErr_SetString(PyExc_ValueError, "NR must be >= 0");
        return NULL;
    }
    if (nc < 0) {
        PyErr_SetString(PyExc_ValueError, "NC must be >= 0");
        return NULL;
    }

    h_array = (PyArrayObject*)PyArray_FROM_OTF(h_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (h_array == NULL) return NULL;

    i32 nh1 = (i32)PyArray_DIM(h_array, 0);
    i32 h_cols = (PyArray_NDIM(h_array) >= 2) ? (i32)PyArray_DIM(h_array, 1) : 1;
    i32 nh2 = (nr + nc > 1) ? h_cols / (nr + nc - 1) : h_cols;

    if (nr + nc > 1 && nh2 * (nr + nc - 1) != h_cols) {
        PyErr_SetString(PyExc_ValueError,
            "H columns must equal NH2 * (NR + NC - 1) for some integer NH2");
        Py_DECREF(h_array);
        return NULL;
    }

    i32 ldh = (nh1 > 1) ? nh1 : 1;
    i32 t_rows = nh1 * nr;
    i32 t_cols = nh2 * nc;
    i32 ldt = (t_rows > 1) ? t_rows : 1;

    f64 *h_data = (f64*)PyArray_DATA(h_array);

    npy_intp dims_t[2] = {t_rows, t_cols};
    npy_intp strides_t[2] = {sizeof(f64), ldt * sizeof(f64)};
    PyObject *t_array = PyArray_New(&PyArray_Type, 2, dims_t, NPY_DOUBLE, strides_t,
                                     NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (t_array == NULL) {
        Py_DECREF(h_array);
        return NULL;
    }

    f64 *t_data = (f64*)PyArray_DATA((PyArrayObject*)t_array);

    tf01od(nh1, nh2, nr, nc, h_data, ldh, t_data, ldt, &info);

    PyObject *result = Py_BuildValue("Oi", t_array, info);

    Py_DECREF(h_array);
    Py_DECREF(t_array);

    return result;
}


PyObject* py_tf01nd(PyObject* self, PyObject* args) {
    const char *uplo;
    PyObject *a_obj, *b_obj, *c_obj, *d_obj, *u_obj, *x_obj;
    PyArrayObject *a_array, *b_array, *c_array, *d_array, *u_array, *x_array;
    i32 info;

    if (!PyArg_ParseTuple(args, "sOOOOOO", &uplo, &a_obj, &b_obj, &c_obj, &d_obj, &u_obj, &x_obj)) {
        return NULL;
    }

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (a_array == NULL) return NULL;

    b_array = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (b_array == NULL) {
        Py_DECREF(a_array);
        return NULL;
    }

    c_array = (PyArrayObject*)PyArray_FROM_OTF(c_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (c_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        return NULL;
    }

    d_array = (PyArrayObject*)PyArray_FROM_OTF(d_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (d_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        return NULL;
    }

    u_array = (PyArrayObject*)PyArray_FROM_OTF(u_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (u_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        return NULL;
    }

    x_array = (PyArrayObject*)PyArray_FROM_OTF(x_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (x_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(b_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        Py_DECREF(u_array);
        return NULL;
    }

    i32 n = (i32)PyArray_DIM(a_array, 0);
    i32 m = (PyArray_NDIM(b_array) >= 2) ? (i32)PyArray_DIM(b_array, 1) : ((n > 0) ? 1 : 0);
    i32 p = (i32)PyArray_DIM(c_array, 0);
    i32 ny = (PyArray_NDIM(u_array) >= 2) ? (i32)PyArray_DIM(u_array, 1) : ((m > 0) ? 1 : 0);

    if (PyArray_DIM(a_array, 1) != n) {
        PyErr_SetString(PyExc_ValueError, "A must be square");
        goto fail;
    }
    if (n > 0 && (i32)PyArray_DIM(b_array, 0) != n) {
        PyErr_SetString(PyExc_ValueError, "B row count must match A row count");
        goto fail;
    }
    if ((i32)PyArray_DIM(c_array, 1) != n) {
        PyErr_SetString(PyExc_ValueError, "C column count must match A column count");
        goto fail;
    }
    if ((i32)PyArray_DIM(d_array, 0) != p) {
        PyErr_SetString(PyExc_ValueError, "D row count must match C row count");
        goto fail;
    }
    if ((i32)PyArray_DIM(d_array, 1) != m) {
        PyErr_SetString(PyExc_ValueError, "D column count must match B column count");
        goto fail;
    }
    if ((i32)PyArray_DIM(u_array, 0) != m) {
        PyErr_SetString(PyExc_ValueError, "U row count must match B column count");
        goto fail;
    }
    if ((i32)PyArray_SIZE(x_array) != n) {
        PyErr_SetString(PyExc_ValueError, "x length must match state dimension");
        goto fail;
    }

    i32 lda = (n > 1) ? n : 1;
    i32 ldb = (n > 1) ? n : 1;
    i32 ldc = (p > 1) ? p : 1;
    i32 ldd = (p > 1) ? p : 1;
    i32 ldu = (m > 1) ? m : 1;
    i32 ldy = (p > 1) ? p : 1;

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);
    f64 *c_data = (f64*)PyArray_DATA(c_array);
    f64 *d_data = (f64*)PyArray_DATA(d_array);
    f64 *u_data = (f64*)PyArray_DATA(u_array);
    f64 *x_data = (f64*)PyArray_DATA(x_array);

    npy_intp dims_y[2] = {p, ny};
    npy_intp strides_y[2] = {sizeof(f64), ldy * sizeof(f64)};
    PyObject *y_array = PyArray_New(&PyArray_Type, 2, dims_y, NPY_DOUBLE, strides_y,
                                     NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (y_array == NULL) {
        goto fail;
    }

    f64 *y_data = (f64*)PyArray_DATA((PyArrayObject*)y_array);

    f64 *dwork = NULL;
    if (n > 0) {
        dwork = (f64*)malloc(n * sizeof(f64));
        if (dwork == NULL) {
            PyErr_SetString(PyExc_MemoryError, "Failed to allocate workspace");
            Py_DECREF(y_array);
            goto fail;
        }
    }

    tf01nd(uplo, n, m, p, ny, a_data, lda, b_data, ldb, c_data, ldc, d_data, ldd,
           u_data, ldu, x_data, y_data, ldy, dwork, &info);

    if (dwork != NULL) {
        free(dwork);
    }

    PyArray_ResolveWritebackIfCopy(x_array);

    PyObject *result = Py_BuildValue("OOi", y_array, x_array, info);

    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(d_array);
    Py_DECREF(u_array);
    Py_DECREF(x_array);
    Py_DECREF(y_array);

    return result;

fail:
    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(c_array);
    Py_DECREF(d_array);
    Py_DECREF(u_array);
    Py_DECREF(x_array);
    return NULL;
}


PyObject* py_tf01pd(PyObject* self, PyObject* args) {
    PyObject *h_obj;
    i32 nr, nc;
    PyArrayObject *h_array = NULL;
    i32 info;

    if (!PyArg_ParseTuple(args, "Oii", &h_obj, &nr, &nc)) {
        return NULL;
    }

    if (nr < 0) {
        PyErr_SetString(PyExc_ValueError, "NR must be >= 0");
        return NULL;
    }
    if (nc < 0) {
        PyErr_SetString(PyExc_ValueError, "NC must be >= 0");
        return NULL;
    }

    h_array = (PyArrayObject*)PyArray_FROM_OTF(h_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (h_array == NULL) return NULL;

    i32 nh1 = (i32)PyArray_DIM(h_array, 0);
    i32 h_cols = (PyArray_NDIM(h_array) >= 2) ? (i32)PyArray_DIM(h_array, 1) : 1;
    i32 nh2 = (nr + nc > 1) ? h_cols / (nr + nc - 1) : h_cols;

    if (nr + nc > 1 && nh2 * (nr + nc - 1) != h_cols) {
        PyErr_SetString(PyExc_ValueError,
            "H columns must equal NH2 * (NR + NC - 1) for some integer NH2");
        Py_DECREF(h_array);
        return NULL;
    }

    i32 ldh = (nh1 > 1) ? nh1 : 1;
    i32 t_rows = nh1 * nr;
    i32 t_cols = nh2 * nc;
    i32 ldt = (t_rows > 1) ? t_rows : 1;

    f64 *h_data = (f64*)PyArray_DATA(h_array);

    npy_intp dims_t[2] = {t_rows, t_cols};
    npy_intp strides_t[2] = {sizeof(f64), ldt * sizeof(f64)};
    PyObject *t_array = PyArray_New(&PyArray_Type, 2, dims_t, NPY_DOUBLE, strides_t,
                                     NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (t_array == NULL) {
        Py_DECREF(h_array);
        return NULL;
    }

    f64 *t_data = (f64*)PyArray_DATA((PyArrayObject*)t_array);

    tf01pd(nh1, nh2, nr, nc, h_data, ldh, t_data, ldt, &info);

    PyObject *result = Py_BuildValue("Oi", t_array, info);

    Py_DECREF(h_array);
    Py_DECREF(t_array);

    return result;
}


/* Python wrapper for tf01qd */
PyObject* py_tf01qd(PyObject* self, PyObject* args) {
    PyObject *iord_obj, *ar_obj, *ma_obj;
    i32 nc, nb, n_param;
    PyArrayObject *iord_array = NULL, *ar_array = NULL, *ma_array = NULL;
    i32 info;

    if (!PyArg_ParseTuple(args, "iiiOOO", &nc, &nb, &n_param, &iord_obj, &ar_obj, &ma_obj)) {
        return NULL;
    }

    if (nc < 0) {
        PyErr_SetString(PyExc_ValueError, "NC must be >= 0");
        return NULL;
    }
    if (nb < 0) {
        PyErr_SetString(PyExc_ValueError, "NB must be >= 0");
        return NULL;
    }
    if (n_param < 0) {
        PyErr_SetString(PyExc_ValueError, "N must be >= 0");
        return NULL;
    }

    iord_array = (PyArrayObject*)PyArray_FROM_OTF(iord_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    if (iord_array == NULL) return NULL;

    ar_array = (PyArrayObject*)PyArray_FROM_OTF(ar_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (ar_array == NULL) {
        Py_DECREF(iord_array);
        return NULL;
    }

    ma_array = (PyArrayObject*)PyArray_FROM_OTF(ma_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (ma_array == NULL) {
        Py_DECREF(iord_array);
        Py_DECREF(ar_array);
        return NULL;
    }

    i32 *iord_data = (i32*)PyArray_DATA(iord_array);
    f64 *ar_data = (f64*)PyArray_DATA(ar_array);
    f64 *ma_data = (f64*)PyArray_DATA(ma_array);

    i32 ldh = (nc > 1) ? nc : 1;
    npy_intp h_cols = (npy_intp)n_param * (npy_intp)nb;
    npy_intp dims_h[2] = {nc, h_cols};
    npy_intp strides_h[2] = {sizeof(f64), ldh * sizeof(f64)};
    PyObject *h_array = PyArray_New(&PyArray_Type, 2, dims_h, NPY_DOUBLE, strides_h,
                                     NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (h_array == NULL) {
        goto fail;
    }

    f64 *h_data = (f64*)PyArray_DATA((PyArrayObject*)h_array);

    tf01qd(nc, nb, n_param, iord_data, ar_data, ma_data, h_data, ldh, &info);

    PyObject *result = Py_BuildValue("Oi", h_array, info);

    Py_DECREF(iord_array);
    Py_DECREF(ar_array);
    Py_DECREF(ma_array);
    Py_DECREF(h_array);

    return result;

fail:
    Py_DECREF(iord_array);
    Py_DECREF(ar_array);
    Py_DECREF(ma_array);
    return NULL;
}
