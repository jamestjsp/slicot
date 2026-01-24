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


PyObject* py_sg03br(PyObject* self, PyObject* args) {
    f64 xr, xi, yr, yi;
    f64 c, sr, si, zr, zi;

    if (!PyArg_ParseTuple(args, "dddd", &xr, &xi, &yr, &yi)) {
        return NULL;
    }

    sg03br(xr, xi, yr, yi, &c, &sr, &si, &zr, &zi);

    return Py_BuildValue("(ddddd)", c, sr, si, zr, zi);
}



/* Python wrapper for sg03ay */
PyObject* py_sg03ay(PyObject* self, PyObject* args) {
    char* trans;
    PyObject *a_obj, *e_obj, *x_obj;
    PyArrayObject *a_array, *e_array, *x_array;
    f64 scale;
    i32 info;

    if (!PyArg_ParseTuple(args, "sOOO", &trans, &a_obj, &e_obj, &x_obj)) {
        return NULL;
    }

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (a_array == NULL) return NULL;

    e_array = (PyArrayObject*)PyArray_FROM_OTF(e_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (e_array == NULL) {
        Py_DECREF(a_array);
        return NULL;
    }

    x_array = (PyArrayObject*)PyArray_FROM_OTF(x_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (x_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(e_array);
        return NULL;
    }

    i32 n = (i32)PyArray_DIM(a_array, 0);
    i32 lda = (n > 1) ? n : 1;
    i32 lde = lda;
    i32 ldx = lda;

    f64* a = (f64*)PyArray_DATA(a_array);
    f64* e = (f64*)PyArray_DATA(e_array);
    f64* x = (f64*)PyArray_DATA(x_array);

    sg03ay(trans, n, a, lda, e, lde, x, ldx, &scale, &info);

    PyArray_ResolveWritebackIfCopy(x_array);

    PyObject* result = Py_BuildValue("Odi", x_array, scale, info);

    Py_DECREF(a_array);
    Py_DECREF(e_array);
    Py_DECREF(x_array);

    return result;
}



/* Python wrapper for sg03ax */
PyObject* py_sg03ax(PyObject* self, PyObject* args) {
    char* trans;
    PyObject *a_obj, *e_obj, *x_obj;
    PyArrayObject *a_array, *e_array, *x_array;
    f64 scale;
    i32 info;

    if (!PyArg_ParseTuple(args, "sOOO", &trans, &a_obj, &e_obj, &x_obj)) {
        return NULL;
    }

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (a_array == NULL) return NULL;

    e_array = (PyArrayObject*)PyArray_FROM_OTF(e_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (e_array == NULL) {
        Py_DECREF(a_array);
        return NULL;
    }

    x_array = (PyArrayObject*)PyArray_FROM_OTF(x_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (x_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(e_array);
        return NULL;
    }

    i32 n = (i32)PyArray_DIM(a_array, 0);
    i32 lda = (n > 1) ? n : 1;
    i32 lde = lda;
    i32 ldx = lda;

    f64* a = (f64*)PyArray_DATA(a_array);
    f64* e = (f64*)PyArray_DATA(e_array);
    f64* x = (f64*)PyArray_DATA(x_array);

    sg03ax(trans, n, a, lda, e, lde, x, ldx, &scale, &info);

    PyArray_ResolveWritebackIfCopy(x_array);

    PyObject* result = Py_BuildValue("Odi", x_array, scale, info);

    Py_DECREF(a_array);
    Py_DECREF(e_array);
    Py_DECREF(x_array);

    return result;
}



/* Python wrapper for sg03bw */
PyObject* py_sg03bw(PyObject* self, PyObject* args) {
    char* trans;
    PyObject *a_obj, *c_obj, *e_obj, *d_obj, *x_obj;
    PyArrayObject *a_array, *c_array, *e_array, *d_array, *x_array;
    f64 scale;
    i32 info;

    if (!PyArg_ParseTuple(args, "sOOOOO", &trans, &a_obj, &e_obj, &c_obj, &d_obj, &x_obj)) {
        return NULL;
    }

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (a_array == NULL) return NULL;

    e_array = (PyArrayObject*)PyArray_FROM_OTF(e_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (e_array == NULL) {
        Py_DECREF(a_array);
        return NULL;
    }

    c_array = (PyArrayObject*)PyArray_FROM_OTF(c_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (c_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(e_array);
        return NULL;
    }

    d_array = (PyArrayObject*)PyArray_FROM_OTF(d_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (d_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(e_array);
        Py_DECREF(c_array);
        return NULL;
    }

    x_array = (PyArrayObject*)PyArray_FROM_OTF(x_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (x_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(e_array);
        Py_DECREF(c_array);
        Py_DECREF(d_array);
        return NULL;
    }

    npy_intp *a_dims = PyArray_DIMS(a_array);
    npy_intp *c_dims = PyArray_DIMS(c_array);
    npy_intp *x_dims = PyArray_DIMS(x_array);

    i32 m = (i32)a_dims[0];
    i32 n = (i32)c_dims[0];
    i32 lda = (i32)a_dims[0];
    i32 ldc = (i32)c_dims[0];
    i32 lde = (i32)PyArray_DIM(e_array, 0);
    i32 ldd = (i32)PyArray_DIM(d_array, 0);
    i32 ldx = (i32)x_dims[0];

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *e_data = (f64*)PyArray_DATA(e_array);
    f64 *c_data = (f64*)PyArray_DATA(c_array);
    f64 *d_data = (f64*)PyArray_DATA(d_array);
    f64 *x_data = (f64*)PyArray_DATA(x_array);

    sg03bw(trans, m, n, a_data, lda, c_data, ldc, e_data, lde, d_data, ldd, x_data, ldx, &scale, &info);

    PyObject *result = Py_BuildValue("Odi", x_array, scale, info);

    Py_DECREF(a_array);
    Py_DECREF(e_array);
    Py_DECREF(c_array);
    Py_DECREF(d_array);
    Py_DECREF(x_array);

    return result;
}



/* Python wrapper for sg03bu */
PyObject* py_sg03bu(PyObject* self, PyObject* args) {
    char *trans;
    PyObject *a_obj, *e_obj, *b_obj;
    PyArrayObject *a_array, *e_array, *b_array;
    f64 scale;
    i32 info;

    if (!PyArg_ParseTuple(args, "sOOO", &trans, &a_obj, &e_obj, &b_obj)) {
        return NULL;
    }

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (a_array == NULL) return NULL;

    e_array = (PyArrayObject*)PyArray_FROM_OTF(e_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (e_array == NULL) {
        Py_DECREF(a_array);
        return NULL;
    }

    b_array = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (b_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(e_array);
        return NULL;
    }

    npy_intp n_rows = PyArray_DIM(b_array, 0);
    npy_intp n_cols = PyArray_DIM(b_array, 1);
    i32 n = (i32)(n_rows < n_cols ? n_rows : n_cols);
    i32 lda = (i32)PyArray_DIM(a_array, 0);
    i32 lde = (i32)PyArray_DIM(e_array, 0);
    i32 ldb = (i32)PyArray_DIM(b_array, 0);

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *e_data = (f64*)PyArray_DATA(e_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);

    i32 ldwork = (n > 1) ? 6 * n - 6 : 1;
    f64 *dwork = (f64*)malloc(ldwork * sizeof(f64));

    if (dwork == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate workspace");
        Py_DECREF(a_array);
        Py_DECREF(e_array);
        Py_DECREF(b_array);
        return NULL;
    }

    sg03bu(trans, n, a_data, lda, e_data, lde, b_data, ldb, &scale, dwork, &info);

    free(dwork);

    PyObject *result = Py_BuildValue("Odi", b_array, scale, info);

    Py_DECREF(a_array);
    Py_DECREF(e_array);
    Py_DECREF(b_array);

    return result;
}



/* Python wrapper for sg03bv */
PyObject* py_sg03bv(PyObject* self, PyObject* args) {
    char *trans;
    PyObject *a_obj, *e_obj, *b_obj;
    PyArrayObject *a_array, *e_array, *b_array;
    f64 scale;
    i32 info;

    if (!PyArg_ParseTuple(args, "sOOO", &trans, &a_obj, &e_obj, &b_obj)) {
        return NULL;
    }

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (a_array == NULL) return NULL;

    e_array = (PyArrayObject*)PyArray_FROM_OTF(e_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (e_array == NULL) {
        Py_DECREF(a_array);
        return NULL;
    }

    b_array = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (b_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(e_array);
        return NULL;
    }

    npy_intp n_rows = PyArray_DIM(b_array, 0);
    npy_intp n_cols = PyArray_DIM(b_array, 1);
    i32 n = (i32)(n_rows < n_cols ? n_rows : n_cols);
    i32 lda = (i32)PyArray_DIM(a_array, 0);
    i32 lde = (i32)PyArray_DIM(e_array, 0);
    i32 ldb = (i32)PyArray_DIM(b_array, 0);

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *e_data = (f64*)PyArray_DATA(e_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);

    i32 ldwork = (n > 1) ? 6 * n - 6 : 1;
    f64 *dwork = (f64*)malloc(ldwork * sizeof(f64));

    if (dwork == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate workspace");
        Py_DECREF(a_array);
        Py_DECREF(e_array);
        Py_DECREF(b_array);
        return NULL;
    }

    sg03bv(trans, n, a_data, lda, e_data, lde, b_data, ldb, &scale, dwork, &info);

    free(dwork);

    PyObject *result = Py_BuildValue("Odi", b_array, scale, info);

    Py_DECREF(a_array);
    Py_DECREF(e_array);
    Py_DECREF(b_array);

    return result;
}



/* Python wrapper for sg03bx */
PyObject* py_sg03bx(PyObject* self, PyObject* args) {
    char *dico, *trans;
    PyObject *a_obj, *e_obj, *b_obj;
    PyArrayObject *a_array, *e_array, *b_array;
    f64 scale;
    i32 info;

    if (!PyArg_ParseTuple(args, "ssOOO", &dico, &trans, &a_obj, &e_obj, &b_obj)) {
        return NULL;
    }

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (a_array == NULL) return NULL;

    e_array = (PyArrayObject*)PyArray_FROM_OTF(e_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (e_array == NULL) {
        Py_DECREF(a_array);
        return NULL;
    }

    b_array = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (b_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(e_array);
        return NULL;
    }

    i32 lda = (i32)PyArray_DIM(a_array, 0);
    i32 lde = (i32)PyArray_DIM(e_array, 0);
    i32 ldb = (i32)PyArray_DIM(b_array, 0);

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *e_data = (f64*)PyArray_DATA(e_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);

    npy_intp dims[2] = {2, 2};
    npy_intp strides[2] = {sizeof(f64), 2 * sizeof(f64)};

    PyObject *u_array = PyArray_New(&PyArray_Type, 2, dims, NPY_DOUBLE, strides,
                                     NULL, 0, NPY_ARRAY_FARRAY, NULL);
    PyObject *m1_array = PyArray_New(&PyArray_Type, 2, dims, NPY_DOUBLE, strides,
                                      NULL, 0, NPY_ARRAY_FARRAY, NULL);
    PyObject *m2_array = PyArray_New(&PyArray_Type, 2, dims, NPY_DOUBLE, strides,
                                      NULL, 0, NPY_ARRAY_FARRAY, NULL);

    if (u_array == NULL || m1_array == NULL || m2_array == NULL) {
        Py_XDECREF(u_array);
        Py_XDECREF(m1_array);
        Py_XDECREF(m2_array);
        Py_DECREF(a_array);
        Py_DECREF(e_array);
        Py_DECREF(b_array);
        return NULL;
    }

    f64 *u_data = (f64*)PyArray_DATA((PyArrayObject*)u_array);
    f64 *m1_data = (f64*)PyArray_DATA((PyArrayObject*)m1_array);
    f64 *m2_data = (f64*)PyArray_DATA((PyArrayObject*)m2_array);

    sg03bx(dico, trans, a_data, lda, e_data, lde, b_data, ldb,
           u_data, 2, &scale, m1_data, 2, m2_data, 2, &info);

    PyObject *result = Py_BuildValue("OdOOi", u_array, scale, m1_array, m2_array, info);

    Py_DECREF(a_array);
    Py_DECREF(e_array);
    Py_DECREF(b_array);
    Py_DECREF(u_array);
    Py_DECREF(m1_array);
    Py_DECREF(m2_array);

    return result;
}



/* Python wrapper for sg03bd */
PyObject* py_sg03bd(PyObject* self, PyObject* args) {
    char *dico, *fact, *trans;
    i32 n, m;
    PyObject *a_obj, *e_obj, *b_obj;
    PyArrayObject *a_array, *e_array, *b_array;
    f64 scale;
    i32 info;

    if (!PyArg_ParseTuple(args, "sssiiOOO", &dico, &fact, &trans, &n, &m, &a_obj, &e_obj, &b_obj)) {
        return NULL;
    }

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (a_array == NULL) return NULL;

    e_array = (PyArrayObject*)PyArray_FROM_OTF(e_obj, NPY_DOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (e_array == NULL) {
        Py_DECREF(a_array);
        return NULL;
    }

    /* B array is modified in place and may need more space than input provides.
     * Allocate workspace of correct size and copy input B into it. */
    b_array = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (b_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(e_array);
        return NULL;
    }
    
    /* Validate B has 2 dimensions */
    if (PyArray_NDIM(b_array) != 2) {
        PyErr_SetString(PyExc_ValueError, "B must be a 2D array");
        Py_DECREF(a_array);
        Py_DECREF(e_array);
        Py_DECREF(b_array);
        return NULL;
    }

    i32 lda = (i32)PyArray_DIM(a_array, 0);
    i32 lde = (i32)PyArray_DIM(e_array, 0);
    
    /* Determine required LDB and allocate workspace for B */
    bool istran = (trans[0] == 'T' || trans[0] == 't');
    i32 ldb_required = istran ? (n > 1 ? n : 1) : ((m > n ? m : n) > 1 ? (m > n ? m : n) : 1);
    i32 b_cols = n;
    
    /* Get input B dimensions */
    i32 b_in_rows = (i32)PyArray_DIM(b_array, 0);
    i32 b_in_cols = (i32)PyArray_DIM(b_array, 1);
    i32 b_in_ld = (i32)PyArray_DIM(b_array, 0);  /* Leading dimension of input */
    
    /* Allocate B workspace */
    npy_intp dims_b[2] = {ldb_required, b_cols};
    npy_intp strides_b[2] = {sizeof(f64), ldb_required * sizeof(f64)};
    PyObject *b_work = PyArray_New(&PyArray_Type, 2, dims_b, NPY_DOUBLE, strides_b,
                                    NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (b_work == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(e_array);
        Py_DECREF(b_array);
        return NULL;
    }
    
    /* Copy input B into workspace */
    f64 *b_work_data = (f64*)PyArray_DATA((PyArrayObject*)b_work);
    f64 *b_data = (f64*)PyArray_DATA(b_array);
    
    /* Zero out workspace first */
    for (i32 j = 0; j < b_cols; j++) {
        for (i32 i = 0; i < ldb_required; i++) {
            b_work_data[i + j*ldb_required] = 0.0;
        }
    }
    
    /* Copy input B (column by column) */
    for (i32 j = 0; j < b_in_cols && j < b_cols; j++) {
        for (i32 i = 0; i < b_in_rows && i < ldb_required; i++) {
            b_work_data[i + j*ldb_required] = b_data[i + j*b_in_ld];
        }
    }
    
    Py_DECREF(b_array);  /* Done with input B */
    i32 ldb = ldb_required;

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *e_data = (f64*)PyArray_DATA(e_array);

    /* Allocate workspace for Q and Z matrices (not returned) */
    i32 ldq = (n > 1) ? n : 1;
    i32 ldz = (n > 1) ? n : 1;
    f64 *q_data = (f64*)malloc(ldq * n * sizeof(f64));
    f64 *z_data = (f64*)malloc(ldz * n * sizeof(f64));

    if (q_data == NULL || z_data == NULL) {
        free(q_data);
        free(z_data);
        Py_DECREF(a_array);
        Py_DECREF(e_array);
        Py_DECREF(b_work);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate Q/Z workspace");
        return NULL;
    }

    npy_intp dims_eig[1] = {n};
    PyObject *alphar_array = PyArray_New(&PyArray_Type, 1, dims_eig, NPY_DOUBLE, NULL,
                                          NULL, 0, NPY_ARRAY_FARRAY, NULL);
    PyObject *alphai_array = PyArray_New(&PyArray_Type, 1, dims_eig, NPY_DOUBLE, NULL,
                                          NULL, 0, NPY_ARRAY_FARRAY, NULL);
    PyObject *beta_array = PyArray_New(&PyArray_Type, 1, dims_eig, NPY_DOUBLE, NULL,
                                        NULL, 0, NPY_ARRAY_FARRAY, NULL);

    if (alphar_array == NULL || alphai_array == NULL || beta_array == NULL) {
        Py_XDECREF(alphar_array);
        Py_XDECREF(alphai_array);
        Py_XDECREF(beta_array);
        free(q_data);
        free(z_data);
        Py_DECREF(a_array);
        Py_DECREF(e_array);
        Py_DECREF(b_work);
        return NULL;
    }

    f64 *alphar_data = (f64*)PyArray_DATA((PyArrayObject*)alphar_array);
    f64 *alphai_data = (f64*)PyArray_DATA((PyArrayObject*)alphai_array);
    f64 *beta_data = (f64*)PyArray_DATA((PyArrayObject*)beta_array);

    i32 ldwork;
    if (fact[0] == 'F' || fact[0] == 'f') {
        ldwork = (2 * n > 6 * n - 6) ? 2 * n : (6 * n - 6);
        ldwork = (ldwork > 1) ? ldwork : 1;
    } else {
        ldwork = (4 * n > 6 * n - 6) ? 4 * n : (6 * n - 6);
        ldwork = (ldwork > 1) ? ldwork : 1;
        i32 mingg = (ldwork > 8 * n + 16) ? ldwork : (8 * n + 16);
        ldwork = mingg;
    }

    f64 *dwork = (f64*)malloc(ldwork * sizeof(f64));
    if (dwork == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate workspace");
        Py_DECREF(alphar_array);
        Py_DECREF(alphai_array);
        Py_DECREF(beta_array);
        free(q_data);
        free(z_data);
        Py_DECREF(a_array);
        Py_DECREF(e_array);
        Py_DECREF(b_work);
        return NULL;
    }

    sg03bd(dico, fact, trans, n, m, a_data, lda, e_data, lde,
           q_data, ldq, z_data, ldz, b_work_data, ldb, &scale,
           alphar_data, alphai_data, beta_data, dwork, ldwork, &info);

    free(dwork);
    free(q_data);
    free(z_data);

    /* Resolve writebackifcopy before decref */
    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(e_array);

    /* sg03bd modifies B in place to produce U (n x n upper triangular).
     * Create a view of the n x n submatrix from b_work.
     * See CLAUDE.md: "CRITICAL: In-place modification - return input array directly"
     */
    npy_intp dims_u[2] = {n, n};
    npy_intp strides_u[2] = {sizeof(f64), ldb * sizeof(f64)};
    PyObject *u_array = PyArray_New(&PyArray_Type, 2, dims_u, NPY_DOUBLE, strides_u,
                                     b_work_data, 0, NPY_ARRAY_FARRAY, NULL);
    if (u_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(e_array);
        Py_DECREF(b_work);
        Py_DECREF(alphar_array);
        Py_DECREF(alphai_array);
        Py_DECREF(beta_array);
        return NULL;
    }
    
    /* Set b_work as the base object so the memory stays alive */
    if (PyArray_SetBaseObject((PyArrayObject*)u_array, (PyObject*)b_work) < 0) {
        Py_DECREF(u_array);
        Py_DECREF(a_array);
        Py_DECREF(e_array);
        Py_DECREF(b_work);
        Py_DECREF(alphar_array);
        Py_DECREF(alphai_array);
        Py_DECREF(beta_array);
        return NULL;
    }

    PyObject *result = Py_BuildValue("OdOOOi", u_array, scale,
                                     alphar_array, alphai_array, beta_array, info);

    /* Py_BuildValue with "O" increments refcounts, so we need to DECREF all arrays
     * that were passed to it to avoid leaks */
    Py_DECREF(a_array);
    Py_DECREF(e_array);
    /* Don't DECREF b_work here - it's now owned by u_array as base object */
    Py_DECREF(u_array);
    Py_DECREF(alphar_array);
    Py_DECREF(alphai_array);
    Py_DECREF(beta_array);

    return result;
}



/* Python wrapper for sg02ad */
PyObject* py_sg02ad(PyObject* self, PyObject* args) {
    const char *dico_str, *jobb_str, *fact_str, *uplo_str, *jobl_str;
    const char *scal_str, *sort_str, *acc_str;
    i32 n, m, p;
    f64 tol;
    PyObject *a_obj, *b_obj, *q_obj, *r_obj, *l_obj, *e_obj;

    if (!PyArg_ParseTuple(args, "ssssssssiiiOOOOOOd",
                          &dico_str, &jobb_str, &fact_str, &uplo_str, &jobl_str,
                          &scal_str, &sort_str, &acc_str, &n, &m, &p,
                          &a_obj, &e_obj, &b_obj, &q_obj, &r_obj, &l_obj, &tol)) {
        return NULL;
    }

    char jobb = toupper((unsigned char)jobb_str[0]);
    bool ljobb = (jobb == 'B');

    if (n < 0) {
        PyErr_SetString(PyExc_ValueError, "n must be non-negative");
        return NULL;
    }
    if (ljobb && m < 0) {
        PyErr_SetString(PyExc_ValueError, "m must be non-negative");
        return NULL;
    }

    PyArrayObject *a_array = NULL, *b_array = NULL, *q_array = NULL;
    PyArrayObject *r_array = NULL, *l_array = NULL, *e_array = NULL;

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY_RO);
    if (!a_array) return NULL;
    i32 lda = (i32)PyArray_DIM(a_array, 0);
    if (lda < 1) lda = 1;
    f64 *a_data = (f64*)PyArray_DATA(a_array);

    e_array = (PyArrayObject*)PyArray_FROM_OTF(e_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY_RO);
    if (!e_array) goto cleanup_sg02;
    i32 lde = (i32)PyArray_DIM(e_array, 0);
    if (lde < 1) lde = 1;
    f64 *e_data = (f64*)PyArray_DATA(e_array);

    b_array = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY_RO);
    if (!b_array) goto cleanup_sg02;
    i32 ldb = (i32)PyArray_DIM(b_array, 0);
    if (ldb < 1) ldb = 1;
    f64 *b_data = (f64*)PyArray_DATA(b_array);

    q_array = (PyArrayObject*)PyArray_FROM_OTF(q_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!q_array) goto cleanup_sg02;
    i32 ldq = (i32)PyArray_DIM(q_array, 0);
    if (ldq < 1) ldq = 1;
    f64 *q_data = (f64*)PyArray_DATA(q_array);

    f64 *r_data = NULL;
    i32 ldr = 1;
    if (ljobb && r_obj != Py_None) {
        r_array = (PyArrayObject*)PyArray_FROM_OTF(r_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
        if (!r_array) goto cleanup_sg02;
        ldr = (i32)PyArray_DIM(r_array, 0);
        if (ldr < 1) ldr = 1;
        r_data = (f64*)PyArray_DATA(r_array);
    }

    f64 *l_data = NULL;
    i32 ldl = 1;
    if (ljobb && l_obj != Py_None) {
        l_array = (PyArrayObject*)PyArray_FROM_OTF(l_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
        if (!l_array) goto cleanup_sg02;
        ldl = (i32)PyArray_DIM(l_array, 0);
        if (ldl < 1) ldl = 1;
        l_data = (f64*)PyArray_DATA(l_array);
    }

    i32 nn = 2 * n;
    i32 nnm = ljobb ? (nn + m) : nn;
    i32 ldx = n > 1 ? n : 1;
    i32 lds = nnm > 1 ? nnm : 1;
    i32 ldt = nnm > 1 ? nnm : 1;
    i32 ldu = nn > 1 ? nn : 1;

    npy_intp x_dims[2] = {n, n};
    npy_intp x_strides[2] = {sizeof(f64), ldx * sizeof(f64)};
    PyArrayObject *x_array = (PyArrayObject*)PyArray_New(&PyArray_Type, 2, x_dims, NPY_DOUBLE,
                                                         x_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);

    npy_intp s_dims[2] = {lds, nnm};
    npy_intp s_strides[2] = {sizeof(f64), lds * sizeof(f64)};
    PyArrayObject *s_array = (PyArrayObject*)PyArray_New(&PyArray_Type, 2, s_dims, NPY_DOUBLE,
                                                         s_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);

    npy_intp t_dims[2] = {ldt, nnm};
    npy_intp t_strides[2] = {sizeof(f64), ldt * sizeof(f64)};
    PyArrayObject *t_array = (PyArrayObject*)PyArray_New(&PyArray_Type, 2, t_dims, NPY_DOUBLE,
                                                         t_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    
    npy_intp u_dims[2] = {ldu, nn};
    npy_intp u_strides[2] = {sizeof(f64), ldu * sizeof(f64)};
    PyArrayObject *u_array = (PyArrayObject*)PyArray_New(&PyArray_Type, 2, u_dims, NPY_DOUBLE,
                                                         u_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);

    npy_intp eig_dims[1] = {nn};
    PyArrayObject *alfar_array = (PyArrayObject*)PyArray_SimpleNew(1, eig_dims, NPY_DOUBLE);
    PyArrayObject *alfai_array = (PyArrayObject*)PyArray_SimpleNew(1, eig_dims, NPY_DOUBLE);
    PyArrayObject *beta_array = (PyArrayObject*)PyArray_SimpleNew(1, eig_dims, NPY_DOUBLE);

    if (!x_array || !s_array || !t_array || !u_array || !alfar_array || !alfai_array || !beta_array) {
        Py_XDECREF(x_array); Py_XDECREF(s_array); Py_XDECREF(t_array); Py_XDECREF(u_array);
        Py_XDECREF(alfar_array); Py_XDECREF(alfai_array); Py_XDECREF(beta_array);
        PyErr_NoMemory();
        goto cleanup_sg02;
    }

    f64 *x_data = (f64*)PyArray_DATA(x_array);
    f64 *s_data = (f64*)PyArray_DATA(s_array);
    f64 *t_data = (f64*)PyArray_DATA(t_array);
    f64 *u_data = (f64*)PyArray_DATA(u_array);
    f64 *alfar = (f64*)PyArray_DATA(alfar_array);
    f64 *alfai = (f64*)PyArray_DATA(alfai_array);
    f64 *beta_arr = (f64*)PyArray_DATA(beta_array);

    memset(x_data, 0, ldx * n * sizeof(f64));
    memset(s_data, 0, lds * nnm * sizeof(f64));
    memset(t_data, 0, ldt * nn * sizeof(f64));
    memset(u_data, 0, ldu * nn * sizeof(f64));
    memset(alfar, 0, nn * sizeof(f64));
    memset(alfai, 0, nn * sizeof(f64));
    memset(beta_arr, 0, nn * sizeof(f64));

    i32 ldwork;
    i32 req1 = 14*n + 23;
    i32 req2 = 16*n;
    ldwork = req1 > req2 ? req1 : req2;
    if (ljobb) {
        i32 ldw = (nnm > 3*m) ? nnm : 3*m;
        ldwork = ldwork > ldw ? ldwork : ldw;
    }
    ldwork = ldwork > 1 ? ldwork : 1;

    i32 liwork = ljobb ? (m > nn ? m : nn) : nn;
    liwork = liwork > 1 ? liwork : 1;

    i32 *iwork = (i32*)calloc(liwork, sizeof(i32));
    f64 *dwork = (f64*)calloc(ldwork, sizeof(f64));

    if (!iwork || !dwork) {
        Py_DECREF(x_array); Py_DECREF(s_array); Py_DECREF(t_array); Py_DECREF(u_array);
        Py_DECREF(alfar_array); Py_DECREF(alfai_array); Py_DECREF(beta_array);
        free(iwork); free(dwork);
        Py_XDECREF(a_array); Py_XDECREF(e_array); Py_XDECREF(b_array);
        Py_XDECREF(q_array); Py_XDECREF(r_array); Py_XDECREF(l_array);
        PyErr_NoMemory();
        goto cleanup_sg02;
    }

    f64 rcondu;
    i32 iwarn, info;

    sg02ad(dico_str, jobb_str, fact_str, uplo_str, jobl_str, scal_str, sort_str, acc_str,
           n, m, p, a_data, lda, e_data, lde, b_data, ldb, q_data, ldq,
           r_data, ldr, l_data, ldl, &rcondu, x_data, ldx, alfar, alfai, beta_arr,
           s_data, lds, t_data, ldt, u_data, ldu, tol, iwork, dwork, ldwork, &iwarn, &info);

    free(iwork);
    free(dwork);

    if (r_array) PyArray_ResolveWritebackIfCopy(r_array);
    if (l_array) PyArray_ResolveWritebackIfCopy(l_array);
    PyArray_ResolveWritebackIfCopy(q_array);

    if (info < 0) {
        Py_DECREF(x_array); Py_DECREF(s_array); Py_DECREF(t_array); Py_DECREF(u_array);
        Py_DECREF(alfar_array); Py_DECREF(alfai_array); Py_DECREF(beta_array);
        Py_XDECREF(a_array); Py_XDECREF(e_array); Py_XDECREF(b_array);
        Py_XDECREF(q_array); Py_XDECREF(r_array); Py_XDECREF(l_array);
        PyErr_Format(PyExc_ValueError, "Parameter %d had an illegal value", -info);
        return NULL;
    }
    
    // Arrays already created and populated by sg02ad

    PyObject *result = Py_BuildValue("OdOOOOOOii", x_array, rcondu, alfar_array, alfai_array,
                                     beta_array, s_array, t_array, u_array, iwarn, info);

    Py_DECREF(x_array); Py_DECREF(alfar_array); Py_DECREF(alfai_array);
    Py_DECREF(beta_array); Py_DECREF(s_array); Py_DECREF(t_array); Py_DECREF(u_array);

    Py_XDECREF(a_array); Py_XDECREF(e_array); Py_XDECREF(b_array);
    Py_XDECREF(q_array); Py_XDECREF(r_array); Py_XDECREF(l_array);

    return result;

cleanup_sg02:
    Py_XDECREF(a_array); Py_XDECREF(e_array); Py_XDECREF(b_array);
    Py_XDECREF(q_array); Py_XDECREF(r_array); Py_XDECREF(l_array);
    return NULL;
}



/* Python wrapper for sg03ad */
PyObject* py_sg03ad(PyObject* self, PyObject* args) {
    const char *dico_str, *job_str, *fact_str, *trans_str, *uplo_str;
    i32 n;
    PyObject *a_obj, *e_obj, *x_obj, *q_obj = NULL, *z_obj = NULL;

    if (!PyArg_ParseTuple(args, "sssssiOOO|OO",
                          &dico_str, &job_str, &fact_str, &trans_str, &uplo_str,
                          &n, &a_obj, &e_obj, &x_obj, &q_obj, &z_obj)) {
        return NULL;
    }

    char fact = toupper((unsigned char)fact_str[0]);
    bool isfact = (fact == 'F');

    if (n < 0) {
        PyErr_SetString(PyExc_ValueError, "n must be non-negative");
        return NULL;
    }

    PyArrayObject *a_array = NULL, *e_array = NULL, *x_array = NULL;
    PyArrayObject *q_array = NULL, *z_array = NULL;

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!a_array) return NULL;
    i32 lda = (i32)PyArray_DIM(a_array, 0);
    if (lda < 1) lda = 1;
    f64 *a_data = (f64*)PyArray_DATA(a_array);

    e_array = (PyArrayObject*)PyArray_FROM_OTF(e_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!e_array) goto cleanup_sg03;
    i32 lde = (i32)PyArray_DIM(e_array, 0);
    if (lde < 1) lde = 1;
    f64 *e_data = (f64*)PyArray_DATA(e_array);

    x_array = (PyArrayObject*)PyArray_FROM_OTF(x_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!x_array) goto cleanup_sg03;
    i32 ldx = (i32)PyArray_DIM(x_array, 0);
    if (ldx < 1) ldx = 1;
    f64 *x_data = (f64*)PyArray_DATA(x_array);

    i32 ldq = n > 1 ? n : 1;
    i32 ldz = n > 1 ? n : 1;
    f64 *q_data = NULL, *z_data = NULL;
    bool q_allocated = false, z_allocated = false;

    if (isfact && q_obj != NULL && q_obj != Py_None) {
        q_array = (PyArrayObject*)PyArray_FROM_OTF(q_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
        if (!q_array) goto cleanup_sg03;
        ldq = (i32)PyArray_DIM(q_array, 0);
        if (ldq < 1) ldq = 1;
        q_data = (f64*)PyArray_DATA(q_array);
    } else {
        q_data = (f64*)calloc(ldq * n, sizeof(f64));
        if (!q_data) {
            PyErr_NoMemory();
            goto cleanup_sg03;
        }
        q_allocated = true;
    }

    if (isfact && z_obj != NULL && z_obj != Py_None) {
        z_array = (PyArrayObject*)PyArray_FROM_OTF(z_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
        if (!z_array) goto cleanup_sg03;
        ldz = (i32)PyArray_DIM(z_array, 0);
        if (ldz < 1) ldz = 1;
        z_data = (f64*)PyArray_DATA(z_array);
    } else {
        z_data = (f64*)calloc(ldz * n, sizeof(f64));
        if (!z_data) {
            if (q_allocated) free(q_data);
            PyErr_NoMemory();
            goto cleanup_sg03;
        }
        z_allocated = true;
    }

    f64 *alphar = (f64*)calloc(n > 0 ? n : 1, sizeof(f64));
    f64 *alphai = (f64*)calloc(n > 0 ? n : 1, sizeof(f64));
    f64 *beta_arr = (f64*)calloc(n > 0 ? n : 1, sizeof(f64));

    if (!alphar || !alphai || !beta_arr) {
        free(alphar); free(alphai); free(beta_arr);
        if (q_allocated) free(q_data);
        if (z_allocated) free(z_data);
        PyErr_NoMemory();
        goto cleanup_sg03;
    }

    i32 liwork = n * n > 1 ? n * n : 1;
    i32 ldwork;
    char job = toupper((unsigned char)job_str[0]);
    bool wantx = (job == 'X');
    if (wantx) {
        ldwork = isfact ? (n > 1 ? n : 1) : (4*n > 1 ? 4*n : 1);
    } else {
        ldwork = isfact ? (2*n*n > 1 ? 2*n*n : 1) : (2*n*n > 4*n ? 2*n*n : 4*n);
    }
    ldwork = ldwork > 8*n + 16 ? ldwork : 8*n + 16;

    i32 *iwork = (i32*)calloc(liwork, sizeof(i32));
    f64 *dwork = (f64*)calloc(ldwork, sizeof(f64));

    if (!iwork || !dwork) {
        free(alphar); free(alphai); free(beta_arr);
        if (q_allocated) free(q_data);
        if (z_allocated) free(z_data);
        free(iwork); free(dwork);
        PyErr_NoMemory();
        goto cleanup_sg03;
    }

    f64 scale, sep = 0.0, ferr = 0.0;
    i32 info;

    sg03ad(dico_str, job_str, fact_str, trans_str, uplo_str, n,
           a_data, lda, e_data, lde, q_data, ldq, z_data, ldz,
           x_data, ldx, &scale, &sep, &ferr, alphar, alphai, beta_arr,
           iwork, dwork, ldwork, &info);

    free(iwork);
    free(dwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(e_array);
    PyArray_ResolveWritebackIfCopy(x_array);
    if (q_array) PyArray_ResolveWritebackIfCopy(q_array);
    if (z_array) PyArray_ResolveWritebackIfCopy(z_array);

    if (info < 0) {
        free(alphar); free(alphai); free(beta_arr);
        if (q_allocated) free(q_data);
        if (z_allocated) free(z_data);
        Py_XDECREF(a_array); Py_XDECREF(e_array); Py_XDECREF(x_array);
        Py_XDECREF(q_array); Py_XDECREF(z_array);
        PyErr_Format(PyExc_ValueError, "Parameter %d had an illegal value", -info);
        return NULL;
    }

    npy_intp mat_dims[2] = {n, n};
    npy_intp eig_dims[1] = {n};

    PyArrayObject *x_out = (PyArrayObject*)PyArray_SimpleNew(2, mat_dims, NPY_DOUBLE);
    PyArrayObject *a_out = (PyArrayObject*)PyArray_SimpleNew(2, mat_dims, NPY_DOUBLE);
    PyArrayObject *e_out = (PyArrayObject*)PyArray_SimpleNew(2, mat_dims, NPY_DOUBLE);
    PyArrayObject *q_out = (PyArrayObject*)PyArray_SimpleNew(2, mat_dims, NPY_DOUBLE);
    PyArrayObject *z_out = (PyArrayObject*)PyArray_SimpleNew(2, mat_dims, NPY_DOUBLE);
    PyArrayObject *ar_out = (PyArrayObject*)PyArray_SimpleNew(1, eig_dims, NPY_DOUBLE);
    PyArrayObject *ai_out = (PyArrayObject*)PyArray_SimpleNew(1, eig_dims, NPY_DOUBLE);
    PyArrayObject *b_out = (PyArrayObject*)PyArray_SimpleNew(1, eig_dims, NPY_DOUBLE);

    if (!x_out || !a_out || !e_out || !q_out || !z_out || !ar_out || !ai_out || !b_out) {
        Py_XDECREF(x_out); Py_XDECREF(a_out); Py_XDECREF(e_out);
        Py_XDECREF(q_out); Py_XDECREF(z_out);
        Py_XDECREF(ar_out); Py_XDECREF(ai_out); Py_XDECREF(b_out);
        free(alphar); free(alphai); free(beta_arr);
        if (q_allocated) free(q_data);
        if (z_allocated) free(z_data);
        goto cleanup_sg03;
    }

    for (i32 j = 0; j < n; j++) {
        for (i32 i = 0; i < n; i++) {
            *((f64*)PyArray_GETPTR2(x_out, i, j)) = x_data[i + j*ldx];
            *((f64*)PyArray_GETPTR2(a_out, i, j)) = a_data[i + j*lda];
            *((f64*)PyArray_GETPTR2(e_out, i, j)) = e_data[i + j*lde];
            *((f64*)PyArray_GETPTR2(q_out, i, j)) = q_data[i + j*ldq];
            *((f64*)PyArray_GETPTR2(z_out, i, j)) = z_data[i + j*ldz];
        }
    }
    for (i32 i = 0; i < n; i++) {
        *((f64*)PyArray_GETPTR1(ar_out, i)) = alphar[i];
        *((f64*)PyArray_GETPTR1(ai_out, i)) = alphai[i];
        *((f64*)PyArray_GETPTR1(b_out, i)) = beta_arr[i];
    }

    free(alphar); free(alphai); free(beta_arr);
    if (q_allocated) free(q_data);
    if (z_allocated) free(z_data);

    PyObject *result = Py_BuildValue("OdddOOOOOOOi", x_out, scale, sep, ferr,
                                     ar_out, ai_out, b_out, a_out, e_out, q_out, z_out, info);

    Py_DECREF(x_out); Py_DECREF(a_out); Py_DECREF(e_out);
    Py_DECREF(q_out); Py_DECREF(z_out);
    Py_DECREF(ar_out); Py_DECREF(ai_out); Py_DECREF(b_out);

    Py_XDECREF(a_array); Py_XDECREF(e_array); Py_XDECREF(x_array);
    Py_XDECREF(q_array); Py_XDECREF(z_array);

    return result;

cleanup_sg03:
    Py_XDECREF(a_array); Py_XDECREF(e_array); Py_XDECREF(x_array);
    Py_XDECREF(q_array); Py_XDECREF(z_array);
    return NULL;
}

PyObject* py_sg02cv(PyObject* self, PyObject* args, PyObject* kwargs) {
    static char* kwlist[] = {"dico", "job", "jobe", "uplo", "trans",
                             "a", "e", "x", "r", NULL};

    const char* dico_str;
    const char* job_str;
    const char* jobe_str;
    const char* uplo_str;
    const char* trans_str;
    PyObject* a_obj;
    PyObject* e_obj = Py_None;
    PyObject* x_obj;
    PyObject* r_obj;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "sssssOOOO", kwlist,
            &dico_str, &job_str, &jobe_str, &uplo_str, &trans_str,
            &a_obj, &e_obj, &x_obj, &r_obj)) {
        return NULL;
    }

    PyArrayObject* a_array = (PyArrayObject*)PyArray_FROM_OTF(
        a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject* x_array = (PyArrayObject*)PyArray_FROM_OTF(
        x_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject* r_array = (PyArrayObject*)PyArray_FROM_OTF(
        r_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);

    if (!a_array || !x_array || !r_array) {
        Py_XDECREF(a_array);
        Py_XDECREF(x_array);
        Py_XDECREF(r_array);
        PyErr_SetString(PyExc_ValueError, "Failed to convert a, x, or r to array");
        return NULL;
    }

    i32 n = (i32)PyArray_DIM(a_array, 0);
    if (n == 0) {
        f64 norm0 = 0.0;
        npy_intp dims0[2] = {0, 0};
        PyArrayObject* r_out = (PyArrayObject*)PyArray_SimpleNew(2, dims0, NPY_DOUBLE);
        npy_intp norm_dims[1] = {2};
        PyArrayObject* norms_out = (PyArrayObject*)PyArray_SimpleNew(1, norm_dims, NPY_DOUBLE);
        *((f64*)PyArray_GETPTR1(norms_out, 0)) = norm0;
        *((f64*)PyArray_GETPTR1(norms_out, 1)) = norm0;

        PyObject* result = Py_BuildValue("OOi", r_out, norms_out, 0);
        Py_DECREF(r_out);
        Py_DECREF(norms_out);
        Py_DECREF(a_array);
        Py_DECREF(x_array);
        Py_DECREF(r_array);
        return result;
    }

    PyArrayObject* e_array = NULL;
    f64* e_data = NULL;
    i32 lde = 1;
    bool ljobe = (jobe_str[0] == 'G' || jobe_str[0] == 'g');

    if (ljobe) {
        if (e_obj == Py_None) {
            Py_DECREF(a_array);
            Py_DECREF(x_array);
            Py_DECREF(r_array);
            PyErr_SetString(PyExc_ValueError, "E matrix required when jobe='G'");
            return NULL;
        }
        e_array = (PyArrayObject*)PyArray_FROM_OTF(
            e_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
        if (!e_array) {
            Py_DECREF(a_array);
            Py_DECREF(x_array);
            Py_DECREF(r_array);
            PyErr_SetString(PyExc_ValueError, "Failed to convert e to array");
            return NULL;
        }
        e_data = (f64*)PyArray_DATA(e_array);
        lde = n;
    }

    f64* a_data = (f64*)PyArray_DATA(a_array);
    f64* x_data = (f64*)PyArray_DATA(x_array);
    f64* r_data = (f64*)PyArray_DATA(r_array);

    i32 lda = n;
    i32 ldx = n;
    i32 ldr = n;

    bool discr = (dico_str[0] == 'D' || dico_str[0] == 'd');
    bool ljobn = (job_str[0] == 'N' || job_str[0] == 'n' ||
                  job_str[0] == 'B' || job_str[0] == 'b');

    i32 nn = n * n;
    i32 ldwork;
    if (ljobn) {
        ldwork = discr ? 2 * nn : nn;
    } else if (!discr && !ljobe) {
        ldwork = 1;
    } else {
        ldwork = nn;
    }
    ldwork = ldwork > 1 ? ldwork : 1;

    f64* dwork = (f64*)calloc(ldwork, sizeof(f64));
    f64 norms[2] = {0.0, 0.0};

    if (!dwork) {
        Py_DECREF(a_array);
        Py_DECREF(x_array);
        Py_DECREF(r_array);
        Py_XDECREF(e_array);
        PyErr_NoMemory();
        return NULL;
    }

    i32 info;

    sg02cv(dico_str, job_str, jobe_str, uplo_str, trans_str, n,
           a_data, lda, e_data, lde, x_data, ldx, r_data, ldr,
           norms, dwork, ldwork, &info);

    free(dwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(x_array);
    PyArray_ResolveWritebackIfCopy(r_array);
    if (e_array) PyArray_ResolveWritebackIfCopy(e_array);

    npy_intp mat_dims[2] = {n, n};
    PyArrayObject* r_out = (PyArrayObject*)PyArray_SimpleNew(2, mat_dims, NPY_DOUBLE);
    if (!r_out) {
        Py_DECREF(a_array);
        Py_DECREF(x_array);
        Py_DECREF(r_array);
        Py_XDECREF(e_array);
        PyErr_NoMemory();
        return NULL;
    }

    for (i32 j = 0; j < n; j++) {
        for (i32 i = 0; i < n; i++) {
            *((f64*)PyArray_GETPTR2(r_out, i, j)) = r_data[i + j * ldr];
        }
    }

    npy_intp norm_dims[1] = {2};
    PyArrayObject* norms_out = (PyArrayObject*)PyArray_SimpleNew(1, norm_dims, NPY_DOUBLE);
    if (!norms_out) {
        Py_DECREF(r_out);
        Py_DECREF(a_array);
        Py_DECREF(x_array);
        Py_DECREF(r_array);
        Py_XDECREF(e_array);
        PyErr_NoMemory();
        return NULL;
    }
    *((f64*)PyArray_GETPTR1(norms_out, 0)) = norms[0];
    *((f64*)PyArray_GETPTR1(norms_out, 1)) = norms[1];

    PyObject* result = Py_BuildValue("OOi", r_out, norms_out, info);

    Py_DECREF(r_out);
    Py_DECREF(norms_out);
    Py_DECREF(a_array);
    Py_DECREF(x_array);
    Py_DECREF(r_array);
    Py_XDECREF(e_array);

    return result;
}

PyObject* py_sg02cw(PyObject* self, PyObject* args, PyObject* kwargs) {
    static char* kwlist[] = {"dico", "job", "jobe", "flag", "jobg", "uplo", "trans",
                             "n", "m", "a", "e", "g", "x", "f", "k", "xe", "q", NULL};

    const char* dico_str;
    const char* job_str;
    const char* jobe_str;
    const char* flag_str;
    const char* jobg_str;
    const char* uplo_str;
    const char* trans_str;
    i32 n, m;
    PyObject* a_obj;
    PyObject* e_obj = Py_None;
    PyObject* g_obj;
    PyObject* x_obj;
    PyObject* f_obj = Py_None;
    PyObject* k_obj = Py_None;
    PyObject* xe_obj = Py_None;
    PyObject* q_obj;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "sssssssiiOOOOOOOO", kwlist,
            &dico_str, &job_str, &jobe_str, &flag_str, &jobg_str, &uplo_str, &trans_str,
            &n, &m, &a_obj, &e_obj, &g_obj, &x_obj, &f_obj, &k_obj, &xe_obj, &q_obj)) {
        return NULL;
    }

    if (n < 0) {
        npy_intp dims0[2] = {0, 0};
        PyArrayObject* r_out = (PyArrayObject*)PyArray_SimpleNew(2, dims0, NPY_DOUBLE);
        PyArrayObject* c_out = (PyArrayObject*)PyArray_SimpleNew(2, dims0, NPY_DOUBLE);
        npy_intp norm_dims[1] = {2};
        PyArrayObject* norms_out = (PyArrayObject*)PyArray_SimpleNew(1, norm_dims, NPY_DOUBLE);
        *((f64*)PyArray_GETPTR1(norms_out, 0)) = 0.0;
        *((f64*)PyArray_GETPTR1(norms_out, 1)) = 0.0;

        PyObject* result = Py_BuildValue("OOOi", r_out, c_out, norms_out, -8);
        Py_DECREF(r_out);
        Py_DECREF(c_out);
        Py_DECREF(norms_out);
        return result;
    }

    if (n == 0) {
        npy_intp dims0[2] = {0, 0};
        PyArrayObject* r_out = (PyArrayObject*)PyArray_SimpleNew(2, dims0, NPY_DOUBLE);
        PyArrayObject* c_out = (PyArrayObject*)PyArray_SimpleNew(2, dims0, NPY_DOUBLE);
        npy_intp norm_dims[1] = {2};
        PyArrayObject* norms_out = (PyArrayObject*)PyArray_SimpleNew(1, norm_dims, NPY_DOUBLE);
        *((f64*)PyArray_GETPTR1(norms_out, 0)) = 0.0;
        *((f64*)PyArray_GETPTR1(norms_out, 1)) = 0.0;

        PyObject* result = Py_BuildValue("OOOi", r_out, c_out, norms_out, 0);
        Py_DECREF(r_out);
        Py_DECREF(c_out);
        Py_DECREF(norms_out);
        return result;
    }

    PyArrayObject* a_array = (PyArrayObject*)PyArray_FROM_OTF(
        a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject* g_array = (PyArrayObject*)PyArray_FROM_OTF(
        g_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject* x_array = (PyArrayObject*)PyArray_FROM_OTF(
        x_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    PyArrayObject* q_array = (PyArrayObject*)PyArray_FROM_OTF(
        q_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);

    if (!a_array || !g_array || !x_array || !q_array) {
        Py_XDECREF(a_array);
        Py_XDECREF(g_array);
        Py_XDECREF(x_array);
        Py_XDECREF(q_array);
        PyErr_SetString(PyExc_ValueError, "Failed to convert a, g, x, or q to array");
        return NULL;
    }

    f64* a_data = (f64*)PyArray_DATA(a_array);
    f64* g_data = (f64*)PyArray_DATA(g_array);
    f64* x_data = (f64*)PyArray_DATA(x_array);
    f64* r_data = (f64*)PyArray_DATA(q_array);

    i32 lda = n;
    i32 ldg = n;
    i32 ldx = n;
    i32 ldr = n;
    i32 ldc = n;

    bool ljobe = (jobe_str[0] == 'G' || jobe_str[0] == 'g');
    bool ljobg_f = (jobg_str[0] == 'F' || jobg_str[0] == 'f');
    bool ljobg_h = (jobg_str[0] == 'H' || jobg_str[0] == 'h');
    bool ljobg_k = (jobg_str[0] == 'K' || jobg_str[0] == 'k');
    bool discr = (dico_str[0] == 'D' || dico_str[0] == 'd');

    PyArrayObject* e_array = NULL;
    f64* e_data = NULL;
    i32 lde = 1;

    if (ljobe) {
        if (e_obj == Py_None) {
            PyArray_DiscardWritebackIfCopy(a_array);
            PyArray_DiscardWritebackIfCopy(g_array);
            PyArray_DiscardWritebackIfCopy(x_array);
            PyArray_DiscardWritebackIfCopy(q_array);
            Py_DECREF(a_array);
            Py_DECREF(g_array);
            Py_DECREF(x_array);
            Py_DECREF(q_array);
            PyErr_SetString(PyExc_ValueError, "E matrix required when jobe='G'");
            return NULL;
        }
        e_array = (PyArrayObject*)PyArray_FROM_OTF(
            e_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
        if (!e_array) {
            PyArray_DiscardWritebackIfCopy(a_array);
            PyArray_DiscardWritebackIfCopy(g_array);
            PyArray_DiscardWritebackIfCopy(x_array);
            PyArray_DiscardWritebackIfCopy(q_array);
            Py_DECREF(a_array);
            Py_DECREF(g_array);
            Py_DECREF(x_array);
            Py_DECREF(q_array);
            PyErr_SetString(PyExc_ValueError, "Failed to convert e to array");
            return NULL;
        }
        e_data = (f64*)PyArray_DATA(e_array);
        lde = n;
    }

    PyArrayObject* f_array = NULL;
    f64* f_data = NULL;
    i32 ldf = 1;

    if (ljobg_f || ljobg_h) {
        if (f_obj == Py_None) {
            PyArray_DiscardWritebackIfCopy(a_array);
            PyArray_DiscardWritebackIfCopy(g_array);
            PyArray_DiscardWritebackIfCopy(x_array);
            PyArray_DiscardWritebackIfCopy(q_array);
            if (e_array) PyArray_DiscardWritebackIfCopy(e_array);
            Py_DECREF(a_array);
            Py_DECREF(g_array);
            Py_DECREF(x_array);
            Py_DECREF(q_array);
            Py_XDECREF(e_array);
            PyErr_SetString(PyExc_ValueError, "F matrix required when jobg='F' or 'H'");
            return NULL;
        }
        f_array = (PyArrayObject*)PyArray_FROM_OTF(
            f_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
        if (!f_array) {
            PyArray_DiscardWritebackIfCopy(a_array);
            PyArray_DiscardWritebackIfCopy(g_array);
            PyArray_DiscardWritebackIfCopy(x_array);
            PyArray_DiscardWritebackIfCopy(q_array);
            if (e_array) PyArray_DiscardWritebackIfCopy(e_array);
            Py_DECREF(a_array);
            Py_DECREF(g_array);
            Py_DECREF(x_array);
            Py_DECREF(q_array);
            Py_XDECREF(e_array);
            PyErr_SetString(PyExc_ValueError, "Failed to convert f to array");
            return NULL;
        }
        f_data = (f64*)PyArray_DATA(f_array);
        ldf = n;
    }

    PyArrayObject* k_array = NULL;
    f64* k_data = NULL;
    i32 ldk = 1;

    if (ljobg_h || ljobg_k) {
        if (k_obj == Py_None) {
            PyArray_DiscardWritebackIfCopy(a_array);
            PyArray_DiscardWritebackIfCopy(g_array);
            PyArray_DiscardWritebackIfCopy(x_array);
            PyArray_DiscardWritebackIfCopy(q_array);
            if (e_array) PyArray_DiscardWritebackIfCopy(e_array);
            if (f_array) PyArray_DiscardWritebackIfCopy(f_array);
            Py_DECREF(a_array);
            Py_DECREF(g_array);
            Py_DECREF(x_array);
            Py_DECREF(q_array);
            Py_XDECREF(e_array);
            Py_XDECREF(f_array);
            PyErr_SetString(PyExc_ValueError, "K matrix required when jobg='H' or 'K'");
            return NULL;
        }
        k_array = (PyArrayObject*)PyArray_FROM_OTF(
            k_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
        if (!k_array) {
            PyArray_DiscardWritebackIfCopy(a_array);
            PyArray_DiscardWritebackIfCopy(g_array);
            PyArray_DiscardWritebackIfCopy(x_array);
            PyArray_DiscardWritebackIfCopy(q_array);
            if (e_array) PyArray_DiscardWritebackIfCopy(e_array);
            if (f_array) PyArray_DiscardWritebackIfCopy(f_array);
            Py_DECREF(a_array);
            Py_DECREF(g_array);
            Py_DECREF(x_array);
            Py_DECREF(q_array);
            Py_XDECREF(e_array);
            Py_XDECREF(f_array);
            PyErr_SetString(PyExc_ValueError, "Failed to convert k to array");
            return NULL;
        }
        k_data = (f64*)PyArray_DATA(k_array);
        ldk = m;
    }

    PyArrayObject* xe_array = NULL;
    f64* xe_data = NULL;
    i32 ldxe = 1;

    if (ljobe && discr) {
        if (xe_obj == Py_None) {
            PyArray_DiscardWritebackIfCopy(a_array);
            PyArray_DiscardWritebackIfCopy(g_array);
            PyArray_DiscardWritebackIfCopy(x_array);
            PyArray_DiscardWritebackIfCopy(q_array);
            if (e_array) PyArray_DiscardWritebackIfCopy(e_array);
            if (f_array) PyArray_DiscardWritebackIfCopy(f_array);
            if (k_array) PyArray_DiscardWritebackIfCopy(k_array);
            Py_DECREF(a_array);
            Py_DECREF(g_array);
            Py_DECREF(x_array);
            Py_DECREF(q_array);
            Py_XDECREF(e_array);
            Py_XDECREF(f_array);
            Py_XDECREF(k_array);
            PyErr_SetString(PyExc_ValueError, "XE matrix required for discrete-time with general E");
            return NULL;
        }
        xe_array = (PyArrayObject*)PyArray_FROM_OTF(
            xe_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
        if (!xe_array) {
            PyArray_DiscardWritebackIfCopy(a_array);
            PyArray_DiscardWritebackIfCopy(g_array);
            PyArray_DiscardWritebackIfCopy(x_array);
            PyArray_DiscardWritebackIfCopy(q_array);
            if (e_array) PyArray_DiscardWritebackIfCopy(e_array);
            if (f_array) PyArray_DiscardWritebackIfCopy(f_array);
            if (k_array) PyArray_DiscardWritebackIfCopy(k_array);
            Py_DECREF(a_array);
            Py_DECREF(g_array);
            Py_DECREF(x_array);
            Py_DECREF(q_array);
            Py_XDECREF(e_array);
            Py_XDECREF(f_array);
            Py_XDECREF(k_array);
            PyErr_SetString(PyExc_ValueError, "Failed to convert xe to array");
            return NULL;
        }
        xe_data = (f64*)PyArray_DATA(xe_array);
        ldxe = n;
    }

    f64* c_data = (f64*)calloc((size_t)n * (size_t)n, sizeof(f64));
    if (!c_data) {
        PyArray_DiscardWritebackIfCopy(a_array);
        PyArray_DiscardWritebackIfCopy(g_array);
        PyArray_DiscardWritebackIfCopy(x_array);
        PyArray_DiscardWritebackIfCopy(q_array);
        if (e_array) PyArray_DiscardWritebackIfCopy(e_array);
        if (f_array) PyArray_DiscardWritebackIfCopy(f_array);
        if (k_array) PyArray_DiscardWritebackIfCopy(k_array);
        if (xe_array) PyArray_DiscardWritebackIfCopy(xe_array);
        Py_DECREF(a_array);
        Py_DECREF(g_array);
        Py_DECREF(x_array);
        Py_DECREF(q_array);
        Py_XDECREF(e_array);
        Py_XDECREF(f_array);
        Py_XDECREF(k_array);
        Py_XDECREF(xe_array);
        PyErr_NoMemory();
        return NULL;
    }

    bool ljobg_g = (jobg_str[0] == 'G' || jobg_str[0] == 'g');
    bool ljobn = (job_str[0] == 'N' || job_str[0] == 'n' ||
                  job_str[0] == 'B' || job_str[0] == 'b');
    bool ljobb = (job_str[0] == 'B' || job_str[0] == 'b');

    i32 nn = n * n;
    i32 nm = n * m;
    i32 ia = (ljobe || discr) ? 1 : 0;
    i32 ib = ljobb ? 1 : 0;

    i32 ldwork;
    if (ljobn || ljobb) {
        if (ljobg_g) {
            ldwork = (ia + ib + 1) * nn;
        } else {
            if (discr) {
                ldwork = ljobn ? 2 * nn : (nn + 2 * nn);
            } else {
                i32 i_val = nn + (ia * ib * nn > nm ? ia * ib * nn : nm);
                i32 j_val = nn + (ia + ib) * nn;
                ldwork = i_val < j_val ? i_val : j_val;
                ldwork = ldwork > (ia + ib + 1) * nn ? ldwork : (ia + ib + 1) * nn;
            }
        }
    } else {
        if (discr) {
            if (ljobg_g) {
                ldwork = 3 * nn;
            } else {
                ldwork = nn + (nn > nm ? nn : nm);
            }
        } else {
            if (ljobg_g) {
                ldwork = (2 * ia + 1) * nn;
            } else {
                i32 i_val = nn + (ia * nn > nm ? ia * nn : nm);
                i32 j_val = (2 * ia + 1) * nn;
                ldwork = i_val > j_val ? i_val : j_val;
            }
        }
    }
    ldwork = ldwork > 1 ? ldwork : 1;

    f64* dwork = (f64*)calloc(ldwork, sizeof(f64));
    f64 norms[2] = {0.0, 0.0};

    if (!dwork) {
        free(c_data);
        PyArray_DiscardWritebackIfCopy(a_array);
        PyArray_DiscardWritebackIfCopy(g_array);
        PyArray_DiscardWritebackIfCopy(x_array);
        PyArray_DiscardWritebackIfCopy(q_array);
        if (e_array) PyArray_DiscardWritebackIfCopy(e_array);
        if (f_array) PyArray_DiscardWritebackIfCopy(f_array);
        if (k_array) PyArray_DiscardWritebackIfCopy(k_array);
        if (xe_array) PyArray_DiscardWritebackIfCopy(xe_array);
        Py_DECREF(a_array);
        Py_DECREF(g_array);
        Py_DECREF(x_array);
        Py_DECREF(q_array);
        Py_XDECREF(e_array);
        Py_XDECREF(f_array);
        Py_XDECREF(k_array);
        Py_XDECREF(xe_array);
        PyErr_NoMemory();
        return NULL;
    }

    i32 info;

    sg02cw(dico_str, job_str, jobe_str, flag_str, jobg_str, uplo_str, trans_str,
           n, m,
           a_data, lda,
           e_data, lde,
           g_data, ldg,
           x_data, ldx,
           f_data, ldf,
           k_data, ldk,
           xe_data, ldxe,
           r_data, ldr,
           c_data, ldc,
           norms,
           dwork, ldwork,
           &info);

    free(dwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(g_array);
    PyArray_ResolveWritebackIfCopy(x_array);
    PyArray_ResolveWritebackIfCopy(q_array);
    if (e_array) PyArray_ResolveWritebackIfCopy(e_array);
    if (f_array) PyArray_ResolveWritebackIfCopy(f_array);
    if (k_array) PyArray_ResolveWritebackIfCopy(k_array);
    if (xe_array) PyArray_ResolveWritebackIfCopy(xe_array);

    npy_intp mat_dims[2] = {n, n};
    PyArrayObject* r_out = (PyArrayObject*)PyArray_SimpleNew(2, mat_dims, NPY_DOUBLE);
    if (!r_out) {
        free(c_data);
        Py_DECREF(a_array);
        Py_DECREF(g_array);
        Py_DECREF(x_array);
        Py_DECREF(q_array);
        Py_XDECREF(e_array);
        Py_XDECREF(f_array);
        Py_XDECREF(k_array);
        Py_XDECREF(xe_array);
        PyErr_NoMemory();
        return NULL;
    }

    for (i32 j = 0; j < n; j++) {
        for (i32 i = 0; i < n; i++) {
            *((f64*)PyArray_GETPTR2(r_out, i, j)) = r_data[i + j * ldr];
        }
    }

    PyArrayObject* c_out = (PyArrayObject*)PyArray_SimpleNew(2, mat_dims, NPY_DOUBLE);
    if (!c_out) {
        free(c_data);
        Py_DECREF(r_out);
        Py_DECREF(a_array);
        Py_DECREF(g_array);
        Py_DECREF(x_array);
        Py_DECREF(q_array);
        Py_XDECREF(e_array);
        Py_XDECREF(f_array);
        Py_XDECREF(k_array);
        Py_XDECREF(xe_array);
        PyErr_NoMemory();
        return NULL;
    }

    for (i32 j = 0; j < n; j++) {
        for (i32 i = 0; i < n; i++) {
            *((f64*)PyArray_GETPTR2(c_out, i, j)) = c_data[i + j * ldc];
        }
    }

    free(c_data);

    npy_intp norm_dims[1] = {2};
    PyArrayObject* norms_out = (PyArrayObject*)PyArray_SimpleNew(1, norm_dims, NPY_DOUBLE);
    if (!norms_out) {
        Py_DECREF(r_out);
        Py_DECREF(c_out);
        Py_DECREF(a_array);
        Py_DECREF(g_array);
        Py_DECREF(x_array);
        Py_DECREF(q_array);
        Py_XDECREF(e_array);
        Py_XDECREF(f_array);
        Py_XDECREF(k_array);
        Py_XDECREF(xe_array);
        PyErr_NoMemory();
        return NULL;
    }
    *((f64*)PyArray_GETPTR1(norms_out, 0)) = norms[0];
    *((f64*)PyArray_GETPTR1(norms_out, 1)) = norms[1];

    PyObject* result = Py_BuildValue("OOOi", r_out, c_out, norms_out, info);

    Py_DECREF(r_out);
    Py_DECREF(c_out);
    Py_DECREF(norms_out);
    Py_DECREF(a_array);
    Py_DECREF(g_array);
    Py_DECREF(x_array);
    Py_DECREF(q_array);
    Py_XDECREF(e_array);
    Py_XDECREF(f_array);
    Py_XDECREF(k_array);
    Py_XDECREF(xe_array);

    return result;
}

PyObject* py_sg02cx(PyObject* self, PyObject* args, PyObject* kwds) {
    static char* kwlist[] = {
        "jobe", "flag", "jobg", "uplo", "trans",
        "n", "m", "e", "r", "s", "g", NULL
    };

    char *jobe, *flag, *jobg, *uplo, *trans;
    int n, m;
    PyObject *e_obj, *r_obj, *s_obj, *g_obj;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "sssssiiOOOO", kwlist,
                                     &jobe, &flag, &jobg, &uplo, &trans,
                                     &n, &m, &e_obj, &r_obj, &s_obj, &g_obj)) {
        return NULL;
    }

    PyArrayObject *e_array = NULL;
    PyArrayObject *r_array = NULL;
    PyArrayObject *s_array = NULL;
    PyArrayObject *g_array = NULL;

    char jobe_c = (char)toupper((unsigned char)jobe[0]);
    char jobg_c = (char)toupper((unsigned char)jobg[0]);
    bool ljobe = (jobe_c == 'G');
    bool ljobl = (jobg_c == 'F' || jobg_c == 'H');

    e_array = (PyArrayObject*)PyArray_FROM_OTF(e_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (e_array == NULL) return NULL;

    r_array = (PyArrayObject*)PyArray_FROM_OTF(r_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (r_array == NULL) {
        Py_DECREF(e_array);
        return NULL;
    }

    s_array = (PyArrayObject*)PyArray_FROM_OTF(s_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (s_array == NULL) {
        Py_DECREF(e_array);
        Py_DECREF(r_array);
        return NULL;
    }

    g_array = (PyArrayObject*)PyArray_FROM_OTF(g_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (g_array == NULL) {
        Py_DECREF(e_array);
        Py_DECREF(r_array);
        Py_DECREF(s_array);
        return NULL;
    }

    i32 lde = (ljobe && !ljobl && n > 0) ? (i32)PyArray_DIM(e_array, 0) : 1;
    i32 ldr = (n > 0) ? (i32)PyArray_DIM(r_array, 0) : 1;
    i32 lds;
    if (jobg_c == 'H') {
        lds = (m > 0) ? (i32)PyArray_DIM(s_array, 0) : 1;
    } else if (ljobl) {
        lds = 1;
    } else {
        lds = (n > 0) ? (i32)PyArray_DIM(s_array, 0) : 1;
    }
    i32 ldg = (n > 0) ? (i32)PyArray_DIM(g_array, 0) : 1;

    const f64* e_data = (const f64*)PyArray_DATA(e_array);
    const f64* r_data = (const f64*)PyArray_DATA(r_array);
    const f64* s_data = (const f64*)PyArray_DATA(s_array);
    f64* g_data = (f64*)PyArray_DATA(g_array);

    i32 nn = n * n;
    i32 nm = n * m;
    i32 ldwork;

    if (ljobl) {
        ldwork = nn + (nn > 51 ? nn : 51);
    } else if (jobg_c == 'G') {
        if (ljobe) {
            ldwork = nn + (2 * nn > 51 ? 2 * nn : 51);
        } else {
            ldwork = nn + (nn > 51 ? nn : 51);
        }
    } else {
        if (ljobe) {
            i32 max_nn_51 = nn > 51 ? nn : 51;
            i32 min_2nn_nm = 2 * nn < nm ? 2 * nn : nm;
            ldwork = nn + (max_nn_51 > min_2nn_nm ? max_nn_51 : min_2nn_nm);
        } else {
            i32 max_nn_nm = nn > nm ? nn : nm;
            ldwork = nn + (max_nn_nm > 51 ? max_nn_nm : 51);
        }
    }
    ldwork += 60;

    f64* dwork = (f64*)calloc(ldwork, sizeof(f64));
    if (!dwork) {
        Py_DECREF(e_array);
        Py_DECREF(r_array);
        Py_DECREF(s_array);
        Py_DECREF(g_array);
        PyErr_NoMemory();
        return NULL;
    }

    f64 alpha, rnorm;
    i32 iwarn, info;

    sg02cx(jobe, flag, jobg, uplo, trans,
           (i32)n, (i32)m,
           e_data, lde,
           r_data, ldr,
           s_data, lds,
           g_data, ldg,
           &alpha, &rnorm,
           dwork, ldwork,
           &iwarn, &info);

    free(dwork);

    PyArray_ResolveWritebackIfCopy(g_array);

    PyObject* result = Py_BuildValue("ddii", alpha, rnorm, iwarn, info);

    Py_DECREF(e_array);
    Py_DECREF(r_array);
    Py_DECREF(s_array);
    Py_DECREF(g_array);

    return result;
}

PyObject* py_sg02nd(PyObject* self, PyObject* args, PyObject* kwargs) {
    (void)self;

    static char* kwlist[] = {
        "dico", "jobe", "job", "jobx", "fact", "uplo", "jobl", "trans",
        "n", "m", "p",
        "a", "e", "b", "r", "ipiv", "l", "x", "rnorm",
        NULL
    };

    const char* dico;
    const char* jobe;
    const char* job;
    const char* jobx;
    const char* fact;
    const char* uplo;
    const char* jobl;
    const char* trans;
    int n, m, p;
    PyObject *a_obj, *e_obj, *b_obj, *r_obj, *ipiv_obj, *l_obj, *x_obj;
    double rnorm_in;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ssssssssiiiOOOOOOOd", kwlist,
                                     &dico, &jobe, &job, &jobx, &fact, &uplo, &jobl, &trans,
                                     &n, &m, &p,
                                     &a_obj, &e_obj, &b_obj, &r_obj, &ipiv_obj, &l_obj, &x_obj,
                                     &rnorm_in)) {
        return NULL;
    }

    char dico_c = toupper((unsigned char)dico[0]);
    char jobe_c = toupper((unsigned char)jobe[0]);
    char job_c = toupper((unsigned char)job[0]);
    char jobx_c = toupper((unsigned char)jobx[0]);
    char fact_c = toupper((unsigned char)fact[0]);
    char jobl_c = toupper((unsigned char)jobl[0]);

    bool discr = (dico_c == 'D');
    bool ljobe = (jobe_c == 'G');
    bool withxe = (jobx_c == 'C');
    bool withh = (job_c == 'H' || job_c == 'F' || job_c == 'C' || job_c == 'D');
    bool withl = (jobl_c == 'N');
    bool lfactd = (fact_c == 'D');

    PyArrayObject *a_array = NULL, *e_array = NULL, *b_array = NULL, *r_array = NULL;
    PyArrayObject *ipiv_array = NULL, *l_array = NULL, *x_array = NULL;

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (a_array == NULL) return NULL;

    e_array = (PyArrayObject*)PyArray_FROM_OTF(e_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (e_array == NULL) { Py_DECREF(a_array); return NULL; }

    b_array = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (b_array == NULL) { Py_DECREF(a_array); Py_DECREF(e_array); return NULL; }

    r_array = (PyArrayObject*)PyArray_FROM_OTF(r_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (r_array == NULL) { Py_DECREF(a_array); Py_DECREF(e_array); Py_DECREF(b_array); return NULL; }

    ipiv_array = (PyArrayObject*)PyArray_FROM_OTF(ipiv_obj, NPY_INT32, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (ipiv_array == NULL) {
        Py_DECREF(a_array); Py_DECREF(e_array); Py_DECREF(b_array); Py_DECREF(r_array);
        return NULL;
    }

    l_array = (PyArrayObject*)PyArray_FROM_OTF(l_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (l_array == NULL) {
        Py_DECREF(a_array); Py_DECREF(e_array); Py_DECREF(b_array); Py_DECREF(r_array);
        Py_DECREF(ipiv_array);
        return NULL;
    }

    x_array = (PyArrayObject*)PyArray_FROM_OTF(x_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (x_array == NULL) {
        Py_DECREF(a_array); Py_DECREF(e_array); Py_DECREF(b_array); Py_DECREF(r_array);
        Py_DECREF(ipiv_array); Py_DECREF(l_array);
        return NULL;
    }

    i32 lda = discr ? (n > 0 ? (i32)PyArray_DIM(a_array, 0) : 1) : 1;
    i32 lde = (!discr && ljobe && n > 0) ? (i32)PyArray_DIM(e_array, 0) : 1;
    i32 ldb = (n > 0) ? (i32)PyArray_DIM(b_array, 0) : 1;
    i32 ldr = (m > 0) ? (i32)PyArray_DIM(r_array, 0) : 1;
    if (lfactd && p > 0 && ldr < p) ldr = p;
    i32 ldl = (withl && n > 0) ? (i32)PyArray_DIM(l_array, 0) : 1;
    i32 ldx = (n > 0) ? (i32)PyArray_DIM(x_array, 0) : 1;
    i32 ldk = (m > 0) ? m : 1;
    i32 ldh = (withh && n > 0) ? n : 1;
    i32 ldxe = (withxe && (discr || ljobe) && n > 0) ? n : 1;

    const f64* a_data = (const f64*)PyArray_DATA(a_array);
    const f64* e_data = (const f64*)PyArray_DATA(e_array);
    f64* b_data = (f64*)PyArray_DATA(b_array);
    f64* r_data = (f64*)PyArray_DATA(r_array);
    i32* ipiv_data = (i32*)PyArray_DATA(ipiv_array);
    const f64* l_data = (const f64*)PyArray_DATA(l_array);
    f64* x_data = (f64*)PyArray_DATA(x_array);

    npy_intp k_dims[2] = {ldk, n};
    npy_intp k_strides[2] = {sizeof(f64), ldk * (npy_intp)sizeof(f64)};
    PyObject* k_array = PyArray_New(&PyArray_Type, 2, k_dims, NPY_DOUBLE, k_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (k_array == NULL) {
        Py_DECREF(a_array); Py_DECREF(e_array); Py_DECREF(b_array); Py_DECREF(r_array);
        Py_DECREF(ipiv_array); Py_DECREF(l_array); Py_DECREF(x_array);
        return NULL;
    }
    f64* k = (f64*)PyArray_DATA((PyArrayObject*)k_array);
    if (ldk * n > 0) memset(k, 0, ldk * n * sizeof(f64));

    PyObject* h_array = NULL;
    f64* h = NULL;
    f64* h_temp = NULL;
    if (withh) {
        npy_intp h_dims[2] = {ldh, m};
        npy_intp h_strides[2] = {sizeof(f64), ldh * (npy_intp)sizeof(f64)};
        h_array = PyArray_New(&PyArray_Type, 2, h_dims, NPY_DOUBLE, h_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        if (h_array == NULL) {
            Py_DECREF(k_array);
            Py_DECREF(a_array); Py_DECREF(e_array); Py_DECREF(b_array); Py_DECREF(r_array);
            Py_DECREF(ipiv_array); Py_DECREF(l_array); Py_DECREF(x_array);
            return NULL;
        }
        h = (f64*)PyArray_DATA((PyArrayObject*)h_array);
        if (ldh * m > 0) memset(h, 0, ldh * m * sizeof(f64));
    } else {
        h_temp = (f64*)calloc(1, sizeof(f64));
        if (!h_temp) {
            Py_DECREF(k_array);
            Py_DECREF(a_array); Py_DECREF(e_array); Py_DECREF(b_array); Py_DECREF(r_array);
            Py_DECREF(ipiv_array); Py_DECREF(l_array); Py_DECREF(x_array);
            PyErr_NoMemory();
            return NULL;
        }
        h = h_temp;
    }

    PyObject* xe_array = NULL;
    f64* xe = NULL;
    f64* xe_temp = NULL;
    if (withxe && (discr || ljobe)) {
        npy_intp xe_dims[2] = {ldxe, n};
        npy_intp xe_strides[2] = {sizeof(f64), ldxe * (npy_intp)sizeof(f64)};
        xe_array = PyArray_New(&PyArray_Type, 2, xe_dims, NPY_DOUBLE, xe_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        if (xe_array == NULL) {
            Py_DECREF(k_array);
            if (h_array) Py_DECREF(h_array);
            free(h_temp);
            Py_DECREF(a_array); Py_DECREF(e_array); Py_DECREF(b_array); Py_DECREF(r_array);
            Py_DECREF(ipiv_array); Py_DECREF(l_array); Py_DECREF(x_array);
            return NULL;
        }
        xe = (f64*)PyArray_DATA((PyArrayObject*)xe_array);
        if (ldxe * n > 0) memset(xe, 0, ldxe * n * sizeof(f64));
    } else {
        xe_temp = (f64*)calloc(1, sizeof(f64));
        if (!xe_temp) {
            Py_DECREF(k_array);
            if (h_array) Py_DECREF(h_array);
            free(h_temp);
            Py_DECREF(a_array); Py_DECREF(e_array); Py_DECREF(b_array); Py_DECREF(r_array);
            Py_DECREF(ipiv_array); Py_DECREF(l_array); Py_DECREF(x_array);
            PyErr_NoMemory();
            return NULL;
        }
        xe = xe_temp;
    }

    bool lfacta = (fact_c == 'C' || fact_c == 'D' || fact_c == 'U');
    bool withcd = (job_c == 'C' || job_c == 'D');
    i32 ldwork;
    if (discr && lfacta && !withcd) {
        ldwork = 2;
        if (3 * m > ldwork) ldwork = 3 * m;
        if (4 * n + 1 > ldwork) ldwork = 4 * n + 1;
        ldwork += n * m + n + 10;
    } else {
        ldwork = 2;
        if (!withxe && (discr || ljobe)) {
            if (n > ldwork) ldwork = n;
        }
        if (fact_c == 'U') {
            if (2 * m > ldwork) ldwork = 2 * m;
        } else {
            if (3 * m > ldwork) ldwork = 3 * m;
        }
        ldwork += n * m + 10;
    }

    f64* dwork = (f64*)calloc(ldwork, sizeof(f64));
    i32* iwork = (i32*)calloc(m > 0 ? m : 1, sizeof(i32));

    if (!dwork || !iwork) {
        free(h_temp); free(xe_temp); free(dwork); free(iwork);
        Py_DECREF(k_array);
        if (h_array) Py_DECREF(h_array);
        if (xe_array) Py_DECREF(xe_array);
        Py_DECREF(a_array); Py_DECREF(e_array); Py_DECREF(b_array); Py_DECREF(r_array);
        Py_DECREF(ipiv_array); Py_DECREF(l_array); Py_DECREF(x_array);
        PyErr_NoMemory();
        return NULL;
    }

    i32 oufact[2] = {0, 0};
    i32 info;

    sg02nd(dico, jobe, job, jobx, fact, uplo, jobl, trans,
           (i32)n, (i32)m, (i32)p,
           a_data, lda,
           e_data, lde,
           b_data, ldb,
           r_data, ldr,
           ipiv_data,
           l_data, ldl,
           x_data, ldx,
           rnorm_in,
           k, ldk,
           h, ldh,
           xe, ldxe,
           oufact,
           dwork, ldwork,
           &info);

    f64 rcond = (ldwork >= 2) ? dwork[1] : 0.0;

    free(dwork);
    free(iwork);
    free(h_temp);
    free(xe_temp);

    PyArray_ResolveWritebackIfCopy(b_array);
    PyArray_ResolveWritebackIfCopy(r_array);
    PyArray_ResolveWritebackIfCopy(ipiv_array);
    PyArray_ResolveWritebackIfCopy(x_array);

    if (!h_array) {
        Py_INCREF(Py_None);
        h_array = Py_None;
    }

    if (!xe_array) {
        Py_INCREF(Py_None);
        xe_array = Py_None;
    }

    npy_intp oufact_dims[1] = {2};
    PyObject* oufact_array = PyArray_SimpleNew(1, oufact_dims, NPY_INT32);
    if (oufact_array == NULL) {
        Py_DECREF(k_array);
        if (h_array != Py_None) Py_DECREF(h_array);
        if (xe_array != Py_None) Py_DECREF(xe_array);
        Py_DECREF(a_array); Py_DECREF(e_array); Py_DECREF(b_array); Py_DECREF(r_array);
        Py_DECREF(ipiv_array); Py_DECREF(l_array); Py_DECREF(x_array);
        return NULL;
    }
    ((i32*)PyArray_DATA((PyArrayObject*)oufact_array))[0] = oufact[0];
    ((i32*)PyArray_DATA((PyArrayObject*)oufact_array))[1] = oufact[1];

    PyObject* result = Py_BuildValue("OOOOdi", k_array, h_array, xe_array, oufact_array, rcond, info);

    Py_DECREF(k_array);
    if (h_array != Py_None) Py_DECREF(h_array);
    if (xe_array != Py_None) Py_DECREF(xe_array);
    Py_DECREF(oufact_array);
    Py_DECREF(a_array);
    Py_DECREF(e_array);
    Py_DECREF(b_array);
    Py_DECREF(r_array);
    Py_DECREF(ipiv_array);
    Py_DECREF(l_array);
    Py_DECREF(x_array);

    return result;
}

/* Python wrapper for sg03bs */
PyObject* py_sg03bs(PyObject* self, PyObject* args) {
    char *trans;
    PyObject *a_obj, *e_obj, *b_obj;
    PyArrayObject *a_array, *e_array, *b_array;
    f64 scale;
    i32 info;

    if (!PyArg_ParseTuple(args, "sOOO", &trans, &a_obj, &e_obj, &b_obj)) {
        return NULL;
    }

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_CDOUBLE,
                                                NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (a_array == NULL) return NULL;

    e_array = (PyArrayObject*)PyArray_FROM_OTF(e_obj, NPY_CDOUBLE,
                                                NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (e_array == NULL) {
        Py_DECREF(a_array);
        return NULL;
    }

    b_array = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_CDOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (b_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(e_array);
        return NULL;
    }

    i32 n = (i32)PyArray_DIM(b_array, 0);
    npy_intp lda_raw = PyArray_DIM(a_array, 0);
    npy_intp lde_raw = PyArray_DIM(e_array, 0);
    npy_intp ldb_raw = PyArray_DIM(b_array, 0);
    i32 lda = (lda_raw > 0) ? (i32)lda_raw : 1;
    i32 lde = (lde_raw > 0) ? (i32)lde_raw : 1;
    i32 ldb = (ldb_raw > 0) ? (i32)ldb_raw : 1;

    c128 *a_data = (c128*)PyArray_DATA(a_array);
    c128 *e_data = (c128*)PyArray_DATA(e_array);
    c128 *b_data = (c128*)PyArray_DATA(b_array);

    i32 ldwork = (n > 1) ? ((n - 1 > 10) ? n - 1 : 10) : 1;
    i32 lzwork = (n > 1) ? 3 * n - 3 : 1;
    f64 *dwork = (f64*)malloc(ldwork * sizeof(f64));
    c128 *zwork = (c128*)PyMem_Calloc(lzwork, sizeof(c128));

    if (dwork == NULL || zwork == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate workspace");
        if (dwork) free(dwork);
        if (zwork) PyMem_Free(zwork);
        Py_DECREF(a_array);
        Py_DECREF(e_array);
        Py_DECREF(b_array);
        return NULL;
    }

    sg03bs(trans, n, a_data, lda, e_data, lde, b_data, ldb, &scale, dwork, zwork, &info);

    free(dwork);
    PyMem_Free(zwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(e_array);
    PyArray_ResolveWritebackIfCopy(b_array);

    PyObject *result = Py_BuildValue("Odi", b_array, scale, info);

    Py_DECREF(a_array);
    Py_DECREF(e_array);
    Py_DECREF(b_array);

    return result;
}

PyObject* py_sg03bt(PyObject* self, PyObject* args)
{
    PyObject *a_obj, *e_obj, *b_obj;
    const char *trans;
    f64 scale;
    i32 info;

    if (!PyArg_ParseTuple(args, "sOOO", &trans, &a_obj, &e_obj, &b_obj)) {
        return NULL;
    }

    PyArrayObject *a_array, *e_array, *b_array;

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_CDOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (a_array == NULL) {
        return NULL;
    }

    e_array = (PyArrayObject*)PyArray_FROM_OTF(e_obj, NPY_CDOUBLE,
                                                NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (e_array == NULL) {
        Py_DECREF(a_array);
        return NULL;
    }

    b_array = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_CDOUBLE,
                                               NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (b_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(e_array);
        return NULL;
    }

    i32 n = (i32)PyArray_DIM(b_array, 0);
    npy_intp lda_raw = PyArray_DIM(a_array, 0);
    npy_intp lde_raw = PyArray_DIM(e_array, 0);
    npy_intp ldb_raw = PyArray_DIM(b_array, 0);
    i32 lda = (lda_raw > 0) ? (i32)lda_raw : 1;
    i32 lde = (lde_raw > 0) ? (i32)lde_raw : 1;
    i32 ldb = (ldb_raw > 0) ? (i32)ldb_raw : 1;

    c128 *a_data = (c128*)PyArray_DATA(a_array);
    c128 *e_data = (c128*)PyArray_DATA(e_array);
    c128 *b_data = (c128*)PyArray_DATA(b_array);

    i32 ldwork = (n > 1) ? n - 1 : 1;
    i32 lzwork = (n > 1) ? 3 * n - 3 : 1;
    f64 *dwork = (f64*)malloc(ldwork * sizeof(f64));
    c128 *zwork = (c128*)PyMem_Calloc(lzwork, sizeof(c128));

    if (dwork == NULL || zwork == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate workspace");
        if (dwork) free(dwork);
        if (zwork) PyMem_Free(zwork);
        Py_DECREF(a_array);
        Py_DECREF(e_array);
        Py_DECREF(b_array);
        return NULL;
    }

    sg03bt(trans, n, a_data, lda, e_data, lde, b_data, ldb, &scale, dwork, zwork, &info);

    free(dwork);
    PyMem_Free(zwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(e_array);
    PyArray_ResolveWritebackIfCopy(b_array);

    PyObject *result = Py_BuildValue("Odi", b_array, scale, info);

    Py_DECREF(a_array);
    Py_DECREF(e_array);
    Py_DECREF(b_array);

    return result;
}

PyObject* py_sg03by(PyObject* self, PyObject* args) {
    (void)self;
    f64 xr, xi, yr, yi;
    if (!PyArg_ParseTuple(args, "dddd", &xr, &xi, &yr, &yi)) {
        return NULL;
    }

    f64 cr, ci, sr, si, z;
    sg03by(xr, xi, yr, yi, &cr, &ci, &sr, &si, &z);

    return Py_BuildValue("ddddd", cr, ci, sr, si, z);
}

PyObject* py_sg03bz(PyObject* self, PyObject* args) {
    (void)self;
    char *dico, *fact, *trans;
    PyObject *a_obj, *e_obj, *q_obj, *z_obj, *b_obj;
    PyArrayObject *a_array, *e_array, *q_array, *z_array, *b_array;
    f64 scale;
    i32 info;

    if (!PyArg_ParseTuple(args, "sssOOOOO", &dico, &fact, &trans,
                          &a_obj, &e_obj, &q_obj, &z_obj, &b_obj)) {
        return NULL;
    }

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_CDOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (a_array == NULL) return NULL;

    e_array = (PyArrayObject*)PyArray_FROM_OTF(e_obj, NPY_CDOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (e_array == NULL) {
        Py_DECREF(a_array);
        return NULL;
    }

    q_array = (PyArrayObject*)PyArray_FROM_OTF(q_obj, NPY_CDOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (q_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(e_array);
        return NULL;
    }

    z_array = (PyArrayObject*)PyArray_FROM_OTF(z_obj, NPY_CDOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (z_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(e_array);
        Py_DECREF(q_array);
        return NULL;
    }

    b_array = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_CDOUBLE, NPY_ARRAY_IN_FARRAY);
    if (b_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(e_array);
        Py_DECREF(q_array);
        Py_DECREF(z_array);
        return NULL;
    }

    i32 n = (i32)PyArray_DIM(a_array, 0);
    i32 m;
    char trans_upper = toupper((unsigned char)*trans);
    if (trans_upper == 'C') {
        m = (PyArray_NDIM(b_array) >= 2) ? (i32)PyArray_DIM(b_array, 1) : 0;
    } else {
        m = (i32)PyArray_DIM(b_array, 0);
    }

    i32 lda = (n > 0) ? n : 1;
    i32 lde = lda;
    i32 ldq = lda;
    i32 ldz = lda;

    i32 maxmn = (m > n) ? m : n;
    i32 ldb;
    if (trans_upper == 'C') {
        ldb = (n > 0) ? n : 1;
    } else {
        ldb = (maxmn > 0) ? maxmn : 1;
    }

    c128 *a_data = (c128*)PyArray_DATA(a_array);
    c128 *e_data = (c128*)PyArray_DATA(e_array);
    c128 *q_data = (c128*)PyArray_DATA(q_array);
    c128 *z_data = (c128*)PyArray_DATA(z_array);
    c128 *b_data = (c128*)PyArray_DATA(b_array);

    i32 b_n1;
    if (trans_upper == 'C') {
        b_n1 = (maxmn > n) ? maxmn : n;
    } else {
        b_n1 = n;
    }
    npy_intp b_work_dims[2] = {ldb, b_n1};
    npy_intp b_work_strides[2] = {sizeof(c128), ldb * sizeof(c128)};
    PyArrayObject *b_work = (PyArrayObject*)PyArray_New(&PyArray_Type, 2, b_work_dims, NPY_CDOUBLE,
                                                         b_work_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (b_work == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(e_array);
        Py_DECREF(q_array);
        Py_DECREF(z_array);
        Py_DECREF(b_array);
        return NULL;
    }
    c128 *b_work_data = (c128*)PyArray_DATA(b_work);
    memset(b_work_data, 0, ldb * b_n1 * sizeof(c128));

    i32 b_rows = (i32)PyArray_DIM(b_array, 0);
    i32 b_cols = (PyArray_NDIM(b_array) >= 2) ? (i32)PyArray_DIM(b_array, 1) : 1;
    for (i32 j = 0; j < b_cols; j++) {
        for (i32 i = 0; i < b_rows; i++) {
            b_work_data[i + j * ldb] = b_data[i + j * b_rows];
        }
    }

    npy_intp alpha_dims[1] = {n};
    PyArrayObject *alpha_array = (PyArrayObject*)PyArray_SimpleNew(1, alpha_dims, NPY_CDOUBLE);
    PyArrayObject *beta_array_out = (PyArrayObject*)PyArray_SimpleNew(1, alpha_dims, NPY_CDOUBLE);
    if (alpha_array == NULL || beta_array_out == NULL) {
        Py_XDECREF(alpha_array);
        Py_XDECREF(beta_array_out);
        Py_DECREF(a_array);
        Py_DECREF(e_array);
        Py_DECREF(q_array);
        Py_DECREF(z_array);
        Py_DECREF(b_array);
        Py_DECREF(b_work);
        return NULL;
    }
    c128 *alpha = (c128*)PyArray_DATA(alpha_array);
    c128 *beta_arr = (c128*)PyArray_DATA(beta_array_out);

    i32 minwrk = 1;
    if (2*n > minwrk) minwrk = 2*n;
    if (3*n - 3 > minwrk) minwrk = 3*n - 3;
    if (minwrk < 1) minwrk = 1;

    i32 ldwork;
    char fact_upper = toupper((unsigned char)*fact);
    char dico_upper = toupper((unsigned char)*dico);
    i32 min_mn = (m < n) ? m : n;
    if (min_mn == 0 || (fact_upper == 'F' && n <= 1)) {
        ldwork = 1;
    } else if (fact_upper == 'F' && dico_upper == 'C') {
        ldwork = (n > 1) ? n - 1 : 1;
    } else if (fact_upper == 'F' && dico_upper == 'D') {
        ldwork = (n > 1) ? ((n - 1 > 10) ? n - 1 : 10) : 10;
    } else {
        ldwork = 8 * n;
        if (ldwork < 1) ldwork = 1;
    }

    i32 lzwork_use = minwrk * 4;
    if (lzwork_use < 64) lzwork_use = 64;

    f64 *dwork = (f64*)malloc(ldwork * sizeof(f64));
    c128 *zwork = (c128*)PyMem_Calloc(lzwork_use, sizeof(c128));

    if (dwork == NULL || zwork == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate workspace");
        if (dwork) free(dwork);
        if (zwork) PyMem_Free(zwork);
        Py_DECREF(alpha_array);
        Py_DECREF(beta_array_out);
        Py_DECREF(a_array);
        Py_DECREF(e_array);
        Py_DECREF(q_array);
        Py_DECREF(z_array);
        Py_DECREF(b_array);
        Py_DECREF(b_work);
        return NULL;
    }

    sg03bz(dico, fact, trans, n, m, a_data, lda, e_data, lde,
           q_data, ldq, z_data, ldz, b_work_data, ldb, &scale,
           alpha, beta_arr, dwork, zwork, lzwork_use, &info);

    free(dwork);
    PyMem_Free(zwork);

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(e_array);
    PyArray_ResolveWritebackIfCopy(q_array);
    PyArray_ResolveWritebackIfCopy(z_array);

    npy_intp dims_u[2] = {n, n};
    npy_intp strides_u[2] = {sizeof(c128), ldb * sizeof(c128)};
    PyObject *u_array = PyArray_New(&PyArray_Type, 2, dims_u, NPY_CDOUBLE, strides_u,
                                     b_work_data, 0, NPY_ARRAY_FARRAY, NULL);
    if (u_array == NULL) {
        Py_DECREF(a_array);
        Py_DECREF(e_array);
        Py_DECREF(q_array);
        Py_DECREF(z_array);
        Py_DECREF(b_array);
        Py_DECREF(b_work);
        Py_DECREF(alpha_array);
        Py_DECREF(beta_array_out);
        return NULL;
    }
    if (PyArray_SetBaseObject((PyArrayObject*)u_array, (PyObject*)b_work) < 0) {
        Py_DECREF(u_array);
        Py_DECREF(a_array);
        Py_DECREF(e_array);
        Py_DECREF(q_array);
        Py_DECREF(z_array);
        Py_DECREF(b_array);
        Py_DECREF(b_work);
        Py_DECREF(alpha_array);
        Py_DECREF(beta_array_out);
        return NULL;
    }

    PyObject *result = Py_BuildValue("OdOOi", u_array, scale, alpha_array, beta_array_out, info);

    Py_DECREF(a_array);
    Py_DECREF(e_array);
    Py_DECREF(q_array);
    Py_DECREF(z_array);
    Py_DECREF(b_array);
    Py_DECREF(u_array);
    Py_DECREF(alpha_array);
    Py_DECREF(beta_array_out);

    return result;
}
