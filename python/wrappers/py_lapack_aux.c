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

PyObject* py_dlatzm(PyObject* self, PyObject* args) {
    const char *side_str;
    i32 m, n, incv;
    f64 tau;
    PyObject *v_obj, *c1_obj, *c2_obj;
    PyArrayObject *v_array = NULL, *c1_array = NULL, *c2_array = NULL;
    f64 *work = NULL;
    f64 *c_combined = NULL;

    if (!PyArg_ParseTuple(args, "siiOidOO", &side_str, &m, &n, &v_obj, &incv,
                          &tau, &c1_obj, &c2_obj)) {
        return NULL;
    }

    if (side_str == NULL || side_str[0] == '\0') {
        PyErr_SetString(PyExc_ValueError, "side must be a non-empty string");
        return NULL;
    }
    char side = toupper(side_str[0]);

    v_array = (PyArrayObject*)PyArray_FROM_OTF(v_obj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
    if (v_array == NULL) {
        goto cleanup;
    }

    c1_array = (PyArrayObject*)PyArray_FROM_OTF(c1_obj, NPY_DOUBLE,
                                                 NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (c1_array == NULL) {
        goto cleanup;
    }

    c2_array = (PyArrayObject*)PyArray_FROM_OTF(c2_obj, NPY_DOUBLE,
                                                 NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (c2_array == NULL) {
        goto cleanup;
    }

    f64 *v_data = (f64*)PyArray_DATA(v_array);
    f64 *c1_data = (f64*)PyArray_DATA(c1_array);
    f64 *c2_data = (f64*)PyArray_DATA(c2_array);

    i32 work_size = (side == 'L') ? n : m;
    if (work_size < 1) work_size = 1;

    work = (f64*)malloc(work_size * sizeof(f64));
    if (work == NULL) {
        PyErr_NoMemory();
        goto cleanup;
    }

    if (side == 'L') {
        i32 ldc = m > 1 ? m : 1;
        c_combined = (f64*)malloc(ldc * n * sizeof(f64));
        if (c_combined == NULL) {
            PyErr_NoMemory();
            goto cleanup;
        }

        for (i32 j = 0; j < n; j++) {
            c_combined[0 + j * ldc] = c1_data[j];
            for (i32 i = 0; i < m - 1; i++) {
                c_combined[(i + 1) + j * ldc] = c2_data[i + j * (m - 1)];
            }
        }

        slicot_dlatzm(&side, m, n, v_data, incv, tau, c_combined, c_combined + 1, ldc, work);

        for (i32 j = 0; j < n; j++) {
            c1_data[j] = c_combined[0 + j * ldc];
            for (i32 i = 0; i < m - 1; i++) {
                c2_data[i + j * (m - 1)] = c_combined[(i + 1) + j * ldc];
            }
        }

        free(c_combined);
        c_combined = NULL;
    } else {
        i32 ldc = m > 1 ? m : 1;
        c_combined = (f64*)malloc(ldc * n * sizeof(f64));
        if (c_combined == NULL) {
            PyErr_NoMemory();
            goto cleanup;
        }

        for (i32 i = 0; i < m; i++) {
            c_combined[i] = c1_data[i];
        }
        for (i32 j = 0; j < n - 1; j++) {
            for (i32 i = 0; i < m; i++) {
                c_combined[i + (j + 1) * ldc] = c2_data[i + j * m];
            }
        }

        slicot_dlatzm(&side, m, n, v_data, incv, tau, c_combined, c_combined + ldc, ldc, work);

        for (i32 i = 0; i < m; i++) {
            c1_data[i] = c_combined[i];
        }
        for (i32 j = 0; j < n - 1; j++) {
            for (i32 i = 0; i < m; i++) {
                c2_data[i + j * m] = c_combined[i + (j + 1) * ldc];
            }
        }

        free(c_combined);
        c_combined = NULL;
    }

    PyArray_ResolveWritebackIfCopy(c1_array);
    PyArray_ResolveWritebackIfCopy(c2_array);

    free(work);

    PyObject *result = Py_BuildValue("OO", c1_array, c2_array);
    Py_DECREF(v_array);
    Py_DECREF(c1_array);
    Py_DECREF(c2_array);
    return result;

cleanup:
    free(work);
    free(c_combined);
    Py_XDECREF(v_array);
    Py_XDECREF(c1_array);
    Py_XDECREF(c2_array);
    return NULL;
}

PyObject* py_dgegv(PyObject* self, PyObject* args) {
    const char *jobvl_str, *jobvr_str;
    PyObject *a_obj, *b_obj;
    i32 lwork = -2;
    PyArrayObject *a_array = NULL, *b_array = NULL;
    PyArrayObject *alphar_array = NULL, *alphai_array = NULL, *beta_array = NULL;
    PyArrayObject *vl_array = NULL, *vr_array = NULL;

    f64 *work = NULL;
    f64 *vl_data = NULL;
    f64 *vr_data = NULL;

    if (!PyArg_ParseTuple(args, "ssOO|i", &jobvl_str, &jobvr_str, &a_obj, &b_obj, &lwork)) {
        return NULL;
    }

    if (jobvl_str == NULL || jobvl_str[0] == '\0') {
        PyErr_SetString(PyExc_ValueError, "jobvl must be 'N' or 'V'");
        return NULL;
    }
    if (jobvr_str == NULL || jobvr_str[0] == '\0') {
        PyErr_SetString(PyExc_ValueError, "jobvr must be 'N' or 'V'");
        return NULL;
    }

    char jobvl = toupper(jobvl_str[0]);
    char jobvr = toupper(jobvr_str[0]);

    if (jobvl != 'N' && jobvl != 'V') {
        PyErr_SetString(PyExc_ValueError, "jobvl must be 'N' or 'V'");
        return NULL;
    }
    if (jobvr != 'N' && jobvr != 'V') {
        PyErr_SetString(PyExc_ValueError, "jobvr must be 'N' or 'V'");
        return NULL;
    }

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_DOUBLE,
                                                NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (a_array == NULL) goto cleanup;

    b_array = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_DOUBLE,
                                                NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (b_array == NULL) goto cleanup;

    if (PyArray_NDIM(a_array) != 2 || PyArray_NDIM(b_array) != 2) {
        PyErr_SetString(PyExc_ValueError, "A and B must be 2D arrays");
        goto cleanup;
    }

    npy_intp *a_dims = PyArray_DIMS(a_array);
    npy_intp *b_dims = PyArray_DIMS(b_array);

    if (a_dims[0] != a_dims[1]) {
        PyErr_SetString(PyExc_ValueError, "A must be square");
        goto cleanup;
    }
    if (b_dims[0] != b_dims[1] || b_dims[0] != a_dims[0]) {
        PyErr_SetString(PyExc_ValueError, "B must be square with same size as A");
        goto cleanup;
    }

    i32 n = (i32)a_dims[0];
    i32 lda = n > 0 ? n : 1;
    i32 ldb = n > 0 ? n : 1;
    i32 ldvl = (jobvl == 'V' && n > 0) ? n : 1;
    i32 ldvr = (jobvr == 'V' && n > 0) ? n : 1;

    f64 *a_data = (f64*)PyArray_DATA(a_array);
    f64 *b_data = (f64*)PyArray_DATA(b_array);

    npy_intp vec_dims[1] = {n};
    alphar_array = (PyArrayObject*)PyArray_SimpleNew(1, vec_dims, NPY_DOUBLE);
    alphai_array = (PyArrayObject*)PyArray_SimpleNew(1, vec_dims, NPY_DOUBLE);
    beta_array = (PyArrayObject*)PyArray_SimpleNew(1, vec_dims, NPY_DOUBLE);
    if (!alphar_array || !alphai_array || !beta_array) {
        PyErr_NoMemory();
        goto cleanup;
    }

    f64 *alphar = (f64*)PyArray_DATA(alphar_array);
    f64 *alphai = (f64*)PyArray_DATA(alphai_array);
    f64 *beta = (f64*)PyArray_DATA(beta_array);

    npy_intp vl_dims[2] = {n > 0 ? n : 1, n > 0 ? n : 1};
    npy_intp vl_strides[2] = {sizeof(f64), ldvl * sizeof(f64)};
    npy_intp vr_strides[2] = {sizeof(f64), ldvr * sizeof(f64)};



    if (jobvl == 'V' && n > 0) {
        vl_data = (f64*)malloc(ldvl * n * sizeof(f64));
        if (!vl_data) { PyErr_NoMemory(); goto cleanup; }
    } else {
        vl_data = (f64*)malloc(sizeof(f64));
        if (!vl_data) { PyErr_NoMemory(); goto cleanup; }
    }

    if (jobvr == 'V' && n > 0) {
        vr_data = (f64*)malloc(ldvr * n * sizeof(f64));
        if (!vr_data) { PyErr_NoMemory(); goto cleanup; }
    } else {
        vr_data = (f64*)malloc(sizeof(f64));
        if (!vr_data) { PyErr_NoMemory(); goto cleanup; }
    }

    i32 actual_lwork;
    if (lwork == -1) {
        actual_lwork = -1;
    } else if (lwork == -2 || lwork < 1) {
        actual_lwork = (n > 0) ? 8 * n : 1;
    } else {
        actual_lwork = lwork;
    }

    work = (f64*)malloc((actual_lwork > 0 ? actual_lwork : 1) * sizeof(f64));
    if (!work) {
        PyErr_NoMemory();
        goto cleanup;
    }

    char jobvl_char[2] = {jobvl, '\0'};
    char jobvr_char[2] = {jobvr, '\0'};

    i32 info = slicot_dgegv(jobvl_char, jobvr_char, n, a_data, lda, b_data, ldb,
                            alphar, alphai, beta, vl_data, ldvl, vr_data, ldvr,
                            work, actual_lwork);

    if (lwork == -1) {
        free(work);
        free(vl_data);
        free(vr_data);
        Py_DECREF(alphar_array);
        Py_DECREF(alphai_array);
        Py_DECREF(beta_array);
        PyArray_ResolveWritebackIfCopy(a_array);
        PyArray_ResolveWritebackIfCopy(b_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);

        npy_intp empty_dims[1] = {0};
        npy_intp empty_2d_dims[2] = {0, 0};
        PyObject *empty_ar = PyArray_SimpleNew(1, empty_dims, NPY_DOUBLE);
        PyObject *empty_vl = PyArray_SimpleNew(2, empty_2d_dims, NPY_DOUBLE);
        PyObject *empty_vr = PyArray_SimpleNew(2, empty_2d_dims, NPY_DOUBLE);
        return Py_BuildValue("OOOOOi", empty_ar, empty_ar, empty_ar, empty_vl, empty_vr, info);
    }

    free(work);
    work = NULL;

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(b_array);

    if (jobvl == 'V' && n > 0) {
        vl_array = (PyArrayObject*)PyArray_New(&PyArray_Type, 2, vl_dims, NPY_DOUBLE,
                                                vl_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        if (vl_array) {
            memcpy(PyArray_DATA(vl_array), vl_data, ldvl * n * sizeof(f64));
        }
        free(vl_data);
        vl_data = NULL;
    } else {
        free(vl_data);
        vl_data = NULL;
        npy_intp empty_vl_dims[2] = {0, 0};
        vl_array = (PyArrayObject*)PyArray_SimpleNew(2, empty_vl_dims, NPY_DOUBLE);
    }

    if (jobvr == 'V' && n > 0) {
        vr_array = (PyArrayObject*)PyArray_New(&PyArray_Type, 2, vl_dims, NPY_DOUBLE,
                                                vr_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        if (vr_array) {
            memcpy(PyArray_DATA(vr_array), vr_data, ldvr * n * sizeof(f64));
        }
        free(vr_data);
        vr_data = NULL;
    } else {
        free(vr_data);
        vr_data = NULL;
        npy_intp empty_vr_dims[2] = {0, 0};
        vr_array = (PyArrayObject*)PyArray_SimpleNew(2, empty_vr_dims, NPY_DOUBLE);
    }

    if (!vl_array || !vr_array) {
        PyErr_NoMemory();
        goto cleanup;
    }

    PyObject *result = Py_BuildValue("OOOOOi",
                                     alphar_array, alphai_array, beta_array,
                                     vl_array, vr_array, info);

    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(alphar_array);
    Py_DECREF(alphai_array);
    Py_DECREF(beta_array);
    Py_DECREF(vl_array);
    Py_DECREF(vr_array);

    return result;

cleanup:
    free(work);
    free(vl_data);
    free(vr_data);
    Py_XDECREF(a_array);
    Py_XDECREF(b_array);
    Py_XDECREF(alphar_array);
    Py_XDECREF(alphai_array);
    Py_XDECREF(beta_array);
    Py_XDECREF(vl_array);
    Py_XDECREF(vr_array);
    return NULL;
}

PyObject* py_zgegs(PyObject* self, PyObject* args) {
    const char *jobvsl_str, *jobvsr_str;
    PyObject *a_obj, *b_obj;
    i32 lwork_arg = -2;
    PyArrayObject *a_array = NULL, *b_array = NULL;
    PyArrayObject *s_array = NULL, *t_array = NULL;
    PyArrayObject *alpha_array = NULL, *beta_array = NULL;
    PyArrayObject *vsl_array = NULL, *vsr_array = NULL;

    c128 *work = NULL;
    f64 *rwork = NULL;
    c128 *vsl_data = NULL;
    c128 *vsr_data = NULL;

    if (!PyArg_ParseTuple(args, "ssOO|i", &jobvsl_str, &jobvsr_str, &a_obj, &b_obj, &lwork_arg)) {
        return NULL;
    }

    if (jobvsl_str == NULL || jobvsl_str[0] == '\0') {
        PyErr_SetString(PyExc_ValueError, "jobvsl must be 'N' or 'V'");
        return NULL;
    }
    if (jobvsr_str == NULL || jobvsr_str[0] == '\0') {
        PyErr_SetString(PyExc_ValueError, "jobvsr must be 'N' or 'V'");
        return NULL;
    }

    char jobvsl = toupper(jobvsl_str[0]);
    char jobvsr = toupper(jobvsr_str[0]);

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_COMPLEX128,
                                                NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (a_array == NULL) goto cleanup;

    b_array = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_COMPLEX128,
                                                NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (b_array == NULL) goto cleanup;

    if (PyArray_NDIM(a_array) != 2 || PyArray_NDIM(b_array) != 2) {
        PyErr_SetString(PyExc_ValueError, "A and B must be 2D arrays");
        goto cleanup;
    }

    npy_intp *a_dims = PyArray_DIMS(a_array);
    npy_intp *b_dims = PyArray_DIMS(b_array);

    if (a_dims[0] != a_dims[1]) {
        PyErr_SetString(PyExc_ValueError, "A must be square");
        goto cleanup;
    }
    if (b_dims[0] != b_dims[1] || b_dims[0] != a_dims[0]) {
        PyErr_SetString(PyExc_ValueError, "B must be square with same size as A");
        goto cleanup;
    }

    i32 n = (i32)a_dims[0];
    i32 lda = n > 0 ? n : 1;
    i32 ldb = n > 0 ? n : 1;
    i32 ldvsl = (jobvsl == 'V' && n > 0) ? n : 1;
    i32 ldvsr = (jobvsr == 'V' && n > 0) ? n : 1;

    c128 *a_data = (c128*)PyArray_DATA(a_array);
    c128 *b_data = (c128*)PyArray_DATA(b_array);

    npy_intp vec_dims[1] = {n};
    alpha_array = (PyArrayObject*)PyArray_SimpleNew(1, vec_dims, NPY_COMPLEX128);
    beta_array = (PyArrayObject*)PyArray_SimpleNew(1, vec_dims, NPY_COMPLEX128);
    if (!alpha_array || !beta_array) {
        PyErr_NoMemory();
        goto cleanup;
    }

    c128 *alpha = (c128*)PyArray_DATA(alpha_array);
    c128 *beta = (c128*)PyArray_DATA(beta_array);

    if (jobvsl == 'V' && n > 0) {
        vsl_data = (c128*)malloc(ldvsl * n * sizeof(c128));
        if (!vsl_data) { PyErr_NoMemory(); goto cleanup; }
    } else {
        vsl_data = (c128*)malloc(sizeof(c128));
        if (!vsl_data) { PyErr_NoMemory(); goto cleanup; }
    }

    if (jobvsr == 'V' && n > 0) {
        vsr_data = (c128*)malloc(ldvsr * n * sizeof(c128));
        if (!vsr_data) { PyErr_NoMemory(); goto cleanup; }
    } else {
        vsr_data = (c128*)malloc(sizeof(c128));
        if (!vsr_data) { PyErr_NoMemory(); goto cleanup; }
    }

    i32 lwork;
    if (lwork_arg == -1) {
        lwork = -1;
    } else if (lwork_arg == -2 || lwork_arg < 1) {
        lwork = (n > 0) ? 2 * n : 1;
    } else {
        lwork = lwork_arg;
    }

    work = (c128*)malloc((lwork > 0 ? lwork : 1) * sizeof(c128));
    if (!work) {
        PyErr_NoMemory();
        goto cleanup;
    }

    i32 rwork_size = (n > 0) ? 3 * n : 1;
    rwork = (f64*)malloc(rwork_size * sizeof(f64));
    if (!rwork) {
        PyErr_NoMemory();
        goto cleanup;
    }

    char jobvsl_char[2] = {jobvsl, '\0'};
    char jobvsr_char[2] = {jobvsr, '\0'};

    i32 info = slicot_zgegs(jobvsl_char, jobvsr_char, n, a_data, lda, b_data, ldb,
                            alpha, beta, vsl_data, ldvsl, vsr_data, ldvsr,
                            work, lwork, rwork);

    if (lwork_arg == -1) {
        free(work);
        free(rwork);
        free(vsl_data);
        free(vsr_data);
        Py_DECREF(alpha_array);
        Py_DECREF(beta_array);
        PyArray_ResolveWritebackIfCopy(a_array);
        PyArray_ResolveWritebackIfCopy(b_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);

        npy_intp empty_dims[1] = {0};
        npy_intp empty_2d_dims[2] = {0, 0};
        PyObject *empty_ar = PyArray_SimpleNew(1, empty_dims, NPY_COMPLEX128);
        PyObject *empty_2d = PyArray_SimpleNew(2, empty_2d_dims, NPY_COMPLEX128);
        return Py_BuildValue("OOOOOOOi", empty_2d, empty_2d, empty_ar, empty_ar, empty_2d, empty_2d, empty_2d, info);
    }

    free(work);
    work = NULL;
    free(rwork);
    rwork = NULL;

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(b_array);

    npy_intp mat_dims[2] = {n > 0 ? n : 1, n > 0 ? n : 1};
    npy_intp mat_strides[2] = {sizeof(c128), lda * sizeof(c128)};

    s_array = (PyArrayObject*)PyArray_New(&PyArray_Type, 2, mat_dims, NPY_COMPLEX128,
                                          mat_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (s_array && n > 0) {
        memcpy(PyArray_DATA(s_array), a_data, lda * n * sizeof(c128));
    }

    t_array = (PyArrayObject*)PyArray_New(&PyArray_Type, 2, mat_dims, NPY_COMPLEX128,
                                          mat_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (t_array && n > 0) {
        memcpy(PyArray_DATA(t_array), b_data, ldb * n * sizeof(c128));
    }

    npy_intp vsl_strides[2] = {sizeof(c128), ldvsl * sizeof(c128)};
    npy_intp vsr_strides[2] = {sizeof(c128), ldvsr * sizeof(c128)};

    if (jobvsl == 'V' && n > 0) {
        vsl_array = (PyArrayObject*)PyArray_New(&PyArray_Type, 2, mat_dims, NPY_COMPLEX128,
                                                vsl_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        if (vsl_array) {
            memcpy(PyArray_DATA(vsl_array), vsl_data, ldvsl * n * sizeof(c128));
        }
        free(vsl_data);
        vsl_data = NULL;
    } else {
        free(vsl_data);
        vsl_data = NULL;
        npy_intp empty_vsl_dims[2] = {0, 0};
        vsl_array = (PyArrayObject*)PyArray_SimpleNew(2, empty_vsl_dims, NPY_COMPLEX128);
    }

    if (jobvsr == 'V' && n > 0) {
        vsr_array = (PyArrayObject*)PyArray_New(&PyArray_Type, 2, mat_dims, NPY_COMPLEX128,
                                                vsr_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        if (vsr_array) {
            memcpy(PyArray_DATA(vsr_array), vsr_data, ldvsr * n * sizeof(c128));
        }
        free(vsr_data);
        vsr_data = NULL;
    } else {
        free(vsr_data);
        vsr_data = NULL;
        npy_intp empty_vsr_dims[2] = {0, 0};
        vsr_array = (PyArrayObject*)PyArray_SimpleNew(2, empty_vsr_dims, NPY_COMPLEX128);
    }

    if (!s_array || !t_array || !vsl_array || !vsr_array) {
        PyErr_NoMemory();
        Py_XDECREF(s_array);
        Py_XDECREF(t_array);
        Py_XDECREF(vsl_array);
        Py_XDECREF(vsr_array);
        goto cleanup;
    }

    PyObject *result = Py_BuildValue("OOOOOOi",
                                     s_array, t_array,
                                     alpha_array, beta_array,
                                     vsl_array, vsr_array, info);

    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(alpha_array);
    Py_DECREF(beta_array);
    Py_DECREF(s_array);
    Py_DECREF(t_array);
    Py_DECREF(vsl_array);
    Py_DECREF(vsr_array);

    return result;

cleanup:
    free(work);
    free(rwork);
    free(vsl_data);
    free(vsr_data);
    Py_XDECREF(a_array);
    Py_XDECREF(b_array);
    Py_XDECREF(alpha_array);
    Py_XDECREF(beta_array);
    Py_XDECREF(s_array);
    Py_XDECREF(t_array);
    Py_XDECREF(vsl_array);
    Py_XDECREF(vsr_array);
    return NULL;
}

PyObject* py_zgegv(PyObject* self, PyObject* args) {
    const char *jobvl_str, *jobvr_str;
    PyObject *a_obj, *b_obj;
    i32 lwork_arg = -2;
    PyArrayObject *a_array = NULL, *b_array = NULL;
    PyArrayObject *alpha_array = NULL, *beta_array = NULL;
    PyArrayObject *vl_array = NULL, *vr_array = NULL;

    c128 *work = NULL;
    f64 *rwork = NULL;
    c128 *vl_data = NULL;
    c128 *vr_data = NULL;

    if (!PyArg_ParseTuple(args, "ssOO|i", &jobvl_str, &jobvr_str, &a_obj, &b_obj, &lwork_arg)) {
        return NULL;
    }

    if (jobvl_str == NULL || jobvl_str[0] == '\0') {
        PyErr_SetString(PyExc_ValueError, "jobvl must be 'N' or 'V'");
        return NULL;
    }
    if (jobvr_str == NULL || jobvr_str[0] == '\0') {
        PyErr_SetString(PyExc_ValueError, "jobvr must be 'N' or 'V'");
        return NULL;
    }

    char jobvl = toupper(jobvl_str[0]);
    char jobvr = toupper(jobvr_str[0]);

    a_array = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_COMPLEX128,
                                                NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (a_array == NULL) goto cleanup;

    b_array = (PyArrayObject*)PyArray_FROM_OTF(b_obj, NPY_COMPLEX128,
                                                NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (b_array == NULL) goto cleanup;

    if (PyArray_NDIM(a_array) != 2 || PyArray_NDIM(b_array) != 2) {
        PyErr_SetString(PyExc_ValueError, "A and B must be 2D arrays");
        goto cleanup;
    }

    npy_intp *a_dims = PyArray_DIMS(a_array);
    npy_intp *b_dims = PyArray_DIMS(b_array);

    if (a_dims[0] != a_dims[1]) {
        PyErr_SetString(PyExc_ValueError, "A must be square");
        goto cleanup;
    }
    if (b_dims[0] != b_dims[1] || b_dims[0] != a_dims[0]) {
        PyErr_SetString(PyExc_ValueError, "B must be square with same size as A");
        goto cleanup;
    }

    i32 n = (i32)a_dims[0];
    i32 lda = n > 0 ? n : 1;
    i32 ldb = n > 0 ? n : 1;
    i32 ldvl = (jobvl == 'V' && n > 0) ? n : 1;
    i32 ldvr = (jobvr == 'V' && n > 0) ? n : 1;

    c128 *a_data = (c128*)PyArray_DATA(a_array);
    c128 *b_data = (c128*)PyArray_DATA(b_array);

    npy_intp vec_dims[1] = {n};
    alpha_array = (PyArrayObject*)PyArray_SimpleNew(1, vec_dims, NPY_COMPLEX128);
    beta_array = (PyArrayObject*)PyArray_SimpleNew(1, vec_dims, NPY_COMPLEX128);
    if (!alpha_array || !beta_array) {
        PyErr_NoMemory();
        goto cleanup;
    }

    c128 *alpha = (c128*)PyArray_DATA(alpha_array);
    c128 *beta = (c128*)PyArray_DATA(beta_array);

    if (jobvl == 'V' && n > 0) {
        vl_data = (c128*)malloc(ldvl * n * sizeof(c128));
        if (!vl_data) { PyErr_NoMemory(); goto cleanup; }
    } else {
        vl_data = (c128*)malloc(sizeof(c128));
        if (!vl_data) { PyErr_NoMemory(); goto cleanup; }
    }

    if (jobvr == 'V' && n > 0) {
        vr_data = (c128*)malloc(ldvr * n * sizeof(c128));
        if (!vr_data) { PyErr_NoMemory(); goto cleanup; }
    } else {
        vr_data = (c128*)malloc(sizeof(c128));
        if (!vr_data) { PyErr_NoMemory(); goto cleanup; }
    }

    i32 lwork;
    if (lwork_arg == -1) {
        lwork = -1;
    } else if (lwork_arg == -2 || lwork_arg < 1) {
        lwork = (n > 0) ? 2 * n : 1;
    } else {
        lwork = lwork_arg;
    }

    work = (c128*)malloc((lwork > 0 ? lwork : 1) * sizeof(c128));
    if (!work) {
        PyErr_NoMemory();
        goto cleanup;
    }

    i32 rwork_size = (n > 0) ? 8 * n : 1;
    rwork = (f64*)malloc(rwork_size * sizeof(f64));
    if (!rwork) {
        PyErr_NoMemory();
        goto cleanup;
    }

    char jobvl_char[2] = {jobvl, '\0'};
    char jobvr_char[2] = {jobvr, '\0'};

    i32 info = slicot_zgegv(jobvl_char, jobvr_char, n, a_data, lda, b_data, ldb,
                            alpha, beta, vl_data, ldvl, vr_data, ldvr,
                            work, lwork, rwork);

    if (lwork_arg == -1) {
        free(work);
        free(rwork);
        free(vl_data);
        free(vr_data);
        Py_DECREF(alpha_array);
        Py_DECREF(beta_array);
        PyArray_ResolveWritebackIfCopy(a_array);
        PyArray_ResolveWritebackIfCopy(b_array);
        Py_DECREF(a_array);
        Py_DECREF(b_array);

        npy_intp empty_dims[1] = {0};
        npy_intp empty_2d_dims[2] = {0, 0};
        PyObject *empty_ar = PyArray_SimpleNew(1, empty_dims, NPY_COMPLEX128);
        PyObject *empty_2d = PyArray_SimpleNew(2, empty_2d_dims, NPY_COMPLEX128);
        return Py_BuildValue("OOOOi", empty_ar, empty_ar, empty_2d, empty_2d, info);
    }

    free(work);
    work = NULL;
    free(rwork);
    rwork = NULL;

    PyArray_ResolveWritebackIfCopy(a_array);
    PyArray_ResolveWritebackIfCopy(b_array);

    npy_intp mat_dims[2] = {n > 0 ? n : 1, n > 0 ? n : 1};
    npy_intp vl_strides[2] = {sizeof(c128), ldvl * sizeof(c128)};
    npy_intp vr_strides[2] = {sizeof(c128), ldvr * sizeof(c128)};

    if (jobvl == 'V' && n > 0) {
        vl_array = (PyArrayObject*)PyArray_New(&PyArray_Type, 2, mat_dims, NPY_COMPLEX128,
                                                vl_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        if (vl_array) {
            memcpy(PyArray_DATA(vl_array), vl_data, ldvl * n * sizeof(c128));
        }
        free(vl_data);
        vl_data = NULL;
    } else {
        free(vl_data);
        vl_data = NULL;
        npy_intp empty_vl_dims[2] = {0, 0};
        vl_array = (PyArrayObject*)PyArray_SimpleNew(2, empty_vl_dims, NPY_COMPLEX128);
    }

    if (jobvr == 'V' && n > 0) {
        vr_array = (PyArrayObject*)PyArray_New(&PyArray_Type, 2, mat_dims, NPY_COMPLEX128,
                                                vr_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
        if (vr_array) {
            memcpy(PyArray_DATA(vr_array), vr_data, ldvr * n * sizeof(c128));
        }
        free(vr_data);
        vr_data = NULL;
    } else {
        free(vr_data);
        vr_data = NULL;
        npy_intp empty_vr_dims[2] = {0, 0};
        vr_array = (PyArrayObject*)PyArray_SimpleNew(2, empty_vr_dims, NPY_COMPLEX128);
    }

    if (!vl_array || !vr_array) {
        PyErr_NoMemory();
        Py_XDECREF(vl_array);
        Py_XDECREF(vr_array);
        goto cleanup;
    }

    PyObject *result = Py_BuildValue("OOOOi",
                                     alpha_array, beta_array,
                                     vl_array, vr_array, info);

    Py_DECREF(a_array);
    Py_DECREF(b_array);
    Py_DECREF(alpha_array);
    Py_DECREF(beta_array);
    Py_DECREF(vl_array);
    Py_DECREF(vr_array);

    return result;

cleanup:
    free(work);
    free(rwork);
    free(vl_data);
    free(vr_data);
    Py_XDECREF(a_array);
    Py_XDECREF(b_array);
    Py_XDECREF(alpha_array);
    Py_XDECREF(beta_array);
    Py_XDECREF(vl_array);
    Py_XDECREF(vr_array);
    return NULL;
}
