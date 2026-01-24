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


/* Python wrapper for ag07bd */
PyObject* py_ag07bd(PyObject* self, PyObject* args) {
    const char *jobe;
    i32 n, m;
    PyObject *a_obj, *b_obj, *c_obj, *d_obj;
    PyObject *e_obj = NULL;

    if (!PyArg_ParseTuple(args, "siiOOOO|O",
            &jobe, &n, &m, &a_obj, &b_obj, &c_obj, &d_obj, &e_obj)) {
        return NULL;
    }

    bool unite = (jobe[0] == 'I' || jobe[0] == 'i');
    bool general = (jobe[0] == 'G' || jobe[0] == 'g');
    i32 nm = n + m;

    if (n < 0 || m < 0 || (!unite && !general)) {
        i32 info = 0;
        f64 dummy = 0.0;
        i32 one = 1;
        ag07bd(jobe, n, m, &dummy, one, &dummy, one, &dummy, one,
               &dummy, one, &dummy, one, &dummy, one, &dummy, one,
               &dummy, one, &dummy, one, &dummy, one, &info);
        return Py_BuildValue("(OOOOOi)", Py_None, Py_None, Py_None, Py_None, Py_None, info);
    }

    PyArrayObject *a_array = NULL, *e_array = NULL, *b_array = NULL;
    PyArrayObject *c_array = NULL, *d_array = NULL;

    if (n > 0) {
        a_array = (PyArrayObject*)PyArray_FROM_OTF(
            a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY_RO);
        if (!a_array) return NULL;
    }

    if (general && n > 0) {
        if (!e_obj) {
            PyErr_SetString(PyExc_ValueError, "E matrix required when jobe='G'");
            Py_XDECREF(a_array);
            return NULL;
        }
        e_array = (PyArrayObject*)PyArray_FROM_OTF(
            e_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY_RO);
        if (!e_array) {
            Py_XDECREF(a_array);
            return NULL;
        }
    }

    if (n > 0 && m > 0) {
        b_array = (PyArrayObject*)PyArray_FROM_OTF(
            b_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY_RO);
        if (!b_array) {
            Py_XDECREF(a_array);
            Py_XDECREF(e_array);
            return NULL;
        }
    }

    if (m > 0) {
        c_array = (PyArrayObject*)PyArray_FROM_OTF(
            c_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY_RO);
        if (!c_array) {
            Py_XDECREF(a_array);
            Py_XDECREF(e_array);
            Py_XDECREF(b_array);
            return NULL;
        }

        d_array = (PyArrayObject*)PyArray_FROM_OTF(
            d_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY_RO);
        if (!d_array) {
            Py_XDECREF(a_array);
            Py_XDECREF(e_array);
            Py_XDECREF(b_array);
            Py_XDECREF(c_array);
            return NULL;
        }
    }

    i32 lda = (n > 0) ? n : 1;
    i32 lde = unite ? 1 : ((n > 0) ? n : 1);
    i32 ldb = (n > 0) ? n : 1;
    i32 ldc = (m > 0) ? m : 1;
    i32 ldd = (m > 0) ? m : 1;
    i32 ldai = (nm > 0) ? nm : 1;
    i32 ldei = (nm > 0) ? nm : 1;
    i32 ldbi = (nm > 0) ? nm : 1;
    i32 ldci = (m > 0) ? m : 1;
    i32 lddi = (m > 0) ? m : 1;

    npy_intp ai_dims[2] = {nm, nm};
    npy_intp ai_strides[2] = {sizeof(f64), ldai * sizeof(f64)};
    PyObject *ai_array = PyArray_New(&PyArray_Type, 2, ai_dims, NPY_DOUBLE,
                                     ai_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (!ai_array) {
        Py_XDECREF(a_array); Py_XDECREF(e_array); Py_XDECREF(b_array);
        Py_XDECREF(c_array); Py_XDECREF(d_array);
        return NULL;
    }

    npy_intp ei_dims[2] = {nm, nm};
    npy_intp ei_strides[2] = {sizeof(f64), ldei * sizeof(f64)};
    PyObject *ei_array = PyArray_New(&PyArray_Type, 2, ei_dims, NPY_DOUBLE,
                                     ei_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (!ei_array) {
        Py_DECREF(ai_array);
        Py_XDECREF(a_array); Py_XDECREF(e_array); Py_XDECREF(b_array);
        Py_XDECREF(c_array); Py_XDECREF(d_array);
        return NULL;
    }

    npy_intp bi_dims[2] = {nm, m};
    npy_intp bi_strides[2] = {sizeof(f64), ldbi * sizeof(f64)};
    PyObject *bi_array = PyArray_New(&PyArray_Type, 2, bi_dims, NPY_DOUBLE,
                                     bi_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (!bi_array) {
        Py_DECREF(ai_array); Py_DECREF(ei_array);
        Py_XDECREF(a_array); Py_XDECREF(e_array); Py_XDECREF(b_array);
        Py_XDECREF(c_array); Py_XDECREF(d_array);
        return NULL;
    }

    npy_intp ci_dims[2] = {m, nm};
    npy_intp ci_strides[2] = {sizeof(f64), ldci * sizeof(f64)};
    PyObject *ci_array = PyArray_New(&PyArray_Type, 2, ci_dims, NPY_DOUBLE,
                                     ci_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (!ci_array) {
        Py_DECREF(ai_array); Py_DECREF(ei_array); Py_DECREF(bi_array);
        Py_XDECREF(a_array); Py_XDECREF(e_array); Py_XDECREF(b_array);
        Py_XDECREF(c_array); Py_XDECREF(d_array);
        return NULL;
    }

    npy_intp di_dims[2] = {m, m};
    npy_intp di_strides[2] = {sizeof(f64), lddi * sizeof(f64)};
    PyObject *di_array = PyArray_New(&PyArray_Type, 2, di_dims, NPY_DOUBLE,
                                     di_strides, NULL, 0, NPY_ARRAY_FARRAY, NULL);
    if (!di_array) {
        Py_DECREF(ai_array); Py_DECREF(ei_array); Py_DECREF(bi_array);
        Py_DECREF(ci_array);
        Py_XDECREF(a_array); Py_XDECREF(e_array); Py_XDECREF(b_array);
        Py_XDECREF(c_array); Py_XDECREF(d_array);
        return NULL;
    }

    f64 *ai = (f64*)PyArray_DATA((PyArrayObject*)ai_array);
    f64 *ei = (f64*)PyArray_DATA((PyArrayObject*)ei_array);
    f64 *bi = (f64*)PyArray_DATA((PyArrayObject*)bi_array);
    f64 *ci = (f64*)PyArray_DATA((PyArrayObject*)ci_array);
    f64 *di = (f64*)PyArray_DATA((PyArrayObject*)di_array);

    if (nm > 0) memset(ai, 0, ldai * nm * sizeof(f64));
    if (nm > 0) memset(ei, 0, ldei * nm * sizeof(f64));
    if (m > 0) {
        memset(bi, 0, ldbi * m * sizeof(f64));
        memset(ci, 0, ldci * nm * sizeof(f64));
        memset(di, 0, lddi * m * sizeof(f64));
    }

    f64 dummy = 0.0;
    const f64 *a_data = a_array ? (const f64*)PyArray_DATA(a_array) : &dummy;
    const f64 *e_data = e_array ? (const f64*)PyArray_DATA(e_array) : &dummy;
    const f64 *b_data = b_array ? (const f64*)PyArray_DATA(b_array) : &dummy;
    const f64 *c_data = c_array ? (const f64*)PyArray_DATA(c_array) : &dummy;
    const f64 *d_data = d_array ? (const f64*)PyArray_DATA(d_array) : &dummy;

    i32 info = 0;

    ag07bd(jobe, n, m, a_data, lda, e_data, lde, b_data, ldb,
           c_data, ldc, d_data, ldd, ai, ldai, ei, ldei,
           bi, ldbi, ci, ldci, di, lddi, &info);

    Py_XDECREF(a_array);
    Py_XDECREF(e_array);
    Py_XDECREF(b_array);
    Py_XDECREF(c_array);
    Py_XDECREF(d_array);

    if (info != 0) {
        Py_DECREF(ai_array); Py_DECREF(ei_array); Py_DECREF(bi_array);
        Py_DECREF(ci_array); Py_DECREF(di_array);
        return Py_BuildValue("(OOOOOi)", Py_None, Py_None, Py_None, Py_None, Py_None, info);
    }

    PyObject *result = Py_BuildValue("(OOOOOi)",
        ai_array, ei_array, bi_array, ci_array, di_array, info);

    Py_DECREF(ai_array);
    Py_DECREF(ei_array);
    Py_DECREF(bi_array);
    Py_DECREF(ci_array);
    Py_DECREF(di_array);

    return result;
}


/* Python wrapper for ag08bd */
PyObject* py_ag08bd(PyObject* self, PyObject* args) {
    const char *equil;
    i32 l, n, m, p;
    f64 tol;
    PyObject *a_obj, *e_obj, *b_obj, *c_obj, *d_obj;

    if (!PyArg_ParseTuple(args, "siiiiOOOOOd",
            &equil, &l, &n, &m, &p, &a_obj, &e_obj, &b_obj, &c_obj, &d_obj, &tol)) {
        return NULL;
    }

    if (l < 0) {
        PyErr_SetString(PyExc_ValueError, "l must be non-negative");
        return NULL;
    }
    if (n < 0) {
        PyErr_SetString(PyExc_ValueError, "n must be non-negative");
        return NULL;
    }
    if (m < 0) {
        PyErr_SetString(PyExc_ValueError, "m must be non-negative");
        return NULL;
    }
    if (p < 0) {
        PyErr_SetString(PyExc_ValueError, "p must be non-negative");
        return NULL;
    }

    PyArrayObject *a_array = NULL, *e_array = NULL, *b_array = NULL;
    PyArrayObject *c_array = NULL, *d_array = NULL;

    if (l > 0 && n > 0) {
        a_array = (PyArrayObject*)PyArray_FROM_OTF(
            a_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
        if (!a_array) return NULL;

        e_array = (PyArrayObject*)PyArray_FROM_OTF(
            e_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
        if (!e_array) {
            Py_DECREF(a_array);
            return NULL;
        }
    }

    if (l > 0 && m > 0) {
        b_array = (PyArrayObject*)PyArray_FROM_OTF(
            b_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
        if (!b_array) {
            Py_XDECREF(a_array);
            Py_XDECREF(e_array);
            return NULL;
        }
    }

    if (p > 0 && n > 0) {
        c_array = (PyArrayObject*)PyArray_FROM_OTF(
            c_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
        if (!c_array) {
            Py_XDECREF(a_array);
            Py_XDECREF(e_array);
            Py_XDECREF(b_array);
            return NULL;
        }
    }

    if (p > 0 && m > 0) {
        d_array = (PyArrayObject*)PyArray_FROM_OTF(
            d_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
        if (!d_array) {
            Py_XDECREF(a_array);
            Py_XDECREF(e_array);
            Py_XDECREF(b_array);
            Py_XDECREF(c_array);
            return NULL;
        }
    }

    i32 lda = (l > 0) ? l : 1;
    i32 lde = (l > 0) ? l : 1;
    i32 ldb = (l > 0 && m > 0) ? l : 1;
    i32 ldc = (p > 0) ? p : 1;
    i32 ldd = (p > 0) ? p : 1;

    i32 ldabcd = (l + p > n + m) ? (l + p) : (n + m);
    if (ldabcd < 1) ldabcd = 1;
    i32 labcd2 = ldabcd * (n + m);

    bool lequil = (equil[0] == 'S' || equil[0] == 's');
    i32 ldwork = labcd2 + (5 * ldabcd > 1 ? 5 * ldabcd : 1);
    if (lequil) {
        i32 equil_work = 4 * (l + n);
        if (equil_work > ldwork) ldwork = equil_work;
    }

    i32 *infz = (i32*)calloc(n + 1 > 1 ? n + 1 : 1, sizeof(i32));
    i32 *kronr = (i32*)calloc(n + m + 1 > 1 ? n + m + 1 : 1, sizeof(i32));
    i32 *infe = (i32*)calloc(1 + (l + p < n + m ? l + p : n + m), sizeof(i32));
    i32 *kronl = (i32*)calloc(l + p + 1 > 1 ? l + p + 1 : 1, sizeof(i32));
    i32 *iwork = (i32*)malloc((n + (m > 1 ? m : 1)) * sizeof(i32));
    f64 *dwork = (f64*)malloc(ldwork * sizeof(f64));

    if (!infz || !kronr || !infe || !kronl || !iwork || !dwork) {
        free(infz); free(kronr); free(infe); free(kronl);
        free(iwork); free(dwork);
        Py_XDECREF(a_array);
        Py_XDECREF(e_array);
        Py_XDECREF(b_array);
        Py_XDECREF(c_array);
        Py_XDECREF(d_array);
        PyErr_NoMemory();
        return NULL;
    }

    f64 dummy = 0.0;
    f64 *a_data = a_array ? (f64*)PyArray_DATA(a_array) : &dummy;
    f64 *e_data = e_array ? (f64*)PyArray_DATA(e_array) : &dummy;
    f64 *b_data = b_array ? (f64*)PyArray_DATA(b_array) : &dummy;
    f64 *c_data = c_array ? (f64*)PyArray_DATA(c_array) : &dummy;
    f64 *d_data = d_array ? (f64*)PyArray_DATA(d_array) : &dummy;

    i32 nfz = 0, nrank = 0, niz = 0, dinfz = 0;
    i32 nkror = 0, ninfe = 0, nkrol = 0, info = 0;

    ag08bd(equil, l, n, m, p, a_data, lda, e_data, lde,
           b_data, ldb, c_data, ldc, d_data, ldd,
           &nfz, &nrank, &niz, &dinfz, &nkror, &ninfe, &nkrol,
           infz, kronr, infe, kronl, tol, iwork, dwork, ldwork, &info);

    free(iwork);
    free(dwork);

    if (a_array) PyArray_ResolveWritebackIfCopy(a_array);
    if (e_array) PyArray_ResolveWritebackIfCopy(e_array);
    if (b_array) PyArray_ResolveWritebackIfCopy(b_array);
    if (c_array) PyArray_ResolveWritebackIfCopy(c_array);
    if (d_array) PyArray_ResolveWritebackIfCopy(d_array);

    npy_intp infz_dims[1] = {dinfz > 0 ? dinfz : 0};
    npy_intp kronr_dims[1] = {nkror > 0 ? nkror : 0};
    npy_intp infe_dims[1] = {ninfe > 0 ? ninfe : 0};
    npy_intp kronl_dims[1] = {nkrol > 0 ? nkrol : 0};

    PyObject *infz_array = PyArray_SimpleNew(1, infz_dims, NPY_INT32);
    PyObject *kronr_array = PyArray_SimpleNew(1, kronr_dims, NPY_INT32);
    PyObject *infe_array = PyArray_SimpleNew(1, infe_dims, NPY_INT32);
    PyObject *kronl_array = PyArray_SimpleNew(1, kronl_dims, NPY_INT32);

    if (dinfz > 0) {
        memcpy(PyArray_DATA((PyArrayObject*)infz_array), infz, dinfz * sizeof(i32));
    }
    if (nkror > 0) {
        memcpy(PyArray_DATA((PyArrayObject*)kronr_array), kronr, nkror * sizeof(i32));
    }
    if (ninfe > 0) {
        memcpy(PyArray_DATA((PyArrayObject*)infe_array), infe, ninfe * sizeof(i32));
    }
    if (nkrol > 0) {
        memcpy(PyArray_DATA((PyArrayObject*)kronl_array), kronl, nkrol * sizeof(i32));
    }

    free(infz);
    free(kronr);
    free(infe);
    free(kronl);

    PyObject *result = Py_BuildValue("(OOiiiiiiiOOOOi)",
        a_array ? (PyObject*)a_array : Py_None,
        e_array ? (PyObject*)e_array : Py_None,
        nfz, nrank, niz, dinfz, nkror, ninfe, nkrol,
        infz_array, kronr_array, infe_array, kronl_array, info);

    Py_XDECREF(a_array);
    Py_XDECREF(e_array);
    Py_XDECREF(b_array);
    Py_XDECREF(c_array);
    Py_XDECREF(d_array);
    Py_DECREF(infz_array);
    Py_DECREF(kronr_array);
    Py_DECREF(infe_array);
    Py_DECREF(kronl_array);

    return result;
}



/* Python wrapper for ag08by */
PyObject* py_ag08by(PyObject* self, PyObject* args) {
    int first_int;
    i32 n, m, p;
    f64 svlmax, tol;
    PyObject *abcd_obj, *e_obj;

    if (!PyArg_ParseTuple(args, "piiidOOd",
            &first_int, &n, &m, &p, &svlmax, &abcd_obj, &e_obj, &tol)) {
        return NULL;
    }

    bool first = (first_int != 0);

    if (n < 0) {
        PyErr_SetString(PyExc_ValueError, "n must be non-negative");
        return NULL;
    }
    if (m < 0) {
        PyErr_SetString(PyExc_ValueError, "m must be non-negative");
        return NULL;
    }
    if (p < 0) {
        PyErr_SetString(PyExc_ValueError, "p must be non-negative");
        return NULL;
    }
    if (!first && m > p) {
        PyErr_SetString(PyExc_ValueError, "m must be <= p when first=False");
        return NULL;
    }
    if (svlmax < 0) {
        PyErr_SetString(PyExc_ValueError, "svlmax must be non-negative");
        return NULL;
    }

    i32 pn = p + n;
    i32 mn = m + n;
    i32 mpm = (p < m) ? p : m;

    PyArrayObject *abcd_array = (PyArrayObject*)PyArray_FROM_OTF(
        abcd_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!abcd_array) return NULL;

    PyArrayObject *e_array = NULL;
    if (n > 0) {
        e_array = (PyArrayObject*)PyArray_FROM_OTF(
            e_obj, NPY_DOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
        if (!e_array) {
            Py_DECREF(abcd_array);
            return NULL;
        }
    }

    i32 ldabcd = pn > 0 ? pn : 1;
    i32 lde = n > 0 ? n : 1;

    i32 ldwork = 1;
    if (p > 0) {
        ldwork = 5 * p;
        if (m > 0) {
            ldwork = (ldwork > (mn - 1)) ? ldwork : (mn - 1);
            if (first) {
                i32 t1 = mpm + ((3*m - 1 > n) ? (3*m - 1) : n);
                ldwork = (ldwork > t1) ? ldwork : t1;
            }
        }
    }
    ldwork = (ldwork > 1) ? ldwork : 1;

    i32 *infz = (i32*)calloc(n + 16, sizeof(i32));
    i32 *kronl = (i32*)calloc(n + 16, sizeof(i32));
    i32 *iwork = (i32*)calloc(m > 1 ? m : 1, sizeof(i32));
    f64 *dwork = (f64*)calloc(ldwork, sizeof(f64));

    if (!infz || !kronl || !iwork || !dwork) {
        free(infz); free(kronl); free(iwork); free(dwork);
        Py_DECREF(abcd_array);
        Py_XDECREF(e_array);
        PyErr_NoMemory();
        return NULL;
    }

    f64 *abcd_data = (f64*)PyArray_DATA(abcd_array);
    f64 dummy = 0.0;
    f64 *e_data = e_array ? (f64*)PyArray_DATA(e_array) : &dummy;

    i32 nr = 0, pr = 0, ninfz = 0, dinfz = 0, nkronl = 0, info = 0;

    ag08by(first, n, m, p, svlmax, abcd_data, ldabcd, e_data, lde,
           &nr, &pr, &ninfz, &dinfz, &nkronl, infz, kronl, tol,
           iwork, dwork, ldwork, &info);

    free(iwork);
    free(dwork);

    PyArray_ResolveWritebackIfCopy(abcd_array);
    if (e_array) PyArray_ResolveWritebackIfCopy(e_array);

    // Cap copy sizes to allocated buffer sizes (infz: n, kronl: n+1)
    i32 infz_copy = (dinfz > 0 && dinfz <= n) ? dinfz : (dinfz > n ? n : 0);
    i32 kronl_copy = (nkronl > 0 && nkronl <= n + 1) ? nkronl : (nkronl > n + 1 ? n + 1 : 0);

    npy_intp infz_dims[1] = {infz_copy};
    npy_intp kronl_dims[1] = {kronl_copy};

    PyObject *infz_array = PyArray_SimpleNew(1, infz_dims, NPY_INT32);
    PyObject *kronl_array = PyArray_SimpleNew(1, kronl_dims, NPY_INT32);

    if (infz_copy > 0) {
        memcpy(PyArray_DATA((PyArrayObject*)infz_array), infz, infz_copy * sizeof(i32));
    }
    if (kronl_copy > 0) {
        memcpy(PyArray_DATA((PyArrayObject*)kronl_array), kronl, kronl_copy * sizeof(i32));
    }

    free(infz);
    free(kronl);

    PyObject *result = Py_BuildValue("(OOiiiiiOOi)",
        abcd_array,
        e_array ? (PyObject*)e_array : Py_None,
        nr, pr, ninfz, dinfz, nkronl,
        infz_array, kronl_array, info);

    Py_DECREF(abcd_array);
    Py_XDECREF(e_array);
    Py_DECREF(infz_array);
    Py_DECREF(kronl_array);

    return result;
}

PyObject* py_ag08bz(PyObject* self, PyObject* args) {
    const char *equil;
    i32 l, n, m, p;
    f64 tol;
    PyObject *a_obj, *e_obj, *b_obj, *c_obj, *d_obj;

    if (!PyArg_ParseTuple(args, "siiiiOOOOOd",
            &equil, &l, &n, &m, &p, &a_obj, &e_obj, &b_obj, &c_obj, &d_obj, &tol)) {
        return NULL;
    }

    if (l < 0) {
        PyErr_SetString(PyExc_ValueError, "l must be non-negative");
        return NULL;
    }
    if (n < 0) {
        PyErr_SetString(PyExc_ValueError, "n must be non-negative");
        return NULL;
    }
    if (m < 0) {
        PyErr_SetString(PyExc_ValueError, "m must be non-negative");
        return NULL;
    }
    if (p < 0) {
        PyErr_SetString(PyExc_ValueError, "p must be non-negative");
        return NULL;
    }

    PyArrayObject *a_array = NULL, *e_array = NULL, *b_array = NULL;
    PyArrayObject *c_array = NULL, *d_array = NULL;

    if (l > 0 && n > 0) {
        a_array = (PyArrayObject*)PyArray_FROM_OTF(
            a_obj, NPY_CDOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
        if (!a_array) return NULL;

        e_array = (PyArrayObject*)PyArray_FROM_OTF(
            e_obj, NPY_CDOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
        if (!e_array) {
            Py_DECREF(a_array);
            return NULL;
        }
    }

    if (l > 0 && m > 0) {
        b_array = (PyArrayObject*)PyArray_FROM_OTF(
            b_obj, NPY_CDOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
        if (!b_array) {
            Py_XDECREF(a_array);
            Py_XDECREF(e_array);
            return NULL;
        }
    }

    if (p > 0 && n > 0) {
        c_array = (PyArrayObject*)PyArray_FROM_OTF(
            c_obj, NPY_CDOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
        if (!c_array) {
            Py_XDECREF(a_array);
            Py_XDECREF(e_array);
            Py_XDECREF(b_array);
            return NULL;
        }
    }

    if (p > 0 && m > 0) {
        d_array = (PyArrayObject*)PyArray_FROM_OTF(
            d_obj, NPY_CDOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
        if (!d_array) {
            Py_XDECREF(a_array);
            Py_XDECREF(e_array);
            Py_XDECREF(b_array);
            Py_XDECREF(c_array);
            return NULL;
        }
    }

    i32 lda = (l > 0) ? l : 1;
    i32 lde = (l > 0) ? l : 1;
    i32 ldb = (l > 0 && m > 0) ? l : 1;
    i32 ldc = (p > 0) ? p : 1;
    i32 ldd = (p > 0) ? p : 1;

    i32 ldabcd = (l + p > n + m) ? (l + p) : (n + m);
    if (ldabcd < 1) ldabcd = 1;
    i32 labcd2 = ldabcd * (n + m);

    bool lequil = (equil[0] == 'S' || equil[0] == 's');
    i32 i0 = (l + p < m + n) ? (l + p) : (m + n);
    i32 i1 = (l < n) ? l : n;
    i32 lzwork = (labcd2 + (i0 + (i1 > 3*(m+n)-1 ? i1 : 3*(m+n)-1)) > labcd2 + 3*(l+p))
               ? labcd2 + (i0 + (i1 > 3*(m+n)-1 ? i1 : 3*(m+n)-1))
               : labcd2 + 3*(l+p);
    if (lzwork < 1) lzwork = 1;

    i32 ldwork = 2 * ldabcd;
    if (lequil) {
        i32 equil_work = 4 * (l + n);
        if (equil_work > ldwork) ldwork = equil_work;
    }

    i32 *infz = (i32*)calloc(n + 1 > 1 ? n + 1 : 1, sizeof(i32));
    i32 *kronr = (i32*)calloc(n + m + 1 > 1 ? n + m + 1 : 1, sizeof(i32));
    i32 *infe = (i32*)calloc(1 + (l + p < n + m ? l + p : n + m), sizeof(i32));
    i32 *kronl = (i32*)calloc(l + p + 1 > 1 ? l + p + 1 : 1, sizeof(i32));
    i32 *iwork = (i32*)malloc((n + (m > 1 ? m : 1)) * sizeof(i32));
    f64 *dwork = (f64*)malloc(ldwork * sizeof(f64));
    c128 *zwork = (c128*)PyMem_Calloc(lzwork, sizeof(c128));

    if (!infz || !kronr || !infe || !kronl || !iwork || !dwork || !zwork) {
        free(infz); free(kronr); free(infe); free(kronl);
        free(iwork); free(dwork); PyMem_Free(zwork);
        Py_XDECREF(a_array);
        Py_XDECREF(e_array);
        Py_XDECREF(b_array);
        Py_XDECREF(c_array);
        Py_XDECREF(d_array);
        PyErr_NoMemory();
        return NULL;
    }

    c128 dummy = 0.0 + 0.0*I;
    c128 *a_data = a_array ? (c128*)PyArray_DATA(a_array) : &dummy;
    c128 *e_data = e_array ? (c128*)PyArray_DATA(e_array) : &dummy;
    c128 *b_data = b_array ? (c128*)PyArray_DATA(b_array) : &dummy;
    c128 *c_data = c_array ? (c128*)PyArray_DATA(c_array) : &dummy;
    c128 *d_data = d_array ? (c128*)PyArray_DATA(d_array) : &dummy;

    i32 nfz = 0, nrank = 0, niz = 0, dinfz = 0;
    i32 nkror = 0, ninfe = 0, nkrol = 0, info = 0;

    ag08bz(equil, l, n, m, p, a_data, lda, e_data, lde,
           b_data, ldb, c_data, ldc, d_data, ldd,
           &nfz, &nrank, &niz, &dinfz, &nkror, &ninfe, &nkrol,
           infz, kronr, infe, kronl, tol, iwork, dwork, zwork, lzwork, &info);

    free(iwork);
    free(dwork);
    PyMem_Free(zwork);

    if (a_array) PyArray_ResolveWritebackIfCopy(a_array);
    if (e_array) PyArray_ResolveWritebackIfCopy(e_array);
    if (b_array) PyArray_ResolveWritebackIfCopy(b_array);
    if (c_array) PyArray_ResolveWritebackIfCopy(c_array);
    if (d_array) PyArray_ResolveWritebackIfCopy(d_array);

    npy_intp infz_dims[1] = {dinfz > 0 ? dinfz : 0};
    npy_intp kronr_dims[1] = {nkror > 0 ? nkror : 0};
    npy_intp infe_dims[1] = {ninfe > 0 ? ninfe : 0};
    npy_intp kronl_dims[1] = {nkrol > 0 ? nkrol : 0};

    PyObject *infz_array = PyArray_SimpleNew(1, infz_dims, NPY_INT32);
    PyObject *kronr_array = PyArray_SimpleNew(1, kronr_dims, NPY_INT32);
    PyObject *infe_array = PyArray_SimpleNew(1, infe_dims, NPY_INT32);
    PyObject *kronl_array = PyArray_SimpleNew(1, kronl_dims, NPY_INT32);

    if (dinfz > 0) {
        memcpy(PyArray_DATA((PyArrayObject*)infz_array), infz, dinfz * sizeof(i32));
    }
    if (nkror > 0) {
        memcpy(PyArray_DATA((PyArrayObject*)kronr_array), kronr, nkror * sizeof(i32));
    }
    if (ninfe > 0) {
        memcpy(PyArray_DATA((PyArrayObject*)infe_array), infe, ninfe * sizeof(i32));
    }
    if (nkrol > 0) {
        memcpy(PyArray_DATA((PyArrayObject*)kronl_array), kronl, nkrol * sizeof(i32));
    }

    free(infz);
    free(kronr);
    free(infe);
    free(kronl);

    PyObject *result = Py_BuildValue("(iiiiiiiOOOOi)",
        nfz, nrank, niz, dinfz, nkror, ninfe, nkrol,
        infz_array, kronr_array, infe_array, kronl_array, info);

    Py_XDECREF(a_array);
    Py_XDECREF(e_array);
    Py_XDECREF(b_array);
    Py_XDECREF(c_array);
    Py_XDECREF(d_array);
    Py_DECREF(infz_array);
    Py_DECREF(kronr_array);
    Py_DECREF(infe_array);
    Py_DECREF(kronl_array);

    return result;
}

/* Python wrapper for ag8byz (complex version) */
PyObject* py_ag8byz(PyObject* self, PyObject* args) {
    int first_int;
    i32 n, m, p;
    f64 svlmax, tol;
    PyObject *abcd_obj, *e_obj;

    if (!PyArg_ParseTuple(args, "piiidOOd",
            &first_int, &n, &m, &p, &svlmax, &abcd_obj, &e_obj, &tol)) {
        return NULL;
    }

    bool first = (first_int != 0);

    if (n < 0) {
        PyErr_SetString(PyExc_ValueError, "n must be non-negative");
        return NULL;
    }
    if (m < 0) {
        PyErr_SetString(PyExc_ValueError, "m must be non-negative");
        return NULL;
    }
    if (p < 0) {
        PyErr_SetString(PyExc_ValueError, "p must be non-negative");
        return NULL;
    }
    if (!first && m > p) {
        PyErr_SetString(PyExc_ValueError, "m must be <= p when first=False");
        return NULL;
    }
    if (svlmax < 0) {
        PyErr_SetString(PyExc_ValueError, "svlmax must be non-negative");
        return NULL;
    }

    i32 pn = p + n;
    i32 mn = m + n;
    i32 mpm = (p < m) ? p : m;

    PyArrayObject *abcd_array = (PyArrayObject*)PyArray_FROM_OTF(
        abcd_obj, NPY_CDOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (!abcd_array) return NULL;

    PyArrayObject *e_array = NULL;
    if (n > 0) {
        e_array = (PyArrayObject*)PyArray_FROM_OTF(
            e_obj, NPY_CDOUBLE, NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
        if (!e_array) {
            Py_DECREF(abcd_array);
            return NULL;
        }
    }

    i32 ldabcd = pn > 0 ? pn : 1;
    i32 lde = n > 0 ? n : 1;

    i32 lzwork = 1;
    if (p > 0) {
        lzwork = 3 * p;
        if (m > 0 || n > 0) {
            i32 mnm1 = mn - 1;
            lzwork = (lzwork > mnm1) ? lzwork : mnm1;
        }
        if (first && m > 0) {
            i32 t1 = mpm + ((3*m - 1 > n) ? (3*m - 1) : n);
            lzwork = (lzwork > t1) ? lzwork : t1;
        }
    }
    lzwork = (lzwork > 1) ? lzwork : 1;

    i32 ldwork = first ? 2 * (m > p ? m : p) : 2 * p;
    if (ldwork < 1) ldwork = 1;

    i32 *infz = (i32*)calloc(n + 16, sizeof(i32));
    i32 *kronl = (i32*)calloc(n + 16, sizeof(i32));
    i32 *iwork = (i32*)calloc(m > 1 ? m : 1, sizeof(i32));
    f64 *dwork = (f64*)calloc(ldwork, sizeof(f64));
    c128 *zwork = (c128*)calloc(lzwork, sizeof(c128));

    if (!infz || !kronl || !iwork || !dwork || !zwork) {
        free(infz); free(kronl); free(iwork); free(dwork); free(zwork);
        Py_DECREF(abcd_array);
        Py_XDECREF(e_array);
        PyErr_NoMemory();
        return NULL;
    }

    c128 *abcd_data = (c128*)PyArray_DATA(abcd_array);
    c128 dummy = 0.0 + 0.0*I;
    c128 *e_data = e_array ? (c128*)PyArray_DATA(e_array) : &dummy;

    i32 nr = 0, pr = 0, ninfz = 0, dinfz = 0, nkronl = 0, info = 0;

    ag8byz(first, n, m, p, svlmax, abcd_data, ldabcd, e_data, lde,
           &nr, &pr, &ninfz, &dinfz, &nkronl, infz, kronl, tol,
           iwork, dwork, zwork, lzwork, &info);

    free(iwork);
    free(dwork);
    free(zwork);

    PyArray_ResolveWritebackIfCopy(abcd_array);
    if (e_array) PyArray_ResolveWritebackIfCopy(e_array);

    // Cap copy sizes to allocated buffer sizes (infz: n, kronl: n+1)
    i32 infz_copy = (dinfz > 0 && dinfz <= n) ? dinfz : (dinfz > n ? n : 0);
    i32 kronl_copy = (nkronl > 0 && nkronl <= n + 1) ? nkronl : (nkronl > n + 1 ? n + 1 : 0);

    npy_intp infz_dims[1] = {infz_copy};
    npy_intp kronl_dims[1] = {kronl_copy};

    PyObject *infz_array = PyArray_SimpleNew(1, infz_dims, NPY_INT32);
    PyObject *kronl_array = PyArray_SimpleNew(1, kronl_dims, NPY_INT32);

    if (infz_copy > 0) {
        memcpy(PyArray_DATA((PyArrayObject*)infz_array), infz, infz_copy * sizeof(i32));
    }
    if (kronl_copy > 0) {
        memcpy(PyArray_DATA((PyArrayObject*)kronl_array), kronl, kronl_copy * sizeof(i32));
    }

    free(infz);
    free(kronl);

    PyObject *result = Py_BuildValue("(OOiiiiiOOi)",
        abcd_array,
        e_array ? (PyObject*)e_array : Py_None,
        nr, pr, ninfz, dinfz, nkronl,
        infz_array, kronl_array, info);

    Py_DECREF(abcd_array);
    Py_XDECREF(e_array);
    Py_DECREF(infz_array);
    Py_DECREF(kronl_array);

    return result;
}

