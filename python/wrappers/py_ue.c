/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#define PY_SSIZE_T_CLEAN
#define PY_ARRAY_UNIQUE_SYMBOL SLICOT_ARRAY_API
#define NO_IMPORT_ARRAY

#include "py_wrappers.h"


PyObject* py_ue01md(PyObject* self, PyObject* args) {
    i32 ispec, n1, n2, n3;
    const char *name;
    const char *opts;

    if (!PyArg_ParseTuple(args, "issiii", &ispec, &name, &opts, &n1, &n2, &n3)) {
        return NULL;
    }

    i32 result = ue01md(ispec, name, opts, n1, n2, n3);

    return PyLong_FromLong(result);
}
