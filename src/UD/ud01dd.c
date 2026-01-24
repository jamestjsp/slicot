/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"

void ud01dd(i32 m, i32 n, i32 nnz, const i32 *rows, const i32 *cols,
            const f64 *vals, f64 *a, i32 lda, i32 *info)
{
    *info = 0;

    i32 max_m_1 = m > 1 ? m : 1;

    if (m < 0) {
        *info = -1;
        return;
    }
    if (n < 0) {
        *info = -2;
        return;
    }
    if (nnz < 0) {
        *info = -3;
        return;
    }
    if (lda < max_m_1) {
        *info = -8;
        return;
    }

    if (m == 0 || n == 0) {
        return;
    }

    f64 zero = 0.0;
    SLC_DLASET("Full", &m, &n, &zero, &zero, a, &lda);

    for (i32 k = 0; k < nnz; k++) {
        i32 i = rows[k];
        i32 j = cols[k];

        if (i < 1 || i > m || j < 1 || j > n) {
            *info = 1;
        } else {
            a[(i - 1) + (j - 1) * lda] = vals[k];
        }
    }
}
