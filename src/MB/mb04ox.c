/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"

void mb04ox(
    const i32 n,
    f64* a,
    const i32 lda,
    f64* x,
    const i32 incx
)
{
    if (n <= 0) {
        return;
    }

    i32 ix = 0;

    for (i32 i = 0; i < n - 1; i++) {
        f64 ci, si, temp;

        SLC_DLARTG(&a[i + i * lda], &x[ix], &ci, &si, &temp);
        a[i + i * lda] = temp;
        ix += incx;

        i32 len = n - i - 1;
        SLC_DROT(&len, &a[i + (i + 1) * lda], &lda, &x[ix], &incx, &ci, &si);
    }

    f64 ci, si, temp;
    SLC_DLARTG(&a[(n - 1) + (n - 1) * lda], &x[ix], &ci, &si, &temp);
    a[(n - 1) + (n - 1) * lda] = temp;
}
