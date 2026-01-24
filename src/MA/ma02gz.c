/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"

void ma02gz(
    const i32 n,
    c128* a,
    const i32 lda,
    const i32 k1,
    const i32 k2,
    const i32* ipiv,
    const i32 incx
)
{
    i32 j, jp, jx;
    i32 one = 1;

    if (incx == 0 || n == 0) {
        return;
    }

    if (incx > 0) {
        jx = k1 - 1;
    } else {
        jx = (1 - k2) * incx;
    }

    if (incx == 1) {
        for (j = k1 - 1; j <= k2 - 1; j++) {
            jp = ipiv[j] - 1;
            if (jp != j) {
                SLC_ZSWAP(&n, &a[0 + j*lda], &one, &a[0 + jp*lda], &one);
            }
        }
    } else if (incx > 1) {
        for (j = k1 - 1; j <= k2 - 1; j++) {
            jp = ipiv[jx] - 1;
            if (jp != j) {
                SLC_ZSWAP(&n, &a[0 + j*lda], &one, &a[0 + jp*lda], &one);
            }
            jx = jx + incx;
        }
    } else {
        for (j = k2 - 1; j >= k1 - 1; j--) {
            jp = ipiv[jx] - 1;
            if (jp != j) {
                SLC_ZSWAP(&n, &a[0 + j*lda], &one, &a[0 + jp*lda], &one);
            }
            jx = jx + incx;
        }
    }
}
