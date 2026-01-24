/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"

void ma02gd(
    const i32 n,
    f64* a,
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
        jx = k1 - 1;  // Convert to 0-based
    } else {
        jx = (1 - k2) * incx;  // For negative incx
    }

    if (incx == 1) {
        for (j = k1 - 1; j <= k2 - 1; j++) {  // 0-based loop
            jp = ipiv[j] - 1;  // Convert 1-based pivot to 0-based
            if (jp != j) {
                SLC_DSWAP(&n, &a[0 + j*lda], &one, &a[0 + jp*lda], &one);
            }
        }
    } else if (incx > 1) {
        for (j = k1 - 1; j <= k2 - 1; j++) {  // 0-based loop
            jp = ipiv[jx] - 1;  // Convert 1-based pivot to 0-based
            if (jp != j) {
                SLC_DSWAP(&n, &a[0 + j*lda], &one, &a[0 + jp*lda], &one);
            }
            jx = jx + incx;
        }
    } else {  // incx < 0
        for (j = k2 - 1; j >= k1 - 1; j--) {  // 0-based reverse loop
            jp = ipiv[jx] - 1;  // Convert 1-based pivot to 0-based
            if (jp != j) {
                SLC_DSWAP(&n, &a[0 + j*lda], &one, &a[0 + jp*lda], &one);
            }
            jx = jx + incx;
        }
    }
}
