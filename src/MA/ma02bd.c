/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"

void ma02bd(const char side, const i32 m, const i32 n, f64* a, const i32 lda)
{
    bool bsides = (side == 'B' || side == 'b');

    if ((side == 'L' || side == 'l' || bsides) && m > 1) {
        i32 m2 = m / 2;
        i32 k = m - m2;
        i32 one = 1;
        i32 minus_one = -1;
        for (i32 j = 0; j < n; j++) {
            SLC_DSWAP(&m2, &a[j * lda], &minus_one, &a[k + j * lda], &one);
        }
    }

    if ((side == 'R' || side == 'r' || bsides) && n > 1) {
        i32 n2 = n / 2;
        i32 k = n - n2;
        i32 minus_lda = -lda;
        for (i32 i = 0; i < m; i++) {
            SLC_DSWAP(&n2, &a[i], &minus_lda, &a[i + k * lda], &lda);
        }
    }
}
