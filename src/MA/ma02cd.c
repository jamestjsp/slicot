/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"

void ma02cd(i32 n, i32 kl, i32 ku, f64 *a, i32 lda) {
    if (n <= 1) {
        return;
    }

    i32 lda1 = lda + 1;
    i32 i1;
    i32 neg_lda1 = -lda1;
    i32 min_kl = (kl < n - 2) ? kl : n - 2;
    i32 min_ku = (ku < n - 2) ? ku : n - 2;

    for (i32 i = 1; i <= min_kl; i++) {
        i1 = (n - i) / 2;
        if (i1 > 0) {
            SLC_DSWAP(&i1,
                      &a[i + 0 * lda],
                      &lda1,
                      &a[(n - i1) + (n - i1 - i) * lda],
                      &neg_lda1);
        }
    }

    for (i32 i = 1; i <= min_ku; i++) {
        i1 = (n - i) / 2;
        if (i1 > 0) {
            SLC_DSWAP(&i1,
                      &a[0 + i * lda],
                      &lda1,
                      &a[(n - i1 - i) + (n - i1) * lda],
                      &neg_lda1);
        }
    }

    i1 = n / 2;
    if (i1 > 0) {
        SLC_DSWAP(&i1,
                  &a[0],
                  &lda1,
                  &a[(n - i1) + (n - i1) * lda],
                  &neg_lda1);
    }
}
