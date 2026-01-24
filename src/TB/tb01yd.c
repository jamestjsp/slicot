/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"

void tb01yd(i32 n, i32 m, i32 p, f64* a, i32 lda,
            f64* b, i32 ldb, f64* c, i32 ldc, i32* info) {
    *info = 0;

    i32 max1n = (1 > n) ? 1 : n;
    i32 max1p = (1 > p) ? 1 : p;

    if (n < 0) {
        *info = -1;
    } else if (m < 0) {
        *info = -2;
    } else if (p < 0) {
        *info = -3;
    } else if (lda < max1n) {
        *info = -5;
    } else if (ldb < 1 || (m > 0 && ldb < n)) {
        *info = -7;
    } else if (ldc < max1p) {
        *info = -9;
    }

    if (*info != 0) {
        return;
    }

    if (n <= 1) {
        return;
    }

    i32 nby2 = n / 2;
    i32 int1 = 1;
    i32 mint1 = -1;

    for (i32 j = 0; j < nby2; j++) {
        SLC_DSWAP(&n, &a[j * lda], &mint1, &a[(n - j - 1) * lda], &int1);
    }

    if ((n % 2) != 0 && n > 2) {
        SLC_DSWAP(&nby2, &a[(nby2 + 1) + nby2 * lda], &mint1, &a[nby2 * lda], &int1);
    }

    if (m > 0) {
        for (i32 j = 0; j < nby2; j++) {
            SLC_DSWAP(&m, &b[j], &ldb, &b[n - j - 1], &ldb);
        }
    }

    if (p > 0) {
        for (i32 j = 0; j < nby2; j++) {
            SLC_DSWAP(&p, &c[j * ldc], &int1, &c[(n - j - 1) * ldc], &int1);
        }
    }
}
