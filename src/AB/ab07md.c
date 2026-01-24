/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 2025, slicot.c contributors
 */

#include "slicot.h"
#include "slicot_blas.h"

i32 ab07md(char jobd, i32 n, i32 m, i32 p, f64* a, i32 lda,
           f64* b, i32 ldb, f64* c, i32 ldc, f64* d, i32 ldd) {

    bool ljobd = (jobd == 'D' || jobd == 'd');
    i32 mplim = (m > p) ? m : p;
    i32 minmp = (m < p) ? m : p;

    if (!ljobd && jobd != 'Z' && jobd != 'z') {
        return -1;
    }
    if (n < 0) {
        return -2;
    }
    if (m < 0) {
        return -3;
    }
    if (p < 0) {
        return -4;
    }
    i32 max1n = (1 > n) ? 1 : n;
    if (lda < max1n) {
        return -6;
    }
    if (ldb < max1n) {
        return -8;
    }
    if ((n > 0 && ldc < ((1 > mplim) ? 1 : mplim)) ||
        (n == 0 && ldc < 1)) {
        return -10;
    }
    if ((ljobd && ldd < ((1 > mplim) ? 1 : mplim)) ||
        (!ljobd && ldd < 1)) {
        return -12;
    }

    i32 max_n_minmp = (n > minmp) ? n : minmp;
    if (max_n_minmp == 0) {
        return 0;
    }

    i32 one = 1;

    if (n > 0) {
        for (i32 j = 0; j < n - 1; j++) {
            i32 len = n - j - 1;
            SLC_DSWAP(&len, &a[(j + 1) + j * lda], &one, &a[j + (j + 1) * lda], &lda);
        }

        for (i32 j = 0; j < mplim; j++) {
            if (j < minmp) {
                SLC_DSWAP(&n, &b[j * ldb], &one, &c[j], &ldc);
            } else if (j >= p) {
                SLC_DCOPY(&n, &b[j * ldb], &one, &c[j], &ldc);
            } else {
                SLC_DCOPY(&n, &c[j], &ldc, &b[j * ldb], &one);
            }
        }
    }

    if (ljobd && minmp > 0) {
        for (i32 j = 0; j < mplim; j++) {
            if (j < minmp - 1) {
                i32 len = minmp - j - 1;
                SLC_DSWAP(&len, &d[(j + 1) + j * ldd], &one, &d[j + (j + 1) * ldd], &ldd);
            } else if (j >= p) {
                SLC_DCOPY(&p, &d[j * ldd], &one, &d[j], &ldd);
            } else if (j >= m) {
                SLC_DCOPY(&m, &d[j], &ldd, &d[j * ldd], &one);
            }
        }
    }

    return 0;
}
