/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"

void tb01xz(
    const char* jobd,
    const i32 n,
    const i32 m,
    const i32 p,
    const i32 kl,
    const i32 ku,
    c128* a,
    const i32 lda,
    c128* b,
    const i32 ldb,
    c128* c,
    const i32 ldc,
    c128* d,
    const i32 ldd,
    i32* info
)
{
    bool ljobd;
    i32 j, maxmp, minmp, nm1;
    const i32 int1 = 1;
    i32 i;
    c128 temp;

    *info = 0;
    ljobd = (jobd[0] == 'D' || jobd[0] == 'd');
    maxmp = (m > p) ? m : p;
    minmp = (m < p) ? m : p;
    nm1 = n - 1;

    if (!ljobd && !(jobd[0] == 'Z' || jobd[0] == 'z')) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (m < 0) {
        *info = -3;
    } else if (p < 0) {
        *info = -4;
    } else if (kl < 0 || (nm1 >= 0 && kl > nm1) || (nm1 < 0 && kl > 0)) {
        *info = -5;
    } else if (ku < 0 || (nm1 >= 0 && ku > nm1) || (nm1 < 0 && ku > 0)) {
        *info = -6;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -8;
    } else if ((maxmp > 0 && ldb < (n > 1 ? n : 1)) || (minmp == 0 && ldb < 1)) {
        *info = -10;
    } else if (ldc < 1 || (n > 0 && ldc < maxmp)) {
        *info = -12;
    } else if (ldd < 1 || (ljobd && ldd < maxmp)) {
        *info = -14;
    }

    if (*info != 0) {
        return;
    }

    if (ljobd) {
        for (j = 0; j < maxmp; j++) {
            if (j < minmp - 1) {
                i32 count = minmp - j - 1;
                SLC_ZSWAP(&count, &d[(j + 1) + j * ldd], &int1, &d[j + (j + 1) * ldd], &ldd);
            } else if (j >= p) {
                SLC_ZCOPY(&p, &d[j * ldd], &int1, &d[j], &ldd);
            } else if (j >= m) {
                SLC_ZCOPY(&m, &d[j], &ldd, &d[j * ldd], &int1);
            }
        }
    }

    if (n == 0) {
        return;
    }

    for (i = 0; i < n; i++) {
        i32 j_start = (i > kl) ? (i - kl) : 0;
        i32 j_end = ((i + ku) < (n - 1)) ? (i + ku) : (n - 1);

        for (j = j_start; j <= j_end; j++) {
            i32 i2 = n - 1 - j;
            i32 j2 = n - 1 - i;
            i32 idx1 = i + j * lda;
            i32 idx2 = i2 + j2 * lda;
            if (idx1 < idx2) {
                temp = a[idx1];
                a[idx1] = a[idx2];
                a[idx2] = temp;
            }
        }
    }

    for (j = 0; j < maxmp; j++) {
        if (j < minmp) {
            for (i = 0; i < n; i++) {
                i32 idx_b = i + j * ldb;
                i32 idx_c = j + (n - 1 - i) * ldc;
                temp = b[idx_b];
                b[idx_b] = c[idx_c];
                c[idx_c] = temp;
            }
        } else if (j >= p) {
            for (i = 0; i < n; i++) {
                c[j + (n - 1 - i) * ldc] = b[i + j * ldb];
            }
        } else {
            for (i = 0; i < n; i++) {
                b[i + j * ldb] = c[j + (n - 1 - i) * ldc];
            }
        }
    }
}
