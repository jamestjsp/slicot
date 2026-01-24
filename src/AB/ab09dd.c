/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 2025, slicot.c contributors
 */

#include "slicot.h"
#include "slicot_blas.h"

i32 ab09dd(const char* dico, i32 n, i32 m, i32 p, i32 nr,
           f64* a, i32 lda, f64* b, i32 ldb,
           f64* c, i32 ldc, f64* d, i32 ldd,
           f64* rcond, i32* iwork, f64* dwork) {

    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    bool discr = (dico[0] == 'D' || dico[0] == 'd');
    bool conti = (dico[0] == 'C' || dico[0] == 'c');

    if (!conti && !discr) {
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
    if (nr < 0 || nr > n) {
        return -5;
    }
    i32 max1n = (1 > n) ? 1 : n;
    if (lda < max1n) {
        return -7;
    }
    if (ldb < max1n) {
        return -9;
    }
    i32 max1p = (1 > p) ? 1 : p;
    if (ldc < max1p) {
        return -11;
    }
    if (ldd < max1p) {
        return -13;
    }

    if (nr == n) {
        *rcond = ONE;
        return 0;
    }

    i32 k = nr;
    i32 ns = n - nr;

    for (i32 j = k; j < n; j++) {
        for (i32 i = k; i < n; i++) {
            a[i + j * lda] = -a[i + j * lda];
        }
        if (discr) {
            a[j + j * lda] += ONE;
        }
    }

    f64 a22nrm = SLC_DLANGE("1", &ns, &ns, &a[k + k * lda], &lda, dwork);

    i32 info = 0;
    SLC_DGETRF(&ns, &ns, &a[k + k * lda], &lda, iwork, &info);
    if (info > 0) {
        *rcond = ZERO;
        return 1;
    }

    SLC_DGECON("1", &ns, &a[k + k * lda], &lda, &a22nrm, rcond, dwork, &iwork[ns], &info);
    f64 eps = SLC_DLAMCH("E");
    if (*rcond <= eps) {
        return 1;
    }

    if (nr > 0) {
        SLC_DGETRS("N", &ns, &nr, &a[k + k * lda], &lda, iwork, &a[k], &lda, &info);
    }

    SLC_DGETRS("N", &ns, &m, &a[k + k * lda], &lda, iwork, &b[k], &ldb, &info);

    if (nr > 0) {
        SLC_DGEMM("N", "N", &nr, &nr, &ns, &ONE, &a[k * lda], &lda,
                  &a[k], &lda, &ONE, a, &lda);

        SLC_DGEMM("N", "N", &nr, &m, &ns, &ONE, &a[k * lda], &lda,
                  &b[k], &ldb, &ONE, b, &ldb);

        SLC_DGEMM("N", "N", &p, &nr, &ns, &ONE, &c[k * ldc], &ldc,
                  &a[k], &lda, &ONE, c, &ldc);
    }

    SLC_DGEMM("N", "N", &p, &m, &ns, &ONE, &c[k * ldc], &ldc,
              &b[k], &ldb, &ONE, d, &ldd);

    return 0;
}
