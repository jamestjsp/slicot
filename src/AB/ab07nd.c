/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 2025, slicot.c contributors
 */

#include "slicot.h"
#include "slicot_blas.h"

i32 ab07nd(i32 n, i32 m, f64* a, i32 lda, f64* b, i32 ldb,
           f64* c, i32 ldc, f64* d, i32 ldd, f64* rcond,
           i32* iwork, f64* dwork, i32 ldwork) {

    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    i32 info = 0;

    if (n < 0) {
        return -1;
    }
    if (m < 0) {
        return -2;
    }
    i32 max1n = (1 > n) ? 1 : n;
    if (lda < max1n) {
        return -4;
    }
    if (ldb < max1n) {
        return -6;
    }
    i32 max1m = (1 > m) ? 1 : m;
    if (ldc < max1m) {
        return -8;
    }
    if (ldd < max1m) {
        return -10;
    }

    i32 minwrk = (1 > 4 * m) ? 1 : 4 * m;
    i32 maxwrk;

    i32 lwork_query = -1;
    i32 ierr;
    SLC_DGETRI(&m, d, &ldd, iwork, dwork, &lwork_query, &ierr);
    i32 opt_getri = (i32)dwork[0];
    i32 nm = n * m;
    maxwrk = minwrk;
    if (opt_getri > maxwrk) maxwrk = opt_getri;
    if (nm > maxwrk) maxwrk = nm;

    bool lquery = (ldwork == -1);
    if (ldwork < minwrk && !lquery) {
        return -14;
    }

    if (lquery) {
        dwork[0] = (f64)maxwrk;
        return 0;
    }

    if (m == 0) {
        *rcond = ONE;
        dwork[0] = ONE;
        return 0;
    }

    SLC_DGETRF(&m, &m, d, &ldd, iwork, &info);
    if (info != 0) {
        *rcond = ZERO;
        return info;
    }

    f64 dnorm = SLC_DLANGE("1", &m, &m, d, &ldd, dwork);
    SLC_DGECON("1", &m, d, &ldd, &dnorm, rcond, dwork, &iwork[m], &ierr);
    f64 eps = SLC_DLAMCH("Epsilon");
    if (*rcond < eps) {
        info = m + 1;
    }

    SLC_DGETRI(&m, d, &ldd, iwork, dwork, &ldwork, &ierr);

    if (n > 0) {
        i32 chunk = ldwork / m;
        bool blas3 = (chunk >= n) && (m > 1);
        bool block = ((chunk < n ? chunk : m) > 1);

        i32 one_int = 1;
        f64 neg_one = -ONE;

        if (blas3) {
            SLC_DLACPY("F", &n, &m, b, &ldb, dwork, &n);
            SLC_DGEMM("N", "N", &n, &m, &m, &neg_one, dwork, &n, d, &ldd, &ZERO, b, &ldb);
        } else if (block) {
            for (i32 i = 0; i < n; i += chunk) {
                i32 bl = n - i;
                if (bl > chunk) bl = chunk;
                SLC_DLACPY("F", &bl, &m, &b[i], &ldb, dwork, &bl);
                SLC_DGEMM("N", "N", &bl, &m, &m, &neg_one, dwork, &bl, d, &ldd, &ZERO, &b[i], &ldb);
            }
        } else {
            for (i32 i = 0; i < n; i++) {
                SLC_DCOPY(&m, &b[i], &ldb, dwork, &one_int);
                SLC_DGEMV("T", &m, &m, &neg_one, d, &ldd, dwork, &one_int, &ZERO, &b[i], &ldb);
            }
        }

        SLC_DGEMM("N", "N", &n, &n, &m, &ONE, b, &ldb, c, &ldc, &ONE, a, &lda);

        if (blas3) {
            SLC_DLACPY("F", &m, &n, c, &ldc, dwork, &m);
            SLC_DGEMM("N", "N", &m, &n, &m, &ONE, d, &ldd, dwork, &m, &ZERO, c, &ldc);
        } else if (block) {
            for (i32 j = 0; j < n; j += chunk) {
                i32 bl = n - j;
                if (bl > chunk) bl = chunk;
                SLC_DLACPY("F", &m, &bl, &c[j * ldc], &ldc, dwork, &m);
                SLC_DGEMM("N", "N", &m, &bl, &m, &ONE, d, &ldd, dwork, &m, &ZERO, &c[j * ldc], &ldc);
            }
        } else {
            for (i32 j = 0; j < n; j++) {
                SLC_DCOPY(&m, &c[j * ldc], &one_int, dwork, &one_int);
                SLC_DGEMV("N", &m, &m, &ONE, d, &ldd, dwork, &one_int, &ZERO, &c[j * ldc], &one_int);
            }
        }
    }

    dwork[0] = (f64)maxwrk;
    return info;
}
