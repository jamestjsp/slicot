/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"

void tb01kx(
    const i32 n,
    const i32 m,
    const i32 p,
    const i32 ndim,
    f64* a,
    const i32 lda,
    f64* b,
    const i32 ldb,
    f64* c,
    const i32 ldc,
    f64* u,
    const i32 ldu,
    f64* v,
    const i32 ldv,
    i32* info
)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    *info = 0;

    i32 max_1_n = (1 > n) ? 1 : n;
    i32 max_1_p = (1 > p) ? 1 : p;

    if (n < 0) {
        *info = -1;
    } else if (m < 0) {
        *info = -2;
    } else if (p < 0) {
        *info = -3;
    } else if (ndim < 0 || ndim > n) {
        *info = -4;
    } else if (lda < max_1_n) {
        *info = -6;
    } else if (ldb < max_1_n) {
        *info = -8;
    } else if (ldc < max_1_p) {
        *info = -10;
    } else if (ldu < max_1_n) {
        *info = -12;
    } else if (ldv < max_1_n) {
        *info = -14;
    }

    if (*info != 0) {
        i32 neginfo = -(*info);
        SLC_XERBLA("TB01KX", &neginfo);
        return;
    }

    if (n == 0) {
        return;
    }

    ma02ad("Full", n, n, u, ldu, v, ldv);

    if (ndim > 0 && ndim < n) {
        i32 nr = n - ndim;
        i32 ndim1 = ndim;  // 0-based index of start of second block
        f64 scale;

        SLC_DTRSYL("N", "N", &(i32){-1}, &ndim, &nr,
                   a, &lda,
                   &a[ndim1 + ndim1*lda], &lda,
                   &a[0 + ndim1*lda], &lda,
                   &scale, info);

        if (*info != 0) {
            return;
        }

        scale = ONE / scale;

        SLC_DGEMM("N", "N", &ndim, &m, &nr, &scale,
                  &a[0 + ndim1*lda], &lda,
                  &b[ndim1 + 0*ldb], &ldb,
                  &ONE, b, &ldb);

        f64 neg_scale = -scale;

        SLC_DGEMM("N", "N", &p, &nr, &ndim, &neg_scale,
                  c, &ldc,
                  &a[0 + ndim1*lda], &lda,
                  &ONE, &c[0 + ndim1*ldc], &ldc);

        SLC_DGEMM("N", "N", &n, &nr, &ndim, &neg_scale,
                  u, &ldu,
                  &a[0 + ndim1*lda], &lda,
                  &ONE, &u[0 + ndim1*ldu], &ldu);

        SLC_DGEMM("N", "N", &ndim, &n, &nr, &scale,
                  &a[0 + ndim1*lda], &lda,
                  &v[ndim1 + 0*ldv], &ldv,
                  &ONE, v, &ldv);

        SLC_DLASET("Full", &ndim, &nr, &ZERO, &ZERO, &a[0 + ndim1*lda], &lda);
    }

    if (n > 2) {
        i32 nm2 = n - 2;
        SLC_DLASET("Lower", &nm2, &nm2, &ZERO, &ZERO, &a[2 + 0*lda], &lda);
    }
}
