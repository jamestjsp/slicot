/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"

void tb01kd(
    const char* dico,
    const char* stdom,
    const char* joba,
    const i32 n,
    const i32 m,
    const i32 p,
    const f64 alpha,
    f64* a,
    const i32 lda,
    f64* b,
    const i32 ldb,
    f64* c,
    const i32 ldc,
    i32* ndim,
    f64* u,
    const i32 ldu,
    f64* wr,
    f64* wi,
    f64* dwork,
    const i32 ldwork,
    i32* info
)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    bool discr = (*dico == 'D' || *dico == 'd');
    bool ljobg = (*joba == 'G' || *joba == 'g');

    *info = 0;

    if (!(*dico == 'C' || *dico == 'c') && !discr) {
        *info = -1;
    } else if (!(*stdom == 'S' || *stdom == 's') && !(*stdom == 'U' || *stdom == 'u')) {
        *info = -2;
    } else if (!(*joba == 'S' || *joba == 's') && !ljobg) {
        *info = -3;
    } else if (n < 0) {
        *info = -4;
    } else if (m < 0) {
        *info = -5;
    } else if (p < 0) {
        *info = -6;
    } else if (discr && alpha < ZERO) {
        *info = -7;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -9;
    } else if (ldb < (n > 1 ? n : 1)) {
        *info = -11;
    } else if (ldc < (p > 1 ? p : 1)) {
        *info = -13;
    } else if (ldu < (n > 1 ? n : 1)) {
        *info = -16;
    } else if ((ldwork < (n > 1 ? n : 1)) || (ljobg && ldwork < (3*n > 1 ? 3*n : 1))) {
        *info = -20;
    }

    if (*info != 0) {
        i32 neginfo = -(*info);
        SLC_XERBLA("TB01KD", &neginfo);
        return;
    }

    *ndim = 0;
    if (n == 0) {
        return;
    }

    tb01ld(dico, stdom, joba, n, m, p, alpha, a, lda, b, ldb, c, ldc,
           ndim, u, ldu, wr, wi, dwork, ldwork, info);

    if (*info != 0) {
        return;
    }

    if (*ndim > 0 && *ndim < n) {
        i32 nr = n - *ndim;
        i32 ndim1 = *ndim;  // 0-based start of second block
        f64 scale;

        SLC_DTRSYL("N", "N", &(i32){-1}, ndim, &nr,
                   a, &lda,
                   &a[ndim1 + ndim1*lda], &lda,
                   &a[0 + ndim1*lda], &lda,
                   &scale, info);

        if (*info != 0) {
            *info = 3;
            return;
        }

        scale = ONE / scale;

        SLC_DGEMM("N", "N", ndim, &m, &nr, &scale,
                  &a[0 + ndim1*lda], &lda,
                  &b[ndim1 + 0*ldb], &ldb,
                  &ONE, b, &ldb);

        f64 neg_scale = -scale;
        SLC_DGEMM("N", "N", &p, &nr, ndim, &neg_scale,
                  c, &ldc,
                  &a[0 + ndim1*lda], &lda,
                  &ONE, &c[0 + ndim1*ldc], &ldc);

        SLC_DGEMM("N", "N", &n, &nr, ndim, &neg_scale,
                  u, &ldu,
                  &a[0 + ndim1*lda], &lda,
                  &ONE, &u[0 + ndim1*ldu], &ldu);

        SLC_DLASET("Full", ndim, &nr, &ZERO, &ZERO, &a[0 + ndim1*lda], &lda);
    }

    if (n > 2) {
        i32 nm2 = n - 2;
        SLC_DLASET("Lower", &nm2, &nm2, &ZERO, &ZERO, &a[2 + 0*lda], &lda);
    }
}
