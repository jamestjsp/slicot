/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <stdlib.h>

static int select_dummy(const f64* wr, const f64* wi) {
    (void)wr; (void)wi;
    return 0;
}

void tb01ld(
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
        SLC_XERBLA("TB01LD", &neginfo);
        return;
    }

    *ndim = 0;
    if (n == 0) {
        return;
    }

    f64 wrkopt = 0.0;

    if (ljobg) {
        i32 sdim;
        int bwork_dummy[1] = {0};

        SLC_DGEES("Vectors", "Not ordered", select_dummy, &n, a, &lda, &sdim,
                  wr, wi, u, &ldu, dwork, &ldwork, bwork_dummy, info);
        wrkopt = dwork[0];
        if (*info != 0) {
            *info = 1;
            return;
        }
    } else {
        i32 nn = n;
        SLC_DLASET("Full", &nn, &nn, &ZERO, &ONE, u, &ldu);
        wrkopt = 0.0;
    }

    i32 nlow = 1;
    i32 nsup = n;
    mb03qd(dico, stdom, "Update", n, nlow, nsup, alpha, a, lda, u, ldu, ndim, dwork, info);
    if (*info != 0) {
        return;
    }

    i32 ierr;
    mb03qx(n, a, lda, wr, wi, &ierr);

    if (ldwork < n * m) {
        for (i32 i = 0; i < m; i++) {
            SLC_DCOPY(&n, &b[0 + i*ldb], &(i32){1}, dwork, &(i32){1});
            SLC_DGEMV("Transpose", &n, &n, &ONE, u, &ldu, dwork, &(i32){1}, &ZERO, &b[0 + i*ldb], &(i32){1});
        }
    } else {
        SLC_DLACPY("Full", &n, &m, b, &ldb, dwork, &n);
        SLC_DGEMM("Transpose", "No transpose", &n, &m, &n, &ONE, u, &ldu, dwork, &n, &ZERO, b, &ldb);
        if ((f64)(n * m) > wrkopt) {
            wrkopt = (f64)(n * m);
        }
    }

    if (ldwork < n * p) {
        for (i32 i = 0; i < p; i++) {
            SLC_DCOPY(&n, &c[i + 0*ldc], &ldc, dwork, &(i32){1});
            SLC_DGEMV("Transpose", &n, &n, &ONE, u, &ldu, dwork, &(i32){1}, &ZERO, &c[i + 0*ldc], &ldc);
        }
    } else {
        i32 ldwp = p > 1 ? p : 1;
        SLC_DLACPY("Full", &p, &n, c, &ldc, dwork, &ldwp);
        SLC_DGEMM("No transpose", "No transpose", &p, &n, &n, &ONE, dwork, &ldwp, u, &ldu, &ZERO, c, &ldc);
        if ((f64)(n * p) > wrkopt) {
            wrkopt = (f64)(n * p);
        }
    }

    dwork[0] = wrkopt;
}
