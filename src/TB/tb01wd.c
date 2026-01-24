/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

/*
 * TB01WD - Orthogonal similarity transformation to real Schur form
 *
 * Purpose:
 *   To reduce the system state matrix A to an upper real Schur form
 *   by using an orthogonal similarity transformation A <- U'*A*U and
 *   to apply the transformation to the matrices B and C: B <- U'*B
 *   and C <- C*U.
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <stdlib.h>
#include <math.h>

static int dummy_select(const f64* wr, const f64* wi) {
    return 0;
}

void tb01wd(
    const i32 n,
    const i32 m,
    const i32 p,
    f64* a,
    const i32 lda,
    f64* b,
    const i32 ldb,
    f64* c,
    const i32 ldc,
    f64* u,
    const i32 ldu,
    f64* wr,
    f64* wi,
    f64* dwork,
    const i32 ldwork,
    i32* info
)
{
    const f64 zero = 0.0;
    const f64 one = 1.0;
    const i32 int1 = 1;
    i32 sdim;
    i32 bwork[1];
    f64 wrkopt;
    i32 i;
    i32 ldwp;

    *info = 0;

    // Validate parameters
    if (n < 0) {
        *info = -1;
    } else if (m < 0) {
        *info = -2;
    } else if (p < 0) {
        *info = -3;
    } else if (lda < (n > 0 ? n : 1)) {
        *info = -5;
    } else if (ldb < (n > 0 ? n : 1)) {
        *info = -7;
    } else if (ldc < (p > 0 ? p : 1)) {
        *info = -9;
    } else if (ldu < (n > 0 ? n : 1)) {
        *info = -11;
    } else if (ldwork < 3 * n) {
        *info = -15;
    }

    if (*info != 0) {
        return;
    }

    // Quick return
    if (n == 0) {
        return;
    }

    // Reduce A to real Schur form using orthogonal similarity transformation
    // A <- U'*A*U, accumulate transformation in U, compute eigenvalues in (WR,WI)
    SLC_DGEES("V", "N", dummy_select, &n, a, &lda, &sdim, wr, wi, u, &ldu,
              dwork, &ldwork, bwork, info);

    wrkopt = dwork[0];
    if (*info != 0) {
        return;
    }

    // Apply transformation: B <- U'*B
    if (m > 0) {
        if (ldwork < n * m) {
            // Not enough workspace for DGEMM - use DGEMV
            for (i = 0; i < m; i++) {
                SLC_DCOPY(&n, &b[i * ldb], &int1, dwork, &int1);
                SLC_DGEMV("T", &n, &n, &one, u, &ldu, dwork, &int1, &zero,
                          &b[i * ldb], &int1);
            }
        } else {
            // Use DGEMM for better performance
            SLC_DLACPY("F", &n, &m, b, &ldb, dwork, &n);
            SLC_DGEMM("T", "N", &n, &m, &n, &one, u, &ldu, dwork, &n, &zero,
                      b, &ldb);
            if (wrkopt < (f64)(n * m)) {
                wrkopt = (f64)(n * m);
            }
        }
    }

    // Apply transformation: C <- C*U
    if (p > 0) {
        if (ldwork < n * p) {
            // Not enough workspace for DGEMM - use DGEMV
            for (i = 0; i < p; i++) {
                SLC_DCOPY(&n, &c[i], &ldc, dwork, &int1);
                SLC_DGEMV("T", &n, &n, &one, u, &ldu, dwork, &int1, &zero,
                          &c[i], &ldc);
            }
        } else {
            // Use DGEMM for better performance
            ldwp = (p > 0 ? p : 1);
            SLC_DLACPY("F", &p, &n, c, &ldc, dwork, &ldwp);
            SLC_DGEMM("N", "N", &p, &n, &n, &one, dwork, &ldwp, u, &ldu,
                      &zero, c, &ldc);
            if (wrkopt < (f64)(n * p)) {
                wrkopt = (f64)(n * p);
            }
        }
    }

    dwork[0] = wrkopt;
}
