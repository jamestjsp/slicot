/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"

void sb08gd(
    const i32 n,
    const i32 m,
    const i32 p,
    f64* a,
    const i32 lda,
    f64* b,
    const i32 ldb,
    f64* c,
    const i32 ldc,
    f64* d,
    const i32 ldd,
    const f64* br,
    const i32 ldbr,
    f64* dr,
    const i32 lddr,
    f64* dwork,
    i32* iwork,
    i32* info
)
{
    const f64 ONE = 1.0;
    const f64 ZERO = 0.0;
    const f64 MONE = -1.0;

    *info = 0;

    i32 max1n = n > 1 ? n : 1;
    i32 max1p = p > 1 ? p : 1;

    if (n < 0) {
        *info = -1;
    } else if (m < 0) {
        *info = -2;
    } else if (p < 0) {
        *info = -3;
    } else if (lda < max1n) {
        *info = -5;
    } else if (ldb < max1n) {
        *info = -7;
    } else if (ldc < max1p) {
        *info = -9;
    } else if (ldd < max1p) {
        *info = -11;
    } else if (ldbr < max1n) {
        *info = -13;
    } else if (lddr < max1p) {
        *info = -15;
    }

    if (*info != 0) {
        i32 neginfo = -(*info);
        SLC_XERBLA("SB08GD", &neginfo);
        return;
    }

    if (p == 0) {
        dwork[0] = ONE;
        return;
    }

    f64 drnorm = SLC_DLANGE("1", &p, &p, dr, &lddr, dwork);

    SLC_DGETRF(&p, &p, dr, &lddr, iwork, info);
    if (*info != 0) {
        *info = 1;
        dwork[0] = ZERO;
        return;
    }

    if (n > 0) {
        SLC_DGETRS("N", &p, &n, dr, &lddr, iwork, c, &ldc, info);
        SLC_DGEMM("N", "N", &n, &n, &p, &MONE, br, &ldbr, c, &ldc, &ONE, a, &lda);
    }

    if (m > 0) {
        SLC_DGETRS("N", &p, &m, dr, &lddr, iwork, d, &ldd, info);
        if (n > 0) {
            SLC_DGEMM("N", "N", &n, &m, &p, &MONE, br, &ldbr, d, &ldd, &ONE, b, &ldb);
        }
    }

    f64 rcond;
    SLC_DGECON("1", &p, dr, &lddr, &drnorm, &rcond, dwork, iwork, info);
    if (rcond <= SLC_DLAMCH("E")) {
        *info = 2;
    }

    dwork[0] = rcond;
}
