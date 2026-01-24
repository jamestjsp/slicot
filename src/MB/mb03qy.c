/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"

void mb03qy(
    const i32 n,
    const i32 l,
    f64* a,
    const i32 lda,
    f64* u,
    const i32 ldu,
    f64* e1,
    f64* e2,
    i32* info
)
{
    const f64 ZERO = 0.0;

    i32 l1;
    f64 ew1, ew2, cs, sn;

    *info = 0;

    if (n < 2) {
        *info = -1;
    } else if (l < 1 || l >= n) {
        *info = -2;
    } else if (lda < n) {
        *info = -4;
    } else if (ldu < n) {
        *info = -6;
    }

    if (*info != 0) {
        i32 neginfo = -(*info);
        SLC_XERBLA("MB03QY", &neginfo);
        return;
    }

    l1 = l;  // l is 1-based in Fortran, convert to 0-based index for the block
    i32 l0 = l - 1;  // 0-based index

    // Compute eigenvalues and Givens rotation using DLANV2
    // DLANV2 reduces the 2x2 block to standard form
    SLC_DLANV2(&a[l0 + l0*lda], &a[l0 + l1*lda], &a[l1 + l0*lda], &a[l1 + l1*lda],
               e1, e2, &ew1, &ew2, &cs, &sn);

    // If eigenvalues are real, e2 gets the second eigenvalue
    if (*e2 == ZERO) {
        *e2 = ew1;
    }

    // Apply the transformation to A: rows L and L+1, columns L+2 to N
    if (l1 < n - 1) {
        i32 ncols = n - l1 - 1;
        SLC_DROT(&ncols, &a[l0 + (l1+1)*lda], &lda, &a[l1 + (l1+1)*lda], &lda, &cs, &sn);
    }

    // Apply the transformation to A: rows 1 to L-1, columns L and L+1
    i32 nrows = l0;
    i32 one = 1;
    if (nrows > 0) {
        SLC_DROT(&nrows, &a[0 + l0*lda], &one, &a[0 + l1*lda], &one, &cs, &sn);
    }

    // Accumulate the transformation in U
    SLC_DROT(&n, &u[0 + l0*ldu], &one, &u[0 + l1*ldu], &one, &cs, &sn);
}
