/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 *
 * TB01TD: Reduce state-space (A,B,C,D) to balanced form
 */

#include "slicot.h"
#include "slicot_blas.h"

void tb01td(i32 n, i32 m, i32 p,
            f64* a, i32 lda,
            f64* b, i32 ldb,
            f64* c, i32 ldc,
            f64* d, i32 ldd,
            i32* low, i32* igh,
            f64* scstat, f64* scin, f64* scout,
            f64* dwork, i32* info)
{
    const f64 one = 1.0;
    i32 int1 = 1;
    i32 max1n = (n > 1) ? n : 1;

    *info = 0;

    /* Validate parameters */
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
    } else if (ldc < ((p > 1) ? p : 1)) {
        *info = -9;
    } else if (ldd < ((p > 1) ? p : 1)) {
        *info = -11;
    }

    if (*info != 0) {
        return;
    }

    /* Quick return if all dimensions zero */
    i32 max_nmp = n;
    if (m > max_nmp) max_nmp = m;
    if (p > max_nmp) max_nmp = p;

    if (max_nmp == 0) {
        *low = 1;
        *igh = n;  /* = 0 */
        return;
    }

    /* Permute states and balance a submatrix of A using DGEBAL */
    SLC_DGEBAL("Both", &n, a, &lda, low, igh, scstat, info);

    /* Use information in SCSTAT to transform B and C */
    for (i32 k = 0; k < n; k++) {
        i32 kold = k;

        /* Check if k is outside the balanced submatrix [low-1, igh-1] */
        /* Note: low/igh are 1-based from DGEBAL */
        if (((*low - 1) > kold) || (kold > (*igh - 1))) {
            /* kold < low-1 or kold > igh-1 */
            if (kold < (*low - 1)) {
                /* Fortran: KOLD = LOW - KOLD where KOLD was 1-based
                 * In C: kold is 0-based, low is 1-based
                 * Fortran kold_f = k_f = k+1
                 * After: KOLD = LOW - (k+1) = low - k - 1
                 * Then kold_c = kold_f - 1 = low - k - 2
                 */
                kold = (*low) - kold - 2;
            }

            /* Get permutation index from scstat (1-based in scstat) */
            i32 knew = (i32)scstat[kold] - 1;  /* Convert to 0-based */

            if (knew != kold) {
                /* Exchange rows kold and knew of B */
                SLC_DSWAP(&m, &b[kold], &ldb, &b[knew], &ldb);

                /* Exchange columns kold and knew of C */
                SLC_DSWAP(&p, &c[kold * ldc], &int1, &c[knew * ldc], &int1);
            }
        }
    }

    /* Scale rows of B and columns of C for balanced submatrix */
    if (*igh != *low) {
        for (i32 k = *low - 1; k < *igh; k++) {  /* low-1 to igh-1 (0-based) */
            f64 scale = scstat[k];
            f64 inv_scale = one / scale;

            /* Scale the k-th row of B by 1/scale */
            SLC_DSCAL(&m, &inv_scale, &b[k], &ldb);

            /* Scale the k-th column of C by scale */
            SLC_DSCAL(&p, &scale, &c[k * ldc], &int1);
        }
    }

    /* Calculate column and row sum norms of balanced A */
    f64 acnorm = SLC_DLANGE("1", &n, &n, a, &lda, dwork);
    f64 arnorm = SLC_DLANGE("I", &n, &n, a, &lda, dwork);

    /* Scale columns of B (inputs) to have norms roughly equal to acnorm */
    tb01ty(1, 0, 0, n, m, acnorm, b, ldb, scin);

    /* Scale rows of C (outputs) to have norms roughly equal to arnorm */
    tb01ty(0, 0, 0, p, n, arnorm, c, ldc, scout);

    /* Apply input and output scalings to D */
    for (i32 j = 0; j < m; j++) {
        f64 scale = scin[j];

        for (i32 i = 0; i < p; i++) {
            d[i + j * ldd] = d[i + j * ldd] * (scale * scout[i]);
        }

        /* Set SCIN(j) = 1/scale */
        scin[j] = one / scale;
    }
}
