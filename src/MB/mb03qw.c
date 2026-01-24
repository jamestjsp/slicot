/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"

void mb03qw(i32 n, i32 l, f64 *a, i32 lda, f64 *e, i32 lde,
            f64 *u, i32 ldu, f64 *v, i32 ldv,
            f64 *alphar, f64 *alphai, f64 *beta, i32 *info)
{
    i32 l1;
    f64 csl, csr, snl, snr;
    i32 int1 = 1;

    *info = 0;

    if (n < 2) {
        *info = -1;
    } else if (l < 1 || l >= n) {
        *info = -2;
    } else if (lda < n) {
        *info = -4;
    } else if (lde < n) {
        *info = -6;
    } else if (ldu < n) {
        *info = -8;
    } else if (ldv < n) {
        *info = -10;
    }

    if (*info != 0) {
        i32 neginfo = -(*info);
        SLC_XERBLA("MB03QW", &neginfo);
        return;
    }

    l1 = l;  // l is already 1-based from Python, l1 = l+1 in Fortran => l in C

    SLC_DLAGV2(&a[(l - 1) + (l - 1) * lda], &lda, &e[(l - 1) + (l - 1) * lde], &lde,
               alphar, alphai, beta, &csl, &snl, &csr, &snr);

    if (l1 < n - 1) {
        i32 nml1 = n - l1 - 1;
        SLC_DROT(&nml1, &a[(l - 1) + (l1 + 1) * lda], &lda,
                 &a[l1 + (l1 + 1) * lda], &lda, &csl, &snl);
        SLC_DROT(&nml1, &e[(l - 1) + (l1 + 1) * lde], &lde,
                 &e[l1 + (l1 + 1) * lde], &lde, &csl, &snl);
    }

    if (l > 1) {
        i32 lm1 = l - 1;
        SLC_DROT(&lm1, &a[(l - 1) * lda], &int1,
                 &a[l1 * lda], &int1, &csr, &snr);
        SLC_DROT(&lm1, &e[(l - 1) * lde], &int1,
                 &e[l1 * lde], &int1, &csr, &snr);
    }

    SLC_DROT(&n, &u[(l - 1) * ldu], &int1, &u[l1 * ldu], &int1, &csl, &snl);
    SLC_DROT(&n, &v[(l - 1) * ldv], &int1, &v[l1 * ldv], &int1, &csr, &snr);
}
