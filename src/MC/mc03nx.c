/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"

void mc03nx(i32 mp, i32 np, i32 dp, const f64 *p, i32 ldp1, i32 ldp2,
            f64 *a, i32 lda, f64 *e, i32 lde)
{
    if (mp <= 0 || np <= 0) {
        return;
    }

    i32 h1 = dp * mp;
    i32 hb = h1 - mp;
    i32 he = hb + np;

    f64 zero = 0.0;
    f64 one = 1.0;
    f64 minusone = -1.0;
    i32 int1 = 1;

    SLC_DLASET("Full", &h1, &he, &zero, &one, a, &lda);
    SLC_DLASET("Full", &mp, &hb, &zero, &zero, e, &lde);
    SLC_DLACPY("Full", &hb, &hb, a, &lda, &e[mp], &lde);

    i32 hb1 = hb;
    SLC_DLACPY("Full", &mp, &np, p, &ldp1, &a[hb1 + hb1 * lda], &lda);

    i32 hi = 0;
    for (i32 k = dp; k >= 1; k--) {
        SLC_DLACPY("Full", &mp, &np, &p[k * ldp1 * ldp2], &ldp1, &e[hi + hb1 * lde], &lde);
        hi += mp;
    }

    for (i32 j = hb1; j < he; j++) {
        SLC_DSCAL(&h1, &minusone, &e[j * lde], &int1);
    }
}
