/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"

void ud01cd(i32 mp, i32 np, i32 dp, i32 nelem, const i32 *rows,
            const i32 *cols, const i32 *degrees, const f64 *coeffs,
            f64 *p, i32 ldp1, i32 ldp2, i32 *info)
{
    *info = 0;

    if (mp < 1) {
        *info = -1;
        return;
    }
    if (np < 1) {
        *info = -2;
        return;
    }
    if (dp < 0) {
        *info = -3;
        return;
    }
    if (ldp1 < mp) {
        *info = -10;
        return;
    }
    if (ldp2 < np) {
        *info = -11;
        return;
    }

    i32 ldp12 = ldp1 * ldp2;
    f64 zero = 0.0;

    for (i32 k = 0; k < dp + 1; k++) {
        SLC_DLASET("Full", &mp, &np, &zero, &zero, &p[k * ldp12], &ldp1);
    }

    i32 coeff_idx = 0;

    for (i32 e = 0; e < nelem; e++) {
        i32 i = rows[e];
        i32 j = cols[e];
        i32 d = degrees[e];

        if (i < 1 || i > mp || j < 1 || j > np || d < 0 || d > dp + 1) {
            *info = 1;
            coeff_idx += d + 1;
            continue;
        }

        i32 i0 = i - 1;
        i32 j0 = j - 1;

        for (i32 k = 0; k <= d; k++) {
            p[i0 + j0 * ldp1 + k * ldp12] = coeffs[coeff_idx++];
        }
    }
}
