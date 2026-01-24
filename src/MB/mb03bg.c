/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"

void mb03bg(i32 k, i32 n, const i32 *amap, const i32 *s, i32 sinv,
            const f64 *a, i32 lda1, i32 lda2, f64 *wr, f64 *wi) {
    const f64 ONE = 1.0;
    const f64 ZERO = 0.0;

    i32 m = n - 1;
    f64 p1 = ONE;
    f64 p3 = ZERO;
    f64 p4 = ONE;

    i32 ldas = lda1 * lda2;

    for (i32 l = 0; l < k - 1; l++) {
        i32 i = amap[l] - 1;
        const f64 *ai = a + i * ldas;

        f64 amm = ai[m - 1 + (m - 1) * lda1];
        f64 amn = ai[m - 1 + (n - 1) * lda1];
        f64 ann = ai[n - 1 + (n - 1) * lda1];

        if (s[i] == sinv) {
            p3 = p1 * amn + p3 * ann;
        } else {
            p3 = (p3 - p1 * amn / amm) / ann;
        }
        p1 = p1 * amm;
        p4 = p4 * ann;
    }

    i32 i = amap[k - 1] - 1;
    const f64 *ai = a + i * ldas;

    f64 amm = ai[m - 1 + (m - 1) * lda1];
    f64 amn = ai[m - 1 + (n - 1) * lda1];
    f64 anm = ai[n - 1 + (m - 1) * lda1];
    f64 ann = ai[n - 1 + (n - 1) * lda1];

    f64 dwork[4];
    dwork[0] = p1 * amm + p3 * anm;
    dwork[1] = p4 * anm;
    dwork[2] = p1 * amn + p3 * ann;
    dwork[3] = p4 * ann;

    f64 z_dummy[1];
    i32 info;
    i32 two = 2;
    i32 one = 1;

    SLC_DLAHQR(&(i32){0}, &(i32){0}, &two, &one, &two, dwork, &two,
               wr, wi, &one, &two, z_dummy, &one, &info);
}
