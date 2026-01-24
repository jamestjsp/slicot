/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>

void mb03be(i32 k, const i32 *amap, const i32 *s, i32 sinv, f64 *a, i32 lda1,
            i32 lda2) {
    const f64 ZERO = 0.0;

    i32 ldas = lda1 * lda2;
    i32 int1 = 1;
    i32 int2 = 2;
    f64 cs, sn, ct, st, temp, ulp;

    for (i32 iter = 0; iter < 20; iter++) {
        mb03ad("Single", k, 2, amap, s, sinv, a, lda1, lda2, &cs, &sn, &ct,
               &st);

        i32 ai = amap[0] - 1;
        f64 *a_slice = a + ai * ldas;
        SLC_DROT(&int2, &a_slice[0], &lda1, &a_slice[1], &lda1, &cs, &sn);

        for (i32 l = k - 1; l >= 1; l--) {
            ai = amap[l] - 1;
            a_slice = a + ai * ldas;

            if (s[ai] == sinv) {
                SLC_DROT(&int2, &a_slice[0], &int1, &a_slice[lda1], &int1, &cs,
                         &sn);
                temp = a_slice[0];
                SLC_DLARTG(&temp, &a_slice[1], &cs, &sn, &a_slice[0]);
                a_slice[1] = ZERO;
                temp = cs * a_slice[lda1] + sn * a_slice[1 + lda1];
                a_slice[1 + lda1] = cs * a_slice[1 + lda1] - sn * a_slice[lda1];
                a_slice[lda1] = temp;
            } else {
                SLC_DROT(&int2, &a_slice[0], &lda1, &a_slice[1], &lda1, &cs,
                         &sn);
                temp = a_slice[1 + lda1];
                SLC_DLARTG(&temp, &a_slice[1], &cs, &sn, &a_slice[1 + lda1]);
                a_slice[1] = ZERO;
                sn = -sn;
                temp = cs * a_slice[0] + sn * a_slice[lda1];
                a_slice[lda1] = cs * a_slice[lda1] - sn * a_slice[0];
                a_slice[0] = temp;
            }
        }

        ai = amap[0] - 1;
        a_slice = a + ai * ldas;
        SLC_DROT(&int2, &a_slice[0], &int1, &a_slice[lda1], &int1, &cs, &sn);

        if (iter == 5) {
            ulp = SLC_DLAMCH("Precision");
            f64 maxabs = fabs(a_slice[0]);
            if (fabs(a_slice[lda1]) > maxabs) maxabs = fabs(a_slice[lda1]);
            if (fabs(a_slice[1 + lda1]) > maxabs)
                maxabs = fabs(a_slice[1 + lda1]);
            if (fabs(a_slice[1]) < ulp * maxabs) {
                return;
            }
        } else if (iter > 9) {
            if (fabs(a_slice[1]) <
                ulp * fmax(fmax(fabs(a_slice[0]), fabs(a_slice[lda1])),
                           fabs(a_slice[1 + lda1]))) {
                return;
            }
        }
    }
}
