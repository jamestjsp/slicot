/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>

void mb03qx(
    const i32 n,
    const f64* t,
    const i32 ldt,
    f64* wr,
    f64* wi,
    i32* info
)
{
    const f64 ZERO = 0.0;

    *info = 0;

    if (n < 0) {
        *info = -1;
    } else if (ldt < (n > 1 ? n : 1)) {
        *info = -3;
    }

    if (*info != 0) {
        i32 neginfo = -(*info);
        SLC_XERBLA("MB03QX", &neginfo);
        return;
    }

    i32 inext = 0;

    for (i32 i = 0; i < n; i++) {
        if (i < inext) {
            continue;
        }

        if (i != n - 1) {
            if (t[(i + 1) + i * ldt] != ZERO) {
                i32 i1 = i + 1;
                f64 a11 = t[i + i * ldt];
                f64 a12 = t[i + i1 * ldt];
                f64 a21 = t[i1 + i * ldt];
                f64 a22 = t[i1 + i1 * ldt];

                f64 cs, sn;
                SLC_DLANV2(&a11, &a12, &a21, &a22, &wr[i], &wi[i],
                           &wr[i1], &wi[i1], &cs, &sn);

                inext = i + 2;
                continue;
            }
        }

        inext = i + 1;
        wr[i] = t[i + i * ldt];
        wi[i] = ZERO;
    }
}
