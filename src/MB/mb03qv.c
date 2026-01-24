/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"

void mb03qv(
    const i32 n,
    const f64* s,
    const i32 lds,
    const f64* t,
    const i32 ldt,
    f64* alphar,
    f64* alphai,
    f64* beta,
    i32* info
)
{
    *info = 0;

    if (n < 0) {
        *info = -1;
        return;
    }
    i32 max_lds = 1 > n ? 1 : n;
    if (lds < max_lds) {
        *info = -3;
        return;
    }
    i32 max_ldt = 1 > n ? 1 : n;
    if (ldt < max_ldt) {
        *info = -5;
        return;
    }

    if (n == 0) {
        return;
    }

    f64 safmin = SLC_DLAMCH("S");

    i32 inext = 0;  // 0-based
    for (i32 i = 0; i < n; i++) {
        if (i < inext) {
            continue;
        }

        if (i != n - 1) {
            if (s[(i + 1) + i * lds] != 0.0) {
                inext = i + 2;
                i32 lds2 = lds;
                i32 ldt2 = ldt;
                SLC_DLAG2(
                    &s[i + i * lds], &lds2,
                    &t[i + i * ldt], &ldt2,
                    &safmin,
                    &beta[i], &beta[i + 1],
                    &alphar[i], &alphar[i + 1],
                    &alphai[i]
                );
                alphai[i + 1] = -alphai[i];
                continue;
            }
        }

        inext = i + 1;
        alphar[i] = s[i + i * lds];
        alphai[i] = 0.0;
        beta[i] = t[i + i * ldt];
    }
}
