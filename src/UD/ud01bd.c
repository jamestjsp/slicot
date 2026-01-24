/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"

void ud01bd(i32 mp, i32 np, i32 dp, const f64 *data, f64 *p,
            i32 ldp1, i32 ldp2, i32 *info)
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
        *info = -6;
        return;
    }
    if (ldp2 < np) {
        *info = -7;
        return;
    }

    i32 data_idx = 0;
    i32 ldp12 = ldp1 * ldp2;

    for (i32 k = 0; k < dp + 1; k++) {
        for (i32 i = 0; i < mp; i++) {
            for (i32 j = 0; j < np; j++) {
                p[i + j * ldp1 + k * ldp12] = data[data_idx++];
            }
        }
    }
}
