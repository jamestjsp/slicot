/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"

void mb03ba(
    const i32 k,
    const i32 h,
    const i32 *s,
    i32 *smult,
    i32 *amap,
    i32 *qmap
)
{
    if (s[h - 1] == -1) {
        *smult = -1;

        for (i32 i = 1; i <= h; i++) {
            amap[i - 1] = h - i + 1;
        }

        for (i32 i = h + 1; i <= k; i++) {
            amap[i - 1] = h + 1 - i + k;
        }

        i32 temp = h % k + 1;

        for (i32 i = temp; i >= 1; i--) {
            qmap[temp - i] = i;
        }

        for (i32 i = k; i >= temp + 1; i--) {
            qmap[temp + k - i] = i;
        }
    } else {
        *smult = 1;

        for (i32 i = h; i <= k; i++) {
            amap[i - h] = i;
            qmap[i - h] = i;
        }

        for (i32 i = 1; i <= h - 1; i++) {
            amap[k - h + i] = i;
            qmap[k - h + i] = i;
        }
    }
}
