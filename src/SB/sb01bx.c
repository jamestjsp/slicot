/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include <math.h>

void sb01bx(
    const bool reig,
    const i32 n,
    const f64 xr,
    const f64 xi,
    f64* wr,
    f64* wi,
    f64* s,
    f64* p
)
{
    i32 i, j, k;
    f64 x, y;

    j = 0;

    if (reig) {
        y = fabs(wr[0] - xr);
        for (i = 1; i < n; i++) {
            x = fabs(wr[i] - xr);
            if (x < y) {
                y = x;
                j = i;
            }
        }

        *s = wr[j];
        k = n - 1 - j;  // Number of elements after j

        if (k > 0) {
            for (i = j; i < j + k; i++) {
                wr[i] = wr[i + 1];
            }
            wr[n - 1] = *s;
        }

        *p = *s;
    } else {
        y = fabs(wr[0] - xr) + fabs(wi[0] - xi);

        for (i = 2; i < n; i += 2) {
            x = fabs(wr[i] - xr) + fabs(wi[i] - xi);
            if (x < y) {
                y = x;
                j = i;
            }
        }

        x = wr[j];
        y = wi[j];
        k = n - 2 - j;  // Number of pairs after selected pair

        if (k > 0) {
            for (i = j; i < j + k; i++) {
                wr[i] = wr[i + 2];
                wi[i] = wi[i + 2];
            }
            wr[n - 2] = x;
            wi[n - 2] = y;
            wr[n - 1] = x;
            wi[n - 1] = -y;
        }

        *s = x + x;
        *p = x * x + y * y;
    }
}
