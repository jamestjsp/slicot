/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include <math.h>

void mc01sw(f64 a, i32 b, f64* m, i32* e)
{
    const f64 zero = 0.0;
    const f64 one = 1.0;

    if (a == zero) {
        *m = zero;
        *e = 0;
        return;
    }

    f64 db = (f64)b;
    *m = fabs(a);
    *e = 0;

    while (*m >= db) {
        *m = (*m) / db;
        *e = (*e) + 1;
    }

    while (*m < one) {
        *m = (*m) * db;
        *e = (*e) - 1;
    }

    if (a < zero) {
        *m = -(*m);
    }
}
