/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <stdlib.h>

void mc01sy(f64 m, i32 e, i32 b, f64* a, bool* ovflow)
{
    const f64 zero = 0.0;
    const f64 one = 1.0;

    *ovflow = false;

    if (m == zero || e == 0) {
        *a = m;
        return;
    }

    i32 emin = (i32)SLC_DLAMCH("M");
    i32 emax = (i32)SLC_DLAMCH("L");
    f64 mt = m;
    i32 et = e;
    f64 b_f = (f64)b;

    while (fabs(mt) >= b_f) {
        mt = mt / b_f;
        et = et + 1;
    }

    while (fabs(mt) < one) {
        mt = mt * b_f;
        et = et - 1;
    }

    if (et < emin) {
        *a = zero;
        return;
    }

    if (et >= emax) {
        *ovflow = true;
        return;
    }

    i32 expon = abs(et);
    *a = mt;
    f64 base = (et < 0) ? one / b_f : b_f;

    while (expon != 0) {
        if (expon % 2 == 0) {
            base = base * base;
            expon = expon / 2;
        } else {
            *a = (*a) * base;
            expon = expon - 1;
        }
    }
}
