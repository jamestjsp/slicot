/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 2002-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>

void tb04bx(i32 ip, i32 iz, f64* a, i32 lda, f64* b, const f64* c, f64 d,
            const f64* pr, const f64* pi, const f64* zr, const f64* zi,
            f64* gain, i32* iwork)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const f64 TWO = 2.0;
    const f64 P1 = 0.1;
    const f64 ONEP1 = 1.1;
    const i32 int1 = 1;

    i32 i, info;
    f64 s0, s;

    if (ip == 0) {
        *gain = ZERO;
        return;
    }

    s0 = ZERO;

    for (i = 0; i < ip; i++) {
        s = fabs(pr[i]);
        if (pi[i] != ZERO) {
            s = s + fabs(pi[i]);
        }
        if (s > s0) s0 = s;
    }

    for (i = 0; i < iz; i++) {
        s = fabs(zr[i]);
        if (zi[i] != ZERO) {
            s = s + fabs(zi[i]);
        }
        if (s > s0) s0 = s;
    }

    s0 = TWO * s0 + P1;
    if (s0 <= ONE) {
        s0 = ONEP1;
    }

    for (i = 0; i < ip; i++) {
        a[i + i * lda] = a[i + i * lda] - s0;
    }

    mb02sd(ip, a, lda, iwork, &info);

    mb02rd("N", ip, 1, a, lda, iwork, b, ip, &info);

    *gain = d - SLC_DDOT(&ip, c, &int1, b, &int1);

    i = 0;
    while (i < ip) {
        if (pi[i] == ZERO) {
            *gain = (*gain) * (s0 - pr[i]);
            i = i + 1;
        } else {
            *gain = (*gain) * (s0 * (s0 - TWO * pr[i]) + pr[i] * pr[i] + pi[i] * pi[i]);
            i = i + 2;
        }
    }

    i = 0;
    while (i < iz) {
        if (zi[i] == ZERO) {
            *gain = (*gain) / (s0 - zr[i]);
            i = i + 1;
        } else {
            *gain = (*gain) / (s0 * (s0 - TWO * zr[i]) + zr[i] * zr[i] + zi[i] * zi[i]);
            i = i + 2;
        }
    }
}
