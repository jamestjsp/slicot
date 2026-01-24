/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <stddef.h>

f64 ma02jz(bool ltran1, bool ltran2, i32 n, const c128 *q1, i32 ldq1,
           const c128 *q2, i32 ldq2, c128 *res, i32 ldres)
{
    const c128 ZERO = 0.0 + 0.0 * I;
    const c128 ONE = 1.0 + 0.0 * I;
    const c128 MONE = -1.0 + 0.0 * I;
    const f64 TWO = 2.0;

    if (n == 0) {
        return 0.0;
    }

    f64 temp;

    if (ltran1) {
        SLC_ZGEMM("N", "C", &n, &n, &n, &ONE, q1, &ldq1, q1, &ldq1,
                  &ZERO, res, &ldres);
    } else {
        SLC_ZGEMM("C", "N", &n, &n, &n, &ONE, q1, &ldq1, q1, &ldq1,
                  &ZERO, res, &ldres);
    }

    if (ltran2) {
        SLC_ZGEMM("N", "C", &n, &n, &n, &ONE, q2, &ldq2, q2, &ldq2,
                  &ONE, res, &ldres);
    } else {
        SLC_ZGEMM("C", "N", &n, &n, &n, &ONE, q2, &ldq2, q2, &ldq2,
                  &ONE, res, &ldres);
    }

    for (i32 i = 0; i < n; i++) {
        res[i + i * ldres] -= ONE;
    }

    temp = SLC_ZLANGE("F", &n, &n, res, &ldres, NULL);

    if (ltran1 && ltran2) {
        SLC_ZGEMM("N", "C", &n, &n, &n, &ONE, q2, &ldq2, q1, &ldq1,
                  &ZERO, res, &ldres);
        SLC_ZGEMM("N", "C", &n, &n, &n, &ONE, q1, &ldq1, q2, &ldq2,
                  &MONE, res, &ldres);
    } else if (ltran1) {
        SLC_ZGEMM("C", "C", &n, &n, &n, &ONE, q2, &ldq2, q1, &ldq1,
                  &ZERO, res, &ldres);
        SLC_ZGEMM("N", "N", &n, &n, &n, &ONE, q1, &ldq1, q2, &ldq2,
                  &MONE, res, &ldres);
    } else if (ltran2) {
        SLC_ZGEMM("N", "N", &n, &n, &n, &ONE, q2, &ldq2, q1, &ldq1,
                  &ZERO, res, &ldres);
        SLC_ZGEMM("C", "C", &n, &n, &n, &ONE, q1, &ldq1, q2, &ldq2,
                  &MONE, res, &ldres);
    } else {
        SLC_ZGEMM("C", "N", &n, &n, &n, &ONE, q2, &ldq2, q1, &ldq1,
                  &ZERO, res, &ldres);
        SLC_ZGEMM("C", "N", &n, &n, &n, &ONE, q1, &ldq1, q2, &ldq2,
                  &MONE, res, &ldres);
    }

    f64 temp2 = SLC_ZLANGE("F", &n, &n, res, &ldres, NULL);
    temp = SLC_DLAPY2(&temp, &temp2);

    return sqrt(TWO) * temp;
}
