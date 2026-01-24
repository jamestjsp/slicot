/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>

void mc01sd(i32 dp, f64 *p, i32 *s, i32 *t, f64 *mant, i32 *e, i32 *iwork, i32 *info)
{
    const f64 ZERO = 0.0;

    if (dp < 0) {
        *info = -1;
        i32 neginf = 1;
        SLC_XERBLA("MC01SD", &neginf);
        return;
    }

    *info = 0;

    i32 lb = 0;
    while (lb <= dp && p[lb] == ZERO) {
        lb++;
    }

    if (lb == dp + 1) {
        *info = 1;
        return;
    }

    i32 ub = dp;
    while (p[ub] == ZERO) {
        ub--;
    }

    i32 beta = (i32)SLC_DLAMCH("Base");

    for (i32 i = 0; i <= dp; i++) {
        mc01sw(p[i], beta, &mant[i], &e[i]);
    }

    i32 m_val = e[lb];
    if (m_val != 0) {
        for (i32 i = lb; i <= ub; i++) {
            if (mant[i] != ZERO) {
                e[i] = e[i] - m_val;
            }
        }
    }
    *s = -m_val;

    if (ub > 0) {
        m_val = (i32)round((f64)e[ub] / (f64)ub);
    }

    for (i32 i = lb; i <= ub; i++) {
        if (mant[i] != ZERO) {
            e[i] = e[i] - m_val * i;
        }
    }
    *t = -m_val;

    i32 v0 = mc01sx(lb + 1, ub + 1, e, mant);
    i32 j = 1;

    for (i32 i = lb; i <= ub; i++) {
        if (mant[i] != ZERO) {
            iwork[i] = e[i] + i;
        }
    }

    i32 v1 = mc01sx(lb + 1, ub + 1, iwork, mant);
    i32 dv = v1 - v0;
    i32 inc = 0;

    if (dv != 0) {
        if (dv > 0) {
            j = 0;
            inc = -1;
            v1 = v0;
            dv = -dv;

            for (i32 i = lb; i <= ub; i++) {
                iwork[i] = e[i];
            }
        } else {
            inc = 1;
        }

        while (dv < 0) {
            v0 = v1;

            for (i32 i = lb; i <= ub; i++) {
                e[i] = iwork[i];
            }

            j = j + inc;

            for (i32 i = lb; i <= ub; i++) {
                iwork[i] = e[i] + inc * i;
            }

            v1 = mc01sx(lb + 1, ub + 1, iwork, mant);
            dv = v1 - v0;
        }

        *t = *t + j - inc;
    }

    bool ovflow;
    for (i32 i = lb; i <= ub; i++) {
        mc01sy(mant[i], e[i], beta, &p[i], &ovflow);
    }
}
