/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <ctype.h>

void mb03ab(const char *shft, i32 k, i32 n, const i32 *amap, const i32 *s,
            i32 sinv, const f64 *a, i32 lda1, i32 lda2, f64 w1, f64 w2,
            f64 *c1, f64 *s1, f64 *c2, f64 *s2) {
    const f64 ONE = 1.0;
    const f64 ZERO = 0.0;

    char shft_upper = (char)toupper((unsigned char)shft[0]);
    bool sgle = (shft_upper == 'S');
    bool isr = (shft_upper != 'C');

    i32 ai = amap[0] - 1;
    i32 ldas = lda1 * lda2;
    const f64 *a_slice = a + ai * ldas;

    f64 temp, tmp, dum;
    f64 alpha, beta, gamma, delta;
    f64 c23, c3, cx, cy, sx, sy, p;
    f64 c1_v, s1_v, c2_v, s2_v, s3;

    SLC_DLARTG(&a_slice[1], &ONE, &c1_v, &s1_v, &temp);
    SLC_DLARTG(&a_slice[0], &temp, &c2_v, &s2_v, &tmp);

    for (i32 i = k - 1; i >= 1; i--) {
        ai = amap[i] - 1;
        a_slice = a + ai * ldas;
        f64 a11 = a_slice[0];
        f64 a12 = a_slice[lda1];
        f64 a22 = a_slice[1 + lda1];

        if (s[ai] == sinv) {
            alpha = a11 * c2_v + a12 * c1_v * s2_v;
            beta = a22 * c1_v;
            gamma = s1_v;
            SLC_DLARTG(&beta, &gamma, &c1_v, &s1_v, &temp);
            f64 arg1 = temp * s2_v;
            SLC_DLARTG(&alpha, &arg1, &c2_v, &s2_v, &dum);
        } else {
            alpha = s2_v * a11;
            beta = c1_v * c2_v * a22 - s2_v * a12;
            gamma = s1_v * a22;
            cx = c1_v;
            sx = s1_v;
            SLC_DLARTG(&cx, &gamma, &c1_v, &s1_v, &temp);
            temp = c1_v * beta + sx * c2_v * s1_v;
            SLC_DLARTG(&temp, &alpha, &c2_v, &s2_v, &dum);
        }
    }

    if (isr) {
        f64 arg1 = c2_v - w2 * s1_v * s2_v;
        f64 arg2 = c1_v * s2_v;
        SLC_DLARTG(&arg1, &arg2, &c2_v, &s2_v, &temp);
        if (sgle) {
            *c1 = c2_v;
            *s1 = s2_v;
            *c2 = ONE;
            *s2 = ZERO;
            return;
        } else {
            cx = c2_v;
            sx = s2_v;
        }
    } else {
        temp = s1_v * s2_v;
        alpha = c2_v - w1 * temp;
        beta = c1_v * s2_v;
        gamma = w2 * temp;
        SLC_DLARTG(&beta, &gamma, &c1_v, &s1_v, &temp);
        SLC_DLARTG(&alpha, &temp, &c2_v, &s2_v, &dum);

        cx = c1_v;
        sx = s1_v;
        cy = c2_v;
        sy = s2_v;
        s2_v = c1_v * s2_v;
    }

    ai = amap[0] - 1;
    a_slice = a + ai * ldas;
    f64 a11 = a_slice[0];
    f64 a12 = a_slice[lda1];
    f64 a22 = a_slice[1 + lda1];
    f64 a21 = a_slice[1];
    f64 a32 = a_slice[2 + lda1];

    alpha = a12 * s2_v + a11 * c2_v;
    beta = a22 * s2_v + a21 * c2_v;
    gamma = a32 * s2_v;

    SLC_DLARTG(&gamma, &ONE, &c1_v, &s1_v, &temp);
    SLC_DLARTG(&beta, &temp, &c3, &s3, &dum);
    f64 arg1 = c3 * beta + s3 * temp;
    SLC_DLARTG(&alpha, &arg1, &c2_v, &s2_v, &dum);

    for (i32 i = k - 1; i >= 1; i--) {
        ai = amap[i] - 1;
        a_slice = a + ai * ldas;
        a11 = a_slice[0];
        a12 = a_slice[lda1];
        a22 = a_slice[1 + lda1];
        f64 a13 = a_slice[2 * lda1];
        f64 a23 = a_slice[1 + 2 * lda1];
        f64 a33 = a_slice[2 + 2 * lda1];

        if (s[ai] == sinv) {
            temp = c1_v * s3;
            alpha = (a13 * temp + a12 * c3) * s2_v + a11 * c2_v;
            beta = (a23 * temp + a22 * c3) * s2_v;
            gamma = a33 * c1_v;
            delta = s1_v;
            SLC_DLARTG(&gamma, &delta, &c1_v, &s1_v, &temp);
            temp = temp * s2_v * s3;
            SLC_DLARTG(&beta, &temp, &c3, &s3, &tmp);
            SLC_DLARTG(&alpha, &tmp, &c2_v, &s2_v, &dum);
        } else {
            c23 = c2_v * c3;
            tmp = c2_v * s3;
            alpha = c1_v * c3 * a33 - s3 * a23;
            beta = s1_v * c3;
            gamma = c1_v * tmp * a33 + c23 * a23 - s2_v * a13;
            delta = s1_v * tmp;
            tmp = c1_v;
            SLC_DLARTG(&tmp, &(f64){s1_v * a33}, &c1_v, &s1_v, &dum);
            temp = alpha * c1_v + beta * s1_v;
            SLC_DLARTG(&temp, &(f64){s3 * a22}, &c3, &s3, &tmp);
            temp = (c23 * a22 - s2_v * a12) * c3 + (gamma * c1_v + delta * s1_v) * s3;
            SLC_DLARTG(&temp, &(f64){s2_v * a11}, &c2_v, &s2_v, &dum);
        }
    }

    if (isr) {
        temp = w1 * s1_v * s3;
        alpha = c2_v - cx * temp * s2_v;
        beta = (c3 - sx * temp) * s2_v;
        gamma = c1_v * s2_v * s3;
    } else {
        p = s1_v * s3;
        alpha = c2_v + (w2 * sx * sy - w1 * cy) * p * s2_v;
        beta = c3 - w1 * cx * sy * p;
        gamma = c1_v * s3;
        p = s2_v;
    }
    SLC_DLARTG(&beta, &gamma, &c2_v, &s2_v, &temp);
    if (!isr) {
        temp = temp * p;
    }
    SLC_DLARTG(&alpha, &temp, &c1_v, &s1_v, &dum);

    *c1 = c1_v;
    *s1 = s1_v;
    *c2 = c2_v;
    *s2 = s2_v;
}
