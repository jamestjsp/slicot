/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <ctype.h>
#include <math.h>

void mb03ad(const char *shft, i32 k, i32 n, const i32 *amap, const i32 *s,
            i32 sinv, const f64 *a, i32 lda1, i32 lda2, f64 *c1, f64 *s1,
            f64 *c2, f64 *s2) {
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const f64 TWO = 2.0;

    char shft_upper = (char)toupper((unsigned char)shft[0]);
    bool sgle = (shft_upper == 'S');

    *c1 = ONE;
    *s1 = ZERO;
    *c2 = ONE / sqrt(TWO);
    *s2 = *c2;

    i32 ldas = lda1 * lda2;

    f64 alpha, beta, gamma, delta, temp;
    f64 c3, s3;
    f64 c1_v = *c1, s1_v = *s1, c2_v = *c2, s2_v = *s2;

    for (i32 i = k - 1; i >= 1; i--) {
        i32 ai = amap[i] - 1;
        const f64 *a_slice = a + ai * ldas;

        f64 a11 = a_slice[0];
        f64 ann = a_slice[(n - 1) + (n - 1) * lda1];
        f64 an1_n = a_slice[(n - 2) + (n - 1) * lda1];
        f64 an1_n1 = a_slice[(n - 2) + (n - 2) * lda1];

        if (s[ai] == sinv) {
            alpha = c2_v * a11;
            gamma = s2_v * ann;
            beta = s2_v * an1_n;
            beta = c1_v * beta + s1_v * an1_n1;
            SLC_DLARTG(&alpha, &gamma, &c2_v, &s2_v, &temp);
            temp = c1_v * temp;
            SLC_DLARTG(&temp, &beta, &c1_v, &s1_v, &alpha);
        } else {
            temp = a11;
            beta = s2_v * temp;
            temp = c2_v * temp;
            alpha = s1_v * temp;
            gamma = ann;
            delta = c2_v * gamma;
            gamma = s2_v * gamma;
            SLC_DLARTG(&delta, &beta, &c2_v, &s2_v, &c3);
            delta = c1_v * an1_n - s1_v * gamma;
            alpha = c2_v * alpha - s2_v * delta;
            gamma = c1_v * an1_n1;
            SLC_DLARTG(&gamma, &alpha, &c1_v, &s1_v, &temp);
        }
    }

    i32 ai = amap[0] - 1;
    const f64 *a_slice = a + ai * ldas;

    f64 a11 = a_slice[0];
    f64 a21 = a_slice[1];
    f64 ann = a_slice[(n - 1) + (n - 1) * lda1];
    f64 an1_n = a_slice[(n - 2) + (n - 1) * lda1];
    f64 an1_n1 = a_slice[(n - 2) + (n - 2) * lda1];
    f64 an_n1 = a_slice[(n - 1) + (n - 2) * lda1];

    alpha = a11 * c2_v - ann * s2_v;
    beta = c1_v * (c2_v * a21);
    gamma = c1_v * (s2_v * an1_n) + s1_v * an1_n1;
    alpha = alpha * c1_v - an_n1 * s1_v;
    SLC_DLARTG(&alpha, &beta, &c1_v, &s1_v, &temp);

    if (sgle) {
        *c1 = c1_v;
        *s1 = s1_v;
        *c2 = ONE;
        *s2 = ZERO;
        return;
    }

    SLC_DLARTG(&temp, &gamma, &c2_v, &s2_v, &alpha);

    alpha = c2_v;
    gamma = (an1_n1 * c1_v) * c2_v + an_n1 * s2_v;
    delta = (an1_n1 * s1_v) * c2_v;
    SLC_DLARTG(&gamma, &delta, &c3, &s3, &temp);
    SLC_DLARTG(&alpha, &temp, &c2_v, &s2_v, &alpha);

    for (i32 i = k - 1; i >= 1; i--) {
        ai = amap[i] - 1;
        a_slice = a + ai * ldas;

        f64 a11_i = a_slice[0];
        f64 a12_i = a_slice[lda1];
        f64 a22_i = a_slice[1 + lda1];
        f64 an1_n1_i = a_slice[(n - 2) + (n - 2) * lda1];

        if (s[ai] == sinv) {
            alpha = (a11_i * c1_v + a12_i * s1_v) * c2_v;
            beta = (a22_i * s1_v) * c2_v;
            gamma = an1_n1_i * s2_v;
            SLC_DLARTG(&alpha, &beta, &c1_v, &s1_v, &temp);
            SLC_DLARTG(&temp, &gamma, &c2_v, &s2_v, &alpha);
        } else {
            alpha = c1_v * a11_i;
            gamma = s1_v * a11_i;
            beta = c1_v * a12_i + s1_v * a22_i;
            delta = -s1_v * a12_i + c1_v * a22_i;
            SLC_DLARTG(&delta, &gamma, &c1_v, &s1_v, &temp);
            alpha = -alpha * s2_v;
            beta = -beta * s2_v;
            alpha = c1_v * alpha + s1_v * beta;
            beta = c2_v * an1_n1_i;
            SLC_DLARTG(&beta, &alpha, &c2_v, &s2_v, &temp);
            s2_v = -s2_v;
        }
    }

    ai = amap[0] - 1;
    a_slice = a + ai * ldas;

    a11 = a_slice[0];
    a21 = a_slice[1];
    f64 a12 = a_slice[lda1];
    f64 a22 = a_slice[1 + lda1];
    f64 a32 = a_slice[2 + lda1];

    alpha = c1_v * a11 + s1_v * a12;
    beta = c1_v * a21 + s1_v * a22;
    gamma = s1_v * a32;
    alpha = c2_v * alpha - s2_v * c3;
    beta = c2_v * beta - s2_v * s3;
    gamma = c2_v * gamma;
    SLC_DLARTG(&beta, &gamma, &c2_v, &s2_v, &temp);
    SLC_DLARTG(&alpha, &temp, &c1_v, &s1_v, &beta);

    *c1 = c1_v;
    *s1 = s1_v;
    *c2 = c2_v;
    *s2 = s2_v;
}
