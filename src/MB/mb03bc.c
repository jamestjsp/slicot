/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>

void mb03bc(i32 k, const i32 *amap, const i32 *s, i32 sinv, f64 *a,
            i32 lda1, i32 lda2, const f64 *macpar, f64 *cv, f64 *sv,
            f64 *dwork) {
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const f64 HALF = 0.5;
    const f64 TWO = 2.0;

    f64 rmax = macpar[0];
    f64 rmxs = sqrt(rmax);
    f64 rmin = macpar[1];
    f64 rmns = sqrt(rmin);
    f64 sfmn = macpar[2];
    f64 eps = macpar[3];
    f64 base = macpar[4];
    f64 twos = sqrt(TWO);

    i32 ldas = lda1 * lda2;
    i32 pw = 0;
    f64 t11 = ONE;
    f64 t12 = ZERO;
    f64 t22 = ONE;

    for (i32 i = 1; i < k; i++) {
        i32 ai = amap[i] - 1;
        f64 *a_slice = a + ai * ldas;
        f64 a11 = a_slice[0];
        f64 a12 = a_slice[lda1];
        f64 a22 = a_slice[1 + lda1];

        if (s[ai] != sinv) {
            f64 temp = a11;
            a11 = a22;
            a22 = temp;
            a12 = -a12;
        }

        f64 mx = fabs(a11) / rmxs;
        f64 mx2 = fabs(t11) / rmxs;
        while (mx * mx2 >= ONE) {
            if (mx >= ONE) {
                mx /= base;
                a11 /= base;
                a22 /= base;
                a12 /= base;
            }
            if (mx2 >= ONE) {
                mx2 /= base;
                t11 /= base;
                t22 /= base;
                t12 /= base;
            }
        }

        mx = fabs(a22) / rmxs;
        mx2 = fabs(t22) / rmxs;
        while (mx * mx2 >= ONE) {
            if (mx >= ONE) {
                mx /= base;
                a11 /= base;
                a22 /= base;
                a12 /= base;
            }
            if (mx2 >= ONE) {
                mx2 /= base;
                t11 /= base;
                t22 /= base;
                t12 /= base;
            }
        }

        mx = fabs(a12) / rmxs;
        mx2 = fabs(t11) / rmxs;
        while (mx * mx2 >= HALF) {
            if (mx >= HALF) {
                mx /= base;
                a11 /= base;
                a22 /= base;
                a12 /= base;
            }
            if (mx2 >= HALF) {
                mx2 /= base;
                t11 /= base;
                t22 /= base;
                t12 /= base;
            }
        }

        mx = fabs(a22) / rmxs;
        mx2 = fabs(t12) / rmxs;
        while (mx * mx2 >= HALF) {
            if (mx >= HALF) {
                mx /= base;
                a11 /= base;
                a22 /= base;
                a12 /= base;
            }
            if (mx2 >= HALF) {
                mx2 /= base;
                t11 /= base;
                t22 /= base;
                t12 /= base;
            }
        }

        mx = fmax(fmax(fabs(a11), fabs(a22)), fabs(a12));
        mx2 = fmax(fmax(fabs(t11), fabs(t22)), fabs(t12));
        if (mx != ZERO && mx2 != ZERO) {
            while ((mx <= ONE / rmns && mx2 <= rmns) ||
                   (mx <= rmns && mx2 <= ONE / rmns)) {
                if (mx <= mx2) {
                    mx *= base;
                    a11 *= base;
                    a22 *= base;
                    a12 *= base;
                } else {
                    mx2 *= base;
                    t11 *= base;
                    t22 *= base;
                    t12 *= base;
                }
            }
        }

        t12 = t11 * a12 + t12 * a22;
        t11 = t11 * a11;
        t22 = t22 * a22;

        if (i < k - 1) {
            dwork[pw] = t11;
            dwork[pw + 1] = t12;
            dwork[pw + 2] = t22;
            pw += 3;
        }
    }

    f64 temp = fmax(fabs(t11 / TWO) + fabs(t12 / TWO), fabs(t22 / TWO));
    if (temp > rmax / (TWO * twos)) {
        temp /= base;
        t11 /= base;
        t12 /= base;
        t22 /= base;
    }

    while (temp < rmax / (TWO * base * twos) && t11 != ZERO && t22 != ZERO) {
        i32 scl = 0;
        if (fabs(t22) <= twos * rmin) {
            scl = 1;
        } else if (eps * fabs(t12) > fabs(t22)) {
            if (sqrt(fabs(t11)) * sqrt(fabs(t22)) <= sqrt(twos) * rmns * sqrt(fabs(t12))) {
                scl = 1;
            }
        } else {
            if (fabs(t11) <= twos * rmin * (ONE + fabs(t12 / t22))) {
                scl = 1;
            }
        }
        if (scl == 1) {
            temp *= base;
            t11 *= base;
            t12 *= base;
            t22 *= base;
        } else {
            break;
        }
    }

    f64 ssmin, ssmax, sr, cr, sl, cl;
    SLC_DLASV2(&t11, &t12, &t22, &ssmin, &ssmax, &sr, &cr, &sl, &cl);

    f64 s11 = t11;
    f64 s22 = t22;

    cv[k - 1] = cr;
    sv[k - 1] = sr;

    for (i32 i = k - 1; i >= 1; i--) {
        i32 ai = amap[i] - 1;
        f64 *a_slice = a + ai * ldas;
        f64 a11, a12, a22;

        if (s[ai] == sinv) {
            a11 = a_slice[0];
            a12 = a_slice[lda1];
            a22 = a_slice[1 + lda1];
        } else {
            a11 = a_slice[1 + lda1];
            a12 = -a_slice[lda1];
            a22 = a_slice[0];
        }

        f64 cc, sc;
        if (i > 1) {
            pw -= 3;
            t11 = dwork[pw];
            t12 = dwork[pw + 1];
            t22 = dwork[pw + 2];

            f64 b11, b12, b22;
            if (fabs(sr * cl * s22) < fabs(sl * cr * s11)) {
                b11 = t22;
                b22 = t11;
                b12 = -t12;
                cc = cl;
                sc = sl;
            } else {
                b11 = a11;
                b12 = a12;
                b22 = a22;
                cc = cr;
                sc = sr;
            }

            f64 mx_b = fmax(fmax(fabs(b11), fabs(b12)), fabs(b22));
            if (mx_b > rmax / TWO) {
                b11 /= TWO;
                b22 /= TWO;
                b12 /= TWO;
            }
            f64 arg1 = b11 * cc + b12 * sc;
            f64 arg2 = sc * b22;
            SLC_DLARTG(&arg1, &arg2, &cc, &sc, &temp);
        } else {
            cc = cl;
            sc = sl;
        }

        if (fabs(sc) < sfmn * fabs(a22)) {
            a_slice[0] = sc * sr * a22 + cc * (cr * a11 + sr * a12);
        } else {
            a_slice[0] = (a22 / sc) * sr;
        }
        if (fabs(sr) < sfmn * fabs(a11)) {
            a_slice[1 + lda1] = sc * sr * a11 + cr * (cc * a22 - sc * a12);
        } else {
            a_slice[1 + lda1] = (a11 / sr) * sc;
        }
        a_slice[lda1] = (a12 * cr - a11 * sr) * cc + a22 * cr * sc;

        if (s[ai] != sinv) {
            temp = a_slice[0];
            a_slice[0] = a_slice[1 + lda1];
            a_slice[1 + lda1] = temp;
            a_slice[lda1] = -a_slice[lda1];
        }

        cr = cc;
        sr = sc;
        cv[i - 1] = cr;
        sv[i - 1] = sr;
        s11 = t11;
        s22 = t22;
    }

    cv[0] = cl;
    sv[0] = sl;
}
