/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 *
 * MC01VD: Compute roots of a quadratic equation with real coefficients
 *
 * Computes roots of: a*x^2 + b*x + c = 0
 *
 * The roots r1, r2 are computed as:
 *   r1 = (-b - sign(b)*sqrt(b^2 - 4*a*c)) / (2*a)
 *   r2 = c / (a*r1)
 * unless a = 0, in which case r1 = -c/b.
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>

void mc01vd(f64 a, f64 b, f64 c, f64* z1re, f64* z1im, f64* z2re, f64* z2im, i32* info)
{
    const f64 zero = 0.0;
    const f64 one = 1.0;
    const f64 four = 4.0;

    bool ovflow;
    i32 beta, ea, eaplec, eb, eb2, ec, ed;
    f64 absa, absb, absc, big, m1, m2, ma, mb, mc_val, md, sfmin, w;

    *info = 0;
    beta = (i32)SLC_DLAMCH("B");
    sfmin = SLC_DLAMCH("S");
    big = one / sfmin;

    if (a == zero) {
        if (b == zero) {
            *info = 1;
        } else {
            ovflow = false;
            *z2re = zero;
            if (c != zero) {
                absb = fabs(b);
                if (absb >= one) {
                    if (fabs(c) >= absb * sfmin) {
                        *z2re = -c / b;
                    }
                } else {
                    if (fabs(c) <= absb * big) {
                        *z2re = -c / b;
                    } else {
                        ovflow = true;
                        *z2re = big;
                        if ((b > zero ? one : -one) * (c > zero ? one : -one) > zero) {
                            *z2re = -big;
                        }
                    }
                }
            }
            if (ovflow) {
                *info = 1;
            } else {
                *z1re = big;
                *z1im = zero;
                *z2im = zero;
                *info = 2;
            }
        }
        return;
    }

    if (c == zero) {
        ovflow = false;
        *z1re = zero;
        if (b != zero) {
            absa = fabs(a);
            if (absa >= one) {
                if (fabs(b) >= absa * sfmin) {
                    *z1re = -b / a;
                }
            } else {
                if (fabs(b) <= absa * big) {
                    *z1re = -b / a;
                } else {
                    ovflow = true;
                    *z1re = big;
                }
            }
        }
        if (ovflow) {
            *info = 3;
        }
        *z1im = zero;
        *z2re = zero;
        *z2im = zero;
        return;
    }

    // A and C are non-zero
    if (b == zero) {
        ovflow = false;
        absc = sqrt(fabs(c));
        absa = sqrt(fabs(a));
        w = zero;
        if (absa >= one) {
            if (absc >= absa * sfmin) {
                w = absc / absa;
            }
        } else {
            if (absc <= absa * big) {
                w = absc / absa;
            } else {
                ovflow = true;
                w = big;
            }
        }
        if (ovflow) {
            *info = 4;
        } else {
            if ((a > zero ? one : -one) * (c > zero ? one : -one) > zero) {
                // Same sign: complex roots
                *z1re = zero;
                *z2re = zero;
                *z1im = w;
                *z2im = -w;
            } else {
                // Opposite sign: real roots
                *z1re = w;
                *z2re = -w;
                *z1im = zero;
                *z2im = zero;
            }
        }
        return;
    }

    // A, B and C are non-zero
    mc01sw(a, beta, &ma, &ea);
    mc01sw(b, beta, &mb, &eb);
    mc01sw(c, beta, &mc_val, &ec);

    // Compute discriminant D = MD * BETA^ED
    eaplec = ea + ec;
    eb2 = 2 * eb;
    if (eaplec > eb2) {
        mc01sy(mb * mb, eb2 - eaplec, beta, &w, &ovflow);
        w = w - four * ma * mc_val;
        mc01sw(w, beta, &md, &ed);
        ed = ed + eaplec;
    } else {
        mc01sy(four * ma * mc_val, eaplec - eb2, beta, &w, &ovflow);
        w = mb * mb - w;
        mc01sw(w, beta, &md, &ed);
        ed = ed + eb2;
    }

    if (ed % 2 != 0) {
        ed = ed + 1;
        md = md / beta;
    }

    // Complex roots (MD < 0)
    if (md < zero) {
        mc01sy(-mb / (2 * ma), eb - ea, beta, z1re, &ovflow);
        if (ovflow) {
            *info = 4;
        } else {
            mc01sy(sqrt(-md) / (2 * ma), ed / 2 - ea, beta, z1im, &ovflow);
            if (ovflow) {
                *info = 4;
            } else {
                *z2re = *z1re;
                *z2im = -(*z1im);
            }
        }
        return;
    }

    // Real roots (MD >= 0)
    md = sqrt(md);
    ed = ed / 2;
    if (ed > eb) {
        mc01sy(fabs(mb), eb - ed, beta, &w, &ovflow);
        w = w + md;
        m1 = -(mb > zero ? one : -one) * w / (2 * ma);
        mc01sy(m1, ed - ea, beta, z1re, &ovflow);
        if (ovflow) {
            *z1re = big;
            *info = 3;
        }
        m2 = -(mb > zero ? one : -one) * 2 * mc_val / w;
        mc01sy(m2, ec - ed, beta, z2re, &ovflow);
    } else {
        mc01sy(md, ed - eb, beta, &w, &ovflow);
        w = w + fabs(mb);
        m1 = -(mb > zero ? one : -one) * w / (2 * ma);
        mc01sy(m1, eb - ea, beta, z1re, &ovflow);
        if (ovflow) {
            *z1re = big;
            *info = 3;
        }
        m2 = -(mb > zero ? one : -one) * 2 * mc_val / w;
        mc01sy(m2, ec - eb, beta, z2re, &ovflow);
    }
    *z1im = zero;
    *z2im = zero;
}
