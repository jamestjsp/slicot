/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>

void sg03br(
    const f64 xr, const f64 xi, const f64 yr, const f64 yi,
    f64* c, f64* sr, f64* si, f64* zr, f64* zi
)
{
    const f64 one = 1.0, two = 2.0, zero = 0.0;

    i32 count, i;
    f64 d, di, dr, eps, safmin, safmn2, safmx2, scale;
    f64 ti, tr, x2, x2s, xis, xrs, y2, y2s, yis, yrs;

    safmin = SLC_DLAMCH("S");
    eps = SLC_DLAMCH("E");
    safmn2 = pow(SLC_DLAMCH("B"), (i32)(log(safmin / eps) / log(SLC_DLAMCH("B")) / two));
    safmx2 = one / safmn2;

    scale = fmax(fmax(fabs(xr), fabs(xi)), fmax(fabs(yr), fabs(yi)));

    xrs = xr;
    xis = xi;
    yrs = yr;
    yis = yi;
    count = 0;

    if (scale >= safmx2) {
        while (scale >= safmx2) {
            count++;
            xrs *= safmn2;
            xis *= safmn2;
            yrs *= safmn2;
            yis *= safmn2;
            scale *= safmn2;
        }
    } else if (scale <= safmn2) {
        if (yr == zero && yi == zero) {
            *c = one;
            *sr = zero;
            *si = zero;
            *zr = xr;
            *zi = xi;
            return;
        }

        while (scale <= safmn2) {
            count--;
            xrs *= safmx2;
            xis *= safmx2;
            yrs *= safmx2;
            yis *= safmx2;
            scale *= safmx2;
        }
    }

    x2 = xrs * xrs + xis * xis;
    y2 = yrs * yrs + yis * yis;

    if (x2 <= fmax(y2, one) * safmin) {
        if (xr == zero && xi == zero) {
            *c = zero;
            *zr = SLC_DLAPY2(&yr, &yi);
            *zi = zero;

            d = SLC_DLAPY2(&yrs, &yis);
            *sr = yrs / d;
            *si = -yis / d;
            return;
        }

        x2s = SLC_DLAPY2(&xrs, &xis);
        y2s = sqrt(y2);
        *c = x2s / y2s;

        if (fmax(fabs(xr), fabs(xi)) > one) {
            d = SLC_DLAPY2(&xr, &xi);
            tr = xr / d;
            ti = xi / d;
        } else {
            dr = safmx2 * xr;
            di = safmx2 * xi;
            d = SLC_DLAPY2(&dr, &di);
            tr = dr / d;
            ti = di / d;
        }

        *sr = tr * (yrs / y2s) + ti * (yis / y2s);
        *si = ti * (yrs / y2s) - tr * (yis / y2s);
        *zr = (*c) * xr + (*sr) * yr - (*si) * yi;
        *zi = (*c) * xi + (*si) * yr + (*sr) * yi;
    } else {
        x2s = sqrt(one + y2 / x2);

        *zr = x2s * xrs;
        *zi = x2s * xis;
        *c = one / x2s;
        d = x2 + y2;

        *sr = (*zr) / d;
        *si = (*zi) / d;
        dr = (*sr) * yrs + (*si) * yis;
        *si = (*si) * yrs - (*sr) * yis;
        *sr = dr;

        if (count != 0) {
            if (count > 0) {
                for (i = 0; i < count; i++) {
                    *zr *= safmx2;
                    *zi *= safmx2;
                }
            } else {
                for (i = 0; i < -count; i++) {
                    *zr *= safmn2;
                    *zi *= safmn2;
                }
            }
        }
    }
}
