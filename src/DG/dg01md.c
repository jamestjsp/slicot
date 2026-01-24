/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include <math.h>
#include <ctype.h>

void dg01md(const char *indi, i32 n, f64 *xr, f64 *xi, i32 *info)
{
    *info = 0;

    char ind = (char)toupper((unsigned char)indi[0]);
    bool lindi = (ind == 'D');

    if (!lindi && ind != 'I') {
        *info = -1;
        return;
    }

    i32 j = 0;
    if (n >= 2) {
        j = n;
        while ((j % 2) == 0) {
            j = j / 2;
        }
    }
    if (j != 1) {
        *info = -2;
        return;
    }

    i32 i, k, l, m;
    f64 tr, ti, whelp, wi, wr, wstpi, wstpr;
    f64 pi2 = 8.0 * atan(1.0);

    if (lindi) {
        pi2 = -pi2;
    }

    j = 0;
    for (i = 0; i < n; i++) {
        if (j > i) {
            tr = xr[i];
            ti = xi[i];
            xr[i] = xr[j];
            xi[i] = xi[j];
            xr[j] = tr;
            xi[j] = ti;
        }
        k = n / 2;
        while (j >= k && k >= 1) {
            j = j - k;
            k = k / 2;
        }
        j = j + k;
    }

    i = 1;
    while (i < n) {
        l = 2 * i;
        whelp = pi2 / (f64)l;
        wstpi = sin(whelp);
        whelp = sin(0.5 * whelp);
        wstpr = -2.0 * whelp * whelp;
        wr = 1.0;
        wi = 0.0;

        for (j = 0; j < i; j++) {
            for (k = j; k < n; k += l) {
                m = k + i;
                tr = wr * xr[m] - wi * xi[m];
                ti = wr * xi[m] + wi * xr[m];
                xr[m] = xr[k] - tr;
                xi[m] = xi[k] - ti;
                xr[k] = xr[k] + tr;
                xi[k] = xi[k] + ti;
            }
            whelp = wr;
            wr = wr + wr * wstpr - wi * wstpi;
            wi = wi + whelp * wstpi + wi * wstpr;
        }
        i = l;
    }
}
