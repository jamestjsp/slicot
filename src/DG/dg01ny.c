/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include <math.h>
#include <ctype.h>

void dg01ny(const char *indi, i32 n, f64 *xr, f64 *xi)
{
    char ind = (char)toupper((unsigned char)indi[0]);
    bool lindi = (ind == 'D');

    f64 pi2 = 8.0 * atan(1.0);
    if (lindi) {
        pi2 = -pi2;
    }

    f64 whelp = pi2 / (f64)(2 * n);
    f64 wstpi = sin(whelp);
    whelp = sin(0.5 * whelp);
    f64 wstpr = -2.0 * whelp * whelp;
    f64 wi = 0.0;
    f64 wr;

    if (lindi) {
        wr = 1.0;
        xr[n] = xr[0];
        xi[n] = xi[0];
    } else {
        wr = -1.0;
    }

    i32 n2 = n / 2 + 1;
    for (i32 i = 0; i < n2; i++) {
        i32 j = n - i;
        f64 ar = xr[i] + xr[j];
        f64 ai = xi[i] - xi[j];
        f64 br = xi[i] + xi[j];
        f64 bi = xr[j] - xr[i];

        if (lindi) {
            ar *= 0.5;
            ai *= 0.5;
            br *= 0.5;
            bi *= 0.5;
        }

        f64 helpr = wr * br - wi * bi;
        f64 helpi = wr * bi + wi * br;
        xr[i] = ar + helpr;
        xi[i] = ai + helpi;
        xr[j] = ar - helpr;
        xi[j] = helpi - ai;

        whelp = wr;
        wr = wr + wr * wstpr - wi * wstpi;
        wi = wi + wi * wstpr + whelp * wstpi;
    }
}
