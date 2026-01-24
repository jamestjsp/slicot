/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include <ctype.h>
#include <math.h>
#include <stdlib.h>

void df01md(const char *sico, i32 n, f64 dt, f64 *a, i32 *info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const f64 TWO = 2.0;
    const f64 FOUR = 4.0;

    *info = 0;

    char sic = (char)toupper((unsigned char)sico[0]);
    bool lsico = (sic == 'S');

    if (!lsico && sic != 'C') {
        *info = -1;
        return;
    }

    i32 m = 0;
    if (n > 4) {
        m = n - 1;
        while ((m % 2) == 0) {
            m = m / 2;
        }
    }
    if (m != 1) {
        *info = -2;
        return;
    }

    m = n - 1;
    i32 md2 = (n + 1) / 2;
    f64 pibym = FOUR * atan(ONE) / (f64)m;

    f64 *dwork = (f64 *)malloc((n + 1) * sizeof(f64));
    if (dwork == NULL) {
        *info = -999;
        return;
    }

    i32 i2 = 0;
    dwork[md2] = ZERO;
    dwork[2 * md2 - 1] = ZERO;

    f64 a0 = ZERO;
    bool lsig;

    if (lsico) {
        lsig = true;
        dwork[0] = -TWO * a[1];
        dwork[md2 - 1] = TWO * a[m - 1];

        for (i32 i = 4; i <= m; i += 2) {
            i2 = i2 + 1;
            dwork[i2] = a[i - 3] - a[i - 1];
            dwork[md2 + i2] = -a[i - 2];
        }
    } else {
        lsig = false;
        dwork[0] = TWO * a[0];
        dwork[md2 - 1] = TWO * a[n - 1];
        a0 = a[1];

        for (i32 i = 4; i <= m; i += 2) {
            i2 = i2 + 1;
            dwork[i2] = TWO * a[i - 2];
            dwork[md2 + i2] = TWO * (a[i - 3] - a[i - 1]);
            a0 = a0 + a[i - 1];
        }

        a0 = TWO * a0;
    }

    i32 info_dg = 0;
    dg01nd("I", md2 - 1, dwork, &dwork[md2], &info_dg);

    if (lsico) {
        a[0] = ZERO;
        a[n - 1] = ZERO;
    } else {
        a[0] = TWO * dt * (dwork[0] + a0);
        a[n - 1] = TWO * dt * (dwork[0] - a0);
    }

    i32 ind1 = md2;
    i32 ind2 = n - 1;

    for (i32 i = 1; i <= m - 1; i += 2) {
        f64 w1 = dwork[ind1];
        f64 w2 = dwork[ind2];
        if (lsig) {
            w2 = -w2;
        }
        f64 w3 = TWO * sin(pibym * (f64)i);
        a[i] = dt * (w1 + w2 - (w1 - w2) / w3);
        ind1 = ind1 + 1;
        ind2 = ind2 - 1;
    }

    ind1 = 1;
    ind2 = md2 - 2;

    for (i32 i = 2; i <= m - 2; i += 2) {
        f64 w1 = dwork[ind1];
        f64 w2 = dwork[ind2];
        if (lsig) {
            w2 = -w2;
        }
        f64 w3 = TWO * sin(pibym * (f64)i);
        a[i] = dt * (w1 + w2 - (w1 - w2) / w3);
        ind1 = ind1 + 1;
        ind2 = ind2 - 1;
    }

    free(dwork);
}
