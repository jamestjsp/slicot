/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <ctype.h>
#include <math.h>

void de01od(const char *conv, i32 n, f64 *a, f64 *b, i32 *info)
{
    const f64 ZERO = 0.0;
    const f64 HALF = 0.5;
    const f64 ONE = 1.0;

    *info = 0;

    char c = (char)toupper((unsigned char)conv[0]);
    bool lconv = (c == 'C');

    if (!lconv && c != 'D') {
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

    dg01md("D", n, a, b, info);

    f64 ast;
    if (lconv) {
        ast = a[0] * b[0];
    } else {
        if (b[0] == ZERO) {
            ast = ZERO;
        } else {
            ast = a[0] / b[0];
        }
    }

    i32 nd2p1 = n / 2 + 1;
    j = nd2p1;

    for (i32 kj = nd2p1; kj <= n; kj++) {
        j--;

        f64 ac = HALF * (a[j] + a[kj - 1]);
        f64 as = HALF * (b[j] - b[kj - 1]);

        f64 bc = HALF * (b[kj - 1] + b[j]);
        f64 bs = HALF * (a[kj - 1] - a[j]);

        f64 cr, ci;
        if (lconv) {
            cr = ac * bc - as * bs;
            ci = as * bc + ac * bs;
        } else {
            f64 max_bc_bs = fabs(bc);
            if (fabs(bs) > max_bc_bs) {
                max_bc_bs = fabs(bs);
            }
            if (max_bc_bs == ZERO) {
                cr = ZERO;
                ci = ZERO;
            } else {
                SLC_DLADIV(&ac, &as, &bc, &bs, &cr, &ci);
            }
        }

        a[j] = cr;
        b[j] = ci;
        a[kj - 1] = cr;
        b[kj - 1] = -ci;
    }

    a[0] = ast;
    b[0] = ZERO;

    dg01md("I", n, a, b, info);

    f64 scale = ONE / (f64)n;
    i32 inc = 1;
    SLC_DSCAL(&n, &scale, a, &inc);
}
