/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <ctype.h>

void de01pd(const char *conv, const char *wght, i32 n, f64 *a, f64 *b, f64 *w, i32 *info)
{
    const f64 HALF = 0.5;
    const f64 TWO = 2.0;

    *info = 0;

    char conv_c = (char)toupper((unsigned char)conv[0]);
    char wght_c = (char)toupper((unsigned char)wght[0]);

    bool lconv = (conv_c == 'C');
    bool lwght = (wght_c == 'A');

    if (!lconv && conv_c != 'D') {
        *info = -1;
        return;
    }
    if (!lwght && wght_c != 'N') {
        *info = -2;
        return;
    }

    i32 m = 0;
    i32 j = 0;
    if (n >= 1) {
        j = n;
        while ((j % 2) == 0) {
            j = j / 2;
            m = m + 1;
        }
        if (j != 1) {
            *info = -3;
            return;
        }
    } else if (n < 0) {
        *info = -3;
        return;
    }

    if (n <= 0) {
        return;
    } else if (n == 1) {
        if (lconv) {
            a[0] = a[0] * b[0];
        } else {
            a[0] = a[0] / b[0];
        }
        return;
    }

    char wght_str[2] = { wght_c, '\0' };
    dg01od("O", wght_str, n, a, w, info);
    wght_str[0] = 'A';
    dg01od("O", wght_str, n, b, w, info);

    i32 len = 1;
    if (lconv) {
        a[0] = TWO * a[0] * b[0];
        a[1] = TWO * a[1] * b[1];

        for (i32 l = 1; l <= m - 1; l++) {
            len = 2 * len;
            i32 r1 = 2 * len;

            for (i32 p1 = len + 1; p1 <= len + len / 2; p1++) {
                i32 p1_idx = p1 - 1;
                i32 r1_idx = r1 - 1;

                f64 t1 = b[p1_idx] + b[r1_idx];
                f64 t2 = b[p1_idx] - b[r1_idx];
                f64 t3 = t2 * a[p1_idx];
                a[p1_idx] = t1 * a[p1_idx] + t2 * a[r1_idx];
                a[r1_idx] = t1 * a[r1_idx] - t3;
                r1 = r1 - 1;
            }
        }
    } else {
        a[0] = HALF * a[0] / b[0];
        a[1] = HALF * a[1] / b[1];

        for (i32 l = 1; l <= m - 1; l++) {
            len = 2 * len;
            i32 r1 = 2 * len;

            for (i32 p1 = len + 1; p1 <= len + len / 2; p1++) {
                i32 p1_idx = p1 - 1;
                i32 r1_idx = r1 - 1;

                f64 t1, t2;
                f64 bp = b[p1_idx] + b[r1_idx];
                f64 br = b[r1_idx] - b[p1_idx];
                SLC_DLADIV(&a[p1_idx], &a[r1_idx], &bp, &br, &t1, &t2);
                a[p1_idx] = t1;
                a[r1_idx] = t2;
                r1 = r1 - 1;
            }
        }
    }

    dg01od("I", "A", n, a, w, info);
    f64 scale;
    if (lconv) {
        scale = HALF / (f64)n;
    } else {
        scale = TWO / (f64)n;
    }
    i32 inc = 1;
    SLC_DSCAL(&n, &scale, a, &inc);
}
