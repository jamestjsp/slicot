/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include <ctype.h>

void dg01nd(const char *indi, i32 n, f64 *xr, f64 *xi, i32 *info)
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

    if (!lindi) {
        dg01ny(indi, n, xr, xi);
    }

    dg01md(indi, n, xr, xi, info);

    if (lindi) {
        dg01ny(indi, n, xr, xi);
    }
}
