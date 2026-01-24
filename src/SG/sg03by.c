/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include <math.h>

void sg03by(
    const f64 xr, const f64 xi,
    const f64 yr, const f64 yi,
    f64* cr, f64* ci,
    f64* sr, f64* si,
    f64* z
)
{
    const f64 one = 1.0, zero = 0.0;

    *z = fmax(fabs(xr), fmax(fabs(xi), fmax(fabs(yr), fabs(yi))));

    if (*z == zero) {
        *cr = one;
        *ci = zero;
        *sr = zero;
        *si = zero;
    } else {
        f64 xr_scaled = xr / (*z);
        f64 xi_scaled = xi / (*z);
        f64 yr_scaled = yr / (*z);
        f64 yi_scaled = yi / (*z);

        *z = (*z) * sqrt(xr_scaled * xr_scaled + xi_scaled * xi_scaled +
                         yr_scaled * yr_scaled + yi_scaled * yi_scaled);
        *cr = xr / (*z);
        *ci = xi / (*z);
        *sr = yr / (*z);
        *si = yi / (*z);
    }
}
