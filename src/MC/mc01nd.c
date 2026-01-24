/*
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * MC01ND - Evaluate real polynomial at complex point using Horner's algorithm
 *
 * Given real polynomial P(x) = p[0] + p[1]*x + ... + p[dp]*x^dp,
 * computes P(x0) where x0 = xr + xi*j is complex.
 */

#include "slicot.h"

void mc01nd(i32 dp, f64 xr, f64 xi, const f64 *p, f64 *vr, f64 *vi, i32 *info)
{
    const f64 ZERO = 0.0;

    *info = 0;

    if (dp < 0) {
        *info = -1;
        return;
    }

    *vr = p[dp];
    *vi = ZERO;

    if (dp == 0)
        return;

    if (xi == ZERO) {
        for (i32 i = dp - 1; i >= 0; i--) {
            *vr = (*vr) * xr + p[i];
        }
    } else {
        for (i32 i = dp - 1; i >= 0; i--) {
            f64 t = (*vr) * xr - (*vi) * xi + p[i];
            *vi = (*vi) * xr + (*vr) * xi;
            *vr = t;
        }
    }
}
