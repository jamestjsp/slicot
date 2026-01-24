/*
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "slicot.h"
#include "slicot_blas.h"

void mc01md(i32 dp, f64 alpha, i32 k, const f64 *p, f64 *q, i32 *info)
{
    const f64 ZERO = 0.0;
    i32 int1 = 1;
    i32 dp_plus_1 = dp + 1;

    *info = 0;

    if (dp < 0) {
        *info = -1;
        return;
    }

    if (k <= 0 || k > dp + 1) {
        *info = -3;
        return;
    }

    SLC_DCOPY(&dp_plus_1, p, &int1, q, &int1);

    if (dp == 0 || alpha == ZERO) {
        return;
    }

    for (i32 j = 0; j < k; j++) {
        for (i32 i = dp - 1; i >= j; i--) {
            q[i] = q[i] + alpha * q[i + 1];
        }
    }
}
