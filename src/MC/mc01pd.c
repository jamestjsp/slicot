/*
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "slicot.h"
#include "slicot_blas.h"

void mc01pd(i32 k, const f64 *rez, const f64 *imz, f64 *p, f64 *dwork, i32 *info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    i32 int1 = 1;

    if (k < 0) {
        *info = -1;
        return;
    }

    *info = 0;
    p[0] = ONE;

    if (k == 0) {
        return;
    }

    i32 i = 1;
    while (i <= k) {
        f64 u = rez[i - 1];
        f64 v = imz[i - 1];
        dwork[0] = ZERO;

        if (v == ZERO) {
            SLC_DCOPY(&i, p, &int1, &dwork[1], &int1);
            f64 neg_u = -u;
            SLC_DAXPY(&i, &neg_u, p, &int1, dwork, &int1);
            i = i + 1;
        } else {
            if (i == k) {
                *info = k;
                return;
            }
            if ((u != rez[i]) || (v != -imz[i])) {
                *info = i + 1;
                return;
            }

            dwork[1] = ZERO;
            SLC_DCOPY(&i, p, &int1, &dwork[2], &int1);
            f64 neg_2u = -(u + u);
            SLC_DAXPY(&i, &neg_2u, p, &int1, &dwork[1], &int1);
            f64 u2_plus_v2 = u * u + v * v;
            SLC_DAXPY(&i, &u2_plus_v2, p, &int1, dwork, &int1);
            i = i + 2;
        }

        SLC_DCOPY(&i, dwork, &int1, p, &int1);
    }
}
