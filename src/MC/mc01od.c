/*
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * MC01OD - Compute complex polynomial coefficients from zeros
 *
 * Computes the coefficients of a complex polynomial P(x) from its zeros:
 *   P(x) = (x - r(1)) * (x - r(2)) * ... * (x - r(K))
 * where r(i) = REZ(i) + j*IMZ(i)
 */

#include "slicot.h"
#include "slicot_blas.h"

void mc01od(i32 k, const f64 *rez, const f64 *imz, f64 *rep, f64 *imp,
            f64 *dwork, i32 *info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    i32 int1 = 1;

    if (k < 0) {
        *info = -1;
        return;
    }

    *info = 0;
    rep[0] = ONE;
    imp[0] = ZERO;

    if (k == 0) {
        return;
    }

    i32 k2 = k + 2;

    for (i32 i = 1; i <= k; i++) {
        f64 u = rez[i - 1];
        f64 v = imz[i - 1];

        dwork[0] = ZERO;
        dwork[k2 - 1] = ZERO;

        SLC_DCOPY(&i, rep, &int1, &dwork[1], &int1);
        SLC_DCOPY(&i, imp, &int1, &dwork[k2], &int1);

        if (u != ZERO) {
            f64 neg_u = -u;
            SLC_DAXPY(&i, &neg_u, rep, &int1, dwork, &int1);
            SLC_DAXPY(&i, &neg_u, imp, &int1, &dwork[k2 - 1], &int1);
        }

        if (v != ZERO) {
            SLC_DAXPY(&i, &v, imp, &int1, dwork, &int1);
            f64 neg_v = -v;
            SLC_DAXPY(&i, &neg_v, rep, &int1, &dwork[k2 - 1], &int1);
        }

        i32 ip1 = i + 1;
        SLC_DCOPY(&ip1, dwork, &int1, rep, &int1);
        SLC_DCOPY(&ip1, &dwork[k2 - 1], &int1, imp, &int1);
    }
}
