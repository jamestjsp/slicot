/*
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "slicot.h"
#include "slicot_blas.h"

void mc01qd(i32 da, i32 *db, const f64 *a, const f64 *b, f64 *rq,
            i32 *iwarn, i32 *info)
{
    const f64 ZERO = 0.0;
    i32 int1 = 1;

    *iwarn = 0;
    *info = 0;

    if (da < -1) {
        *info = -1;
        return;
    }
    if (*db < 0) {
        *info = -2;
        return;
    }

    while (*db >= 0 && b[*db] == ZERO) {
        (*db)--;
        (*iwarn)++;
    }

    if (*db == -1) {
        *info = 1;
        return;
    }

    if (da >= 0) {
        i32 n = da;
        i32 len = n + 1;
        SLC_DCOPY(&len, a, &int1, rq, &int1);

        while (n >= *db) {
            if (rq[n] != ZERO) {
                f64 q = rq[n] / b[*db];
                f64 neg_q = -q;
                SLC_DAXPY(db, &neg_q, b, &int1, &rq[n - *db], &int1);
                rq[n] = q;
            }
            n--;
        }
    }
}
