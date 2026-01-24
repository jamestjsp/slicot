// SPDX-License-Identifier: BSD-3-Clause

#include "slicot/mb03.h"
#include <math.h>

i32 mb03nd(i32 n, f64 theta, const f64 *q2, const f64 *e2, f64 pivmin, i32 *info) {
    *info = 0;

    if (n < 0) {
        *info = -1;
        return 0;
    }

    if (n == 0 || theta < 0.0) {
        return 0;
    }

    i32 numeig = n;
    f64 t = -theta;
    f64 r = t;

    if (fabs(r) < pivmin) {
        r = -pivmin;
    }

    for (i32 j = 0; j < n - 1; j++) {
        r = t - q2[j] / r;
        if (fabs(r) < pivmin) {
            r = -pivmin;
        }
        if (r > 0.0) {
            numeig--;
        }

        r = t - e2[j] / r;
        if (fabs(r) < pivmin) {
            r = -pivmin;
        }
        if (r > 0.0) {
            numeig--;
        }
    }

    r = t - q2[n - 1] / r;
    if (fabs(r) < pivmin) {
        r = -pivmin;
    }
    if (r > 0.0) {
        numeig--;
    }

    return numeig;
}
