// SPDX-License-Identifier: BSD-3-Clause

#include "slicot/mb03.h"
#include "slicot_blas.h"
#include <math.h>

void mb03md(i32 n, i32 *l, f64 *theta, const f64 *q, const f64 *e,
            const f64 *q2, const f64 *e2, f64 pivmin, f64 tol, f64 reltol,
            i32 *iwarn, i32 *info) {
    const f64 ZERO = 0.0;
    const f64 TWO = 2.0;
    const f64 FUDGE = 2.0;

    *iwarn = 0;
    *info = 0;

    if (n < 0) {
        *info = -1;
        return;
    }
    if (*l < 0 || *l > n) {
        *info = -2;
        return;
    }

    if (n == 0) {
        return;
    }

    if (*l == 0) {
        *theta = ZERO;
    }

    if (*theta < ZERO) {
        if (*l == 1) {
            *theta = mb03my(n, q, 1);
            if (n == 1) {
                return;
            }
        } else {
            *theta = fabs(q[n - *l]);
        }
    }

    i32 dummy_info = 0;
    i32 num = mb03nd(n, *theta, q2, e2, pivmin, &dummy_info);
    if (num == *l) {
        return;
    }

    f64 y, z;
    i32 numz;
    f64 th;

    if (num < *l) {
        th = fabs(q[0]);
        z = ZERO;
        y = *theta;
        numz = n;

        for (i32 i = 0; i < n - 1; i++) {
            f64 h = fabs(q[i + 1]);
            f64 max_th_h = (th > h) ? th : h;
            f64 gershgorin = max_th_h + fabs(e[i]);
            if (gershgorin > z) {
                z = gershgorin;
            }
            th = h;
        }

        f64 eps = SLC_DLAMCH("Epsilon");
        z = z + FUDGE * fabs(z) * eps * (f64)n + FUDGE * pivmin;
    } else {
        z = *theta;
        y = ZERO;
        numz = num;
    }

    f64 abs_y = fabs(y);
    f64 abs_z = fabs(z);
    f64 max_yz = (abs_y > abs_z) ? abs_y : abs_z;
    f64 threshold_tol = (tol > pivmin) ? tol : pivmin;
    f64 rel_threshold = reltol * max_yz;
    f64 threshold = (threshold_tol > rel_threshold) ? threshold_tol : rel_threshold;

    while (num != *l && fabs(z - y) > threshold) {
        th = (y + z) / TWO;
        num = mb03nd(n, th, q2, e2, pivmin, &dummy_info);

        if (num < *l) {
            y = th;
        } else {
            z = th;
            numz = num;
        }

        abs_y = fabs(y);
        abs_z = fabs(z);
        max_yz = (abs_y > abs_z) ? abs_y : abs_z;
        rel_threshold = reltol * max_yz;
        threshold = (threshold_tol > rel_threshold) ? threshold_tol : rel_threshold;
    }

    if (num != *l) {
        *l = numz;
        *theta = z;
        *iwarn = 1;
    } else {
        *theta = th;
    }
}
