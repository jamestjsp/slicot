// SPDX-License-Identifier: BSD-3-Clause

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>

void ma01dd(f64 ar1, f64 ai1, f64 ar2, f64 ai2, f64 eps, f64 safemn, f64 *d) {
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const f64 TWO = 2.0;
    const f64 FOUR = 4.0;

    f64 par = FOUR - TWO * eps;
    f64 big = par / safemn;

    if (big * safemn > par) {
        big = ONE / safemn;
    }

    f64 mx1 = fmax(fabs(ar1), fabs(ai1));
    f64 mx2 = fmax(fabs(ar2), fabs(ai2));
    f64 mx = fmax(mx1, mx2);

    f64 d1, d2;

    if (mx == ZERO) {
        *d = ZERO;
        return;
    } else if (mx < big) {
        if (mx2 == ZERO) {
            *d = SLC_DLAPY2(&ar1, &ai1);
            return;
        } else if (mx1 == ZERO) {
            *d = SLC_DLAPY2(&ar2, &ai2);
            return;
        } else {
            f64 dr = ar1 - ar2;
            f64 di = ai1 - ai2;
            d1 = SLC_DLAPY2(&dr, &di);
        }
    } else {
        d1 = big;
    }

    if (mx > ONE / big) {
        f64 ap1 = SLC_DLAPY2(&ar1, &ai1);
        f64 ap2 = SLC_DLAPY2(&ar2, &ai2);

        if (mx1 <= big && mx2 <= big) {
            f64 dr = (ar1 / ap1) / ap1 - (ar2 / ap2) / ap2;
            f64 di = (ai2 / ap2) / ap2 - (ai1 / ap1) / ap1;
            d2 = SLC_DLAPY2(&dr, &di);
        } else if (mx1 <= big) {
            d2 = ONE / ap1;
        } else if (mx2 <= big) {
            d2 = ONE / ap2;
        } else {
            d2 = ZERO;
        }
    } else {
        d2 = big;
    }

    *d = fmin(d1, d2);
}
