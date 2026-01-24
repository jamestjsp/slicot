// SPDX-License-Identifier: BSD-3-Clause

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>

void ma01dz(f64 ar1, f64 ai1, f64 b1, f64 ar2, f64 ai2, f64 b2,
            f64 eps, f64 safemn, f64 *d1, f64 *d2, i32 *iwarn) {
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const f64 TWO = 2.0;
    const f64 FOUR = 4.0;

    *iwarn = 0;

    f64 par = FOUR - TWO * eps;
    f64 big = par / safemn;

    if (big * safemn > par) {
        big = ONE / safemn;
    }

    f64 mx1 = fmax(fabs(ar1), fabs(ai1));
    f64 mx2 = fmax(fabs(ar2), fabs(ai2));

    if (b1 == ZERO) {
        if (mx1 == ZERO) {
            *d1 = ZERO;
            *d2 = ZERO;
            *iwarn = 1;
        } else {
            if (b2 == ZERO) {
                *d1 = ZERO;
                if (mx2 == ZERO) {
                    *d2 = ZERO;
                    *iwarn = 1;
                } else {
                    *d2 = ONE;
                }
            } else if (b2 > ONE) {
                if (mx2 > b2 / big) {
                    *d1 = b2 / SLC_DLAPY2(&ar2, &ai2);
                    *d2 = ONE;
                } else {
                    *d1 = ONE;
                    *d2 = ZERO;
                }
            } else if (mx2 > ZERO) {
                *d1 = b2 / SLC_DLAPY2(&ar2, &ai2);
                *d2 = ONE;
            } else {
                *d1 = ONE;
                *d2 = ZERO;
            }
        }
    } else if (b2 == ZERO) {
        if (mx2 == ZERO) {
            *d1 = ZERO;
            *d2 = ZERO;
            *iwarn = 1;
        } else {
            if (b1 > ONE) {
                if (mx1 > b1 / big) {
                    *d1 = b1 / SLC_DLAPY2(&ar1, &ai1);
                    *d2 = ONE;
                } else {
                    *d1 = ONE;
                    *d2 = ZERO;
                }
            } else if (mx1 > ZERO) {
                *d1 = b1 / SLC_DLAPY2(&ar1, &ai1);
                *d2 = ONE;
            } else {
                *d1 = ONE;
                *d2 = ZERO;
            }
        }
    } else {
        bool inf1, inf2, zer1, zer2;
        f64 ap1, ap2;

        if (b1 >= ONE) {
            inf1 = false;
            f64 t1 = ar1 / b1;
            f64 t2 = ai1 / b1;
            ap1 = SLC_DLAPY2(&t1, &t2);
            zer1 = (ap1 < ONE / big);
        } else {
            zer1 = false;
            inf1 = (mx1 > b1 * big);
            if (!inf1) {
                f64 t1 = ar1 / b1;
                f64 t2 = ai1 / b1;
                ap1 = SLC_DLAPY2(&t1, &t2);
            }
        }

        if (b2 >= ONE) {
            inf2 = false;
            f64 t1 = ar2 / b2;
            f64 t2 = ai2 / b2;
            ap2 = SLC_DLAPY2(&t1, &t2);
            zer2 = (ap2 < ONE / big);
        } else {
            zer2 = false;
            inf2 = (mx2 > b2 * big);
            if (!inf2) {
                f64 t1 = ar2 / b2;
                f64 t2 = ai2 / b2;
                ap2 = SLC_DLAPY2(&t1, &t2);
            }
        }

        *d2 = ONE;
        if (zer1 && zer2) {
            *d1 = ZERO;
        } else if (zer1) {
            if (!inf2) {
                *d1 = ap2;
            } else {
                *d1 = ONE;
                *d2 = ZERO;
            }
        } else if (zer2) {
            if (!inf1) {
                *d1 = ap1;
            } else {
                *d1 = ONE;
                *d2 = ZERO;
            }
        } else if (inf1) {
            if (inf2) {
                *d1 = ZERO;
            } else {
                *d1 = b2 / SLC_DLAPY2(&ar2, &ai2);
            }
        } else if (inf2) {
            *d1 = b1 / SLC_DLAPY2(&ar1, &ai1);
        } else {
            f64 pr1 = ar1 / b1;
            f64 pi1 = ai1 / b1;
            f64 pr2 = ar2 / b2;
            f64 pi2 = ai2 / b2;

            f64 dr = pr1 - pr2;
            f64 di = pi1 - pi2;
            f64 diff = SLC_DLAPY2(&dr, &di);

            f64 invdr = (pr1 / ap1) / ap1 - (pr2 / ap2) / ap2;
            f64 invdi = (pi2 / ap2) / ap2 - (pi1 / ap1) / ap1;
            f64 invdiff = SLC_DLAPY2(&invdr, &invdi);

            *d1 = fmin(diff, invdiff);
        }
    }
}
