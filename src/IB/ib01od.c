/*
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * IB01OD - Estimate system order from Hankel singular values
 *
 * Estimates system order based on singular values of triangular factor
 * from QR factorization of concatenated block Hankel matrices.
 */

#include "slicot.h"
#include <math.h>
#include <float.h>
#include <stdbool.h>

i32 SLC_IB01OD(char ctrl, i32 nobr, i32 l, const f64 *sv, i32 *n,
               f64 tol, i32 *iwarn, i32 *info)
{
    bool contrl = (ctrl == 'C' || ctrl == 'c');
    i32 lnobr = l * nobr;

    *iwarn = 0;
    *info = 0;

    /* Parameter validation */
    if (!(contrl || ctrl == 'N' || ctrl == 'n')) {
        *info = -1;
        return *info;
    }
    if (nobr <= 0) {
        *info = -2;
        return *info;
    }
    if (l <= 0) {
        *info = -3;
        return *info;
    }

    /* Set TOL if necessary */
    f64 toll = tol;
    if (toll == 0.0) {
        toll = DBL_EPSILON * sv[0] * (f64)nobr;
    }

    /* Obtain the system order */
    *n = 0;
    if (sv[0] != 0.0) {
        *n = nobr;
        if (toll >= 0.0) {
            /* Estimate n based on tolerance TOLL */
            for (i32 i = 0; i < nobr - 1; i++) {
                if (sv[i + 1] < toll) {
                    *n = i + 1;
                    goto done;
                }
            }
        } else {
            /* Estimate n based on largest logarithmic gap */
            f64 gap = 0.0;
            for (i32 i = 0; i < nobr - 1; i++) {
                f64 rnrm = sv[i + 1];
                if (rnrm != 0.0) {
                    rnrm = log10(sv[i]) - log10(rnrm);
                    if (rnrm > gap) {
                        gap = rnrm;
                        *n = i + 1;
                    }
                } else {
                    if (gap == 0.0) {
                        *n = i + 1;
                    }
                    goto done;
                }
            }
        }
    }

done:
    if (*n == 0) {
        /* Return with N = 0 if all singular values are zero */
        *iwarn = 3;
        return 0;
    }

    if (contrl) {
        /* Ask confirmation of the system order */
        i32 ierr;
        SLC_IB01OY(lnobr, nobr - 1, n, sv, &ierr);
    }

    return 0;
}
