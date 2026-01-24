/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Translated from SLICOT Fortran77 to C11
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>

void mb03pd(const char *jobrq, i32 m, i32 n, f64 *a, i32 lda, i32 *jpvt,
            f64 rcond, f64 svlmax, f64 *tau, i32 *rank, f64 *sval,
            f64 *dwork, i32 *info)
{
    const i32 IMAX = 1, IMIN = 2;
    const f64 ZERO = 0.0, ONE = 1.0;

    bool ljobrq;
    i32 i, ismax, ismin, jwork, mn;
    f64 c1, c2, s1, s2, smax, smaxpr, smin, sminpr;
    i32 int1 = 1;

    ljobrq = (jobrq[0] == 'R' || jobrq[0] == 'r');
    mn = (m < n) ? m : n;

    *info = 0;
    if (!ljobrq && jobrq[0] != 'N' && jobrq[0] != 'n') {
        *info = -1;
    } else if (m < 0) {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    } else if (lda < (m > 1 ? m : 1)) {
        *info = -5;
    } else if (rcond < ZERO) {
        *info = -7;
    } else if (svlmax < ZERO) {
        *info = -8;
    }

    if (*info != 0) {
        return;
    }

    if (mn == 0) {
        *rank = 0;
        sval[0] = ZERO;
        sval[1] = ZERO;
        sval[2] = ZERO;
        return;
    }

    if (ljobrq) {
        mb04gd(m, n, a, lda, jpvt, tau, dwork, info);
    }

    smax = fabs(a[(m - 1) + (n - 1) * lda]);
    if (smax == ZERO || svlmax * rcond > smax) {
        *rank = 0;
        sval[0] = smax;
        sval[1] = ZERO;
        sval[2] = ZERO;
    } else {
        ismin = mn - 1;
        ismax = 2 * mn - 1;
        jwork = ismax + 1;
        dwork[ismin] = ONE;
        dwork[ismax] = ONE;
        *rank = 1;
        smin = smax;
        sminpr = smin;

        while (*rank < mn) {
            i32 row_src = m - (*rank) - 1;
            i32 col_start = n - (*rank);

            SLC_DCOPY(rank, &a[row_src + col_start * lda], &lda,
                      &dwork[jwork], &int1);

            i32 rank_val = *rank;
            SLC_DLAIC1(&IMIN, &rank_val, &dwork[ismin], &smin,
                       &dwork[jwork], &a[row_src + (n - (*rank) - 1) * lda],
                       &sminpr, &s1, &c1);
            SLC_DLAIC1(&IMAX, &rank_val, &dwork[ismax], &smax,
                       &dwork[jwork], &a[row_src + (n - (*rank) - 1) * lda],
                       &smaxpr, &s2, &c2);

            if (svlmax * rcond <= smaxpr) {
                if (svlmax * rcond <= sminpr) {
                    if (smaxpr * rcond <= sminpr) {
                        for (i = 0; i < *rank; i++) {
                            dwork[ismin + i] = s1 * dwork[ismin + i];
                            dwork[ismax + i] = s2 * dwork[ismax + i];
                        }
                        ismin = ismin - 1;
                        ismax = ismax - 1;
                        dwork[ismin] = c1;
                        dwork[ismax] = c2;
                        smin = sminpr;
                        smax = smaxpr;
                        (*rank)++;
                        continue;
                    }
                }
            }
            break;
        }
        sval[0] = smax;
        sval[1] = smin;
        sval[2] = sminpr;
    }
}
