/*
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "slicot.h"
#include "slicot_blas.h"

void mb04wp(const i32 n, const i32 ilo, f64 *u1, const i32 ldu1,
            f64 *u2, const i32 ldu2, const f64 *cs, const f64 *tau,
            f64 *dwork, const i32 ldwork, i32 *info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    i32 i, j, ierr;
    i32 minwrk, wrkopt;
    i32 nh;
    bool lquery;

    *info = 0;
    nh = n - ilo;

    i32 max_1_n = (1 > n) ? 1 : n;

    if (n < 0) {
        *info = -1;
    } else if (ilo < 1 || ilo > (n > 1 ? n : 1)) {
        *info = -2;
    } else if (ldu1 < max_1_n) {
        *info = -4;
    } else if (ldu2 < max_1_n) {
        *info = -6;
    } else {
        lquery = (ldwork == -1);
        minwrk = (1 > 2 * nh) ? 1 : 2 * nh;

        if (lquery) {
            if (n == 0) {
                wrkopt = 1;
            } else {
                mb04wd(false, false, nh, nh, nh,
                       u1, ldu1, u2, ldu2, cs, tau, dwork, -1, &ierr);
                wrkopt = (minwrk > (i32)dwork[0]) ? minwrk : (i32)dwork[0];
            }
            dwork[0] = (f64)wrkopt;
            return;
        } else if (ldwork < minwrk) {
            dwork[0] = (f64)minwrk;
            *info = -10;
        }
    }

    if (*info != 0) {
        return;
    }

    if (n == 0) {
        dwork[0] = ONE;
        return;
    }

    /* Shift the vectors which define the elementary reflectors one
       column to the right, and set the first ilo rows and columns to
       those of the unit matrix. */

    /* Process U1: shift reflector vectors right */
    for (j = n - 1; j >= ilo; j--) {
        /* Set U1(0:j-1, j) = 0 */
        for (i = 0; i < j; i++) {
            u1[i + j * ldu1] = ZERO;
        }
        /* Shift U1(j+1:n-1, j-1) to U1(j+1:n-1, j) */
        for (i = j + 1; i < n; i++) {
            u1[i + j * ldu1] = u1[i + (j - 1) * ldu1];
        }
    }

    /* Set first ilo columns of U1 to identity */
    SLC_DLASET("All", &n, &ilo, &ZERO, &ONE, u1, &ldu1);

    /* Process U2: shift reflector vectors right */
    for (j = n - 1; j >= ilo; j--) {
        /* Set U2(0:j-1, j) = 0 */
        for (i = 0; i < j; i++) {
            u2[i + j * ldu2] = ZERO;
        }
        /* Shift U2(j:n-1, j-1) to U2(j:n-1, j) */
        for (i = j; i < n; i++) {
            u2[i + j * ldu2] = u2[i + (j - 1) * ldu2];
        }
    }

    /* Set first ilo columns of U2 to zero */
    SLC_DLASET("All", &n, &ilo, &ZERO, &ZERO, u2, &ldu2);

    if (nh > 0) {
        /* Apply blocked algorithm */
        mb04wd(false, false, nh, nh, nh,
               &u1[ilo + ilo * ldu1], ldu1,
               &u2[ilo + ilo * ldu2], ldu2,
               &cs[ilo - 1], &tau[ilo - 1],
               dwork, ldwork, &ierr);
    } else {
        dwork[0] = ONE;
    }
}
