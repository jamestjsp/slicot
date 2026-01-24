/*
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * MB04WR - Generate orthogonal symplectic matrices U or V from symplectic
 *          reflectors and Givens rotations.
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <string.h>

void mb04wr(const char *job, const char *trans, const i32 n, const i32 ilo,
            f64 *q1, const i32 ldq1, f64 *q2, const i32 ldq2,
            const f64 *cs, const f64 *tau, f64 *dwork, const i32 ldwork,
            i32 *info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    i32 i, j, ierr;
    i32 minwrk, wrkopt;
    i32 nh;
    bool lquery, compu, ltran;

    *info = 0;

    compu = (job[0] == 'U' || job[0] == 'u');
    ltran = (trans[0] == 'T' || trans[0] == 't' ||
             trans[0] == 'C' || trans[0] == 'c');

    i32 max_1_n = (1 > n) ? 1 : n;

    if (!compu && !(job[0] == 'V' || job[0] == 'v')) {
        *info = -1;
    } else if (!ltran && !(trans[0] == 'N' || trans[0] == 'n')) {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    } else if (ilo < 1 || ilo > (n > 1 ? n : 1)) {
        *info = -4;
    } else if (ldq1 < max_1_n) {
        *info = -6;
    } else if (ldq2 < max_1_n) {
        *info = -8;
    } else {
        lquery = (ldwork == -1);

        if (compu) {
            nh = n - ilo + 1;
        } else {
            nh = n - ilo;
        }

        minwrk = (1 > 2 * nh) ? 1 : 2 * nh;

        if (lquery) {
            if (nh <= 0) {
                wrkopt = 1;
            } else {
                mb04wd(ltran, ltran, nh, nh, nh,
                       q1, ldq1, q2, ldq2, cs, tau, dwork, -1, &ierr);
                wrkopt = (minwrk > (i32)dwork[0]) ? minwrk : (i32)dwork[0];
            }
            dwork[0] = (f64)wrkopt;
            return;
        } else if (ldwork < minwrk) {
            dwork[0] = (f64)minwrk;
            *info = -12;
        }
    }

    if (*info != 0) {
        return;
    }

    if (n == 0) {
        dwork[0] = ONE;
        return;
    }

    if (compu) {
        i32 ilo_m1 = ilo - 1;
        i32 n_ilo_p1 = n - ilo + 1;

        SLC_DLASET("All", &n, &ilo_m1, &ZERO, &ONE, q1, &ldq1);

        SLC_DLASET("All", &ilo_m1, &n_ilo_p1, &ZERO, &ZERO, &q1[(ilo - 1) * ldq1], &ldq1);

        SLC_DLASET("All", &n, &ilo_m1, &ZERO, &ZERO, q2, &ldq2);

        SLC_DLASET("All", &ilo_m1, &n_ilo_p1, &ZERO, &ZERO, &q2[(ilo - 1) * ldq2], &ldq2);
    }

    if (compu && !ltran) {
        mb04wd(false, false, nh, nh, nh,
               &q1[(ilo - 1) + (ilo - 1) * ldq1], ldq1,
               &q2[(ilo - 1) + (ilo - 1) * ldq2], ldq2,
               &cs[ilo - 1], &tau[ilo - 1],
               dwork, ldwork, &ierr);
    } else if (compu && ltran) {
        mb04wd(true, false, nh, nh, nh,
               &q1[(ilo - 1) + (ilo - 1) * ldq1], ldq1,
               &q2[(ilo - 1) + (ilo - 1) * ldq2], ldq2,
               &cs[ilo - 1], &tau[ilo - 1],
               dwork, ldwork, &ierr);
    } else if (!compu && !ltran) {
        /* Shift vectors one row down and set first ilo rows/columns to identity/zero */

        /* Process Q1: shift reflector vectors one row down */
        for (i = 0; i < n; i++) {
            /* Set Q1(n-1:max(i,ilo-1)+1, i) = 0 (from top going down to below diagonal) */
            for (j = n - 1; j >= ((i > (ilo - 1)) ? i : (ilo - 1)) + 1; j--) {
                q1[j + i * ldq1] = ZERO;
            }
            /* Shift Q1(max(i,ilo-1):ilo, i) one row down */
            for (j = (i > (ilo - 1)) ? i : (ilo - 1); j >= ilo; j--) {
                q1[j + i * ldq1] = q1[(j - 1) + i * ldq1];
            }
            /* Set Q1(0:ilo-1, i) = 0 */
            for (j = ilo - 1; j >= 0; j--) {
                q1[j + i * ldq1] = ZERO;
            }
            /* Set diagonal to 1 for first ilo columns */
            if (i < ilo) {
                q1[i + i * ldq1] = ONE;
            }
        }

        /* Process Q2: shift reflector vectors one row down */
        for (i = 0; i < n; i++) {
            /* Set Q2(n-1:max(i,ilo-1)+1, i) = 0 */
            for (j = n - 1; j >= ((i > (ilo - 1)) ? i : (ilo - 1)) + 1; j--) {
                q2[j + i * ldq2] = ZERO;
            }
            /* Shift Q2(max(i,ilo-1):ilo, i) one row down */
            for (j = (i > (ilo - 1)) ? i : (ilo - 1); j >= ilo; j--) {
                q2[j + i * ldq2] = q2[(j - 1) + i * ldq2];
            }
            /* Set Q2(0:ilo-1, i) = 0 */
            for (j = ilo - 1; j >= 0; j--) {
                q2[j + i * ldq2] = ZERO;
            }
        }

        if (nh > 0) {
            mb04wd(true, true, nh, nh, nh,
                   &q1[ilo + ilo * ldq1], ldq1,
                   &q2[ilo + ilo * ldq2], ldq2,
                   &cs[ilo - 1], &tau[ilo - 1],
                   dwork, ldwork, &ierr);
        } else {
            dwork[0] = ONE;
        }
    } else if (!compu && ltran) {
        /* Shift vectors one column right and set first ilo rows/columns */

        /* Process Q1: shift reflector vectors one column right */
        for (j = n - 1; j >= ilo; j--) {
            /* Set Q1(0:j-1, j) = 0 */
            for (i = 0; i < j; i++) {
                q1[i + j * ldq1] = ZERO;
            }
            /* Shift Q1(j+1:n-1, j-1) to Q1(j+1:n-1, j) */
            for (i = j + 1; i < n; i++) {
                q1[i + j * ldq1] = q1[i + (j - 1) * ldq1];
            }
        }

        /* Set first ilo columns to identity */
        SLC_DLASET("All", &n, &ilo, &ZERO, &ONE, q1, &ldq1);

        /* Process Q2: shift reflector vectors one row down */
        for (i = 0; i < n; i++) {
            for (j = n - 1; j >= ((i > (ilo - 1)) ? i : (ilo - 1)) + 1; j--) {
                q2[j + i * ldq2] = ZERO;
            }
            for (j = (i > (ilo - 1)) ? i : (ilo - 1); j >= ilo; j--) {
                q2[j + i * ldq2] = q2[(j - 1) + i * ldq2];
            }
            for (j = ilo - 1; j >= 0; j--) {
                q2[j + i * ldq2] = ZERO;
            }
        }

        if (nh > 0) {
            mb04wd(false, true, nh, nh, nh,
                   &q1[ilo + ilo * ldq1], ldq1,
                   &q2[ilo + ilo * ldq2], ldq2,
                   &cs[ilo - 1], &tau[ilo - 1],
                   dwork, ldwork, &ierr);
        } else {
            dwork[0] = ONE;
        }
    }
}
