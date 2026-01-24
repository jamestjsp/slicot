/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <stdlib.h>
#include <math.h>

void mb03od(
    const char* jobqr,
    const i32 m,
    const i32 n,
    f64* a,
    const i32 lda,
    i32* jpvt,
    const f64 rcond,
    const f64 svlmax,
    f64* tau,
    i32* rank,
    f64* sval,
    f64* dwork,
    const i32 ldwork,
    i32* info
)
{
    const i32 IMAX = 1;
    const i32 IMIN = 2;
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    bool ljobqr, lquery;
    i32 ismax, ismin, maxwrk, minwrk, mn;
    f64 c1, c2, s1, s2, smax, smaxpr, smin, sminpr;

    ljobqr = (*jobqr == 'Q' || *jobqr == 'q');
    mn = (m < n) ? m : n;

    if (ljobqr) {
        minwrk = 3 * n + 1;
    } else {
        minwrk = (2 * mn > 1) ? 2 * mn : 1;
    }
    maxwrk = minwrk;

    *info = 0;

    if (!ljobqr && *jobqr != 'N' && *jobqr != 'n') {
        *info = -1;
    } else if (m < 0) {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    } else if (lda < ((m > 1) ? m : 1)) {
        *info = -5;
    } else if (rcond < ZERO) {
        *info = -7;
    } else if (svlmax < ZERO) {
        *info = -8;
    } else {
        lquery = (ldwork == -1);
        if (ljobqr) {
            i32 lwork_query = -1;
            i32 info_temp = 0;
            SLC_DGEQP3(&m, &n, a, &lda, jpvt, tau, dwork, &lwork_query, &info_temp);
            i32 qp3_opt = (i32)dwork[0];
            maxwrk = (maxwrk > qp3_opt) ? maxwrk : qp3_opt;
        }
        if (ldwork < minwrk && !lquery) {
            *info = -13;
        }
    }

    if (*info != 0) {
        return;
    } else if (lquery) {
        dwork[0] = (f64)maxwrk;
        return;
    }

    if (mn == 0) {
        *rank = 0;
        sval[0] = ZERO;
        sval[1] = ZERO;
        sval[2] = ZERO;
        dwork[0] = ONE;
        return;
    }

    if (ljobqr) {
        SLC_DGEQP3(&m, &n, a, &lda, jpvt, tau, dwork, &ldwork, info);
    }

    ismin = 0;
    ismax = mn;
    dwork[ismin] = ONE;
    dwork[ismax] = ONE;
    smax = fabs(a[0]);
    smin = smax;

    if (smax == ZERO || svlmax * rcond > smax) {
        *rank = 0;
        sval[0] = smax;
        sval[1] = ZERO;
        sval[2] = ZERO;
    } else {
        *rank = 1;
        sminpr = smin;

        while (*rank < mn) {
            i32 i_cur = *rank;

            SLC_DLAIC1(&IMIN, rank, &dwork[ismin], &smin, &a[i_cur * lda],
                       &a[i_cur + i_cur * lda], &sminpr, &s1, &c1);
            SLC_DLAIC1(&IMAX, rank, &dwork[ismax], &smax, &a[i_cur * lda],
                       &a[i_cur + i_cur * lda], &smaxpr, &s2, &c2);

            if (svlmax * rcond <= smaxpr) {
                if (svlmax * rcond <= sminpr) {
                    if (smaxpr * rcond <= sminpr) {
                        for (i32 i = 0; i < *rank; i++) {
                            dwork[ismin + i] = s1 * dwork[ismin + i];
                            dwork[ismax + i] = s2 * dwork[ismax + i];
                        }
                        dwork[ismin + *rank] = c1;
                        dwork[ismax + *rank] = c2;
                        smin = sminpr;
                        smax = smaxpr;
                        (*rank)++;
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            } else {
                break;
            }
        }
        sval[0] = smax;
        sval[1] = smin;
        sval[2] = sminpr;
    }

    dwork[0] = (f64)maxwrk;
}
