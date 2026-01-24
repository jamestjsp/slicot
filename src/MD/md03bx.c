/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>

void md03bx(
    i32 m,
    i32 n,
    f64 fnorm,
    f64* j,
    i32* ldj,
    f64* e,
    f64* jnorms,
    f64* gnorm,
    i32* ipvt,
    f64* dwork,
    i32 ldwork,
    i32* info
)
{
    const f64 zero = 0.0;
    const f64 one = 1.0;
    i32 i, itau, jwork, l, wrkopt;
    f64 sum;

    *info = 0;

    if (m < 0) {
        *info = -1;
    } else if (n < 0 || m < n) {
        *info = -2;
    } else if (fnorm < zero) {
        *info = -3;
    } else if (*ldj < (m > 0 ? m : 1)) {
        *info = -5;
    } else {
        if (n == 0 || m == 1) {
            jwork = 1;
        } else {
            jwork = 4*n + 1;
        }
        if (ldwork < jwork) {
            *info = -11;
        }
    }

    if (*info != 0) {
        return;
    }

    *gnorm = zero;
    if (n == 0) {
        *ldj = 1;
        dwork[0] = one;
        return;
    } else if (m == 1) {
        jnorms[0] = fabs(j[0]);
        if (fnorm*j[0] != zero) {
            *gnorm = fabs(e[0]/fnorm);
        }
        *ldj = 1;
        ipvt[0] = 1;
        dwork[0] = one;
        return;
    }

    for (i = 0; i < n; i++) {
        ipvt[i] = 0;
    }

    itau = 0;
    jwork = itau + n;
    wrkopt = 1;

    SLC_DGEQP3(&m, &n, j, ldj, ipvt, &dwork[itau], &dwork[jwork],
               &(i32){ldwork - jwork}, info);
    wrkopt = (i32)dwork[jwork] + jwork > wrkopt ? (i32)dwork[jwork] + jwork : wrkopt;

    {
        i32 one_int = 1;
        SLC_DORMQR("L", "T", &m, &one_int, &n, j, ldj, &dwork[itau], e,
                   &m, &dwork[jwork], &(i32){ldwork - jwork}, info);
    }
    wrkopt = (i32)dwork[jwork] + jwork > wrkopt ? (i32)dwork[jwork] + jwork : wrkopt;

    if (*ldj > n) {
        SLC_DLACPY("U", &n, &n, j, ldj, j, &n);
        *ldj = n;
    }

    if (fnorm != zero) {
        for (i = 0; i < n; i++) {
            l = ipvt[i] - 1;
            i32 len = i + 1;
            i32 inc = 1;
            jnorms[l] = SLC_DNRM2(&len, &j[i*(*ldj)], &inc);
            if (jnorms[l] != zero) {
                sum = SLC_DDOT(&len, &j[i*(*ldj)], &inc, e, &inc) / fnorm;
                *gnorm = fabs(sum/jnorms[l]) > *gnorm ? fabs(sum/jnorms[l]) : *gnorm;
            }
        }
    } else {
        for (i = 0; i < n; i++) {
            l = ipvt[i] - 1;
            i32 len = i + 1;
            i32 inc = 1;
            jnorms[l] = SLC_DNRM2(&len, &j[i*(*ldj)], &inc);
        }
    }

    dwork[0] = (f64)wrkopt;
}
