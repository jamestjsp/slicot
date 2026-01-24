/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>

void sb03ou(
    const bool discr,
    const bool ltrans,
    const i32 n,
    const i32 m,
    const f64* a,
    const i32 lda,
    f64* b,
    const i32 ldb,
    f64* tau,
    f64* u,
    const i32 ldu,
    f64* scale,
    f64* dwork,
    const i32 ldwork,
    i32* info
)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    *info = 0;

    if (n < 0) {
        *info = -3;
    } else if (m < 0) {
        *info = -4;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -6;
    } else if ((!ltrans && ldb < (m > 1 ? m : 1)) || (ltrans && ldb < (n > 1 ? n : 1))) {
        *info = -8;
    } else if (ldu < (n > 1 ? n : 1)) {
        *info = -11;
    } else if (ldwork < (4*n > 1 ? 4*n : 1)) {
        *info = -14;
    }

    if (*info != 0) {
        i32 neginfo = -(*info);
        SLC_XERBLA("SB03OU", &neginfo);
        return;
    }

    i32 mn = n < m ? n : m;
    if (mn == 0) {
        *scale = ONE;
        dwork[0] = ONE;
        return;
    }

    i32 wrkopt;

    if (ltrans) {
        SLC_DGERQF(&n, &m, b, &ldb, tau, dwork, &ldwork, info);
        wrkopt = (i32)dwork[0];

        if (m >= n) {
            i32 cols = n;
            i32 moff = m - n;
            SLC_DLACPY("Upper", &mn, &cols, &b[0 + moff*ldb], &ldb, u, &ldu);
        } else {
            for (i32 i = m - 1; i >= 0; i--) {
                i32 len = n - m + i + 1;
                i32 jdst = n - m + i;
                SLC_DCOPY(&len, &b[0 + i*ldb], &(i32){1}, &u[0 + jdst*ldu], &(i32){1});
            }
            i32 cols = n - m;
            SLC_DLASET("Full", &n, &cols, &ZERO, &ZERO, u, &ldu);
        }
    } else {
        SLC_DGEQRF(&m, &n, b, &ldb, tau, dwork, &ldwork, info);
        SLC_DLACPY("Upper", &mn, &n, b, &ldb, u, &ldu);
        if (m < n) {
            i32 rows = n - m;
            i32 cols = n - m;
            SLC_DLASET("Upper", &rows, &cols, &ZERO, &ZERO, &u[m + m*ldu], &ldu);
        }
    }
    wrkopt = (i32)dwork[0];

    sb03ot(discr, ltrans, n, (f64*)a, lda, u, ldu, scale, dwork, info);
    if (*info != 0 && *info != 1) {
        return;
    }

    if (ltrans) {
        for (i32 j = 0; j < n; j++) {
            if (u[j + j*ldu] < ZERO) {
                for (i32 i = 0; i <= j; i++) {
                    u[i + j*ldu] = -u[i + j*ldu];
                }
            }
        }
    } else {
        for (i32 j = 0; j < n; j++) {
            dwork[j] = u[j + j*ldu];
        }
        for (i32 j = 0; j < n; j++) {
            for (i32 i = 0; i <= j; i++) {
                if (dwork[i] < ZERO) {
                    u[i + j*ldu] = -u[i + j*ldu];
                }
            }
        }
    }

    dwork[0] = (f64)(wrkopt > 4*n ? wrkopt : 4*n);
}
