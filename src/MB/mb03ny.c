/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <complex.h>

f64 mb03ny(
    const i32 n,
    const f64 omega,
    f64* a,
    const i32 lda,
    f64* s,
    f64* dwork,
    const i32 ldwork,
    c128* cwork,
    const i32 lcwork,
    i32* info
)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const c128 CONE = 1.0 + 0.0 * I;
    const c128 RTMONE = 0.0 + 1.0 * I;

    *info = 0;

    if (n < 0) {
        *info = -1;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -4;
    } else if (ldwork < (5 * n > 1 ? 5 * n : 1)) {
        *info = -7;
    } else if (lcwork < 1 || (omega != ZERO && lcwork < n * n + 3 * n)) {
        *info = -9;
    }

    if (*info != 0) {
        i32 neginfo = -(*info);
        SLC_XERBLA("MB03NY", &neginfo);
        return ZERO;
    }

    if (n == 0) {
        dwork[0] = ONE;
        if (omega != ZERO) {
            cwork[0] = CONE;
        }
        return ZERO;
    }

    i32 n2 = n * n;

    if (omega == ZERO) {
        f64 dummy[1];
        i32 int1 = 1;
        i32 lwork_query = -1;

        SLC_DGESVD("N", "N", &n, &n, a, &lda, s, dummy, &int1, dummy, &int1,
                   dwork, &lwork_query, info);
        i32 optimal_lwork = (i32)dwork[0];

        i32 lwork = ldwork > optimal_lwork ? ldwork : optimal_lwork;
        if (lwork > ldwork) {
            lwork = ldwork;
        }

        SLC_DGESVD("N", "N", &n, &n, a, &lda, s, dummy, &int1, dummy, &int1,
                   dwork, &lwork, info);
        if (*info != 0) {
            *info = 2;
            return ZERO;
        }
    } else {
        i32 ic = 0;
        for (i32 j = 0; j < n; j++) {
            for (i32 i = 0; i < n; i++) {
                cwork[ic] = a[i + j * lda];
                ic++;
            }
            cwork[j * n + j] -= omega * RTMONE;
        }

        c128 zdummy[1];
        i32 int1 = 1;
        i32 lwork_query = -1;
        i32 lcwork_work = lcwork - n2;

        SLC_ZGESVD("N", "N", &n, &n, cwork, &n, s, zdummy, &int1, zdummy, &int1,
                   &cwork[n2], &lwork_query, dwork, info);
        i32 optimal_lcwork = (i32)creal(cwork[n2]);

        i32 lwork_c = lcwork_work > optimal_lcwork ? optimal_lcwork : lcwork_work;

        SLC_ZGESVD("N", "N", &n, &n, cwork, &n, s, zdummy, &int1, zdummy, &int1,
                   &cwork[n2], &lwork_c, dwork, info);
        if (*info != 0) {
            *info = 2;
            return ZERO;
        }

        cwork[0] = cwork[n2] + (f64)(n2) * CONE;
        dwork[0] = (f64)(5 * n);
    }

    return s[n - 1];
}
