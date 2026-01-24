/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>

void mb05md(
    const char* balanc,
    const i32 n,
    const f64 delta,
    f64* a,
    const i32 lda,
    f64* v,
    const i32 ldv,
    f64* y,
    const i32 ldy,
    f64* valr,
    f64* vali,
    i32* iwork,
    f64* dwork,
    const i32 ldwork,
    i32* info
)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const i32 int1 = 1;

    bool scale = (*balanc == 'S' || *balanc == 's');
    bool no_scale = (*balanc == 'N' || *balanc == 'n');

    *info = 0;

    if (!scale && !no_scale) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -5;
    } else if (ldv < (n > 1 ? n : 1)) {
        *info = -7;
    } else if (ldy < (n > 1 ? n : 1)) {
        *info = -9;
    } else if (ldwork < (4 * n > 1 ? 4 * n : 1)) {
        *info = -14;
    }

    if (*info != 0) {
        return;
    }

    if (n == 0) {
        dwork[0] = ONE;
        return;
    }

    mb05my(balanc, n, a, lda, valr, vali, v, ldv, y, ldy, dwork, ldwork, info);

    if (*info > 0) {
        return;
    }

    f64 wrkopt = dwork[0];

    if (scale) {
        for (i32 i = 0; i < n; i++) {
            dwork[i] = dwork[i + 1];
        }
    }

    for (i32 i = 0; i < n; i++) {
        if (v[i + i * ldv] == ZERO) {
            *info = n + 1;
            return;
        }
    }

    f64 rcond;
    SLC_DTRCON("1", "U", "N", &n, v, &ldv, &rcond, &dwork[n], iwork, info);

    f64 eps = SLC_DLAMCH("E");
    if (rcond < eps) {
        dwork[1] = rcond;
        *info = n + 2;
        return;
    }

    SLC_DLACPY("F", &n, &n, y, &ldy, a, &lda);
    SLC_DTRMM("R", "U", "N", "N", &n, &n, &ONE, v, &ldv, a, &lda);

    if (scale) {
        i32 ilo = 1;
        i32 ihi = n;
        i32 ierr;
        SLC_DGEBAK(balanc, "R", &n, &ilo, &ihi, dwork, &n, a, &lda, &ierr);
    }

    for (i32 i = 1; i < n; i++) {
        SLC_DSWAP(&i, &y[i], &ldy, &y[i * ldy], &int1);
    }

    SLC_DTRSM("L", "U", "N", "N", &n, &n, &ONE, v, &ldv, y, &ldy);

    if (scale) {
        for (i32 i = 0; i < n; i++) {
            f64 tempr = ONE / dwork[i];
            SLC_DSCAL(&n, &tempr, &y[i * ldy], &int1);
        }
    }

    SLC_DLACPY("F", &n, &n, a, &lda, v, &ldv);

    i32 i = 0;
    while (i < n) {
        if (vali[i] == ZERO) {
            f64 tempr = exp(valr[i] * delta);
            SLC_DSCAL(&n, &tempr, &y[i], &ldy);
            i++;
        } else {
            f64 tempr = valr[i] * delta;
            f64 tempi = vali[i] * delta;
            f64 exp_tempr = exp(tempr);

            f64 tmp[4];
            tmp[0] = cos(tempi) * exp_tempr;
            tmp[1] = sin(tempi) * exp_tempr;
            tmp[2] = -tmp[1];
            tmp[3] = tmp[0];

            i32 two = 2;
            SLC_DLACPY("F", &two, &n, &y[i], &ldy, dwork, &two);
            SLC_DGEMM("N", "N", &two, &n, &two, &ONE, tmp, &two, dwork, &two, &ZERO, &y[i], &ldy);
            i += 2;
        }
    }

    SLC_DGEMM("N", "N", &n, &n, &n, &ONE, v, &ldv, y, &ldy, &ZERO, a, &lda);

    dwork[0] = wrkopt;
    dwork[1] = rcond;
}
