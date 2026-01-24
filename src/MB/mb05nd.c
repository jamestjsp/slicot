/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>

void mb05nd(
    const i32 n,
    const f64 delta,
    const f64* a,
    const i32 lda,
    f64* ex,
    const i32 ldex,
    f64* exint,
    const i32 ldexin,
    const f64 tol,
    i32* iwork,
    f64* dwork,
    const i32 ldwork,
    i32* info
)
{
    const f64 ZERO = 0.0;
    const f64 HALF = 0.5;
    const f64 ONE = 1.0;
    const f64 ONE64 = 1.64;
    const f64 THREE = 3.0;
    const f64 FOUR8 = 4.8;

    i32 nn = n * n;

    *info = 0;

    if (n < 0) {
        *info = -1;
    } else if (lda < ((n > 1) ? n : 1)) {
        *info = -4;
    } else if (ldex < ((n > 1) ? n : 1)) {
        *info = -6;
    } else if (ldexin < ((n > 1) ? n : 1)) {
        *info = -8;
    } else if (ldwork < ((nn + n > 1) ? nn + n : 1)) {
        *info = -12;
    }

    if (*info != 0) {
        return;
    }

    dwork[0] = ONE;
    if (n == 0) {
        return;
    }

    SLC_DLASET("Full", &n, &n, &ZERO, &ZERO, ex, &ldex);
    SLC_DLASET("Full", &n, &n, &ZERO, &ZERO, exint, &ldexin);

    if (delta == ZERO) {
        SLC_DLASET("Upper", &n, &n, &ZERO, &ONE, ex, &ldex);
        return;
    }

    if (n == 1) {
        ex[0] = exp(delta * a[0]);
        if (a[0] == ZERO) {
            exint[0] = delta;
        } else {
            exint[0] = (ONE / a[0]) * ex[0] - (ONE / a[0]);
        }
        return;
    }

    f64 eps = SLC_DLAMCH("Epsilon");
    f64 small = SLC_DLAMCH("Safe minimum") / eps;

    f64 fnorm = delta * SLC_DLANGE("Frobenius", &n, &n, a, &lda, dwork);

    if (fnorm > sqrt(ONE / small)) {
        *info = n + 1;
        return;
    }

    i32 jscal = 0;
    f64 delsc = delta;

    while (fnorm >= HALF) {
        jscal++;
        delsc *= HALF;
        fnorm *= HALF;
    }

    f64 fnorm2 = fnorm * fnorm;
    i32 iq = 1;
    f64 qmax = fnorm / THREE;
    f64 err = (delta / delsc) * fnorm2 * fnorm2 / FOUR8;

    while (err > tol * ((f64)(2 * iq + 3) - fnorm) / ONE64 && qmax >= eps) {
        iq++;
        qmax = qmax * (f64)(iq + 1) * fnorm / (f64)(2 * iq * (2 * iq + 1));
        if (qmax >= eps) {
            err = err * fnorm2 * (f64)(2 * iq + 5) /
                  (f64)((2 * iq + 3) * (2 * iq + 3) * (2 * iq + 4));
        }
    }

    i32 i2iq1 = 2 * iq + 1;
    f64 f2iq1 = (f64)i2iq1;
    f64 coeffd = -(f64)iq / f2iq1;
    f64 coeffn = HALF / f2iq1;

    i32 ij = 0;
    for (i32 j = 0; j < n; j++) {
        for (i32 i = 0; i < n; i++) {
            dwork[ij] = delsc * a[i + j * lda];
            exint[i + j * ldexin] = coeffn * dwork[ij];
            ex[i + j * ldex] = coeffd * dwork[ij];
            ij++;
        }
        exint[j + j * ldexin] += ONE;
        ex[j + j * ldex] += ONE;
    }

    for (i32 kk = 2; kk <= iq; kk++) {
        coeffd = -coeffd * (f64)(iq + 1 - kk) / (f64)(kk * (i2iq1 + 1 - kk));
        if (kk % 2 == 0) {
            coeffn = coeffd / (f64)(kk + 1);
        } else {
            coeffn = -coeffd / (f64)(i2iq1 - kk);
        }
        ij = 0;

        if (ldwork >= 2 * nn) {
            SLC_DGEMM("N", "N", &n, &n, &n, &delsc, a, &lda, dwork, &n,
                      &ZERO, &dwork[nn], &n);

            i32 inc1 = 1;
            SLC_DCOPY(&nn, &dwork[nn], &inc1, dwork, &inc1);

            for (i32 j = 0; j < n; j++) {
                SLC_DAXPY(&n, &coeffn, &dwork[ij], &inc1, &exint[j * ldexin], &inc1);
                SLC_DAXPY(&n, &coeffd, &dwork[ij], &inc1, &ex[j * ldex], &inc1);
                ij += n;
            }
        } else {
            i32 inc1 = 1;
            for (i32 j = 0; j < n; j++) {
                SLC_DGEMV("N", &n, &n, &ONE, a, &lda, &dwork[ij], &inc1,
                          &ZERO, &dwork[nn], &inc1);
                SLC_DCOPY(&n, &dwork[nn], &inc1, &dwork[ij], &inc1);
                SLC_DSCAL(&n, &delsc, &dwork[ij], &inc1);
                SLC_DAXPY(&n, &coeffn, &dwork[ij], &inc1, &exint[j * ldexin], &inc1);
                SLC_DAXPY(&n, &coeffd, &dwork[ij], &inc1, &ex[j * ldex], &inc1);
                ij += n;
            }
        }
    }

    SLC_DGESV(&n, &n, ex, &ldex, iwork, exint, &ldexin, info);
    if (*info != 0) {
        return;
    }

    i32 inc1 = 1;
    for (i32 j = 0; j < n; j++) {
        SLC_DSCAL(&n, &delsc, &exint[j * ldexin], &inc1);
    }

    SLC_DGEMM("N", "N", &n, &n, &n, &ONE, exint, &ldexin, a, &lda,
              &ZERO, ex, &ldex);

    for (i32 j = 0; j < n; j++) {
        ex[j + j * ldex] += ONE;
    }

    for (i32 l = 0; l < jscal; l++) {
        SLC_DLACPY("Full", &n, &n, exint, &ldexin, dwork, &n);
        SLC_DGEMM("N", "N", &n, &n, &n, &ONE, dwork, &n, ex, &ldex,
                  &ONE, exint, &ldexin);
        SLC_DLACPY("Full", &n, &n, ex, &ldex, dwork, &n);
        SLC_DGEMM("N", "N", &n, &n, &n, &ONE, dwork, &n, dwork, &n,
                  &ZERO, ex, &ldex);
    }

    dwork[0] = (f64)(2 * nn);
}
