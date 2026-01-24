/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>

void mb04pu(const i32 n, const i32 ilo, f64 *a, const i32 lda,
            f64 *qg, const i32 ldqg, f64 *cs, f64 *tau,
            f64 *dwork, const i32 ldwork, i32 *info)
{
    const f64 ZERO = 0.0;
    const f64 HALF = 0.5;
    const f64 ONE = 1.0;
    const f64 MONE = -1.0;

    i32 i;
    f64 alpha, c, mu, nu, s, temp, ttemp;
    i32 int1 = 1;
    i32 max_1_n = (1 > n) ? 1 : n;

    *info = 0;
    if (n < 0) {
        *info = -1;
    } else if (ilo < 1 || ilo > max_1_n) {
        *info = -2;
    } else if (lda < max_1_n) {
        *info = -4;
    } else if (ldqg < max_1_n) {
        *info = -6;
    } else if (ldwork < max_1_n) {
        dwork[0] = (f64)max_1_n;
        *info = -10;
    }

    if (*info != 0) {
        return;
    }

    if (n <= ilo) {
        dwork[0] = ONE;
        return;
    }

    for (i = ilo; i <= n - 1; i++) {
        i32 i_c = i - 1;
        i32 n_mi = n - i;
        i32 min_i2n_c = (i + 1 < n - 1) ? (i + 1) : (n - 1);

        alpha = qg[i_c + 1 + i_c * ldqg];
        SLC_DLARFG(&n_mi, &alpha, &qg[min_i2n_c + i_c * ldqg], &int1, &nu);

        if (nu != ZERO) {
            qg[i_c + 1 + i_c * ldqg] = ONE;

            SLC_DSYMV("Lower", &n_mi, &nu, &qg[(i_c + 1) + (i_c + 1) * ldqg], &ldqg,
                      &qg[i_c + 1 + i_c * ldqg], &int1, &ZERO, dwork, &int1);

            f64 dot = SLC_DDOT(&n_mi, dwork, &int1, &qg[i_c + 1 + i_c * ldqg], &int1);
            mu = -HALF * nu * dot;
            SLC_DAXPY(&n_mi, &mu, &qg[i_c + 1 + i_c * ldqg], &int1, dwork, &int1);

            SLC_DSYR2("Lower", &n_mi, &MONE, &qg[i_c + 1 + i_c * ldqg], &int1,
                      dwork, &int1, &qg[(i_c + 1) + (i_c + 1) * ldqg], &ldqg);

            SLC_DLARF("Right", &i, &n_mi, &qg[i_c + 1 + i_c * ldqg], &int1, &nu,
                      &qg[0 + (i_c + 2) * ldqg], &ldqg, dwork);

            SLC_DSYMV("Upper", &n_mi, &nu, &qg[(i_c + 1) + (i_c + 2) * ldqg], &ldqg,
                      &qg[i_c + 1 + i_c * ldqg], &int1, &ZERO, dwork, &int1);

            dot = SLC_DDOT(&n_mi, dwork, &int1, &qg[i_c + 1 + i_c * ldqg], &int1);
            mu = -HALF * nu * dot;
            SLC_DAXPY(&n_mi, &mu, &qg[i_c + 1 + i_c * ldqg], &int1, dwork, &int1);

            SLC_DSYR2("Upper", &n_mi, &MONE, &qg[i_c + 1 + i_c * ldqg], &int1,
                      dwork, &int1, &qg[(i_c + 1) + (i_c + 2) * ldqg], &ldqg);

            i32 n_mi_p1 = n - i + 1;
            SLC_DLARF("Left", &n_mi, &n_mi_p1, &qg[i_c + 1 + i_c * ldqg], &int1, &nu,
                      &a[(i_c + 1) + i_c * lda], &lda, dwork);

            SLC_DLARF("Right", &n, &n_mi, &qg[i_c + 1 + i_c * ldqg], &int1, &nu,
                      &a[0 + (i_c + 1) * lda], &lda, dwork);
        }
        qg[i_c + 1 + i_c * ldqg] = nu;

        temp = a[i_c + 1 + i_c * lda];
        SLC_DLARTG(&temp, &alpha, &c, &s, &a[i_c + 1 + i_c * lda]);

        i32 n_mi_m1 = n - i - 1;
        if (n_mi_m1 > 0) {
            SLC_DROT(&n_mi_m1, &a[(i_c + 1) + (i_c + 2) * lda], &lda,
                     &qg[(i_c + 2) + (i_c + 1) * ldqg], &int1, &c, &s);
        }

        SLC_DROT(&i, &a[0 + (i_c + 1) * lda], &int1,
                 &qg[0 + (i_c + 2) * ldqg], &int1, &c, &s);

        if (n_mi_m1 > 0) {
            SLC_DROT(&n_mi_m1, &a[(i_c + 2) + (i_c + 1) * lda], &int1,
                     &qg[(i_c + 1) + (i_c + 3) * ldqg], &ldqg, &c, &s);
        }

        temp = a[(i_c + 1) + (i_c + 1) * lda];
        ttemp = qg[(i_c + 1) + (i_c + 2) * ldqg];
        a[(i_c + 1) + (i_c + 1) * lda] = c * temp + s * qg[(i_c + 1) + (i_c + 1) * ldqg];
        qg[(i_c + 1) + (i_c + 2) * ldqg] = c * ttemp - s * temp;
        qg[(i_c + 1) + (i_c + 1) * ldqg] = -s * temp + c * qg[(i_c + 1) + (i_c + 1) * ldqg];
        ttemp = -s * ttemp - c * temp;
        temp = a[(i_c + 1) + (i_c + 1) * lda];
        qg[(i_c + 1) + (i_c + 1) * ldqg] = c * qg[(i_c + 1) + (i_c + 1) * ldqg] + s * ttemp;
        a[(i_c + 1) + (i_c + 1) * lda] = c * temp + s * qg[(i_c + 1) + (i_c + 2) * ldqg];
        qg[(i_c + 1) + (i_c + 2) * ldqg] = -s * temp + c * qg[(i_c + 1) + (i_c + 2) * ldqg];
        cs[2 * (i - 1)] = c;
        cs[2 * (i - 1) + 1] = s;

        SLC_DLARFG(&n_mi, &a[i_c + 1 + i_c * lda],
                   &a[min_i2n_c + i_c * lda], &int1, &nu);

        if (nu != ZERO) {
            temp = a[i_c + 1 + i_c * lda];
            a[i_c + 1 + i_c * lda] = ONE;

            SLC_DLARF("Left", &n_mi, &n_mi, &a[i_c + 1 + i_c * lda], &int1, &nu,
                      &a[(i_c + 1) + (i_c + 1) * lda], &lda, dwork);

            SLC_DLARF("Right", &n, &n_mi, &a[i_c + 1 + i_c * lda], &int1, &nu,
                      &a[0 + (i_c + 1) * lda], &lda, dwork);

            SLC_DSYMV("Lower", &n_mi, &nu, &qg[(i_c + 1) + (i_c + 1) * ldqg], &ldqg,
                      &a[i_c + 1 + i_c * lda], &int1, &ZERO, dwork, &int1);

            f64 dot = SLC_DDOT(&n_mi, dwork, &int1, &a[i_c + 1 + i_c * lda], &int1);
            mu = -HALF * nu * dot;
            SLC_DAXPY(&n_mi, &mu, &a[i_c + 1 + i_c * lda], &int1, dwork, &int1);

            SLC_DSYR2("Lower", &n_mi, &MONE, &a[i_c + 1 + i_c * lda], &int1,
                      dwork, &int1, &qg[(i_c + 1) + (i_c + 1) * ldqg], &ldqg);

            SLC_DLARF("Right", &i, &n_mi, &a[i_c + 1 + i_c * lda], &int1, &nu,
                      &qg[0 + (i_c + 2) * ldqg], &ldqg, dwork);

            SLC_DSYMV("Upper", &n_mi, &nu, &qg[(i_c + 1) + (i_c + 2) * ldqg], &ldqg,
                      &a[i_c + 1 + i_c * lda], &int1, &ZERO, dwork, &int1);

            dot = SLC_DDOT(&n_mi, dwork, &int1, &a[i_c + 1 + i_c * lda], &int1);
            mu = -HALF * nu * dot;
            SLC_DAXPY(&n_mi, &mu, &a[i_c + 1 + i_c * lda], &int1, dwork, &int1);

            SLC_DSYR2("Upper", &n_mi, &MONE, &a[i_c + 1 + i_c * lda], &int1,
                      dwork, &int1, &qg[(i_c + 1) + (i_c + 2) * ldqg], &ldqg);

            a[i_c + 1 + i_c * lda] = temp;
        }
        tau[i - 1] = nu;
    }

    dwork[0] = (f64)max_1_n;
}
