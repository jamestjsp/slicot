/**
 * @file mb04ru.c
 * @brief Reduction of skew-Hamiltonian matrix to PVL form (unblocked).
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "slicot.h"
#include "slicot_blas.h"

void mb04ru(
    const i32 n,
    const i32 ilo,
    f64* a, const i32 lda,
    f64* qg, const i32 ldqg,
    f64* cs,
    f64* tau,
    f64* dwork, const i32 ldwork,
    i32* info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    i32 int1 = 1;

    i32 i, min_i2n;
    f64 alpha, c, nu, s, temp;

    *info = 0;

    if (n < 0) {
        *info = -1;
    } else if (ilo < 1 || ilo > n + 1) {
        *info = -2;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -4;
    } else if (ldqg < (n > 1 ? n : 1)) {
        *info = -6;
    } else if (ldwork < (n > 1 ? n : 1)) {
        dwork[0] = (f64)(n > 1 ? n : 1);
        *info = -10;
    }

    if (*info != 0) {
        return;
    }

    if (n <= ilo) {
        dwork[0] = ONE;
        return;
    }

    for (i = ilo - 1; i < n - 1; i++) {
        i32 n_minus_i = n - i - 1;
        min_i2n = (i + 2 < n) ? i + 2 : n - 1;

        alpha = qg[(i + 1) + i * ldqg];
        SLC_DLARFG(&n_minus_i, &alpha, &qg[min_i2n + i * ldqg], &int1, &nu);

        if (nu != ZERO) {
            qg[(i + 1) + i * ldqg] = ONE;

            i32 info_mb01;
            mb01md('L', n_minus_i, nu, &qg[(i + 1) + (i + 1) * ldqg], ldqg,
                   &qg[(i + 1) + i * ldqg], 1, ZERO, dwork, 1, &info_mb01);

            mb01nd('L', n_minus_i, ONE, &qg[(i + 1) + i * ldqg], 1, dwork, 1,
                   &qg[(i + 1) + (i + 1) * ldqg], ldqg, &info_mb01);

            i32 i_rows = i + 1;
            SLC_DLARF("R", &i_rows, &n_minus_i, &qg[(i + 1) + i * ldqg], &int1, &nu,
                      &qg[(i + 2) * ldqg], &ldqg, dwork);

            mb01md('U', n_minus_i, nu, &qg[(i + 1) + (i + 2) * ldqg], ldqg,
                   &qg[(i + 1) + i * ldqg], 1, ZERO, dwork, 1, &info_mb01);

            mb01nd('U', n_minus_i, ONE, &qg[(i + 1) + i * ldqg], 1, dwork, 1,
                   &qg[(i + 1) + (i + 2) * ldqg], ldqg, &info_mb01);

            i32 n_minus_i_plus1 = n - i;
            SLC_DLARF("L", &n_minus_i, &n_minus_i_plus1, &qg[(i + 1) + i * ldqg], &int1, &nu,
                      &a[(i + 1) + i * lda], &lda, dwork);

            SLC_DLARF("R", &n, &n_minus_i, &qg[(i + 1) + i * ldqg], &int1, &nu,
                      &a[(i + 1) * lda], &lda, dwork);
        }
        qg[(i + 1) + i * ldqg] = nu;

        temp = a[(i + 1) + i * lda];
        SLC_DLARTG(&temp, &alpha, &c, &s, &a[(i + 1) + i * lda]);

        if (n - i - 2 > 0) {
            i32 len = n - i - 2;
            f64 neg_s = -s;
            SLC_DROT(&len, &a[(i + 1) + (i + 2) * lda], &lda, &qg[(i + 2) + (i + 1) * ldqg], &int1, &c, &neg_s);
        }

        if (i + 1 > 0) {
            i32 len = i + 1;
            SLC_DROT(&len, &a[(i + 1) * lda], &int1, &qg[(i + 2) * ldqg], &int1, &c, &s);
        }

        if (n - i - 2 > 0) {
            i32 len = n - i - 2;
            f64 neg_s = -s;
            SLC_DROT(&len, &a[(i + 2) + (i + 1) * lda], &int1, &qg[(i + 1) + (i + 3) * ldqg], &ldqg, &c, &neg_s);
        }

        cs[2 * i] = c;
        cs[2 * i + 1] = s;

        SLC_DLARFG(&n_minus_i, &a[(i + 1) + i * lda], &a[min_i2n + i * lda], &int1, &nu);

        if (nu != ZERO) {
            temp = a[(i + 1) + i * lda];
            a[(i + 1) + i * lda] = ONE;

            SLC_DLARF("L", &n_minus_i, &n_minus_i, &a[(i + 1) + i * lda], &int1, &nu,
                      &a[(i + 1) + (i + 1) * lda], &lda, dwork);

            SLC_DLARF("R", &n, &n_minus_i, &a[(i + 1) + i * lda], &int1, &nu,
                      &a[(i + 1) * lda], &lda, dwork);

            i32 info_mb01;
            mb01md('L', n_minus_i, nu, &qg[(i + 1) + (i + 1) * ldqg], ldqg,
                   &a[(i + 1) + i * lda], 1, ZERO, dwork, 1, &info_mb01);

            mb01nd('L', n_minus_i, ONE, &a[(i + 1) + i * lda], 1, dwork, 1,
                   &qg[(i + 1) + (i + 1) * ldqg], ldqg, &info_mb01);

            i32 i_rows = i + 1;
            SLC_DLARF("R", &i_rows, &n_minus_i, &a[(i + 1) + i * lda], &int1, &nu,
                      &qg[(i + 2) * ldqg], &ldqg, dwork);

            mb01md('U', n_minus_i, nu, &qg[(i + 1) + (i + 2) * ldqg], ldqg,
                   &a[(i + 1) + i * lda], 1, ZERO, dwork, 1, &info_mb01);

            mb01nd('U', n_minus_i, ONE, &a[(i + 1) + i * lda], 1, dwork, 1,
                   &qg[(i + 1) + (i + 2) * ldqg], ldqg, &info_mb01);

            a[(i + 1) + i * lda] = temp;
        }
        tau[i] = nu;
    }

    dwork[0] = (f64)(n > 1 ? n : 1);
}
