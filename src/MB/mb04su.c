/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"

void mb04su(const i32 m, const i32 n, f64 *a, const i32 lda,
            f64 *b, const i32 ldb, f64 *cs, f64 *tau,
            f64 *dwork, const i32 ldwork, i32 *info)
{
    const f64 ONE = 1.0;
    i32 i, k;
    f64 alpha, nu, temp;
    i32 int1 = 1;

    *info = 0;

    i32 max_1_m = (1 > m) ? 1 : m;
    i32 max_1_n = (1 > n) ? 1 : n;

    if (m < 0) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (lda < max_1_m) {
        *info = -4;
    } else if (ldb < max_1_m) {
        *info = -6;
    } else if (ldwork < max_1_n) {
        dwork[0] = (f64)max_1_n;
        *info = -10;
    }

    if (*info != 0) {
        return;
    }

    k = (m < n) ? m : n;
    if (k == 0) {
        dwork[0] = ONE;
        return;
    }

    for (i = 0; i < k; i++) {
        i32 m_mi = m - i;
        i32 n_mi = n - i;
        i32 n_mi_m1 = n - i - 1;
        i32 min_ip1_m_c = (i + 1 < m - 1) ? (i + 1) : (m - 1);

        alpha = b[i + i * ldb];
        SLC_DLARFG(&m_mi, &alpha, &b[min_ip1_m_c + i * ldb], &int1, &nu);

        b[i + i * ldb] = ONE;
        SLC_DLARF("Left", &m_mi, &n_mi, &b[i + i * ldb], &int1, &nu,
                  &a[i + i * lda], &lda, dwork);
        if (i < n - 1) {
            SLC_DLARF("Left", &m_mi, &n_mi_m1, &b[i + i * ldb], &int1, &nu,
                      &b[i + (i + 1) * ldb], &ldb, dwork);
        }
        b[i + i * ldb] = nu;

        temp = a[i + i * lda];
        SLC_DLARTG(&temp, &alpha, &cs[2 * i], &cs[2 * i + 1], &a[i + i * lda]);

        if (i < n - 1) {
            SLC_DROT(&n_mi_m1, &a[i + (i + 1) * lda], &lda,
                     &b[i + (i + 1) * ldb], &ldb, &cs[2 * i], &cs[2 * i + 1]);
        }

        SLC_DLARFG(&m_mi, &a[i + i * lda], &a[min_ip1_m_c + i * lda], &int1, &tau[i]);

        if (i < n - 1) {
            temp = a[i + i * lda];
            a[i + i * lda] = ONE;
            SLC_DLARF("Left", &m_mi, &n_mi_m1, &a[i + i * lda], &int1, &tau[i],
                      &a[i + (i + 1) * lda], &lda, dwork);
            SLC_DLARF("Left", &m_mi, &n_mi_m1, &a[i + i * lda], &int1, &tau[i],
                      &b[i + (i + 1) * ldb], &ldb, dwork);
            a[i + i * lda] = temp;
        }
    }

    dwork[0] = (f64)max_1_n;
}
