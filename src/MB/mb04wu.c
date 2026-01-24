/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"

void mb04wu(bool tranq1, bool tranq2, const i32 m, const i32 n, const i32 k,
            f64 *q1, const i32 ldq1, f64 *q2, const i32 ldq2,
            const f64 *cs, const f64 *tau, f64 *dwork,
            const i32 ldwork, i32 *info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    i32 i, j;
    f64 nu;

    *info = 0;

    i32 max_1_m = (1 > m) ? 1 : m;
    i32 max_1_n = (1 > n) ? 1 : n;

    if (m < 0) {
        *info = -3;
    } else if (n < 0 || n > m) {
        *info = -4;
    } else if (k < 0 || k > n) {
        *info = -5;
    } else if ((tranq1 && ldq1 < max_1_n) || (!tranq1 && ldq1 < max_1_m)) {
        *info = -7;
    } else if ((tranq2 && ldq2 < max_1_n) || (!tranq2 && ldq2 < max_1_m)) {
        *info = -9;
    } else if (ldwork < (1 > m + n ? 1 : m + n)) {
        dwork[0] = (f64)(1 > m + n ? 1 : m + n);
        *info = -13;
    }

    if (*info != 0) {
        return;
    }

    if (n == 0) {
        dwork[0] = ONE;
        return;
    }

    i32 int1 = 1;

    for (j = k; j < n; j++) {
        for (i = 0; i < m; i++) {
            q1[i + j * ldq1] = ZERO;
        }
        q1[j + j * ldq1] = ONE;
    }

    i32 n_minus_k = n - k;
    if (n_minus_k > 0) {
        SLC_DLASET("All", &m, &n_minus_k, &ZERO, &ZERO, &q2[k * ldq2], &ldq2);
    }

    if (tranq1 && tranq2) {
        for (i = k - 1; i >= 0; i--) {
            i32 m_mi = m - i;
            i32 n_mi = n - i;
            i32 n_mi_m1 = n - i - 1;

            SLC_DCOPY(&m_mi, &q2[i + i * ldq2], &ldq2, dwork, &int1);

            if (i < n - 1) {
                q1[i + i * ldq1] = ONE;
                SLC_DLARF("Right", &n_mi_m1, &m_mi, &q1[i + i * ldq1], &ldq1, &tau[i],
                          &q1[(i + 1) + i * ldq1], &ldq1, &dwork[m]);
                SLC_DLARF("Right", &n_mi_m1, &m_mi, &q1[i + i * ldq1], &ldq1, &tau[i],
                          &q2[(i + 1) + i * ldq2], &ldq2, &dwork[m]);
            }
            if (i < m - 1) {
                i32 m_mi_m1 = m - i - 1;
                f64 neg_tau = -tau[i];
                SLC_DSCAL(&m_mi_m1, &neg_tau, &q1[i + (i + 1) * ldq1], &ldq1);
            }
            q1[i + i * ldq1] = ONE - tau[i];

            for (j = 0; j < i; j++) {
                q1[i + j * ldq1] = ZERO;
            }
            for (j = 0; j < m; j++) {
                q2[i + j * ldq2] = ZERO;
            }

            SLC_DROT(&n_mi, &q1[i + i * ldq1], &int1, &q2[i + i * ldq2], &int1,
                     &cs[2 * i], &cs[2 * i + 1]);

            nu = dwork[0];
            dwork[0] = ONE;
            SLC_DLARF("Right", &n_mi, &m_mi, dwork, &int1, &nu,
                      &q1[i + i * ldq1], &ldq1, &dwork[m]);
            SLC_DLARF("Right", &n_mi, &m_mi, dwork, &int1, &nu,
                      &q2[i + i * ldq2], &ldq2, &dwork[m]);
        }
    } else if (tranq1) {
        for (i = k - 1; i >= 0; i--) {
            i32 m_mi = m - i;
            i32 n_mi = n - i;
            i32 n_mi_m1 = n - i - 1;

            SLC_DCOPY(&m_mi, &q2[i + i * ldq2], &int1, dwork, &int1);

            if (i < n - 1) {
                q1[i + i * ldq1] = ONE;
                SLC_DLARF("Right", &n_mi_m1, &m_mi, &q1[i + i * ldq1], &ldq1, &tau[i],
                          &q1[(i + 1) + i * ldq1], &ldq1, &dwork[m]);
                SLC_DLARF("Left", &m_mi, &n_mi_m1, &q1[i + i * ldq1], &ldq1, &tau[i],
                          &q2[i + (i + 1) * ldq2], &ldq2, &dwork[m]);
            }
            if (i < m - 1) {
                i32 m_mi_m1 = m - i - 1;
                f64 neg_tau = -tau[i];
                SLC_DSCAL(&m_mi_m1, &neg_tau, &q1[i + (i + 1) * ldq1], &ldq1);
            }
            q1[i + i * ldq1] = ONE - tau[i];

            for (j = 0; j < i; j++) {
                q1[i + j * ldq1] = ZERO;
            }
            for (j = 0; j < m; j++) {
                q2[j + i * ldq2] = ZERO;
            }

            SLC_DROT(&n_mi, &q1[i + i * ldq1], &int1, &q2[i + i * ldq2], &ldq2,
                     &cs[2 * i], &cs[2 * i + 1]);

            nu = dwork[0];
            dwork[0] = ONE;
            SLC_DLARF("Right", &n_mi, &m_mi, dwork, &int1, &nu,
                      &q1[i + i * ldq1], &ldq1, &dwork[m]);
            SLC_DLARF("Left", &m_mi, &n_mi, dwork, &int1, &nu,
                      &q2[i + i * ldq2], &ldq2, &dwork[m]);
        }
    } else if (tranq2) {
        for (i = k - 1; i >= 0; i--) {
            i32 m_mi = m - i;
            i32 n_mi = n - i;
            i32 n_mi_m1 = n - i - 1;

            SLC_DCOPY(&m_mi, &q2[i + i * ldq2], &ldq2, dwork, &int1);

            if (i < n - 1) {
                q1[i + i * ldq1] = ONE;
                SLC_DLARF("Left", &m_mi, &n_mi_m1, &q1[i + i * ldq1], &int1, &tau[i],
                          &q1[i + (i + 1) * ldq1], &ldq1, &dwork[m]);
                SLC_DLARF("Right", &n_mi_m1, &m_mi, &q1[i + i * ldq1], &int1, &tau[i],
                          &q2[(i + 1) + i * ldq2], &ldq2, &dwork[m]);
            }
            if (i < m - 1) {
                i32 m_mi_m1 = m - i - 1;
                f64 neg_tau = -tau[i];
                SLC_DSCAL(&m_mi_m1, &neg_tau, &q1[(i + 1) + i * ldq1], &int1);
            }
            q1[i + i * ldq1] = ONE - tau[i];

            for (j = 0; j < i; j++) {
                q1[j + i * ldq1] = ZERO;
            }
            for (j = 0; j < m; j++) {
                q2[i + j * ldq2] = ZERO;
            }

            SLC_DROT(&n_mi, &q1[i + i * ldq1], &ldq1, &q2[i + i * ldq2], &int1,
                     &cs[2 * i], &cs[2 * i + 1]);

            nu = dwork[0];
            dwork[0] = ONE;
            SLC_DLARF("Left", &m_mi, &n_mi, dwork, &int1, &nu,
                      &q1[i + i * ldq1], &ldq1, &dwork[m]);
            SLC_DLARF("Right", &n_mi, &m_mi, dwork, &int1, &nu,
                      &q2[i + i * ldq2], &ldq2, &dwork[m]);
        }
    } else {
        for (i = k - 1; i >= 0; i--) {
            i32 m_mi = m - i;
            i32 n_mi = n - i;
            i32 n_mi_m1 = n - i - 1;

            SLC_DCOPY(&m_mi, &q2[i + i * ldq2], &int1, dwork, &int1);

            if (i < n - 1) {
                q1[i + i * ldq1] = ONE;
                SLC_DLARF("Left", &m_mi, &n_mi_m1, &q1[i + i * ldq1], &int1, &tau[i],
                          &q1[i + (i + 1) * ldq1], &ldq1, &dwork[m]);
                SLC_DLARF("Left", &m_mi, &n_mi_m1, &q1[i + i * ldq1], &int1, &tau[i],
                          &q2[i + (i + 1) * ldq2], &ldq2, &dwork[m]);
            }
            if (i < m - 1) {
                i32 m_mi_m1 = m - i - 1;
                f64 neg_tau = -tau[i];
                SLC_DSCAL(&m_mi_m1, &neg_tau, &q1[(i + 1) + i * ldq1], &int1);
            }
            q1[i + i * ldq1] = ONE - tau[i];

            for (j = 0; j < i; j++) {
                q1[j + i * ldq1] = ZERO;
            }
            for (j = 0; j < m; j++) {
                q2[j + i * ldq2] = ZERO;
            }

            SLC_DROT(&n_mi, &q1[i + i * ldq1], &ldq1, &q2[i + i * ldq2], &ldq2,
                     &cs[2 * i], &cs[2 * i + 1]);

            nu = dwork[0];
            dwork[0] = ONE;
            SLC_DLARF("Left", &m_mi, &n_mi, dwork, &int1, &nu,
                      &q1[i + i * ldq1], &ldq1, &dwork[m]);
            SLC_DLARF("Left", &m_mi, &n_mi, dwork, &int1, &nu,
                      &q2[i + i * ldq2], &ldq2, &dwork[m]);
        }
    }

    dwork[0] = (f64)(1 > m + n ? 1 : m + n);
}
