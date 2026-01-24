/*
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <stdlib.h>

void mb04wd(bool tranq1, bool tranq2, const i32 m, const i32 n, const i32 k,
            f64 *q1, const i32 ldq1, f64 *q2, const i32 ldq2,
            const f64 *cs, const f64 *tau, f64 *dwork,
            const i32 ldwork, i32 *info)
{
    const f64 ONE = 1.0;
    i32 ierr;
    i32 i, ib;
    i32 ki, kk, nb, nbmin, nx;
    i32 minwrk, wrkopt;
    i32 pdrs, pdt, pdw;
    bool lquery;

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
    } else {
        lquery = (ldwork == -1);
        minwrk = (1 > m + n) ? 1 : m + n;
        if (ldwork < minwrk && !lquery) {
            dwork[0] = (f64)minwrk;
            *info = -13;
        } else {
            if (m == 0 || n == 0) {
                wrkopt = 1;
            } else {
                i32 neg1 = -1;
                SLC_DORGQR(&m, &n, &k, dwork, &m, dwork, dwork, &neg1, &ierr);
                wrkopt = (minwrk > (i32)dwork[0]) ? minwrk : (i32)dwork[0];
                nb = (i32)(wrkopt / n);
                i32 opt2 = 8 * n * nb + 15 * nb * nb;
                wrkopt = (wrkopt > opt2) ? wrkopt : opt2;
            }
            if (lquery) {
                dwork[0] = (f64)wrkopt;
                return;
            }
        }
    }

    if (*info != 0) {
        return;
    }

    if (n == 0) {
        dwork[0] = ONE;
        return;
    }

    nbmin = 2;
    nx = 0;
    if (nb > 1 && nb < k) {
        nx = 0;
        if (nx < k) {
            if (ldwork < wrkopt) {
                f64 disc = 16.0 * n * n + 15.0 * ldwork;
                nb = (i32)((sqrt(disc) - 4.0 * n) / 15.0);
                nbmin = 2;
            }
        }
    }

    if (nb >= nbmin && nb < k && nx < k) {
        ki = ((k - nx - 1) / nb) * nb;
        kk = (k < ki + nb) ? k : ki + nb;
    } else {
        kk = 0;
    }

    if (kk < n) {
        i32 m_kk = m - kk;
        i32 n_kk = n - kk;
        i32 k_kk = k - kk;
        i32 ldq1_adj = ldq1;
        i32 ldq2_adj = ldq2;

        mb04wu(tranq1, tranq2, m_kk, n_kk, k_kk,
               &q1[kk + kk * ldq1], ldq1_adj, &q2[kk + kk * ldq2], ldq2_adj,
               &cs[2 * kk], &tau[kk], dwork, ldwork, &ierr);
    }

    if (kk > 0) {
        pdrs = 0;
        pdt = pdrs + 6 * nb * nb;
        pdw = pdt + 9 * nb * nb;

        if (tranq1 && tranq2) {
            for (i = ki; i >= 0; i -= nb) {
                ib = (nb < k - i) ? nb : k - i;

                if (i + ib < n) {
                    mb04qf("Forward", "Rowwise", "Rowwise", m - i, ib,
                           &q1[i + i * ldq1], ldq1, &q2[i + i * ldq2], ldq2,
                           &cs[2 * i], &tau[i], &dwork[pdrs], nb,
                           &dwork[pdt], nb, &dwork[pdw]);

                    mb04qc("Zero Structure", "Transpose", "Transpose",
                           "No Transpose", "Forward", "Rowwise", "Rowwise",
                           m - i, n - i - ib, ib,
                           &q1[i + i * ldq1], ldq1, &q2[i + i * ldq2], ldq2,
                           &dwork[pdrs], nb, &dwork[pdt], nb,
                           &q2[(i + ib) + i * ldq2], ldq2,
                           &q1[(i + ib) + i * ldq1], ldq1, &dwork[pdw]);
                }

                mb04wu(true, true, m - i, ib, ib,
                       &q1[i + i * ldq1], ldq1, &q2[i + i * ldq2], ldq2,
                       &cs[2 * i], &tau[i], dwork, ldwork, &ierr);
            }
        } else if (tranq1) {
            for (i = ki; i >= 0; i -= nb) {
                ib = (nb < k - i) ? nb : k - i;

                if (i + ib < n) {
                    mb04qf("Forward", "Rowwise", "Columnwise", m - i, ib,
                           &q1[i + i * ldq1], ldq1, &q2[i + i * ldq2], ldq2,
                           &cs[2 * i], &tau[i], &dwork[pdrs], nb,
                           &dwork[pdt], nb, &dwork[pdw]);

                    mb04qc("Zero Structure", "No Transpose", "Transpose",
                           "No Transpose", "Forward", "Rowwise", "Columnwise",
                           m - i, n - i - ib, ib,
                           &q1[i + i * ldq1], ldq1, &q2[i + i * ldq2], ldq2,
                           &dwork[pdrs], nb, &dwork[pdt], nb,
                           &q2[i + (i + ib) * ldq2], ldq2,
                           &q1[(i + ib) + i * ldq1], ldq1, &dwork[pdw]);
                }

                mb04wu(true, false, m - i, ib, ib,
                       &q1[i + i * ldq1], ldq1, &q2[i + i * ldq2], ldq2,
                       &cs[2 * i], &tau[i], dwork, ldwork, &ierr);
            }
        } else if (tranq2) {
            for (i = ki; i >= 0; i -= nb) {
                ib = (nb < k - i) ? nb : k - i;

                if (i + ib < n) {
                    mb04qf("Forward", "Columnwise", "Rowwise", m - i, ib,
                           &q1[i + i * ldq1], ldq1, &q2[i + i * ldq2], ldq2,
                           &cs[2 * i], &tau[i], &dwork[pdrs], nb,
                           &dwork[pdt], nb, &dwork[pdw]);

                    mb04qc("Zero Structure", "Transpose", "No Transpose",
                           "No Transpose", "Forward", "Columnwise", "Rowwise",
                           m - i, n - i - ib, ib,
                           &q1[i + i * ldq1], ldq1, &q2[i + i * ldq2], ldq2,
                           &dwork[pdrs], nb, &dwork[pdt], nb,
                           &q2[(i + ib) + i * ldq2], ldq2,
                           &q1[i + (i + ib) * ldq1], ldq1, &dwork[pdw]);
                }

                mb04wu(false, true, m - i, ib, ib,
                       &q1[i + i * ldq1], ldq1, &q2[i + i * ldq2], ldq2,
                       &cs[2 * i], &tau[i], dwork, ldwork, &ierr);
            }
        } else {
            for (i = ki; i >= 0; i -= nb) {
                ib = (nb < k - i) ? nb : k - i;

                if (i + ib < n) {
                    mb04qf("Forward", "Columnwise", "Columnwise", m - i, ib,
                           &q1[i + i * ldq1], ldq1, &q2[i + i * ldq2], ldq2,
                           &cs[2 * i], &tau[i], &dwork[pdrs], nb,
                           &dwork[pdt], nb, &dwork[pdw]);

                    mb04qc("Zero Structure", "No Transpose", "No Transpose",
                           "No Transpose", "Forward", "Columnwise", "Columnwise",
                           m - i, n - i - ib, ib,
                           &q1[i + i * ldq1], ldq1, &q2[i + i * ldq2], ldq2,
                           &dwork[pdrs], nb, &dwork[pdt], nb,
                           &q2[i + (i + ib) * ldq2], ldq2,
                           &q1[i + (i + ib) * ldq1], ldq1, &dwork[pdw]);
                }

                mb04wu(false, false, m - i, ib, ib,
                       &q1[i + i * ldq1], ldq1, &q2[i + i * ldq2], ldq2,
                       &cs[2 * i], &tau[i], dwork, ldwork, &ierr);
            }
        }
    }

    dwork[0] = (f64)wrkopt;
}
