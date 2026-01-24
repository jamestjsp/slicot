// SPDX-License-Identifier: BSD-3-Clause

#include "slicot.h"
#include "slicot_blas.h"
#include <complex.h>
#include <math.h>
#include <string.h>

void mb04iz(i32 n, i32 m, i32 p, i32 l, c128 *a, i32 lda, c128 *b, i32 ldb,
            c128 *tau, c128 *zwork, i32 lzwork, i32 *info)
{
    const c128 zero = 0.0 + 0.0 * I;
    const c128 one = 1.0 + 0.0 * I;

    *info = 0;
    bool lquery = (lzwork == -1);

    if (n < 0) {
        *info = -1;
    } else if (m < 0) {
        *info = -2;
    } else if (p < 0) {
        *info = -3;
    } else if (l < 0) {
        *info = -4;
    } else if (lda < (n > 0 ? n : 1)) {
        *info = -6;
    } else if (ldb < 1 || (l > 0 && ldb < n)) {
        *info = -8;
    } else {
        i32 minwork = 1;
        if (m > 1) {
            i32 val = m - 1;
            if (val > minwork) minwork = val;
        }
        if (m > p) {
            i32 val = m - p;
            if (val > minwork) minwork = val;
        }
        if (l > minwork) minwork = l;

        if (lquery) {
            i32 wrkopt = minwork;
            if (m > p) {
                i32 info_qrf = 0;
                i32 lwork_query = -1;
                c128 work_tmp;
                i32 nmp = n - p;
                i32 mmp = m - p;
                SLC_ZGEQRF(&nmp, &mmp, a, &lda, tau, &work_tmp, &lwork_query, &info_qrf);
                i32 zgeqrf_opt = (i32)creal(work_tmp);
                if (zgeqrf_opt > wrkopt) wrkopt = zgeqrf_opt;

                if (l > 0) {
                    i32 info_mqr = 0;
                    i32 minval = (n < m ? n : m) - p;
                    SLC_ZUNMQR("Left", "Conjugate", &nmp, &l, &minval,
                              a, &lda, tau, b, &ldb, &work_tmp, &lwork_query,
                              &info_mqr);
                    i32 zunmqr_opt = (i32)creal(work_tmp);
                    if (zunmqr_opt > wrkopt) wrkopt = zunmqr_opt;
                }
            }
            zwork[0] = (c128)wrkopt;
        } else if (lzwork < minwork) {
            *info = -11;
        }
    }

    if (*info != 0) {
        return;
    } else if (lquery) {
        return;
    }

    i32 minval = (m < n ? m : n);
    if (minval == 0) {
        zwork[0] = one;
        return;
    } else if (n <= p + 1) {
        for (i32 i = 0; i < minval; i++) {
            tau[i] = zero;
        }
        zwork[0] = one;
        return;
    }

    i32 wrkopt = 1;
    i32 pmin = (p < m ? p : m);
    i32 int1 = 1;

    for (i32 i = 0; i < pmin; i++) {
        i32 vec_len = n - p;
        i32 offset = i + i * lda;

        SLC_ZLARFG(&vec_len, &a[offset], &a[offset + 1], &int1, &tau[i]);

        if (cabs(tau[i]) != 0.0) {
            c128 first = a[offset];
            a[offset] = one;

            if (i < m - 1) {
                i32 ncols = m - i - 1;
                c128 tau_conj = conj(tau[i]);
                SLC_ZLARF("Left", &vec_len, &ncols, &a[offset], &int1,
                         &tau_conj, &a[offset + lda], &lda, zwork);
            }

            if (l > 0) {
                c128 tau_conj = conj(tau[i]);
                SLC_ZLARF("Left", &vec_len, &l, &a[offset], &int1,
                         &tau_conj, &b[i], &ldb, zwork);
            }

            a[offset] = first;
        }
    }

    if (m > 1) {
        if (wrkopt < m - 1) wrkopt = m - 1;
    }
    if (l > wrkopt) wrkopt = l;

    if (m > p) {
        i32 nrows = n - p;
        i32 ncols = m - p;
        i32 offset = p + p * lda;
        i32 info_qrf = 0;

        SLC_ZGEQRF(&nrows, &ncols, &a[offset], &lda, &tau[p], zwork, &lzwork,
                  &info_qrf);
        i32 opt_qrf = (i32)creal(zwork[0]);
        if (opt_qrf > wrkopt) wrkopt = opt_qrf;

        if (l > 0) {
            i32 minval2 = (n < m ? n : m) - p;
            i32 info_mqr = 0;

            SLC_ZUNMQR("Left", "Conjugate", &nrows, &l, &minval2, &a[offset],
                      &lda, &tau[p], &b[p], &ldb, zwork, &lzwork, &info_mqr);
            i32 opt_mqr = (i32)creal(zwork[0]);
            if (opt_mqr > wrkopt) wrkopt = opt_mqr;
        }
    }

    zwork[0] = (c128)wrkopt;
}
