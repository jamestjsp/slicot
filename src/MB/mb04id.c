// SPDX-License-Identifier: BSD-3-Clause

#include "slicot.h"
#include "slicot_blas.h"
#include <string.h>
#include <math.h>

void mb04id(i32 n, i32 m, i32 p, i32 l, f64 *a, i32 lda, f64 *b, i32 ldb,
            f64 *tau, f64 *dwork, i32 ldwork, i32 *info)
{
    const f64 zero = 0.0;
    const f64 one = 1.0;

    *info = 0;
    bool lquery = (ldwork == -1);

    // Parameter validation
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
                // Query DGEQRF
                i32 info_qrf = 0;
                i32 lwork_query = -1;
                f64 work_tmp;
                SLC_DGEQRF(&(i32){n - p}, &(i32){m - p}, a, &lda, tau,
                          &work_tmp, &lwork_query, &info_qrf);
                i32 dgeqrf_opt = (i32)work_tmp;
                if (dgeqrf_opt > wrkopt) wrkopt = dgeqrf_opt;

                if (l > 0) {
                    // Query DORMQR
                    i32 info_mqr = 0;
                    i32 minval = (n < m ? n : m) - p;
                    SLC_DORMQR("Left", "Transpose", &(i32){n - p}, &l, &minval,
                              a, &lda, tau, b, &ldb, &work_tmp, &lwork_query,
                              &info_mqr);
                    i32 dormqr_opt = (i32)work_tmp;
                    if (dormqr_opt > wrkopt) wrkopt = dormqr_opt;
                }
            }
            dwork[0] = (f64)wrkopt;
        } else if (ldwork < minwork) {
            *info = -11;
        }
    }

    if (*info != 0) {
        return;
    } else if (lquery) {
        return;
    }

    // Quick return if possible
    i32 minval = (m < n ? m : n);
    if (minval == 0) {
        dwork[0] = one;
        return;
    } else if (n <= p + 1) {
        for (i32 i = 0; i < minval; i++) {
            tau[i] = zero;
        }
        dwork[0] = one;
        return;
    }

    // Annihilate subdiagonal elements exploiting structure
    i32 wrkopt = 1;
    i32 pmin = (p < m ? p : m);

    for (i32 i = 0; i < pmin; i++) {
        // Generate Householder reflector for column i
        // Reflector acts on rows i to n-1, exploiting that rows n-p to n-1
        // have zeros in first p columns
        i32 vec_len = n - p;
        i32 offset = i + i * lda;

        SLC_DLARFG(&vec_len, &a[offset], &a[offset + 1], &(i32){1}, &tau[i]);

        if (tau[i] != zero) {
            f64 first = a[offset];
            a[offset] = one;

            // Apply to remaining columns of A
            if (i < m - 1) {
                i32 ncols = m - i - 1;
                SLC_DLARF("Left", &vec_len, &ncols, &a[offset], &(i32){1},
                         &tau[i], &a[offset + lda], &lda, dwork);
            }

            // Apply to B if present
            if (l > 0) {
                SLC_DLARF("Left", &vec_len, &l, &a[offset], &(i32){1},
                         &tau[i], &b[i], &ldb, dwork);
            }

            a[offset] = first;
        }
    }

    if (m > 1) {
        if (wrkopt < m - 1) wrkopt = m - 1;
    }
    if (l > wrkopt) wrkopt = l;

    // Fast QR factorization of remaining submatrix
    if (m > p) {
        i32 nrows = n - p;
        i32 ncols = m - p;
        i32 offset = p + p * lda;
        i32 info_qrf = 0;

        SLC_DGEQRF(&nrows, &ncols, &a[offset], &lda, &tau[p], dwork, &ldwork,
                  &info_qrf);
        i32 opt_qrf = (i32)dwork[0];
        if (opt_qrf > wrkopt) wrkopt = opt_qrf;

        if (l > 0) {
            // Apply transformations to B
            i32 minval2 = (n < m ? n : m) - p;
            i32 info_mqr = 0;

            SLC_DORMQR("Left", "Transpose", &nrows, &l, &minval2, &a[offset],
                      &lda, &tau[p], &b[p], &ldb, dwork, &ldwork, &info_mqr);
            i32 opt_mqr = (i32)dwork[0];
            if (opt_mqr > wrkopt) wrkopt = opt_mqr;
        }
    }

    dwork[0] = (f64)wrkopt;
}
