/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <ctype.h>
#include <stdbool.h>

void mb02cx(const char* typet, i32 p, i32 q, i32 k,
            f64* a, i32 lda, f64* b, i32 ldb,
            f64* cs, i32 lcs, f64* dwork, i32 ldwork, i32* info)
{
    const f64 zero = 0.0;
    const f64 one = 1.0;

    char typet_u = (char)toupper((unsigned char)typet[0]);
    bool isrow = (typet_u == 'R');

    *info = 0;

    if (!isrow && typet_u != 'C') {
        *info = -1;
    } else if (p < 0) {
        *info = -2;
    } else if (q < 0) {
        *info = -3;
    } else if (k < 0 || k > p) {
        *info = -4;
    } else if (lda < 1 || (isrow && lda < p) || (!isrow && lda < k)) {
        *info = -6;
    } else if (ldb < 1 || (isrow && ldb < q) || (!isrow && ldb < k)) {
        *info = -8;
    } else if (lcs < 2 * k + (k < q ? k : q)) {
        *info = -10;
    } else if (ldwork < (k > 1 ? k : 1)) {
        dwork[0] = (f64)(k > 1 ? k : 1);
        *info = -12;
    }

    if (*info != 0) {
        return;
    }

    i32 minQK = (q < k) ? q : k;
    if (minQK == 0) {
        dwork[0] = one;
        return;
    }

    i32 int1 = 1;
    i32 ierr = 0;
    f64 maxwrk = one;

    if (isrow) {
        SLC_DGEQRF(&q, &k, b, &ldb, &cs[2 * k], dwork, &ldwork, &ierr);
        maxwrk = dwork[0];

        for (i32 i = 0; i < k; i++) {
            f64 alpha, tau;

            if (q > 1) {
                i32 qi_min = (i + 1 < q) ? (i + 1) : q;
                SLC_DLARFG(&qi_min, &b[0 + i * ldb], &b[1 + i * ldb], &int1, &tau);
                alpha = b[0 + i * ldb];
                b[0 + i * ldb] = one;
                if (k > i + 1) {
                    i32 cols = k - i - 1;
                    SLC_DLARF("L", &qi_min, &cols, &b[0 + i * ldb], &int1, &tau,
                              &b[0 + (i + 1) * ldb], &ldb, dwork);
                }
                b[0 + i * ldb] = alpha;
            } else {
                alpha = b[0 + i * ldb];
                tau = zero;
            }

            f64 beta = a[i + i * lda];
            f64 c, s;
            i32 ma_info = 0;
            ma02fd(&beta, alpha, &c, &s, &ma_info);
            if (ma_info != 0) {
                *info = 1;
                return;
            }

            cs[i * 2] = c;
            cs[i * 2 + 1] = s;

            i32 len = k - i;
            f64 inv_c = one / c;
            f64 neg_s_c = -s / c;
            f64 neg_s = -s;

            SLC_DSCAL(&len, &inv_c, &a[i + i * lda], &lda);
            SLC_DAXPY(&len, &neg_s_c, &b[0 + i * ldb], &ldb, &a[i + i * lda], &lda);
            SLC_DSCAL(&len, &c, &b[0 + i * ldb], &ldb);
            SLC_DAXPY(&len, &neg_s, &a[i + i * lda], &lda, &b[0 + i * ldb], &ldb);
            b[0 + i * ldb] = tau;
        }
    } else {
        SLC_DGELQF(&k, &q, b, &ldb, &cs[2 * k], dwork, &ldwork, &ierr);
        maxwrk = dwork[0];

        for (i32 i = 0; i < k; i++) {
            f64 alpha, tau;

            if (q > 1) {
                i32 qi_min = (i + 1 < q) ? (i + 1) : q;
                SLC_DLARFG(&qi_min, &b[i + 0 * ldb], &b[i + 1 * ldb], &ldb, &tau);
                alpha = b[i + 0 * ldb];
                b[i + 0 * ldb] = one;
                if (k > i + 1) {
                    i32 rows = k - i - 1;
                    SLC_DLARF("R", &rows, &qi_min, &b[i], &ldb, &tau,
                              &b[i + 1], &ldb, dwork);
                }
                b[i + 0 * ldb] = alpha;
            } else {
                alpha = b[i + 0 * ldb];
                tau = zero;
            }

            f64 beta = a[i + i * lda];
            f64 c, s;
            i32 ma_info = 0;
            ma02fd(&beta, alpha, &c, &s, &ma_info);
            if (ma_info != 0) {
                *info = 1;
                return;
            }

            cs[i * 2] = c;
            cs[i * 2 + 1] = s;

            i32 len = k - i;
            f64 inv_c = one / c;
            f64 neg_s_c = -s / c;
            f64 neg_s = -s;

            SLC_DSCAL(&len, &inv_c, &a[i + i * lda], &int1);
            SLC_DAXPY(&len, &neg_s_c, &b[i], &int1, &a[i + i * lda], &int1);
            SLC_DSCAL(&len, &c, &b[i], &int1);
            SLC_DAXPY(&len, &neg_s, &a[i + i * lda], &int1, &b[i], &int1);
            b[i + 0 * ldb] = tau;
        }
    }

    dwork[0] = maxwrk;
}
