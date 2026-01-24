/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <ctype.h>
#include <stdbool.h>

void mb02cy(const char* typet, const char* strucg, i32 p, i32 q, i32 n, i32 k,
            f64* a, i32 lda, f64* b, i32 ldb, f64* h, i32 ldh,
            f64* cs, i32 lcs, f64* dwork, i32 ldwork, i32* info)
{
    const f64 zero = 0.0;
    const f64 one = 1.0;

    char typet_u = (char)toupper((unsigned char)typet[0]);
    char strucg_u = (char)toupper((unsigned char)strucg[0]);

    bool isrow = (typet_u == 'R');
    bool islwr = (strucg_u == 'T');

    *info = 0;

    if (!isrow && typet_u != 'C') {
        *info = -1;
    } else if (!islwr && strucg_u != 'N') {
        *info = -2;
    } else if (p < 0) {
        *info = -3;
    } else if (q < 0) {
        *info = -4;
    } else if (n < 0) {
        *info = -5;
    } else if (k < 0 || k > p) {
        *info = -6;
    } else if (lda < 1 || (isrow && lda < p) || (!isrow && lda < n)) {
        *info = -8;
    } else if (ldb < 1 || (isrow && ldb < q) || (!isrow && ldb < n)) {
        *info = -10;
    } else if (ldh < 1 || (isrow && ldh < q) || (!isrow && ldh < k)) {
        *info = -12;
    } else if (lcs < 2 * k + (k < q ? k : q)) {
        *info = -14;
    } else if (ldwork < (n > 1 ? n : 1)) {
        dwork[0] = (f64)(n > 1 ? n : 1);
        *info = -16;
    }

    if (*info != 0) {
        return;
    }

    i32 min_nkq = n;
    if (k < min_nkq) min_nkq = k;
    if (q < min_nkq) min_nkq = q;

    if (min_nkq == 0) {
        dwork[0] = one;
        return;
    }

    i32 maxwrk = 1;
    i32 ierr = 0;
    i32 int1 = 1;

    if (isrow) {
        if (islwr) {
            for (i32 i = 0; i < k; i++) {
                i32 ci = n - k + i;
                i32 qi_min = (i + 1 < q) ? (i + 1) : q;

                f64 tau = h[0 + i * ldh];
                h[0 + i * ldh] = one;
                SLC_DLARF("L", &qi_min, &ci, &h[0 + i * ldh], &int1, &tau, b, &ldb, dwork);
                h[0 + i * ldh] = tau;

                f64 c = cs[i * 2];
                f64 s = cs[i * 2 + 1];

                f64 inv_c = one / c;
                f64 neg_s_c = -s / c;

                SLC_DSCAL(&ci, &inv_c, &a[i], &lda);
                SLC_DAXPY(&ci, &neg_s_c, b, &ldb, &a[i], &lda);
                SLC_DSCAL(&ci, &c, b, &ldb);
                f64 neg_s = -s;
                SLC_DAXPY(&ci, &neg_s, &a[i], &lda, b, &ldb);

                i32 idx = n - k + i;
                b[0 + idx * ldb] = neg_s_c * a[i + idx * lda];
                a[i + idx * lda] = inv_c * a[i + idx * lda];

                if (q > 1) {
                    i32 rows = q - 1;
                    SLC_DLASET("A", &rows, &int1, &zero, &zero, &b[1 + idx * ldb], &ldb);
                }
            }
        } else {
            i32 minKQ = (k < q) ? k : q;
            SLC_DORMQR("L", "T", &q, &n, &minKQ, h, &ldh, &cs[2 * k], b, &ldb, dwork, &ldwork, &ierr);
            maxwrk = (i32)dwork[0];

            for (i32 i = 0; i < k; i++) {
                i32 qi_min = (i + 1 < q) ? (i + 1) : q;

                f64 tau = h[0 + i * ldh];
                h[0 + i * ldh] = one;
                SLC_DLARF("L", &qi_min, &n, &h[0 + i * ldh], &int1, &tau, b, &ldb, dwork);
                h[0 + i * ldh] = tau;

                f64 c = cs[i * 2];
                f64 s = cs[i * 2 + 1];

                f64 inv_c = one / c;
                f64 neg_s_c = -s / c;

                SLC_DSCAL(&n, &inv_c, &a[i], &lda);
                SLC_DAXPY(&n, &neg_s_c, b, &ldb, &a[i], &lda);
                SLC_DSCAL(&n, &c, b, &ldb);
                f64 neg_s = -s;
                SLC_DAXPY(&n, &neg_s, &a[i], &lda, b, &ldb);
            }
        }
    } else {
        if (islwr) {
            for (i32 i = 0; i < k; i++) {
                i32 ci = n - k + i;
                i32 qi_min = (i + 1 < q) ? (i + 1) : q;

                f64 tau = h[i + 0 * ldh];
                h[i + 0 * ldh] = one;
                SLC_DLARF("R", &ci, &qi_min, &h[i], &ldh, &tau, b, &ldb, dwork);
                h[i + 0 * ldh] = tau;

                f64 c = cs[i * 2];
                f64 s = cs[i * 2 + 1];

                f64 inv_c = one / c;
                f64 neg_s_c = -s / c;

                SLC_DSCAL(&ci, &inv_c, &a[0 + i * lda], &int1);
                SLC_DAXPY(&ci, &neg_s_c, b, &int1, &a[0 + i * lda], &int1);
                SLC_DSCAL(&ci, &c, b, &int1);
                f64 neg_s = -s;
                SLC_DAXPY(&ci, &neg_s, &a[0 + i * lda], &int1, b, &int1);

                i32 idx = n - k + i;
                b[idx + 0 * ldb] = neg_s_c * a[idx + i * lda];
                a[idx + i * lda] = inv_c * a[idx + i * lda];

                if (q > 1) {
                    i32 cols = q - 1;
                    SLC_DLASET("A", &int1, &cols, &zero, &zero, &b[idx + 1 * ldb], &ldb);
                }
            }
        } else {
            i32 minKQ = (k < q) ? k : q;
            SLC_DORMLQ("R", "T", &n, &q, &minKQ, h, &ldh, &cs[2 * k], b, &ldb, dwork, &ldwork, &ierr);
            maxwrk = (i32)dwork[0];

            for (i32 i = 0; i < k; i++) {
                i32 qi_min = (i + 1 < q) ? (i + 1) : q;

                f64 tau = h[i + 0 * ldh];
                h[i + 0 * ldh] = one;
                SLC_DLARF("R", &n, &qi_min, &h[i], &ldh, &tau, b, &ldb, dwork);
                h[i + 0 * ldh] = tau;

                f64 c = cs[i * 2];
                f64 s = cs[i * 2 + 1];

                f64 inv_c = one / c;
                f64 neg_s_c = -s / c;

                SLC_DSCAL(&n, &inv_c, &a[0 + i * lda], &int1);
                SLC_DAXPY(&n, &neg_s_c, b, &int1, &a[0 + i * lda], &int1);
                SLC_DSCAL(&n, &c, b, &int1);
                f64 neg_s = -s;
                SLC_DAXPY(&n, &neg_s, &a[0 + i * lda], &int1, b, &int1);
            }
        }
    }

    dwork[0] = (f64)(maxwrk > n ? maxwrk : n);
}
