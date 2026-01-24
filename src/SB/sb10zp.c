/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>

void sb10zp(
    i32 discfl,
    i32* n,
    f64* a,
    i32 lda,
    f64* b,
    f64* c,
    f64* d,
    i32* iwork,
    f64* dwork,
    i32 ldwork,
    i32* info
)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const f64 NEGONE = -1.0;
    i32 int1 = 1;

    *info = 0;

    i32 n_val = *n;
    i32 max1n = (1 > n_val) ? 1 : n_val;
    i32 min1n = (1 < n_val) ? 1 : n_val;
    i32 minwork = n_val * n_val + 5 * n_val;
    i32 minwork2 = 6 * n_val + 1 + min1n;
    if (minwork2 > minwork) minwork = minwork2;

    if (discfl != 0 && discfl != 1) {
        *info = -1;
    } else if (n_val < 0) {
        *info = -2;
    } else if (lda < max1n) {
        *info = -4;
    } else if (ldwork < minwork) {
        *info = -10;
    }

    if (*info != 0) {
        return;
    }

    if (n_val == 0) {
        dwork[0] = ONE;
        return;
    }

    i32 rep  = 0;
    i32 imp  = rep + n_val;
    i32 rez  = imp + n_val;
    i32 imz  = rez + n_val;
    i32 iwa  = rez;
    i32 idw1 = iwa + n_val * n_val;
    i32 ldw1 = ldwork - idw1;

    i32 info2;
    i32 maxwrk = 0;

    if (discfl == 1) {
        info2 = ab04md('D', n_val, 1, 1, ONE, ONE, a, lda, b, lda, c, 1, d, 1,
                       iwork, dwork, ldwork);
        if (info2 != 0) {
            *info = 1;
            return;
        }
        maxwrk = (i32)dwork[0];
    }

    f64 scald = d[0];
    f64 scalc = sqrt(fabs(scald));
    f64 scalb = (scald >= 0) ? scalc : -scalc;

    SLC_DLACPY("Full", &n_val, &n_val, a, &lda, &dwork[iwa], &n_val);

    SLC_DGEEV("N", "N", &n_val, &dwork[iwa], &n_val, &dwork[rep], &dwork[imp],
              &dwork[idw1], &int1, &dwork[idw1], &int1, &dwork[idw1], &ldw1, &info2);
    if (info2 != 0) {
        *info = 2;
        return;
    }
    i32 wk = (i32)dwork[idw1] + idw1;
    if (wk > maxwrk) maxwrk = wk;

    f64 rcond;
    info2 = ab07nd(n_val, 1, a, lda, b, lda, c, 1, d, 1, &rcond, iwork, &dwork[idw1], ldw1);
    if (info2 != 0) {
        *info = 3;
        return;
    }
    wk = (i32)dwork[idw1] + idw1;
    if (wk > maxwrk) maxwrk = wk;

    idw1 = imz + n_val;
    ldw1 = ldwork - idw1;

    SLC_DGEEV("N", "N", &n_val, a, &lda, &dwork[rez], &dwork[imz],
              &dwork[idw1], &int1, &dwork[idw1], &int1, &dwork[idw1], &ldw1, &info2);
    if (info2 != 0) {
        *info = 4;
        return;
    }
    wk = (i32)dwork[idw1] + idw1;
    if (wk > maxwrk) maxwrk = wk;

    for (i32 i = 0; i < n_val; i++) {
        if (dwork[rep + i] > ZERO) {
            dwork[rep + i] = -dwork[rep + i];
        }
        if (dwork[rez + i] > ZERO) {
            dwork[rez + i] = -dwork[rez + i];
        }
    }

    i32 iwp  = idw1;
    i32 idw2 = iwp + n_val + 1;
    i32 iwps = 0;

    mc01pd(n_val, &dwork[rep], &dwork[imp], &dwork[iwp], &dwork[idw2], &info2);

    i32 np1 = n_val + 1;
    SLC_DCOPY(&np1, &dwork[iwp], &int1, &dwork[iwps], &int1);
    for (i32 i = 0; i < (np1 / 2); i++) {
        f64 tmp = dwork[iwps + i];
        dwork[iwps + i] = dwork[iwps + np1 - 1 - i];
        dwork[iwps + np1 - 1 - i] = tmp;
    }

    i32 iwq = idw1;
    i32 iwqs = iwps + n_val + 1;
    i32 idw3 = iwqs + n_val + 1;

    mc01pd(n_val, &dwork[rez], &dwork[imz], &dwork[iwq], &dwork[idw2], &info2);

    SLC_DCOPY(&np1, &dwork[iwq], &int1, &dwork[iwqs], &int1);
    for (i32 i = 0; i < (np1 / 2); i++) {
        f64 tmp = dwork[iwqs + i];
        dwork[iwqs + i] = dwork[iwqs + np1 - 1 - i];
        dwork[iwqs + np1 - 1 - i] = tmp;
    }

    i32 index_arr[1] = {n_val};
    i32 nr;
    i32 lddcoe = 1;
    i32 lduco1 = 1;
    i32 lduco2 = 1;
    i32 ldwork_td = ldwork - idw3;

    td04ad("R", 1, 1, index_arr, &dwork[iwps], lddcoe, &dwork[iwqs], lduco1, lduco2,
           &nr, a, lda, b, lda, c, 1, d, 1, NEGONE, iwork, &dwork[idw3], ldwork_td, &info2);
    if (info2 != 0) {
        *info = 5;
        return;
    }
    *n = nr;
    wk = (i32)dwork[idw3] + idw3;
    if (wk > maxwrk) maxwrk = wk;

    if (nr > 0) {
        SLC_DSCAL(&nr, &scalb, b, &int1);
        c[nr - 1] = scalc * c[nr - 1];
    }

    d[0] = scald;

    if (discfl == 1) {
        info2 = ab04md('C', nr, 1, 1, ONE, ONE, a, lda, b, lda, c, 1, d, 1,
                       iwork, dwork, ldwork);
        if (info2 != 0) {
            *info = 6;
            return;
        }
    }

    dwork[0] = (f64)maxwrk;
}
