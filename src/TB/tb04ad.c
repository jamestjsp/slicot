/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <stdbool.h>

static bool lsame(char ca, char cb) {
    if (ca >= 'a' && ca <= 'z') ca -= 32;
    if (cb >= 'a' && cb <= 'z') cb -= 32;
    return ca == cb;
}

static i32 imax(i32 a, i32 b) {
    return a > b ? a : b;
}

void tb04ad(const char* rowcol, i32 n, i32 m, i32 p,
            f64* a, i32 lda, f64* b, i32 ldb,
            f64* c, i32 ldc, f64* d, i32 ldd,
            i32* nr, i32* index, f64* dcoeff, i32 lddcoe,
            f64* ucoeff, i32 lduco1, i32 lduco2,
            f64 tol1, f64 tol2,
            i32* iwork, f64* dwork, i32 ldwork, i32* info) {

    const f64 ZERO = 0.0;
    const i32 int1 = 1;

    bool lrocor, lrococ;
    i32 i, ia, itau, j, jwork, k, kdcoef, maxmp, maxmpn, mplim, mwork, n1, pwork;

    *info = 0;
    lrocor = lsame(*rowcol, 'R');
    lrococ = lsame(*rowcol, 'C');
    maxmp = imax(m, p);
    mplim = imax(1, maxmp);
    maxmpn = imax(maxmp, n);
    n1 = imax(1, n);

    if (lrocor) {
        pwork = p;
        mwork = m;
    } else {
        pwork = m;
        mwork = p;
    }

    if (!lrocor && !lrococ) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (m < 0) {
        *info = -3;
    } else if (p < 0) {
        *info = -4;
    } else if (lda < n1) {
        *info = -6;
    } else if (ldb < n1) {
        *info = -8;
    } else if ((lrococ && ldc < mplim) || ldc < imax(1, p)) {
        *info = -10;
    } else if ((lrococ && ldd < mplim) || ldd < imax(1, p)) {
        *info = -12;
    } else if (lddcoe < imax(1, pwork)) {
        *info = -16;
    } else if (lduco1 < imax(1, pwork)) {
        *info = -18;
    } else if (lduco2 < imax(1, mwork)) {
        *info = -19;
    } else {
        i32 min_ldwork = n * (n + 1) +
            imax(imax(n * mwork + 2 * n + imax(n, mwork), 3 * mwork), pwork);
        if (ldwork < imax(1, min_ldwork)) {
            *info = -24;
        }
    }

    if (*info != 0) {
        return;
    }

    if (maxmpn == 0) {
        return;
    }

    ia = 0;
    itau = ia + n * n;
    jwork = itau + n;

    if (lrococ) {
        ab07md('D', n, m, p, a, lda, b, ldb, c, ldc, d, ldd);
    }

    for (k = 0; k < n + 1; k++) {
        SLC_DLASET("F", &pwork, &mwork, &ZERO, &ZERO, &ucoeff[k * lduco1 * lduco2], &lduco1);
    }

    tb04ay(n, mwork, pwork, a, lda, b, ldb, c, ldc, d, ldd,
           nr, index, dcoeff, lddcoe, ucoeff, lduco1, lduco2,
           &dwork[ia], n1, &dwork[itau], tol1, tol2, iwork,
           &dwork[jwork], ldwork - jwork, info);
    dwork[0] = dwork[jwork] + (f64)(jwork);

    if (lrococ) {
        tb01xd("D", n, mwork, pwork, iwork[0] + iwork[1] - 1, n - 1,
               a, lda, b, ldb, c, ldc, d, ldd, info);

        if (mplim != 1) {
            kdcoef = 0;
            for (i = 0; i < pwork; i++) {
                if (index[i] > kdcoef) kdcoef = index[i];
            }
            kdcoef = kdcoef + 1;

            for (k = 0; k < kdcoef; k++) {
                for (j = 0; j < mplim - 1; j++) {
                    i32 mplim_mj = mplim - j - 1;
                    SLC_DSWAP(&mplim_mj, &ucoeff[j + 1 + j * lduco1 + k * lduco1 * lduco2], &int1,
                              &ucoeff[j + (j + 1) * lduco1 + k * lduco1 * lduco2], &lduco1);
                }
            }
        }
    }
}
