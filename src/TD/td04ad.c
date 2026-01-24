/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <ctype.h>

static inline i32 max_i32(i32 a, i32 b) {
    return a > b ? a : b;
}

void td04ad(
    const char* rowcol,
    const i32 m,
    const i32 p,
    const i32* index,
    f64* dcoeff,
    const i32 lddcoe,
    f64* ucoeff,
    const i32 lduco1,
    const i32 lduco2,
    i32* nr,
    f64* a,
    const i32 lda,
    f64* b,
    const i32 ldb,
    f64* c,
    const i32 ldc,
    f64* d,
    const i32 ldd,
    const f64 tol,
    i32* iwork,
    f64* dwork,
    const i32 ldwork,
    i32* info
)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    char rc = (char)toupper((unsigned char)rowcol[0]);
    bool lrocor = (rc == 'R');
    bool lrococ = (rc == 'C');
    i32 mplim = max_i32(1, max_i32(m, p));

    *info = 0;

    if (!lrocor && !lrococ) {
        *info = -1;
    } else if (m < 0) {
        *info = -2;
    } else if (p < 0) {
        *info = -3;
    } else if ((lrocor && lddcoe < max_i32(1, p)) ||
               (lrococ && lddcoe < max_i32(1, m))) {
        *info = -6;
    } else if ((lrocor && lduco1 < max_i32(1, p)) ||
               (lrococ && lduco1 < mplim)) {
        *info = -8;
    } else if ((lrocor && lduco2 < max_i32(1, m)) ||
               (lrococ && lduco2 < mplim)) {
        *info = -9;
    }

    i32 n = 0;
    i32 kdcoef = 0;
    i32 pwork, mwork;

    if (*info == 0) {
        if (lrocor) {
            pwork = p;
            mwork = m;
        } else {
            pwork = m;
            mwork = p;
        }

        for (i32 i = 0; i < pwork; i++) {
            kdcoef = max_i32(kdcoef, index[i]);
            n += index[i];
        }
        kdcoef = kdcoef + 1;

        if (lda < max_i32(1, n)) {
            *info = -12;
        } else if (ldb < max_i32(1, n)) {
            *info = -14;
        } else if (ldc < mplim) {
            *info = -16;
        } else if ((lrocor && ldd < max_i32(1, p)) ||
                   (lrococ && ldd < mplim)) {
            *info = -18;
        } else if (ldwork < max_i32(1, n + max_i32(n, max_i32(3 * m, 3 * p)))) {
            *info = -22;
        }
    }

    if (*info != 0) {
        return;
    }

    if (max_i32(n, max_i32(m, p)) == 0) {
        *nr = 0;
        dwork[0] = ONE;
        return;
    }

    i32 int1 = 1;
    i32 jstop = 0;

    if (lrococ) {
        if (p < m) {
            i32 rows = m - p;
            for (i32 k = 0; k < kdcoef; k++) {
                SLC_DLASET("Full", &rows, &mplim, &ZERO, &ZERO,
                           &ucoeff[p + k * lduco1 * lduco2], &lduco1);
            }
        } else if (p > m) {
            i32 cols = p - m;
            for (i32 k = 0; k < kdcoef; k++) {
                SLC_DLASET("Full", &mplim, &cols, &ZERO, &ZERO,
                           &ucoeff[m * lduco1 + k * lduco1 * lduco2], &lduco1);
            }
        }

        if (mplim != 1) {
            jstop = mplim - 1;

            for (i32 k = 0; k < kdcoef; k++) {
                for (i32 j = 0; j < jstop; j++) {
                    i32 count = mplim - j - 1;
                    SLC_DSWAP(&count,
                              &ucoeff[(j + 1) + j * lduco1 + k * lduco1 * lduco2], &int1,
                              &ucoeff[j + (j + 1) * lduco1 + k * lduco1 * lduco2], &lduco1);
                }
            }
        }
    }

    td03ay(mwork, pwork, index, dcoeff, lddcoe, ucoeff, lduco1, lduco2,
           n, a, lda, b, ldb, c, ldc, d, ldd, info);

    if (*info > 0) {
        return;
    }

    tb01pd("Minimal", "Scale", n, mwork, pwork, a, lda, b, ldb, c, ldc,
           nr, tol, iwork, dwork, ldwork, info);

    if (lrococ) {
        i32 kl = 0;
        if (iwork[0] > 0) {
            kl = iwork[0] - 1;
            if (*nr > 0 && iwork[1] > 0) {
                kl = kl + iwork[1];
            }
        }

        i32 ku = max_i32(0, *nr - 1);
        i32 info_tmp = 0;

        tb01xd("D", *nr, mwork, pwork, kl, ku, a, lda, b, ldb, c, ldc, d, ldd, &info_tmp);

        if (mplim != 1) {
            for (i32 k = 0; k < kdcoef; k++) {
                for (i32 j = 0; j < jstop; j++) {
                    i32 count = mplim - j - 1;
                    SLC_DSWAP(&count,
                              &ucoeff[(j + 1) + j * lduco1 + k * lduco1 * lduco2], &int1,
                              &ucoeff[j + (j + 1) * lduco1 + k * lduco1 * lduco2], &lduco1);
                }
            }
        }
    }
}
