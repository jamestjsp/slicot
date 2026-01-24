/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"

void tf01rd(const i32 na, const i32 nb, const i32 nc, const i32 n,
            const f64* a, const i32 lda, const f64* b, const i32 ldb,
            const f64* c, const i32 ldc, f64* h, const i32 ldh,
            f64* dwork, const i32 ldwork, i32* info)
{
    const f64 zero = 0.0;
    const f64 one = 1.0;
    i32 jwork, ldw, k, col;

    *info = 0;

    i32 na_min = (na > 1) ? na : 1;
    i32 nc_min = (nc > 1) ? nc : 1;

    if (na < 0) {
        *info = -1;
    } else if (nb < 0) {
        *info = -2;
    } else if (nc < 0) {
        *info = -3;
    } else if (n < 0) {
        *info = -4;
    } else if (lda < na_min) {
        *info = -6;
    } else if (ldb < na_min) {
        *info = -8;
    } else if (ldc < nc_min) {
        *info = -10;
    } else if (ldh < nc_min) {
        *info = -12;
    } else if (ldwork < (2 * na * nc > 1 ? 2 * na * nc : 1)) {
        *info = -14;
    }

    if (*info != 0) {
        return;
    }

    i32 minval = na;
    if (nb < minval) minval = nb;
    if (nc < minval) minval = nc;
    if (n < minval) minval = n;
    if (minval == 0) {
        return;
    }

    jwork = nc * na;
    ldw = (nc > 1) ? nc : 1;

    SLC_DLACPY("Full", &nc, &na, c, &ldc, &dwork[jwork], &ldw);

    for (k = 0; k < n; k++) {
        SLC_DLACPY("Full", &nc, &na, &dwork[jwork], &ldw, dwork, &ldw);

        col = k * nb;
        SLC_DGEMM("No transpose", "No transpose", &nc, &nb, &na, &one,
                  dwork, &ldw, b, &ldb, &zero, &h[col * ldh], &ldh);

        if (k != n - 1) {
            SLC_DGEMM("No transpose", "No transpose", &nc, &na, &na, &one,
                      dwork, &ldw, a, &lda, &zero, &dwork[jwork], &ldw);
        }
    }
}
