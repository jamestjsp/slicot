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

void tb01ux(const char* compz, i32 n, i32 m, i32 p,
            f64* a, i32 lda, f64* b, i32 ldb, f64* c, i32 ldc,
            f64* z, i32 ldz, i32* nobsv, i32* nlblck, i32* ctau,
            f64 tol, i32* iwork, f64* dwork, i32* info) {

    const f64 ONE = 1.0;

    bool ilz;
    i32 lba, ldwork;
    f64 dum[1];

    *info = 0;
    ilz = lsame(*compz, 'I');

    if (!ilz && !lsame(*compz, 'N')) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (m < 0) {
        *info = -3;
    } else if (p < 0) {
        *info = -4;
    } else if (lda < imax(1, n)) {
        *info = -6;
    } else if (ldb < 1 || (imax(m, p) > 0 && ldb < n)) {
        *info = -8;
    } else if (ldc < 1 || (ldc < imax(m, p) && n > 0)) {
        *info = -10;
    } else if (ldz < 1 || (ilz && ldz < n)) {
        *info = -12;
    } else if (tol >= ONE) {
        *info = -16;
    }

    if (*info != 0) {
        return;
    }

    *nobsv = 0;
    *nlblck = 0;

    ab07md('Z', n, m, p, a, lda, b, ldb, c, ldc, dum, 1);

    ldwork = imax(1, imax(n, imax(3 * p, m)));
    i32 info_sub;
    tb01ud(compz, n, p, m, a, lda, b, ldb, c, ldc,
           nobsv, nlblck, ctau, z, ldz, dwork, tol,
           iwork, &dwork[n], ldwork, &info_sub);
    (void)info_sub;

    if (*nlblck > 1) {
        lba = ctau[0] + ctau[1] - 1;
    } else if (*nlblck == 1) {
        lba = ctau[0] - 1;
    } else {
        lba = 0;
    }

    lba = imax(lba, n - *nobsv - 1);
    tb01xd("Z", n, p, m, lba, imax(0, n - 1), a, lda, b, ldb,
           c, ldc, dum, 1, &info_sub);

    if (ilz) {
        ma02bd('R', n, n, z, ldz);
    }

    dwork[0] = dwork[n];
}
