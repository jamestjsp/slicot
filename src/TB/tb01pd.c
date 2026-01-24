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

static inline i32 min_i32(i32 a, i32 b) {
    return a < b ? a : b;
}

void tb01pd(const char* job, const char* equil, i32 n, i32 m, i32 p,
            f64* a, i32 lda, f64* b, i32 ldb, f64* c, i32 ldc,
            i32* nr, f64 tol, i32* iwork, f64* dwork, i32 ldwork, i32* info) {

    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const i32 LDIZ = 1;

    char job_upper = (char)toupper((unsigned char)job[0]);
    char equil_upper = (char)toupper((unsigned char)equil[0]);

    bool lnjobc = (job_upper != 'C');
    bool lnjobo = (job_upper != 'O');
    bool lequil = (equil_upper == 'S');

    i32 maxmp = max_i32(m, p);

    *info = 0;

    if (lnjobc && lnjobo && job_upper != 'M') {
        *info = -1;
    } else if (!lequil && equil_upper != 'N') {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    } else if (m < 0) {
        *info = -4;
    } else if (p < 0) {
        *info = -5;
    } else if (lda < max_i32(1, n)) {
        *info = -7;
    } else if (ldb < max_i32(1, n)) {
        *info = -9;
    } else if (ldc < 1 || (n > 0 && ldc < maxmp)) {
        *info = -11;
    } else if (ldwork < max_i32(1, n + max_i32(n, 3 * maxmp))) {
        *info = -16;
    }

    if (*info != 0) {
        return;
    }

    if (n == 0 || (lnjobc && min_i32(n, p) == 0) ||
                  (lnjobo && min_i32(n, m) == 0)) {
        *nr = 0;

        for (i32 i = 0; i < n; i++) {
            iwork[i] = 0;
        }

        dwork[0] = ONE;
        return;
    }

    i32 iz = 0;
    i32 itau = 0;
    i32 jwork = itau + n;
    i32 wrkopt;
    i32 ncont;
    i32 indcon;
    i32 info_tmp = 0;

    if (lequil) {
        f64 maxred = ZERO;
        tb01id("A", n, m, p, &maxred, a, lda, b, ldb, c, ldc, dwork, &info_tmp);
        wrkopt = n;
    } else {
        wrkopt = 1;
    }

    if (lnjobo) {
        i32 ldwork_remaining = ldwork - jwork;
        tb01ud("N", n, m, p, a, lda, b, ldb, c, ldc,
               &ncont, &indcon, iwork, &dwork[iz], LDIZ, &dwork[itau], tol,
               &iwork[n], &dwork[jwork], ldwork_remaining, &info_tmp);

        wrkopt = max_i32(wrkopt, (i32)dwork[jwork] + jwork);
    } else {
        ncont = n;
        indcon = 0;
    }

    if (lnjobc) {
        f64 dummy;
        (void)ab07md('Z', ncont, m, p, a, lda, b, ldb, c, ldc, &dummy, 1);

        i32 ldwork_remaining = ldwork - jwork;
        tb01ud("N", ncont, p, m, a, lda, b, ldb, c, ldc,
               nr, &indcon, iwork, &dwork[iz], LDIZ, &dwork[itau], tol,
               &iwork[n], &dwork[jwork], ldwork_remaining, &info_tmp);

        wrkopt = max_i32(wrkopt, (i32)dwork[jwork] + jwork);

        i32 kl;
        if (indcon > 0) {
            kl = iwork[0] - 1;
            if (indcon >= 2) {
                kl = kl + iwork[1];
            }
        } else {
            kl = 0;
        }

        i32 ku = max_i32(0, *nr - 1);
        f64 d_dummy;
        tb01xd("Z", *nr, p, m, kl, ku, a, lda, b, ldb, c, ldc, &d_dummy, 1, &info_tmp);
    } else {
        *nr = ncont;
    }

    for (i32 i = indcon; i < n; i++) {
        iwork[i] = 0;
    }

    dwork[0] = (f64)wrkopt;
}
