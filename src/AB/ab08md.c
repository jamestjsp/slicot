/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>

static inline bool lsame(char ca, char cb) {
    if (ca >= 'a' && ca <= 'z') ca -= 32;
    if (cb >= 'a' && cb <= 'z') cb -= 32;
    return ca == cb;
}

static inline i32 max_i32(i32 a, i32 b) {
    return a > b ? a : b;
}

static inline i32 min_i32(i32 a, i32 b) {
    return a < b ? a : b;
}

static inline i32 max3_i32(i32 a, i32 b, i32 c) {
    return max_i32(a, max_i32(b, c));
}

void ab08md(const char* equil, i32 n, i32 m, i32 p, f64* a, i32 lda,
            f64* b, i32 ldb, f64* c, i32 ldc, f64* d, i32 ldd,
            i32* rank, f64 tol, i32* iwork, f64* dwork, i32 ldwork,
            i32* info) {

    const f64 ZERO = 0.0, ONE = 1.0;

    i32 np = n + p;
    i32 nm = n + m;
    bool lequil = lsame(*equil, 'S');
    bool lquery = (ldwork == -1);
    i32 wrkopt = np * nm;

    *info = 0;

    if (!lequil && !lsame(*equil, 'N')) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (m < 0) {
        *info = -3;
    } else if (p < 0) {
        *info = -4;
    } else if (lda < max_i32(1, n)) {
        *info = -6;
    } else if (ldb < max_i32(1, n)) {
        *info = -8;
    } else if (ldc < max_i32(1, p)) {
        *info = -10;
    } else if (ldd < max_i32(1, p)) {
        *info = -12;
    } else {
        i32 kw = wrkopt + max3_i32(min_i32(p, m) + max_i32(3*m - 1, n), 1,
                                    min_i32(p, n) + max3_i32(3*p - 1, np, nm));
        if (lquery) {
            i32 ro = p, sigma = 0, ninfz = 0;
            i32 mu, nu, nkrol;
            ab08nx(n, m, p, &ro, &sigma, ZERO, dwork, max_i32(1, np),
                   &ninfz, iwork, iwork, &mu, &nu, &nkrol, tol, iwork,
                   dwork, -1, info);
            wrkopt = max_i32(kw, wrkopt + (i32)dwork[0]);
        } else if (ldwork < kw) {
            *info = -17;
        }
    }

    if (*info != 0) {
        return;
    } else if (lquery) {
        dwork[0] = (f64)wrkopt;
        return;
    }

    if (min_i32(m, p) == 0) {
        *rank = 0;
        dwork[0] = ONE;
        return;
    }

    for (i32 i = 0; i < 2*n + 1; i++) {
        iwork[i] = 0;
    }

    sl_int np_int = np, nm_int = nm, n_int = n, m_int = m, p_int = p;
    sl_int lda_int = lda, ldb_int = ldb, ldc_int = ldc, ldd_int = ldd;
    sl_int ldnp = np;

    SLC_DLACPY("Full", &n_int, &m_int, b, &ldb_int, dwork, &ldnp);
    SLC_DLACPY("Full", &p_int, &m_int, d, &ldd_int, &dwork[n], &ldnp);
    SLC_DLACPY("Full", &n_int, &n_int, a, &lda_int, &dwork[np * m], &ldnp);
    SLC_DLACPY("Full", &p_int, &n_int, c, &ldc_int, &dwork[np * m + n], &ldnp);

    i32 kw = wrkopt;
    f64 maxred, svlmax, thresh, toler;

    if (lequil) {
        maxred = ZERO;
        tb01id("A", n, m, p, &maxred, &dwork[np * m], np, dwork, np,
               &dwork[np * m + n], np, &dwork[kw], info);
        wrkopt = wrkopt + n;
    }

    thresh = sqrt((f64)(np * nm)) * SLC_DLAMCH("Precision");
    toler = tol;
    if (toler < thresh) toler = thresh;

    svlmax = SLC_DLANGE("Frobenius", &np_int, &nm_int, dwork, &ldnp, &dwork[kw]);

    i32 ro = p;
    i32 sigma = 0;
    i32 ninfz = 0;
    i32 mu, nu, nkrol;

    ab08nx(n, m, p, &ro, &sigma, svlmax, dwork, np, &ninfz, iwork,
           &iwork[n], &mu, &nu, &nkrol, toler, &iwork[2*n + 1],
           &dwork[kw], ldwork - kw + 1, info);

    *rank = mu;

    dwork[0] = (f64)max_i32(wrkopt, (i32)dwork[kw] + kw - 1);
}
