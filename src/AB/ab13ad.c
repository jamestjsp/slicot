/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <stdbool.h>

f64 ab13ad(
    const char* dico,
    const char* equil,
    const i32 n,
    const i32 m,
    const i32 p,
    const f64 alpha,
    f64* a,
    const i32 lda,
    f64* b,
    const i32 ldb,
    f64* c,
    const i32 ldc,
    i32* ns,
    f64* hsv,
    f64* dwork,
    const i32 ldwork,
    i32* info
)
{
    const f64 C100 = 100.0;
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    *info = 0;
    bool discr = (*dico == 'D' || *dico == 'd');

    if (!discr && !(*dico == 'C' || *dico == 'c')) {
        *info = -1;
    } else if (!(*equil == 'S' || *equil == 's') &&
               !(*equil == 'N' || *equil == 'n')) {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    } else if (m < 0) {
        *info = -4;
    } else if (p < 0) {
        *info = -5;
    } else if ((discr && (alpha < ZERO || alpha > ONE)) ||
               (!discr && alpha > ZERO)) {
        *info = -6;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -8;
    } else if (ldb < (n > 1 ? n : 1)) {
        *info = -10;
    } else if (ldc < (p > 1 ? p : 1)) {
        *info = -12;
    } else {
        i32 max_nmp = n;
        if (m > max_nmp) max_nmp = m;
        if (p > max_nmp) max_nmp = p;
        i32 minwork = n * (max_nmp + 5) + (n * (n + 1)) / 2;
        if (minwork < 1) minwork = 1;
        if (ldwork < minwork) {
            *info = -16;
        }
    }

    if (*info != 0) {
        i32 neginfo = -(*info);
        SLC_XERBLA("AB13AD", &neginfo);
        return ZERO;
    }

    i32 mn = n;
    if (m < mn) mn = m;
    if (p < mn) mn = p;
    if (mn == 0) {
        *ns = 0;
        dwork[0] = ONE;
        return ZERO;
    }

    if (*equil == 'S' || *equil == 's') {
        f64 maxred = C100;
        i32 tb01id_info = 0;
        tb01id("All", n, m, p, &maxred, a, lda, b, ldb, c, ldc,
               dwork, &tb01id_info);
    }

    f64 alpwrk = alpha;
    f64 eps_sqrt = sqrt(SLC_DLAMCH("E"));
    if (discr) {
        if (alpha == ONE) alpwrk = ONE - eps_sqrt;
    } else {
        if (alpha == ZERO) alpwrk = -eps_sqrt;
    }

    i32 kt = 0;
    i32 kw1 = n * n;
    i32 kw2 = kw1 + n;
    i32 kw = kw2 + n;

    i32 ierr = 0;
    tb01kd(dico, "Stable", "General", n, m, p, alpwrk, a, lda,
           b, ldb, c, ldc, ns, &dwork[kt], n, &dwork[kw1],
           &dwork[kw2], &dwork[kw], ldwork - kw, &ierr);

    if (ierr != 0) {
        if (ierr != 3) {
            *info = 1;
        } else {
            *info = 2;
        }
        return ZERO;
    }

    f64 wrkopt = dwork[kw] + (f64)(kw);

    f64 result;
    if (*ns == 0) {
        result = ZERO;
    } else {
        result = ab13ax(dico, *ns, m, p, a, lda, b, ldb, c, ldc, hsv,
                        dwork, ldwork, &ierr);

        if (ierr != 0) {
            *info = ierr + 2;
            return ZERO;
        }

        f64 tmp = dwork[0];
        if (tmp > wrkopt) wrkopt = tmp;
    }

    dwork[0] = wrkopt;

    return result;
}
