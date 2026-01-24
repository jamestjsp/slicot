/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>

void ab09md(
    const char* dico,
    const char* job,
    const char* equil,
    const char* ordsel,
    i32 n,
    i32 m,
    i32 p,
    i32* nr,
    f64 alpha,
    f64* a,
    i32 lda,
    f64* b,
    i32 ldb,
    f64* c,
    i32 ldc,
    i32* ns,
    f64* hsv,
    f64 tol,
    i32* iwork,
    f64* dwork,
    i32 ldwork,
    i32* iwarn,
    i32* info
)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const f64 C100 = 100.0;

    *info = 0;
    *iwarn = 0;

    bool discr = (*dico == 'D' || *dico == 'd');
    bool fixord = (*ordsel == 'F' || *ordsel == 'f');

    if (!(*dico == 'C' || *dico == 'c') && !discr) {
        *info = -1;
    } else if (!(*job == 'B' || *job == 'b') && !(*job == 'N' || *job == 'n')) {
        *info = -2;
    } else if (!(*equil == 'S' || *equil == 's') && !(*equil == 'N' || *equil == 'n')) {
        *info = -3;
    } else if (!fixord && !(*ordsel == 'A' || *ordsel == 'a')) {
        *info = -4;
    } else if (n < 0) {
        *info = -5;
    } else if (m < 0) {
        *info = -6;
    } else if (p < 0) {
        *info = -7;
    } else if (fixord && (*nr < 0 || *nr > n)) {
        *info = -8;
    } else if ((discr && (alpha < ZERO || alpha > ONE)) ||
               (!discr && alpha > ZERO)) {
        *info = -9;
    } else {
        i32 max1n = (1 > n) ? 1 : n;
        i32 max1p = (1 > p) ? 1 : p;
        if (lda < max1n) {
            *info = -11;
        } else if (ldb < max1n) {
            *info = -13;
        } else if (ldc < max1p) {
            *info = -15;
        } else {
            i32 max_nmp = n;
            if (m > max_nmp) max_nmp = m;
            if (p > max_nmp) max_nmp = p;
            i32 minwork = n * (2 * n + max_nmp + 5) + (n * (n + 1)) / 2;
            if (minwork < 1) minwork = 1;
            if (ldwork < minwork) {
                *info = -21;
            }
        }
    }

    if (*info != 0) {
        i32 neginfo = -(*info);
        SLC_XERBLA("AB09MD", &neginfo);
        return;
    }

    i32 min_nmp = n;
    if (m < min_nmp) min_nmp = m;
    if (p < min_nmp) min_nmp = p;
    if (min_nmp == 0) {
        *nr = 0;
        dwork[0] = ONE;
        return;
    }

    if (*equil == 'S' || *equil == 's') {
        f64 maxred = C100;
        tb01id("All", n, m, p, &maxred, a, lda, b, ldb, c, ldc, dwork, info);
    }

    f64 alpwrk = alpha;
    if (discr) {
        if (alpha == ONE) alpwrk = ONE - sqrt(SLC_DLAMCH("E"));
    } else {
        if (alpha == ZERO) alpwrk = -sqrt(SLC_DLAMCH("E"));
    }

    i32 nn = n * n;
    i32 ku = 0;
    i32 kwr = ku + nn;
    i32 kwi = kwr + n;
    i32 kw = kwi + n;
    i32 lwr = ldwork - kw;

    i32 nu;
    i32 ierr = 0;
    tb01kd(dico, "Unstable", "General", n, m, p, alpwrk, a, lda,
           b, ldb, c, ldc, &nu, &dwork[ku], n, &dwork[kwr],
           &dwork[kwi], &dwork[kw], lwr, &ierr);

    if (ierr != 0) {
        if (ierr != 3) {
            *info = 1;
        } else {
            *info = 2;
        }
        return;
    }

    i32 wrkopt = (i32)dwork[kw] + kw;

    i32 iwarnl = 0;
    *ns = n - nu;
    i32 nra;
    if (fixord) {
        nra = (*nr - nu) > 0 ? (*nr - nu) : 0;
        if (*nr < nu)
            iwarnl = 2;
    } else {
        nra = 0;
    }

    if (*ns == 0) {
        *nr = nu;
        dwork[0] = (f64)wrkopt;
        return;
    }

    i32 nu1 = nu;

    i32 kt = 0;
    i32 kti = kt + nn;
    kw = kti + nn;

    i32 iwarn_ax;
    ab09ax(dico, job, ordsel, *ns, m, p, &nra, &a[nu1 + nu1 * lda], lda,
           &b[nu1], ldb, &c[nu1 * ldc], ldc, hsv, &dwork[kt], n,
           &dwork[kti], n, tol, iwork, &dwork[kw], ldwork - kw,
           &iwarn_ax, &ierr);
    *iwarn = (iwarn_ax > iwarnl) ? iwarn_ax : iwarnl;

    if (ierr != 0) {
        *info = ierr + 1;
        return;
    }

    *nr = nra + nu;

    i32 wrkopt2 = (i32)dwork[kw] + kw;
    dwork[0] = (f64)((wrkopt > wrkopt2) ? wrkopt : wrkopt2);
}
