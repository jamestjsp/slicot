/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>

void ab09nd(const char* dico, const char* job, const char* equil,
            const char* ordsel, i32 n, i32 m, i32 p, i32* nr, f64 alpha,
            f64* a, i32 lda, f64* b, i32 ldb, f64* c, i32 ldc,
            f64* d, i32 ldd, i32* ns, f64* hsv, f64 tol1, f64 tol2,
            i32* iwork, f64* dwork, i32 ldwork, i32* iwarn, i32* info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const f64 C100 = 100.0;

    bool discr = (dico[0] == 'D' || dico[0] == 'd');
    bool conti = (dico[0] == 'C' || dico[0] == 'c');
    bool fixord = (ordsel[0] == 'F' || ordsel[0] == 'f');
    bool autosel = (ordsel[0] == 'A' || ordsel[0] == 'a');
    bool balj = (job[0] == 'B' || job[0] == 'b');
    bool balnot = (job[0] == 'N' || job[0] == 'n');
    bool doeq = (equil[0] == 'S' || equil[0] == 's');
    bool noeq = (equil[0] == 'N' || equil[0] == 'n');

    *info = 0;
    *iwarn = 0;

    i32 max1n = (1 > n) ? 1 : n;
    i32 max1p = (1 > p) ? 1 : p;
    i32 maxnmp = n;
    if (m > maxnmp) maxnmp = m;
    if (p > maxnmp) maxnmp = p;

    i32 minwrk = n * (2*n + maxnmp + 5) + (n * (n + 1)) / 2;
    if (minwrk < 1) minwrk = 1;

    if (!conti && !discr) {
        *info = -1;
    } else if (!balj && !balnot) {
        *info = -2;
    } else if (!doeq && !noeq) {
        *info = -3;
    } else if (!fixord && !autosel) {
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
    } else if (lda < max1n) {
        *info = -11;
    } else if (ldb < max1n) {
        *info = -13;
    } else if (ldc < max1p) {
        *info = -15;
    } else if (ldd < max1p) {
        *info = -17;
    } else if (tol2 > ZERO && tol2 > tol1) {
        *info = -21;
    } else if (ldwork < minwrk) {
        *info = -24;
    }

    if (*info != 0) {
        i32 neginfo = -(*info);
        SLC_XERBLA("AB09ND", &neginfo);
        return;
    }

    i32 minval = n;
    if (m < minval) minval = m;
    if (p < minval) minval = p;

    if (minval == 0) {
        *nr = 0;
        iwork[0] = 0;
        dwork[0] = ONE;
        return;
    }

    if (doeq) {
        f64 maxred = C100;
        tb01id("All", n, m, p, &maxred, a, lda, b, ldb, c, ldc, dwork, info);
    }

    f64 alpwrk = alpha;
    if (discr) {
        if (alpha == ONE) {
            alpwrk = ONE - sqrt(SLC_DLAMCH("E"));
        }
    } else {
        if (alpha == ZERO) {
            alpwrk = -sqrt(SLC_DLAMCH("E"));
        }
    }

    i32 nn = n * n;
    i32 ku = 0;
    i32 kwr = ku + nn;
    i32 kwi = kwr + n;
    i32 kw = kwi + n;
    i32 lwr = ldwork - kw;

    i32 nu;
    i32 ierr;
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
        nra = (*nr - nu > 0) ? (*nr - nu) : 0;
        if (*nr < nu) {
            iwarnl = 2;
        }
    } else {
        nra = 0;
    }

    if (*ns == 0) {
        *nr = nu;
        iwork[0] = 0;
        dwork[0] = (f64)wrkopt;
        return;
    }

    i32 nu1 = nu;

    i32 kt = 0;
    i32 kti = kt + nn;
    kw = kti + nn;

    ab09bx(dico, job, ordsel, *ns, m, p, &nra,
           &a[nu1 + nu1*lda], lda,
           &b[nu1], ldb,
           &c[nu1*ldc], ldc,
           d, ldd, hsv,
           &dwork[kt], n, &dwork[kti], n,
           tol1, tol2, iwork, &dwork[kw], ldwork - kw,
           iwarn, &ierr);

    if (*iwarn < iwarnl) {
        *iwarn = iwarnl;
    }

    if (ierr != 0) {
        *info = ierr + 1;
        return;
    }

    *nr = nra + nu;

    i32 tmp = (i32)dwork[kw] + kw;
    if (tmp > wrkopt) wrkopt = tmp;
    dwork[0] = (f64)wrkopt;
}
