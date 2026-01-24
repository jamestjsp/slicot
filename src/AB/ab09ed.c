/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>

void ab09ed(const char* dico, const char* equil, const char* ordsel,
            i32 n, i32 m, i32 p, i32* nr, f64 alpha,
            f64* a, i32 lda, f64* b, i32 ldb, f64* c, i32 ldc,
            f64* d, i32 ldd, i32* ns, f64* hsv, f64 tol1, f64 tol2,
            i32* iwork, f64* dwork, i32 ldwork, i32* iwarn, i32* info)
{
    const f64 C100 = 100.0;
    const f64 ONE = 1.0;
    const f64 ZERO = 0.0;

    bool discr = (dico[0] == 'D' || dico[0] == 'd');
    bool conti = (dico[0] == 'C' || dico[0] == 'c');
    bool scale = (equil[0] == 'S' || equil[0] == 's');
    bool noscale = (equil[0] == 'N' || equil[0] == 'n');
    bool fixord = (ordsel[0] == 'F' || ordsel[0] == 'f');
    bool autosel = (ordsel[0] == 'A' || ordsel[0] == 'a');

    *info = 0;
    *iwarn = 0;

    i32 max1n = (1 > n) ? 1 : n;
    i32 max1p = (1 > p) ? 1 : p;
    i32 maxnmp = n;
    if (m > maxnmp) maxnmp = m;
    if (p > maxnmp) maxnmp = p;
    i32 minnm = n < m ? n : m;

    i32 ldw1 = n * (2 * n + maxnmp + 5) + (n * (n + 1)) / 2;
    i32 ldw2 = n * (m + p + 2) + 2 * m * p + minnm;
    i32 tmp1 = 3 * m + 1;
    i32 tmp2 = minnm + p;
    if (tmp2 > tmp1) ldw2 += tmp2;
    else ldw2 += tmp1;
    i32 minwrk = (ldw1 > ldw2) ? ldw1 : ldw2;
    if (minwrk < 1) minwrk = 1;

    if (!conti && !discr) {
        *info = -1;
    } else if (!scale && !noscale) {
        *info = -2;
    } else if (!fixord && !autosel) {
        *info = -3;
    } else if (n < 0) {
        *info = -4;
    } else if (m < 0) {
        *info = -5;
    } else if (p < 0) {
        *info = -6;
    } else if (fixord && (*nr < 0 || *nr > n)) {
        *info = -7;
    } else if ((discr && (alpha < ZERO || alpha > ONE)) ||
               (!discr && alpha > ZERO)) {
        *info = -8;
    } else if (lda < max1n) {
        *info = -10;
    } else if (ldb < max1n) {
        *info = -12;
    } else if (ldc < max1p) {
        *info = -14;
    } else if (ldd < max1p) {
        *info = -16;
    } else if (tol2 > ZERO && tol2 > tol1) {
        *info = -20;
    } else if (ldwork < minwrk) {
        *info = -23;
    }

    if (*info != 0) {
        i32 neginfo = -(*info);
        SLC_XERBLA("AB09ED", &neginfo);
        return;
    }

    i32 min_nmp = n;
    if (m < min_nmp) min_nmp = m;
    if (p < min_nmp) min_nmp = p;
    if (min_nmp == 0) {
        *nr = 0;
        *ns = 0;
        iwork[0] = 0;
        dwork[0] = ONE;
        return;
    }

    if (scale) {
        f64 maxred = C100;
        i32 tb01id_info;
        tb01id("All", n, m, p, &maxred, a, lda, b, ldb, c, ldc,
               dwork, &tb01id_info);
    }

    f64 alpwrk = alpha;
    if (discr) {
        if (alpha == ONE) alpwrk = ONE - sqrt(SLC_DLAMCH("E"));
    } else {
        if (alpha == ZERO) alpwrk = -sqrt(SLC_DLAMCH("E"));
    }

    i32 ku = 0;
    i32 kl = ku + n * n;
    i32 ki = kl + n;
    i32 kw = ki + n;

    i32 nu;
    i32 ierr;
    tb01kd(dico, "Unstable", "General", n, m, p, alpwrk, a, lda,
           b, ldb, c, ldc, &nu, &dwork[ku], n, &dwork[kl],
           &dwork[ki], &dwork[kw], ldwork - kw, &ierr);

    if (ierr != 0) {
        if (ierr != 3) {
            *info = 1;
        } else {
            *info = 2;
        }
        return;
    }

    f64 wrkopt = dwork[kw] + (f64)(kw);

    i32 iwarnl = 0;
    *ns = n - nu;
    i32 nra;
    if (fixord) {
        nra = (*nr - nu > 0) ? *nr - nu : 0;
        if (*nr < nu) {
            iwarnl = 2;
        }
    } else {
        nra = 0;
    }

    if (*ns == 0) {
        *nr = nu;
        iwork[0] = 0;
        dwork[0] = wrkopt;
        return;
    }

    i32 nu1 = nu;
    ab09cx(dico, ordsel, *ns, m, p, &nra, &a[nu1 + nu1 * lda], lda,
           &b[nu1], ldb, &c[nu1 * ldc], ldc, d, ldd, hsv, tol1,
           tol2, iwork, dwork, ldwork, iwarn, &ierr);

    *iwarn = (*iwarn > iwarnl) ? *iwarn : iwarnl;
    if (ierr != 0) {
        *info = ierr + 2;
        return;
    }

    *nr = nra + nu;

    dwork[0] = (wrkopt > dwork[0]) ? wrkopt : dwork[0];
}
