/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"

void ab09ad(
    const char* dico,
    const char* job,
    const char* equil,
    const char* ordsel,
    i32 n,
    i32 m,
    i32 p,
    i32* nr,
    f64* a,
    i32 lda,
    f64* b,
    i32 ldb,
    f64* c,
    i32 ldc,
    f64* hsv,
    f64 tol,
    i32* iwork,
    f64* dwork,
    i32 ldwork,
    i32* iwarn,
    i32* info
)
{
    const f64 ONE = 1.0;
    const f64 C100 = 100.0;

    *info = 0;
    *iwarn = 0;

    bool fixord = (ordsel[0] == 'F' || ordsel[0] == 'f');

    if (!(dico[0] == 'C' || dico[0] == 'c' || dico[0] == 'D' || dico[0] == 'd')) {
        *info = -1;
    } else if (!(job[0] == 'B' || job[0] == 'b' || job[0] == 'N' || job[0] == 'n')) {
        *info = -2;
    } else if (!(equil[0] == 'S' || equil[0] == 's' || equil[0] == 'N' || equil[0] == 'n')) {
        *info = -3;
    } else if (!fixord && !(ordsel[0] == 'A' || ordsel[0] == 'a')) {
        *info = -4;
    } else if (n < 0) {
        *info = -5;
    } else if (m < 0) {
        *info = -6;
    } else if (p < 0) {
        *info = -7;
    } else if (fixord && (*nr < 0 || *nr > n)) {
        *info = -8;
    } else {
        i32 max1n = (1 > n) ? 1 : n;
        i32 max1p = (1 > p) ? 1 : p;
        if (lda < max1n) {
            *info = -10;
        } else if (ldb < max1n) {
            *info = -12;
        } else if (ldc < max1p) {
            *info = -14;
        } else {
            i32 max_nmp = n;
            if (m > max_nmp) max_nmp = m;
            if (p > max_nmp) max_nmp = p;
            i32 minwork = n * (2 * n + max_nmp + 5) + (n * (n + 1)) / 2;
            if (minwork < 1) minwork = 1;
            if (ldwork < minwork) {
                *info = -19;
            }
        }
    }

    if (*info != 0) {
        i32 neginfo = -(*info);
        SLC_XERBLA("AB09AD", &neginfo);
        return;
    }

    i32 min_nmp = n;
    if (m < min_nmp) min_nmp = m;
    if (p < min_nmp) min_nmp = p;
    if (min_nmp == 0 || (fixord && *nr == 0)) {
        *nr = 0;
        dwork[0] = ONE;
        return;
    }

    i32 nn = n * n;
    i32 kt = 0;
    i32 kr = kt + nn;
    i32 ki = kr + n;
    i32 kw = ki + n;

    if (equil[0] == 'S' || equil[0] == 's') {
        f64 maxred = C100;
        tb01id("A", n, m, p, &maxred, a, lda, b, ldb, c, ldc, dwork, info);
    }

    i32 ierr = 0;
    tb01wd(n, m, p, a, lda, b, ldb, c, ldc, &dwork[kt], n,
           &dwork[kr], &dwork[ki], &dwork[kw], ldwork - kw, &ierr);
    if (ierr != 0) {
        *info = 1;
        return;
    }

    f64 wrkopt = dwork[kw] + (f64)(kw);
    i32 kti = kt + nn;
    kw = kti + nn;

    ab09ax(dico, job, ordsel, n, m, p, nr, a, lda, b, ldb, c, ldc, hsv,
           &dwork[kt], n, &dwork[kti], n, tol, iwork, &dwork[kw], ldwork - kw,
           iwarn, &ierr);

    if (ierr != 0) {
        *info = ierr + 1;
        return;
    }

    f64 wrkopt2 = dwork[kw] + (f64)(kw);
    if (wrkopt2 > wrkopt) wrkopt = wrkopt2;
    dwork[0] = wrkopt;
}
