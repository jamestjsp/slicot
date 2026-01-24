/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>

void ab09bd(
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
    f64* d,
    i32 ldd,
    f64* hsv,
    f64 tol1,
    f64 tol2,
    i32* iwork,
    f64* dwork,
    i32 ldwork,
    i32* iwarn,
    i32* info)
{
    const f64 zero = 0.0;
    const f64 one = 1.0;
    const f64 c100 = 100.0;

    bool fixord;
    i32 ierr, ki, kr, kt, kti, kw, nn;
    f64 maxred, wrkopt;

    *info = 0;
    *iwarn = 0;
    fixord = (ordsel[0] == 'F' || ordsel[0] == 'f');

    /* Parameter validation */
    if (dico[0] != 'C' && dico[0] != 'c' && dico[0] != 'D' && dico[0] != 'd') {
        *info = -1;
    } else if (job[0] != 'B' && job[0] != 'b' && job[0] != 'N' && job[0] != 'n') {
        *info = -2;
    } else if (equil[0] != 'S' && equil[0] != 's' && equil[0] != 'N' && equil[0] != 'n') {
        *info = -3;
    } else if (!fixord && ordsel[0] != 'A' && ordsel[0] != 'a') {
        *info = -4;
    } else if (n < 0) {
        *info = -5;
    } else if (m < 0) {
        *info = -6;
    } else if (p < 0) {
        *info = -7;
    } else if (fixord && (*nr < 0 || *nr > n)) {
        *info = -8;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -10;
    } else if (ldb < (n > 1 ? n : 1)) {
        *info = -12;
    } else if (ldc < (p > 1 ? p : 1)) {
        *info = -14;
    } else if (ldd < (p > 1 ? p : 1)) {
        *info = -16;
    } else if (tol2 > zero && tol2 > tol1) {
        *info = -19;
    } else {
        /* Compute minimum workspace */
        i32 maxnmp = n;
        if (m > maxnmp) maxnmp = m;
        if (p > maxnmp) maxnmp = p;
        i32 minwork = n * (2 * n + maxnmp + 5) + (n * (n + 1)) / 2;
        if (minwork < 1) minwork = 1;
        if (ldwork < minwork) {
            *info = -22;
        }
    }

    if (*info != 0) {
        return;
    }

    /* Quick return if possible */
    if (n == 0 || m == 0 || p == 0) {
        *nr = 0;
        iwork[0] = 0;
        dwork[0] = one;
        return;
    }

    /* Allocate working storage */
    nn = n * n;
    kt = 0;
    kr = kt + nn;
    ki = kr + n;
    kw = ki + n;

    if (equil[0] == 'S' || equil[0] == 's') {
        /* Scale simultaneously the matrices A, B and C:
           A <- inv(D)*A*D, B <- inv(D)*B and C <- C*D, where D is diagonal */
        maxred = c100;
        tb01id("All", n, m, p, &maxred, a, lda, b, ldb, c, ldc, dwork, info);
    }

    /* Reduce A to real Schur form using orthogonal similarity transformation
       A <- T'*A*T and apply the transformation to B and C:
       B <- T'*B and C <- C*T */
    tb01wd(n, m, p, a, lda, b, ldb, c, ldc, &dwork[kt], n,
           &dwork[kr], &dwork[ki], &dwork[kw], ldwork - kw, &ierr);
    if (ierr != 0) {
        *info = 1;
        return;
    }

    wrkopt = dwork[kw] + (f64)kw;

    kti = kt + nn;
    kw = kti + nn;

    ab09bx(dico, job, ordsel, n, m, p, nr, a, lda, b, ldb,
           c, ldc, d, ldd, hsv, &dwork[kt], n, &dwork[kti], n,
           tol1, tol2, iwork, &dwork[kw], ldwork - kw, iwarn, &ierr);

    if (ierr != 0) {
        *info = ierr + 1;
        return;
    }

    f64 wrkopt2 = dwork[kw] + (f64)kw;
    dwork[0] = wrkopt > wrkopt2 ? wrkopt : wrkopt2;
}
