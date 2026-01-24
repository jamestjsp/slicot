/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>

void ab09gd(
    const char* dico,
    const char* jobcf,
    const char* fact,
    const char* jobmr,
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
    f64* d,
    i32 ldd,
    i32* nq,
    f64* hsv,
    f64 tol1,
    f64 tol2,
    f64 tol3,
    i32* iwork,
    f64* dwork,
    i32 ldwork,
    i32* iwarn,
    i32* info)
{
    const f64 zero = 0.0;
    const f64 one = 1.0;
    const f64 c100 = 100.0;

    bool discr, fixord, left, stabd;
    i32 ierr, iwarnk, kb, kbr, kbt, kc, kcr, kd, kdr, kdt, kt, kti, kw;
    i32 lw1, lw2, lw3, lw4, lwr, maxmp, mp, ndr, nminr, pm;
    f64 maxred, wrkopt;
    f64 alpha_arr[2];

    *info = 0;
    *iwarn = 0;
    discr = (dico[0] == 'D' || dico[0] == 'd');
    fixord = (ordsel[0] == 'F' || ordsel[0] == 'f');
    left = (jobcf[0] == 'L' || jobcf[0] == 'l');
    stabd = (fact[0] == 'S' || fact[0] == 's');
    maxmp = m > p ? m : p;

    lwr = 2 * n * n + n * ((n > m + p ? n : m + p) + 5) + (n * (n + 1)) / 2;
    lw1 = n * (2 * maxmp + p) + maxmp * (maxmp + p);
    lw2 = lw1 + (n * p + (n * (n + 5) > p * (p + 2) ? n * (n + 5) : p * (p + 2)) > 4 * p ?
                 (n * p + (n * (n + 5) > p * (p + 2) ? n * (n + 5) : p * (p + 2))) :
                 (n * p + 4 * p > 4 * m ? n * p + 4 * p : 4 * m));
    if (lw2 < lw1 + lwr) lw2 = lw1 + lwr;
    lw1 = lw1 + (n * p + (n * (n + 5) > 5 * p ? n * (n + 5) : 5 * p) > 4 * m ?
                 (n * p + (n * (n + 5) > 5 * p ? n * (n + 5) : 5 * p)) : 4 * m);
    if (lw1 < n * (2 * maxmp + p) + maxmp * (maxmp + p) + lwr)
        lw1 = n * (2 * maxmp + p) + maxmp * (maxmp + p) + lwr;
    lw3 = (n + m) * (m + p) + (5 * m > 4 * p ? 5 * m : 4 * p);
    if (lw3 < (n + m) * (m + p) + lwr) lw3 = (n + m) * (m + p) + lwr;
    lw4 = (n + m) * (m + p) + (m * (m + 2) > 4 * m ? m * (m + 2) : 4 * m);
    if (lw4 < (n + m) * (m + p) + 4 * p) lw4 = (n + m) * (m + p) + 4 * p;
    if (lw4 < (n + m) * (m + p) + lwr) lw4 = (n + m) * (m + p) + lwr;

    /* Parameter validation */
    if (dico[0] != 'C' && dico[0] != 'c' && dico[0] != 'D' && dico[0] != 'd') {
        *info = -1;
    } else if (jobcf[0] != 'L' && jobcf[0] != 'l' && jobcf[0] != 'R' && jobcf[0] != 'r') {
        *info = -2;
    } else if (fact[0] != 'S' && fact[0] != 's' && fact[0] != 'I' && fact[0] != 'i') {
        *info = -3;
    } else if (jobmr[0] != 'B' && jobmr[0] != 'b' && jobmr[0] != 'N' && jobmr[0] != 'n') {
        *info = -4;
    } else if (equil[0] != 'S' && equil[0] != 's' && equil[0] != 'N' && equil[0] != 'n') {
        *info = -5;
    } else if (!fixord && ordsel[0] != 'A' && ordsel[0] != 'a') {
        *info = -6;
    } else if (stabd && ((!discr && alpha >= zero) ||
                         (discr && (alpha < zero || alpha >= one)))) {
        *info = -7;
    } else if (n < 0) {
        *info = -8;
    } else if (m < 0) {
        *info = -9;
    } else if (p < 0) {
        *info = -10;
    } else if (fixord && (*nr < 0 || *nr > n)) {
        *info = -11;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -13;
    } else if (ldb < (n > 1 ? n : 1)) {
        *info = -15;
    } else if (ldc < (p > 1 ? p : 1)) {
        *info = -17;
    } else if (ldd < (p > 1 ? p : 1)) {
        *info = -19;
    } else if (tol2 > zero && tol2 > tol1) {
        *info = -23;
    } else if (ldwork < 1 ||
               (stabd && left && ldwork < lw1) ||
               (!stabd && left && ldwork < lw2) ||
               (stabd && !left && ldwork < lw3) ||
               (!stabd && !left && ldwork < lw4)) {
        *info = -27;
    }

    if (*info != 0) {
        return;
    }

    /* Quick return if possible */
    i32 minval = n < m ? n : m;
    if (minval > p) minval = p;
    if (minval == 0) {
        *nr = 0;
        *nq = 0;
        iwork[0] = 0;
        dwork[0] = one;
        return;
    }

    if (equil[0] == 'S' || equil[0] == 's') {
        /* Scale simultaneously A, B, C:
           A <- inv(D)*A*D, B <- inv(D)*B, C <- C*D */
        maxred = c100;
        tb01id("A", n, m, p, &maxred, a, lda, b, ldb, c, ldc, dwork, info);
    }

    /* Perform coprime factor model reduction */
    kd = 0;
    i32 lwr_work;

    if (left) {
        /* Compute a LCF G = R^{-1}*Q */
        mp = m + p;
        kdr = kd + maxmp * maxmp;
        kc = kdr + maxmp * p;
        kb = kc + maxmp * n;
        kbr = kb + n * maxmp;
        kw = kbr + n * p;
        lwr_work = ldwork - kw;

        SLC_DLACPY("Full", &n, &m, b, &ldb, &dwork[kb], &n);
        SLC_DLACPY("Full", &p, &n, c, &ldc, &dwork[kc], &maxmp);
        SLC_DLACPY("Full", &p, &m, d, &ldd, &dwork[kd], &maxmp);

        if (stabd) {
            /* Compute LCF with prescribed stability degree */
            alpha_arr[0] = alpha;
            alpha_arr[1] = alpha;
            sb08ed(dico, n, m, p, alpha_arr, a, lda, &dwork[kb], n,
                   &dwork[kc], maxmp, &dwork[kd], maxmp, nq, &ndr,
                   &dwork[kbr], n, &dwork[kdr], maxmp, tol3,
                   &dwork[kw], lwr_work, iwarn, info);
        } else {
            /* Compute LCF with inner denominator */
            sb08cd(dico, n, m, p, a, lda, &dwork[kb], n,
                   &dwork[kc], maxmp, &dwork[kd], maxmp, nq, &ndr,
                   &dwork[kbr], n, &dwork[kdr], maxmp, tol3,
                   &dwork[kw], lwr_work, iwarn, info);
        }

        *iwarn = 10 * (*iwarn);
        if (*info != 0) return;

        wrkopt = dwork[kw] + (f64)kw;

        if (*nq == 0) {
            *nr = 0;
            iwork[0] = 0;
            dwork[0] = wrkopt;
            return;
        }

        if (maxmp > m) {
            /* Form matrices (BQ, BR) and (DQ, DR) in consecutive columns */
            kbt = kbr;
            kbr = kb + n * m;
            kdt = kdr;
            kdr = kd + maxmp * m;
            SLC_DLACPY("Full", nq, &p, &dwork[kbt], &n, &dwork[kbr], &n);
            SLC_DLACPY("Full", &p, &p, &dwork[kdt], &maxmp, &dwork[kdr], &maxmp);
        }

        /* Perform model reduction on (Q, R) to determine (Qr, Rr) */
        kt = kw;
        kti = kt + (*nq) * (*nq);
        kw = kti + (*nq) * (*nq);

        ab09bx(dico, jobmr, ordsel, *nq, mp, p, nr, a, lda,
               &dwork[kb], n, &dwork[kc], maxmp, &dwork[kd], maxmp,
               hsv, &dwork[kt], n, &dwork[kti], n, tol1, tol2,
               iwork, &dwork[kw], ldwork - kw, &iwarnk, &ierr);

        *iwarn = *iwarn + iwarnk;
        if (ierr != 0) {
            *info = 4;
            return;
        }

        nminr = iwork[0];
        f64 wrkopt2 = dwork[kw] + (f64)kw;
        if (wrkopt2 > wrkopt) wrkopt = wrkopt2;

        /* Compute reduced order system Gr = Rr^{-1}*Qr */
        kw = kt;
        sb08gd(*nr, m, p, a, lda, &dwork[kb], n, &dwork[kc], maxmp,
               &dwork[kd], maxmp, &dwork[kbr], n, &dwork[kdr], maxmp,
               &dwork[kw], iwork, info);

        /* Copy reduced system matrices Br, Cr, Dr to B, C, D */
        SLC_DLACPY("Full", nr, &m, &dwork[kb], &n, b, &ldb);
        SLC_DLACPY("Full", &p, nr, &dwork[kc], &maxmp, c, &ldc);
        SLC_DLACPY("Full", &p, &m, &dwork[kd], &maxmp, d, &ldd);
    } else {
        /* Compute a RCF G = Q*R^{-1} */
        pm = p + m;
        kdr = kd + p;
        kc = kd + pm * m;
        kcr = kc + p;
        kw = kc + pm * n;
        lwr_work = ldwork - kw;

        SLC_DLACPY("Full", &p, &n, c, &ldc, &dwork[kc], &pm);
        SLC_DLACPY("Full", &p, &m, d, &ldd, &dwork[kd], &pm);

        if (stabd) {
            /* Compute RCF with prescribed stability degree */
            alpha_arr[0] = alpha;
            alpha_arr[1] = alpha;
            sb08fd(dico, n, m, p, alpha_arr, a, lda, b, ldb,
                   &dwork[kc], pm, &dwork[kd], pm, nq, &ndr,
                   &dwork[kcr], pm, &dwork[kdr], pm, tol3,
                   &dwork[kw], lwr_work, iwarn, info);
        } else {
            /* Compute RCF with inner denominator */
            sb08dd(dico, n, m, p, a, lda, b, ldb,
                   &dwork[kc], pm, &dwork[kd], pm, nq, &ndr,
                   &dwork[kcr], pm, &dwork[kdr], pm, tol3,
                   &dwork[kw], lwr_work, iwarn, info);
        }

        *iwarn = 10 * (*iwarn);
        if (*info != 0) return;

        wrkopt = dwork[kw] + (f64)kw;

        if (*nq == 0) {
            *nr = 0;
            iwork[0] = 0;
            dwork[0] = wrkopt;
            return;
        }

        /* Perform model reduction on (Q; R) to determine (Qr; Rr) */
        kt = kw;
        kti = kt + (*nq) * (*nq);
        kw = kti + (*nq) * (*nq);

        ab09bx(dico, jobmr, ordsel, *nq, m, pm, nr, a, lda,
               b, ldb, &dwork[kc], pm, &dwork[kd], pm, hsv,
               &dwork[kt], n, &dwork[kti], n, tol1, tol2, iwork,
               &dwork[kw], ldwork - kw, &iwarnk, &ierr);

        *iwarn = *iwarn + iwarnk;
        if (ierr != 0) {
            *info = 4;
            return;
        }

        nminr = iwork[0];
        f64 wrkopt2 = dwork[kw] + (f64)kw;
        if (wrkopt2 > wrkopt) wrkopt = wrkopt2;

        /* Compute reduced order system Gr = Qr*Rr^{-1} */
        kw = kt;
        sb08hd(*nr, m, p, a, lda, b, ldb, &dwork[kc], pm,
               &dwork[kd], pm, &dwork[kcr], pm, &dwork[kdr], pm,
               &dwork[kw], info);

        /* Copy reduced system matrices Cr and Dr to C and D */
        SLC_DLACPY("Full", &p, nr, &dwork[kc], &pm, c, &ldc);
        SLC_DLACPY("Full", &p, &m, &dwork[kd], &pm, d, &ldd);
    }

    iwork[0] = nminr;
    dwork[0] = wrkopt;
}
