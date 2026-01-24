/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

/*
 * AB09FD - Balance & Truncate model reduction for unstable systems
 *          with coprime factorization.
 *
 * Purpose:
 *   To compute a reduced order model (Ar,Br,Cr) for an original
 *   state-space representation (A,B,C) by using either the square-root
 *   or the balancing-free square-root Balance & Truncate (B & T)
 *   model reduction method in conjunction with stable coprime
 *   factorization techniques.
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <stdlib.h>

void ab09fd(
    const char* dico,
    const char* jobcf,
    const char* fact,
    const char* jobmr,
    const char* equil,
    const char* ordsel,
    const i32 n,
    const i32 m,
    const i32 p,
    i32* nr,
    const f64 alpha,
    f64* a,
    const i32 lda,
    f64* b,
    const i32 ldb,
    f64* c,
    const i32 ldc,
    i32* nq,
    f64* hsv,
    const f64 tol1,
    const f64 tol2,
    i32* iwork,
    f64* dwork,
    const i32 ldwork,
    i32* iwarn,
    i32* info
)
{
    const f64 C100 = 100.0;
    const f64 ONE = 1.0;
    const f64 ZERO = 0.0;

    *info = 0;
    *iwarn = 0;

    bool discr = (*dico == 'D' || *dico == 'd');
    bool conti = (*dico == 'C' || *dico == 'c');
    bool fixord = (*ordsel == 'F' || *ordsel == 'f');
    bool autosel = (*ordsel == 'A' || *ordsel == 'a');
    bool left = (*jobcf == 'L' || *jobcf == 'l');
    bool right = (*jobcf == 'R' || *jobcf == 'r');
    bool stabd = (*fact == 'S' || *fact == 's');
    bool inner = (*fact == 'I' || *fact == 'i');
    bool bal = (*jobmr == 'B' || *jobmr == 'b');
    bool bfree = (*jobmr == 'N' || *jobmr == 'n');
    bool scale = (*equil == 'S' || *equil == 's');
    bool noscale = (*equil == 'N' || *equil == 'n');

    i32 maxmp = (m > p) ? m : p;

    i32 lwr = 2 * n * n + n * ((n > (m + p)) ? n : (m + p)) + 5 * n + (n * (n + 1)) / 2;
    i32 lw1 = n * (2 * maxmp + p) + maxmp * (maxmp + p);
    i32 lw2_add1 = n * p + (n * (n + 5) > (p * (p + 2)) ? n * (n + 5) : (p * (p + 2)));
    lw2_add1 = (lw2_add1 > 4 * p) ? lw2_add1 : 4 * p;
    lw2_add1 = (lw2_add1 > 4 * m) ? lw2_add1 : 4 * m;
    i32 lw2 = lw1 + ((lw2_add1 > lwr) ? lw2_add1 : lwr);

    i32 lw1_add1 = n * p + ((n * (n + 5) > 5 * p) ? n * (n + 5) : 5 * p);
    lw1_add1 = (lw1_add1 > 4 * m) ? lw1_add1 : 4 * m;
    lw1 = lw1 + ((lw1_add1 > lwr) ? lw1_add1 : lwr);

    i32 lw3 = (n + m) * (m + p) + (((5 * m > 4 * p) ? 5 * m : 4 * p) > lwr ? ((5 * m > 4 * p) ? 5 * m : 4 * p) : lwr);

    i32 lw4_add1 = (m * (m + 2) > 4 * m) ? m * (m + 2) : 4 * m;
    lw4_add1 = (lw4_add1 > 4 * p) ? lw4_add1 : 4 * p;
    i32 lw4 = (n + m) * (m + p) + ((lw4_add1 > lwr) ? lw4_add1 : lwr);

    if (!conti && !discr) {
        *info = -1;
    } else if (!left && !right) {
        *info = -2;
    } else if (!stabd && !inner) {
        *info = -3;
    } else if (!bal && !bfree) {
        *info = -4;
    } else if (!scale && !noscale) {
        *info = -5;
    } else if (!fixord && !autosel) {
        *info = -6;
    } else if (stabd && ((!discr && alpha >= ZERO) ||
               (discr && (alpha < ZERO || alpha >= ONE)))) {
        *info = -7;
    } else if (n < 0) {
        *info = -8;
    } else if (m < 0) {
        *info = -9;
    } else if (p < 0) {
        *info = -10;
    } else if (fixord && (*nr < 0 || *nr > n)) {
        *info = -11;
    } else {
        i32 max1n = (n > 1) ? n : 1;
        i32 max1p = (p > 1) ? p : 1;
        if (lda < max1n) {
            *info = -13;
        } else if (ldb < max1n) {
            *info = -15;
        } else if (ldc < max1p) {
            *info = -17;
        } else if (ldwork < 1 ||
                   (stabd && left && ldwork < lw1) ||
                   (!stabd && left && ldwork < lw2) ||
                   (stabd && !left && ldwork < lw3) ||
                   (!stabd && !left && ldwork < lw4)) {
            *info = -24;
        }
    }

    if (*info != 0) {
        i32 neginfo = -(*info);
        SLC_XERBLA("AB09FD", &neginfo);
        return;
    }

    i32 minnmp = n;
    if (m < minnmp) minnmp = m;
    if (p < minnmp) minnmp = p;
    if (minnmp == 0 || (fixord && *nr == 0)) {
        *nr = 0;
        *nq = 0;
        dwork[0] = ONE;
        return;
    }

    if (scale) {
        f64 maxred = C100;
        tb01id("A", n, m, p, &maxred, a, lda, b, ldb, c, ldc, dwork, info);
    }

    i32 kd, kdr, kc, kb, kbr, kw, lwr_rem;
    i32 iwarnk;
    i32 ndr;
    i32 wrkopt;
    i32 ierr;
    f64 alpha_arr[2];

    if (left) {
        i32 mp = m + p;
        kd = 0;
        kdr = kd + maxmp * maxmp;
        kc = kdr + maxmp * p;
        kb = kc + maxmp * n;
        kbr = kb + n * maxmp;
        kw = kbr + n * p;
        lwr_rem = ldwork - kw;

        SLC_DLACPY("Full", &n, &m, b, &ldb, &dwork[kb], &n);
        SLC_DLACPY("Full", &p, &n, c, &ldc, &dwork[kc], &maxmp);
        SLC_DLASET("Full", &p, &m, &ZERO, &ZERO, &dwork[kd], &maxmp);

        if (stabd) {
            alpha_arr[0] = alpha;
            alpha_arr[1] = alpha;
            sb08ed(dico, n, m, p, alpha_arr, a, lda, &dwork[kb], n,
                   &dwork[kc], maxmp, &dwork[kd], maxmp, nq, &ndr,
                   &dwork[kbr], n, &dwork[kdr], maxmp, tol2,
                   &dwork[kw], lwr_rem, iwarn, info);
        } else {
            sb08cd(dico, n, m, p, a, lda, &dwork[kb], n,
                   &dwork[kc], maxmp, &dwork[kd], maxmp, nq, &ndr,
                   &dwork[kbr], n, &dwork[kdr], maxmp, tol2,
                   &dwork[kw], lwr_rem, iwarn, info);
        }

        *iwarn = 10 * (*iwarn);
        if (*info != 0) {
            return;
        }

        wrkopt = (i32)dwork[kw] + kw;

        if (*nq == 0) {
            *nr = 0;
            dwork[0] = (f64)wrkopt;
            return;
        }

        if (maxmp > m) {
            i32 kbt = kbr;
            kbr = kb + n * m;
            i32 kdt = kdr;
            kdr = kd + maxmp * m;
            SLC_DLACPY("Full", nq, &p, &dwork[kbt], &n, &dwork[kbr], &n);
            SLC_DLACPY("Full", &p, &p, &dwork[kdt], &maxmp, &dwork[kdr], &maxmp);
        }

        i32 kt = kw;
        i32 kti = kt + (*nq) * (*nq);
        kw = kti + (*nq) * (*nq);

        ab09ax(dico, jobmr, ordsel, *nq, mp, p, nr, a, lda,
               &dwork[kb], n, &dwork[kc], maxmp, hsv, &dwork[kt],
               n, &dwork[kti], n, tol1, iwork, &dwork[kw],
               ldwork - kw, &iwarnk, &ierr);

        *iwarn = *iwarn + iwarnk;
        if (ierr != 0) {
            *info = 4;
            return;
        }

        wrkopt = (wrkopt > ((i32)dwork[kw] + kw)) ? wrkopt : ((i32)dwork[kw] + kw);

        kw = kt;
        sb08gd(*nr, m, p, a, lda, &dwork[kb], n, &dwork[kc], maxmp,
               &dwork[kd], maxmp, &dwork[kbr], n, &dwork[kdr], maxmp,
               &dwork[kw], iwork, info);

        SLC_DLACPY("Full", nr, &m, &dwork[kb], &n, b, &ldb);
        SLC_DLACPY("Full", &p, nr, &dwork[kc], &maxmp, c, &ldc);

    } else {
        i32 pm = p + m;
        kd = 0;
        kdr = kd + p;
        kc = kd + pm * m;
        i32 kcr = kc + p;
        kw = kc + pm * n;
        lwr_rem = ldwork - kw;

        SLC_DLACPY("Full", &p, &n, c, &ldc, &dwork[kc], &pm);
        SLC_DLASET("Full", &p, &m, &ZERO, &ZERO, &dwork[kd], &pm);

        if (stabd) {
            alpha_arr[0] = alpha;
            alpha_arr[1] = alpha;
            sb08fd(dico, n, m, p, alpha_arr, a, lda, b, ldb,
                   &dwork[kc], pm, &dwork[kd], pm, nq, &ndr,
                   &dwork[kcr], pm, &dwork[kdr], pm, tol2,
                   &dwork[kw], lwr_rem, iwarn, info);
        } else {
            sb08dd(dico, n, m, p, a, lda, b, ldb,
                   &dwork[kc], pm, &dwork[kd], pm, nq, &ndr,
                   &dwork[kcr], pm, &dwork[kdr], pm, tol2,
                   &dwork[kw], lwr_rem, iwarn, info);
        }

        *iwarn = 10 * (*iwarn);
        if (*info != 0) {
            return;
        }

        wrkopt = (i32)dwork[kw] + kw;

        if (*nq == 0) {
            *nr = 0;
            dwork[0] = (f64)wrkopt;
            return;
        }

        i32 kt = kw;
        i32 kti = kt + (*nq) * (*nq);
        kw = kti + (*nq) * (*nq);

        ab09ax(dico, jobmr, ordsel, *nq, m, pm, nr, a, lda,
               b, ldb, &dwork[kc], pm, hsv, &dwork[kt],
               n, &dwork[kti], n, tol1, iwork, &dwork[kw],
               ldwork - kw, &iwarnk, &ierr);

        *iwarn = *iwarn + iwarnk;
        if (ierr != 0) {
            *info = 4;
            return;
        }

        wrkopt = (wrkopt > ((i32)dwork[kw] + kw)) ? wrkopt : ((i32)dwork[kw] + kw);

        kw = kt;
        sb08hd(*nr, m, p, a, lda, b, ldb, &dwork[kc], pm,
               &dwork[kd], pm, &dwork[kcr], pm, &dwork[kdr], pm,
               &dwork[kw], info);

        SLC_DLACPY("Full", &p, nr, &dwork[kc], &pm, c, &ldc);
    }

    dwork[0] = (f64)wrkopt;
}
