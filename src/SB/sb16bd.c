/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <stdlib.h>

void sb16bd(
    const char* dico,
    const char* jobd,
    const char* jobmr,
    const char* jobcf,
    const char* equil,
    const char* ordsel,
    const i32 n,
    const i32 m,
    const i32 p,
    i32* ncr,
    f64* a,
    const i32 lda,
    const f64* b,
    const i32 ldb,
    const f64* c,
    const i32 ldc,
    const f64* d,
    const i32 ldd,
    f64* f,
    const i32 ldf,
    f64* g,
    const i32 ldg,
    f64* dc,
    const i32 lddc,
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
    const f64 ONE = 1.0;
    const f64 ZERO = 0.0;

    *info = 0;
    *iwarn = 0;

    bool discr = (dico[0] == 'D' || dico[0] == 'd');
    bool withd = (jobd[0] == 'D' || jobd[0] == 'd');
    bool bta = (jobmr[0] == 'B' || jobmr[0] == 'b' ||
                jobmr[0] == 'F' || jobmr[0] == 'f');
    bool spa = (jobmr[0] == 'S' || jobmr[0] == 's' ||
                jobmr[0] == 'P' || jobmr[0] == 'p');
    bool bal = (jobmr[0] == 'B' || jobmr[0] == 'b' ||
                jobmr[0] == 'S' || jobmr[0] == 's');
    bool left = (jobcf[0] == 'L' || jobcf[0] == 'l');
    bool lequil = (equil[0] == 'S' || equil[0] == 's');
    bool fixord = (ordsel[0] == 'F' || ordsel[0] == 'f');

    i32 mp = m + p;
    i32 lwr = n * (2 * n + ((n > mp) ? n : mp) + 5) + (n * (n + 1)) / 2;
    if (lwr < 1) lwr = 1;
    i32 lw1 = (n + m) * mp + ((lwr > 4 * m) ? lwr : 4 * m);
    i32 lw2 = (n + p) * mp + ((lwr > 4 * p) ? lwr : 4 * p);

    i32 max1n = (1 > n) ? 1 : n;
    i32 max1m = (1 > m) ? 1 : m;
    i32 max1p = (1 > p) ? 1 : p;

    if (!(dico[0] == 'C' || dico[0] == 'c' || discr)) {
        *info = -1;
    } else if (!(withd || jobd[0] == 'Z' || jobd[0] == 'z')) {
        *info = -2;
    } else if (!(bta || spa)) {
        *info = -3;
    } else if (!(left || jobcf[0] == 'R' || jobcf[0] == 'r')) {
        *info = -4;
    } else if (!(lequil || equil[0] == 'N' || equil[0] == 'n')) {
        *info = -5;
    } else if (!(fixord || ordsel[0] == 'A' || ordsel[0] == 'a')) {
        *info = -6;
    } else if (n < 0) {
        *info = -7;
    } else if (m < 0) {
        *info = -8;
    } else if (p < 0) {
        *info = -9;
    } else if (fixord && (*ncr < 0 || *ncr > n)) {
        *info = -10;
    } else if (lda < max1n) {
        *info = -12;
    } else if (ldb < max1n) {
        *info = -14;
    } else if (ldc < max1p) {
        *info = -16;
    } else if (ldd < 1 || (withd && ldd < (i32)p)) {
        *info = -18;
    } else if (ldf < max1m) {
        *info = -20;
    } else if (ldg < max1n) {
        *info = -22;
    } else if (lddc < max1m) {
        *info = -24;
    } else if (!fixord && tol2 > ZERO && tol2 > tol1) {
        *info = -27;
    } else {
        bool need_full_work = (!fixord || *ncr < n);
        if ((need_full_work && ((left && ldwork < lw1) || (!left && ldwork < lw2))) ||
            (fixord && *ncr == n && ldwork < p * n)) {
            *info = -30;
        }
    }

    if (*info != 0) {
        i32 neginfo = -(*info);
        SLC_XERBLA("SB16BD", &neginfo);
        return;
    }

    i32 min_nmp = n;
    if (m < min_nmp) min_nmp = m;
    if (p < min_nmp) min_nmp = p;

    if (min_nmp == 0 || (fixord && bta && *ncr == 0)) {
        *ncr = 0;
        dwork[0] = ONE;
        return;
    }

    if (*ncr == n) {
        SLC_DLACPY("F", &p, &n, c, &ldc, dwork, &p);
        if (withd) {
            SLC_DGEMM("N", "N", &p, &n, &m, &ONE, d, &ldd, f, &ldf, &ONE, dwork, &p);
        }
        SLC_DGEMM("N", "N", &n, &n, &p, &ONE, g, &ldg, dwork, &p, &ONE, a, &lda);
        SLC_DGEMM("N", "N", &n, &n, &m, &ONE, b, &ldb, f, &ldf, &ONE, a, &lda);

        dwork[0] = (f64)(p * n);
        return;
    }

    char job[2] = {bal ? 'B' : 'N', '\0'};

    if (left) {
        SLC_DGEMM("N", "N", &n, &n, &p, &ONE, g, &ldg, c, &ldc, &ONE, a, &lda);

        i32 kbe = 0;
        i32 kde = kbe + n * (p + m);
        i32 ldbe = max1n;
        i32 ldde = m;

        SLC_DLACPY("F", &n, &p, g, &ldg, &dwork[kbe], &ldbe);
        SLC_DLACPY("F", &n, &m, b, &ldb, &dwork[kbe + n * p], &ldbe);
        if (withd) {
            SLC_DGEMM("N", "N", &n, &m, &p, &ONE, g, &ldg, d, &ldd, &ONE,
                      &dwork[kbe + n * p], &ldbe);
        }
        SLC_DLASET("F", &m, &p, &ZERO, &ZERO, &dwork[kde], &ldde);
        SLC_DLASET("F", &m, &m, &ZERO, &ONE, &dwork[kde + m * p], &ldde);

        i32 kw = kde + m * (p + m);
        i32 remaining_work = ldwork - kw;

        if (bta) {
            ab09ad(dico, job, equil, ordsel, n, mp, m, ncr,
                   a, lda, &dwork[kbe], ldbe, f, ldf, hsv, tol1,
                   iwork, &dwork[kw], remaining_work, iwarn, info);
        } else {
            ab09bd(dico, job, equil, ordsel, n, mp, m, ncr,
                   a, lda, &dwork[kbe], ldbe, f, ldf, &dwork[kde], ldde,
                   hsv, tol1, tol2, iwork, &dwork[kw], remaining_work,
                   iwarn, info);
        }

        if (*info != 0) {
            return;
        }

        i32 wrkopt = (i32)dwork[kw] + kw;

        sb08gd(*ncr, p, m, a, lda, &dwork[kbe], ldbe, f, ldf,
               &dwork[kde], ldde, &dwork[kbe + n * p], ldbe,
               &dwork[kde + m * p], ldde, &dwork[kw], iwork, info);

        SLC_DLACPY("F", ncr, &p, &dwork[kbe], &ldbe, g, &ldg);
        SLC_DLACPY("F", &m, &p, &dwork[kde], &ldde, dc, &lddc);

        dwork[0] = (f64)wrkopt;
    } else {
        SLC_DGEMM("N", "N", &n, &n, &m, &ONE, b, &ldb, f, &ldf, &ONE, a, &lda);

        i32 kce = 0;
        i32 kde = kce + n * (p + m);
        i32 ldce = mp;
        i32 ldde = ldce;

        SLC_DLACPY("F", &m, &n, f, &ldf, &dwork[kce], &ldce);
        SLC_DLACPY("F", &p, &n, c, &ldc, &dwork[kce + m], &ldce);
        if (withd) {
            SLC_DGEMM("N", "N", &p, &n, &m, &ONE, d, &ldd, f, &ldf, &ONE,
                      &dwork[kce + m], &ldce);
        }
        SLC_DLASET("F", &m, &p, &ZERO, &ZERO, &dwork[kde], &ldde);
        SLC_DLASET("F", &p, &p, &ZERO, &ONE, &dwork[kde + m], &ldde);

        i32 kw = kde + p * (p + m);
        i32 remaining_work = ldwork - kw;

        if (bta) {
            ab09ad(dico, job, equil, ordsel, n, p, mp, ncr,
                   a, lda, g, ldg, &dwork[kce], ldce, hsv, tol1,
                   iwork, &dwork[kw], remaining_work, iwarn, info);
        } else {
            ab09bd(dico, job, equil, ordsel, n, p, mp, ncr,
                   a, lda, g, ldg, &dwork[kce], ldce, &dwork[kde], ldde,
                   hsv, tol1, tol2, iwork, &dwork[kw], remaining_work,
                   iwarn, info);
        }

        if (*info != 0) {
            if (*info != 3) {
                *info = *info + 3;
            }
            return;
        }

        i32 wrkopt = (i32)dwork[kw] + kw;

        sb08hd(*ncr, p, m, a, lda, g, ldg, &dwork[kce], ldce,
               &dwork[kde], ldde, &dwork[kce + m], ldce,
               &dwork[kde + m], ldde, &dwork[kw], info);

        SLC_DLACPY("F", &m, ncr, &dwork[kce], &ldce, f, &ldf);
        SLC_DLACPY("F", &m, &p, &dwork[kde], &ldde, dc, &lddc);

        dwork[0] = (f64)wrkopt;
    }
}
