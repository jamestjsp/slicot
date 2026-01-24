/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <ctype.h>
#include <stdbool.h>

void ab09hd(
    const char* dico,
    const char* job,
    const char* equil,
    const char* ordsel,
    const i32 n,
    const i32 m,
    const i32 p,
    i32* nr,
    const f64 alpha,
    const f64 beta,
    f64* a,
    const i32 lda,
    f64* b,
    const i32 ldb,
    f64* c,
    const i32 ldc,
    f64* d,
    const i32 ldd,
    i32* ns,
    f64* hsv,
    const f64 tol1,
    const f64 tol2,
    i32* iwork,
    f64* dwork,
    const i32 ldwork,
    i32* bwork,
    i32* iwarn,
    i32* info
)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const f64 TWO = 2.0;
    const f64 TWOBY3 = TWO / 3.0;
    const f64 C100 = 100.0;

    *info = 0;
    *iwarn = 0;

    char dico_upper = (char)toupper((unsigned char)dico[0]);
    char job_upper = (char)toupper((unsigned char)job[0]);
    char equil_upper = (char)toupper((unsigned char)equil[0]);
    char ordsel_upper = (char)toupper((unsigned char)ordsel[0]);

    bool discr = (dico_upper == 'D');
    bool fixord = (ordsel_upper == 'F');
    bool lequil = (equil_upper == 'S');
    bool bta = (job_upper == 'B' || job_upper == 'F');
    bool spa = (job_upper == 'S' || job_upper == 'P');

    i32 mb = m;
    if (beta > ZERO) {
        mb = m + p;
    }

    i32 max_n_mb = n > mb ? n : mb;
    i32 max_n_mb_p = max_n_mb > p ? max_n_mb : p;
    i32 lw1 = n * (max_n_mb_p + 5);
    i32 lw2a = p * (mb + 2);
    i32 lw2b = 10 * n * (n + 1);
    i32 lw2 = 2 * n * p + (lw2a > lw2b ? lw2a : lw2b);
    i32 lw = lw1 > lw2 ? lw1 : lw2;
    if (lw < 2) lw = 2;
    lw = 2 * n * n + mb * (n + p) + lw;

    if (dico_upper != 'C' && dico_upper != 'D') {
        *info = -1;
    } else if (!bta && !spa) {
        *info = -2;
    } else if (equil_upper != 'S' && equil_upper != 'N') {
        *info = -3;
    } else if (ordsel_upper != 'F' && ordsel_upper != 'A') {
        *info = -4;
    } else if (n < 0) {
        *info = -5;
    } else if (m < 0) {
        *info = -6;
    } else if (p < 0 || (beta == ZERO && p > m)) {
        *info = -7;
    } else if (fixord && (*nr < 0 || *nr > n)) {
        *info = -8;
    } else if ((discr && (alpha < ZERO || alpha > ONE)) ||
               (!discr && alpha > ZERO)) {
        *info = -9;
    } else if (beta < ZERO) {
        *info = -10;
    } else {
        i32 max_1_n = n > 1 ? n : 1;
        i32 max_1_p = p > 1 ? p : 1;
        if (lda < max_1_n) {
            *info = -12;
        } else if (ldb < max_1_n) {
            *info = -14;
        } else if (ldc < max_1_p) {
            *info = -16;
        } else if (ldd < max_1_p) {
            *info = -18;
        } else if (tol1 >= ONE) {
            *info = -21;
        } else if ((tol2 > ZERO && !fixord && tol2 > tol1) || tol2 >= ONE) {
            *info = -22;
        } else if (ldwork < lw) {
            *info = -25;
        }
    }

    if (*info != 0) {
        i32 neginfo = -(*info);
        SLC_XERBLA("AB09HD", &neginfo);
        return;
    }

    i32 minval = n;
    if (m < minval) minval = m;
    if (p < minval) minval = p;

    if (minval == 0 || (bta && fixord && *nr == 0)) {
        *nr = 0;
        *ns = 0;
        iwork[0] = 0;
        dwork[0] = TWO;
        dwork[1] = ONE;
        return;
    }

    if (lequil) {
        f64 maxred = C100;
        tb01id("All", n, m, p, &maxred, a, lda, b, ldb, c, ldc, dwork, info);
    }

    i32 nn = n * n;
    i32 ku = 0;
    i32 kwr = ku + nn;
    i32 kwi = kwr + n;
    i32 kw = kwi + n;
    i32 lwr = ldwork - kw;

    i32 nu = 0;
    i32 ierr = 0;
    tb01kd(dico, "Unstable", "General", n, m, p, alpha, a, lda, b, ldb, c, ldc,
           &nu, &dwork[ku], n, &dwork[kwr], &dwork[kwi], &dwork[kw], lwr, &ierr);

    if (ierr != 0) {
        if (ierr != 3) {
            *info = 1;
        } else {
            *info = 8;
        }
        return;
    }

    i32 wrkopt = (i32)dwork[kw] + kw;

    i32 iwarnl = 0;
    *ns = n - nu;
    i32 nra;

    if (fixord) {
        nra = *nr - nu;
        if (nra < 0) nra = 0;
        if (*nr < nu) {
            iwarnl = 3;
        }
    } else {
        nra = 0;
    }

    if (*ns == 0) {
        *nr = nu;
        iwork[0] = *ns;
        dwork[0] = (f64)wrkopt;
        dwork[1] = ONE;
        return;
    }

    i32 nu1 = nu;

    i32 kb = 0;
    i32 kd = kb + n * mb;
    i32 kt = kd + p * mb;
    i32 kti = kt + n * n;
    kw = kti + n * n;

    SLC_DLACPY("F", ns, &m, &b[nu1], &ldb, &dwork[kb], &n);
    SLC_DLACPY("F", &p, &m, d, &ldd, &dwork[kd], &p);

    if (beta > ZERO) {
        SLC_DLASET("F", ns, &p, &ZERO, &ZERO, &dwork[kb + n * m], &n);
        SLC_DLASET("F", &p, &p, &ZERO, &beta, &dwork[kd + p * m], &p);
    }

    if (discr) {
        ierr = ab04md('D', *ns, mb, p, ONE, ONE, &a[nu1 + nu1 * lda], lda,
                      &dwork[kb], n, &c[nu1 * ldc], ldc, &dwork[kd], p,
                      iwork, &dwork[kt], ldwork - kt);
        if (ierr == 0) {
            i32 tmp = (i32)dwork[kt] + kt;
            if (tmp > wrkopt) wrkopt = tmp;
        }
    }

    f64 scalec, scaleo;
    ab09hy(*ns, mb, p, &a[nu1 + nu1 * lda], lda, &dwork[kb], n,
           &c[nu1 * ldc], ldc, &dwork[kd], p, &scalec, &scaleo,
           &dwork[kti], n, &dwork[kt], n, iwork, &dwork[kw],
           ldwork - kw, bwork, info);

    if (*info != 0) {
        return;
    }

    i32 tmp = (i32)dwork[kw] + kw;
    if (tmp > wrkopt) wrkopt = tmp;
    f64 ricond = dwork[kw + 1];

    f64 epsm = SLC_DLAMCH("Epsilon");
    i32 nmr;
    f64 tol1_eff = tol1 > (f64)n * epsm ? tol1 : (f64)n * epsm;

    i32 iwarn_ab09ix = 0;
    ierr = ab09ix("C", job, "Schur", ordsel, *ns, mb, p, &nra, scalec, scaleo,
                  &a[nu1 + nu1 * lda], lda, &dwork[kb], n, &c[nu1 * ldc], ldc,
                  &dwork[kd], p, &dwork[kti], n, &dwork[kt], n, &nmr, hsv,
                  tol1_eff, tol2, iwork, &dwork[kw], ldwork - kw, &iwarn_ab09ix);

    *iwarn = iwarn_ab09ix > iwarnl ? iwarn_ab09ix : iwarnl;

    if (ierr != 0) {
        *info = 7;
        return;
    }

    tmp = (i32)dwork[kw] + kw;
    if (tmp > wrkopt) wrkopt = tmp;

    if (nra < *ns && hsv[nra] >= ONE - pow(epsm, TWOBY3)) {
        *info = 9;
        return;
    }

    if (discr) {
        ierr = ab04md('C', nra, mb, p, ONE, ONE, &a[nu1 + nu1 * lda], lda,
                      &dwork[kb], n, &c[nu1 * ldc], ldc, &dwork[kd], p,
                      iwork, dwork, ldwork);
        if (ierr == 0) {
            tmp = (i32)dwork[0];
            if (tmp > wrkopt) wrkopt = tmp;
        }
    }

    SLC_DLACPY("F", &nra, &m, &dwork[kb], &n, &b[nu1], &ldb);
    SLC_DLACPY("F", &p, &m, &dwork[kd], &p, d, &ldd);

    *nr = nra + nu;

    iwork[0] = nmr;
    dwork[0] = (f64)wrkopt;
    dwork[1] = ricond;
}
