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

void ab09hx(
    const char* dico,
    const char* job,
    const char* ordsel,
    const i32 n,
    const i32 m,
    const i32 p,
    i32* nr,
    f64* a,
    const i32 lda,
    f64* b,
    const i32 ldb,
    f64* c,
    const i32 ldc,
    f64* d,
    const i32 ldd,
    f64* hsv,
    f64* t,
    const i32 ldt,
    f64* ti,
    const i32 ldti,
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

    *info = 0;
    *iwarn = 0;

    char dico_upper = (char)toupper((unsigned char)dico[0]);
    char job_upper = (char)toupper((unsigned char)job[0]);
    char ordsel_upper = (char)toupper((unsigned char)ordsel[0]);

    bool discr = (dico_upper == 'D');
    bool bta = (job_upper == 'B' || job_upper == 'F');
    bool spa = (job_upper == 'S' || job_upper == 'P');
    bool bal = (job_upper == 'B' || job_upper == 'S');
    bool fixord = (ordsel_upper == 'F');

    i32 max_n_m = n > m ? n : m;
    i32 max_n_m_p = max_n_m > p ? max_n_m : p;
    i32 lw1 = n * (max_n_m_p + 5);
    i32 lw2a = p * (m + 2);
    i32 lw2b = 10 * n * (n + 1);
    i32 lw2 = 2 * n * p + (lw2a > lw2b ? lw2a : lw2b);
    i32 lw = lw1 > lw2 ? lw1 : lw2;
    if (lw < 2) lw = 2;

    if (dico_upper != 'C' && dico_upper != 'D') {
        *info = -1;
    } else if (!bta && !spa) {
        *info = -2;
    } else if (ordsel_upper != 'F' && ordsel_upper != 'A') {
        *info = -3;
    } else if (n < 0) {
        *info = -4;
    } else if (m < 0) {
        *info = -5;
    } else if (p < 0 || p > m) {
        *info = -6;
    } else if (fixord && (*nr < 0 || *nr > n)) {
        *info = -7;
    } else {
        i32 max_1_n = n > 1 ? n : 1;
        i32 max_1_p = p > 1 ? p : 1;
        if (lda < max_1_n) {
            *info = -9;
        } else if (ldb < max_1_n) {
            *info = -11;
        } else if (ldc < max_1_p) {
            *info = -13;
        } else if (ldd < max_1_p) {
            *info = -15;
        } else if (ldt < max_1_n) {
            *info = -18;
        } else if (ldti < max_1_n) {
            *info = -20;
        } else if (tol2 > ZERO && !fixord && tol2 > tol1) {
            *info = -22;
        } else if (ldwork < lw) {
            *info = -25;
        }
    }

    if (*info != 0) {
        i32 neginfo = -(*info);
        SLC_XERBLA("AB09HX", &neginfo);
        return;
    }

    i32 minval = n;
    if (m < minval) minval = m;
    if (p < minval) minval = p;

    if (minval == 0) {
        *nr = 0;
        iwork[0] = 0;
        dwork[0] = TWO;
        dwork[1] = ONE;
        return;
    }

    i32 wrkopt = 0;
    i32 ierr = 0;

    if (discr) {
        ierr = ab04md('D', n, m, p, ONE, ONE, a, lda, b, ldb, c, ldc, d, ldd,
                      iwork, dwork, ldwork);
        if (ierr != 0) {
            *info = 1;
            return;
        }
        i32 tmp = (i32)dwork[0];
        if (tmp > wrkopt) wrkopt = tmp;
        if (n > wrkopt) wrkopt = n;
    }

    f64 scalec, scaleo;
    ab09hy(n, m, p, a, lda, b, ldb, c, ldc, d, ldd,
           &scalec, &scaleo, ti, ldti, t, ldt, iwork,
           dwork, ldwork, bwork, info);
    if (*info != 0) {
        return;
    }

    i32 tmp = (i32)dwork[0];
    if (tmp > wrkopt) wrkopt = tmp;
    f64 ricond = dwork[1];

    i32 ku = 0;
    i32 kv = ku + n * n;
    i32 kw = kv + n * n;

    SLC_DLACPY("U", &n, &n, ti, &ldti, &dwork[kv], &n);

    i32 int1 = 1;
    for (i32 j = 0; j < n; j++) {
        i32 jp1 = j + 1;
        SLC_DTRMV("U", "N", "N", &jp1, t, &ldt, &ti[j * ldti], &int1);
    }

    i32 ldwn = n;
    mb03ud('V', 'V', n, ti, ldti, &dwork[ku], ldwn, hsv,
           &dwork[kw], ldwork - kw, &ierr);
    if (ierr != 0) {
        *info = 7;
        return;
    }

    tmp = (i32)dwork[kw] + kw;
    if (tmp > wrkopt) wrkopt = tmp;

    f64 scale = ONE / (scalec * scaleo);
    SLC_DSCAL(&n, &scale, hsv, &int1);

    f64 epsm = SLC_DLAMCH("E");
    f64 toldef = (f64)n * epsm;
    f64 atol = toldef;

    if (fixord) {
        if (*nr > 0) {
            if (hsv[*nr - 1] <= atol) {
                *nr = 0;
                *iwarn = 1;
                fixord = false;
            }
        }
    } else {
        atol = tol1 > toldef ? tol1 : toldef;
        *nr = 0;
    }

    if (!fixord) {
        for (i32 j = 0; j < n; j++) {
            if (hsv[j] <= atol) break;
            (*nr)++;
        }
    }

    i32 nr1 = *nr;
    i32 nminr = *nr;
    if (*nr < n) {
        if (spa) {
            f64 tol2_eff = tol2 > toldef ? tol2 : toldef;
            atol = tol2_eff;
        }
        for (i32 j = nr1; j < n; j++) {
            if (hsv[j] <= atol) break;
            nminr++;
        }
    }

    if (*nr == 0) {
        if (spa) {
            f64 rcond_spa;
            ab09dd("C", n, m, p, *nr, a, lda, b, ldb, c, ldc, d, ldd,
                   &rcond_spa, iwork, dwork);
            iwork[0] = nminr;
        } else {
            iwork[0] = 0;
        }
        dwork[0] = (f64)wrkopt;
        dwork[1] = ricond;
        return;
    }

    i32 ns = nminr - *nr;

    SLC_DTRMM("L", "U", "T", "N", &n, &nminr, &ONE, t, &ldt, &dwork[ku], &ldwn);

    ma02ad("F", nminr, n, ti, ldti, t, ldt);
    SLC_DTRMM("L", "U", "N", "N", &n, &nminr, &ONE, &dwork[kv], &ldwn, t, &ldt);

    i32 ktau = kv;
    i32 ij;

    if (bal) {
        ij = ku;
        for (i32 j = 0; j < *nr; j++) {
            f64 temp = ONE / sqrt(hsv[j]);
            SLC_DSCAL(&n, &temp, &t[j * ldt], &int1);
            SLC_DSCAL(&n, &temp, &dwork[ij], &int1);
            ij += n;
        }
    } else {
        kw = ktau + *nr;
        i32 ldw = ldwork - kw;

        SLC_DGEQRF(&n, nr, t, &ldt, &dwork[ktau], &dwork[kw], &ldw, &ierr);
        SLC_DORGQR(&n, nr, nr, t, &ldt, &dwork[ktau], &dwork[kw], &ldw, &ierr);

        SLC_DGEQRF(&n, nr, &dwork[ku], &ldwn, &dwork[ktau], &dwork[kw], &ldw, &ierr);
        tmp = (i32)dwork[kw] + kw;
        if (tmp > wrkopt) wrkopt = tmp;

        SLC_DORGQR(&n, nr, nr, &dwork[ku], &ldwn, &dwork[ktau], &dwork[kw], &ldw, &ierr);
        tmp = (i32)dwork[kw] + kw;
        if (tmp > wrkopt) wrkopt = tmp;
    }

    if (ns > 0) {
        kw = ktau + ns;
        i32 ldw = ldwork - kw;

        SLC_DGEQRF(&n, &ns, &t[nr1 * ldt], &ldt, &dwork[ktau], &dwork[kw], &ldw, &ierr);
        SLC_DORGQR(&n, &ns, &ns, &t[nr1 * ldt], &ldt, &dwork[ktau], &dwork[kw], &ldw, &ierr);

        i32 ku_ns = ku + n * nr1;
        SLC_DGEQRF(&n, &ns, &dwork[ku_ns], &ldwn, &dwork[ktau], &dwork[kw], &ldw, &ierr);
        tmp = (i32)dwork[kw] + kw;
        if (tmp > wrkopt) wrkopt = tmp;

        SLC_DORGQR(&n, &ns, &ns, &dwork[ku_ns], &ldwn, &dwork[ktau], &dwork[kw], &ldw, &ierr);
        tmp = (i32)dwork[kw] + kw;
        if (tmp > wrkopt) wrkopt = tmp;
    }

    ma02ad("F", n, nminr, &dwork[ku], ldwn, ti, ldti);

    if (!bal) {
        SLC_DGEMM("N", "N", nr, nr, &n, &ONE, ti, &ldti, t, &ldt, &ZERO, &dwork[ku], &ldwn);
        SLC_DGETRF(nr, nr, &dwork[ku], &ldwn, iwork, &ierr);
        SLC_DGETRS("N", nr, &n, &dwork[ku], &ldwn, iwork, ti, &ldti, &ierr);

        if (ns > 0) {
            SLC_DGEMM("N", "N", &ns, &ns, &n, &ONE, &ti[nr1], &ldti, &t[nr1 * ldt], &ldt, &ZERO, &dwork[ku], &ldwn);
            SLC_DGETRF(&ns, &ns, &dwork[ku], &ldwn, iwork, &ierr);
            SLC_DGETRS("N", &ns, &n, &dwork[ku], &ldwn, iwork, &ti[nr1], &ldti, &ierr);
        }
    }

    ij = ku;
    for (i32 j = 0; j < n; j++) {
        i32 k = j + 1;
        if (k > n) k = n;
        SLC_DGEMV("N", &nminr, &k, &ONE, ti, &ldti, &a[j * lda], &int1, &ZERO, &dwork[ij], &int1);
        ij += n;
    }

    SLC_DGEMM("N", "N", &nminr, &nminr, &n, &ONE, &dwork[ku], &ldwn, t, &ldt, &ZERO, a, &lda);

    SLC_DLACPY("F", &n, &m, b, &ldb, &dwork[ku], &ldwn);
    SLC_DGEMM("N", "N", &nminr, &m, &n, &ONE, ti, &ldti, &dwork[ku], &ldwn, &ZERO, b, &ldb);

    SLC_DLACPY("F", &p, &n, c, &ldc, &dwork[ku], &p);
    SLC_DGEMM("N", "N", &p, &nminr, &n, &ONE, &dwork[ku], &p, t, &ldt, &ZERO, c, &ldc);

    f64 rcond_spa;
    ab09dd("C", nminr, m, p, *nr, a, lda, b, ldb, c, ldc, d, ldd,
           &rcond_spa, iwork, dwork);

    if (discr) {
        ierr = ab04md('C', *nr, m, p, ONE, ONE, a, lda, b, ldb, c, ldc, d, ldd,
                      iwork, dwork, ldwork);
        if (ierr == 0) {
            tmp = (i32)dwork[0];
            if (tmp > wrkopt) wrkopt = tmp;
        }
    }

    iwork[0] = nminr;
    dwork[0] = (f64)wrkopt;
    dwork[1] = ricond;
}
