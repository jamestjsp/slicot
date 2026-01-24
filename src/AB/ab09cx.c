/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

static void local_dlacpy_safe(i32 m, i32 n, const f64 *a, i32 lda, f64 *b, i32 ldb) {
    for (i32 j = 0; j < n; j++) {
        memmove(&b[j * ldb], &a[j * lda], (size_t)m * sizeof(f64));
    }
}

void ab09cx(const char* dico, const char* ordsel,
            i32 n, i32 m, i32 p, i32* nr, f64* a, i32 lda,
            f64* b, i32 ldb, f64* c, i32 ldc, f64* d, i32 ldd,
            f64* hsv, f64 tol1, f64 tol2, i32* iwork,
            f64* dwork, i32 ldwork, i32* iwarn, i32* info)
{
    const f64 ONE = 1.0;
    const f64 ZERO = 0.0;

    bool discr = (dico[0] == 'D' || dico[0] == 'd');
    bool conti = (dico[0] == 'C' || dico[0] == 'c');
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
    } else if (!fixord && !autosel) {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    } else if (m < 0) {
        *info = -4;
    } else if (p < 0) {
        *info = -5;
    } else if (fixord && (*nr < 0 || *nr > n)) {
        *info = -6;
    } else if (lda < max1n) {
        *info = -8;
    } else if (ldb < max1n) {
        *info = -10;
    } else if (ldc < max1p) {
        *info = -12;
    } else if (ldd < max1p) {
        *info = -14;
    } else if (tol2 > ZERO && tol2 > tol1) {
        *info = -17;
    } else if (ldwork < minwrk) {
        *info = -20;
    }

    if (*info != 0) {
        i32 neginfo = -(*info);
        SLC_XERBLA("AB09CX", &neginfo);
        return;
    }

    if (n == 0 || m == 0 || p == 0) {
        *nr = 0;
        iwork[0] = 0;
        dwork[0] = ONE;
        return;
    }

    f64 rtol = (f64)n * SLC_DLAMCH("Epsilon");
    f64 srrtol = sqrt(rtol);

    i32 kt = 0;
    i32 kti = kt + n * n;
    i32 kw = kti + n * n;

    i32 ierr = 0;
    i32 nminr;
    ab09ax(dico, "Balanced", "Automatic", n, m, p, &nminr, a, lda, b, ldb, c, ldc,
           hsv, &dwork[kt], n, &dwork[kti], n, tol2, iwork, &dwork[kw], ldwork - kw,
           iwarn, &ierr);
    if (ierr != 0) {
        *info = ierr;
        return;
    }
    i32 wrkopt = (i32)dwork[kw] + kw;

    f64 atol = rtol * hsv[0];
    if (fixord) {
        if (*nr > 0) {
            if (*nr > nminr) {
                *nr = nminr;
                *iwarn = 1;
            }
        }
    } else {
        if (tol1 > atol) atol = tol1;
        *nr = 0;
        for (i32 i = 0; i < nminr; i++) {
            if (hsv[i] <= atol) break;
            (*nr)++;
        }
    }

    if (*nr == nminr) {
        iwork[0] = nminr;
        dwork[0] = (f64)wrkopt;
        kw = n * (n + 2);
        i32 tb01wd_info;
        tb01wd(nminr, m, p, a, lda, b, ldb, c, ldc, &dwork[2 * n], n,
               dwork, &dwork[n], &dwork[kw], ldwork - kw, &tb01wd_info);
        if (tb01wd_info != 0) {
            *info = 3;
            return;
        }
        return;
    }

    f64 skp = hsv[*nr];

    i32 nr_local = *nr;
    while (nr_local > 0) {
        if (fabs(hsv[nr_local - 1] - skp) <= srrtol * skp) {
            nr_local--;
        } else {
            break;
        }
    }
    *nr = nr_local;

    i32 kr = 1;
    i32 nkr1 = (*nr + 1 < nminr) ? *nr + 1 + kr : nminr;
    for (i32 i = *nr + 1; i < nminr; i++) {
        if (fabs(hsv[i] - skp) > srrtol * skp) break;
        kr++;
        nkr1 = (i + 1 < nminr) ? i + 2 : nminr;
    }
    nkr1 = *nr + kr;
    if (nkr1 > nminr) nkr1 = nminr;

    if (discr) {
        i32 ab04_info = ab04md('D', nminr, m, p, ONE, ONE, a, lda, b, ldb, c, ldc, d, ldd,
                                iwork, dwork, ldwork);
        (void)ab04_info;
        i32 tmp = (i32)dwork[0];
        if (tmp > wrkopt) wrkopt = tmp;
    }

    i32 nu = nminr - *nr - kr;
    i32 na = *nr + nu;
    i32 ldb1 = na;
    i32 ldc1 = p;
    i32 ldb2 = kr;
    i32 ldc2t = (kr > m) ? kr : m;
    i32 nr1 = *nr;
    i32 nkr1_idx = (*nr + kr < nminr) ? *nr + kr : nminr;

    i32 khsvp = 0;
    i32 khsvp2 = khsvp + na;
    i32 ku = khsvp2 + na;
    i32 kb1 = ku + p * m;
    i32 kb2 = kb1 + ldb1 * m;
    i32 kc1 = kb2 + ldb2 * m;
    i32 kc2t = kc1 + ldc1 * na;
    kw = kc2t + ldc2t * p;

    SLC_DLACPY("Full", &kr, &m, &b[nr1], &ldb, &dwork[kb2], &ldb2);
    ma02ad("Full", p, kr, &c[nr1 * ldc], ldc, &dwork[kc2t], ldc2t);

    if (*nr > 0) {
        for (i32 i = 0; i < *nr; i++) {
            dwork[khsvp + i] = hsv[i];
        }
        for (i32 i = 0; i < nu; i++) {
            dwork[khsvp + *nr + i] = hsv[nkr1_idx + i];
        }

        // Use memmove-based copy for potentially overlapping in-place copies
        local_dlacpy_safe(nminr, nu, &a[nkr1_idx * lda], lda, &a[nr1 * lda], lda);
        local_dlacpy_safe(nu, na, &a[nkr1_idx], lda, &a[nr1], lda);
        local_dlacpy_safe(nu, m, &b[nkr1_idx], ldb, &b[nr1], ldb);
        SLC_DLACPY("Full", &p, &nu, &c[nkr1_idx * ldc], &ldc, &c[nr1 * ldc], &ldc);

        SLC_DLACPY("Full", &na, &m, b, &ldb, &dwork[kb1], &ldb1);
        SLC_DLACPY("Full", &p, &na, c, &ldc, &dwork[kc1], &ldc1);
    }

    for (i32 j = 0; j < m; j++) {
        iwork[j] = 0;
    }

    i32 irank;
    i32 lwork_gelsy = ldwork - kw;
    SLC_DGELSY(&kr, &m, &p, &dwork[kb2], &ldb2, &dwork[kc2t], &ldc2t,
               iwork, &rtol, &irank, &dwork[kw], &lwork_gelsy, &ierr);
    i32 tmp = (i32)dwork[kw] + kw;
    if (tmp > wrkopt) wrkopt = tmp;

    ma02ad("Full", m, p, &dwork[kc2t], ldc2t, &dwork[ku], p);

    i32 iu = ku;
    i32 one = 1;
    for (i32 j = 0; j < m; j++) {
        SLC_DAXPY(&p, &skp, &dwork[iu], &one, &d[j * ldd], &one);
        iu += p;
    }

    if (*nr > 0) {
        f64 skp2 = skp * skp;

        i32 i1 = khsvp2;
        for (i32 i = khsvp; i < khsvp + na; i++) {
            dwork[i1] = ONE / (dwork[i] * dwork[i] - skp2);
            i1++;
        }

        mb01sd('C', p, na, c, ldc, dwork, &dwork[khsvp]);
        f64 neg_skp = -skp;
        SLC_DGEMM("NoTranspose", "Transpose", &p, &na, &m, &neg_skp,
                  &dwork[ku], &p, &dwork[kb1], &ldb1, &ONE, c, &ldc);

        mb01sd('R', na, m, b, ldb, &dwork[khsvp], dwork);
        SLC_DGEMM("Transpose", "NoTranspose", &na, &m, &p, &neg_skp,
                  &dwork[kc1], &ldc1, &dwork[ku], &p, &ONE, b, &ldb);
        mb01sd('R', na, m, b, ldb, &dwork[khsvp2], dwork);

        for (i32 j = 1; j < na; j++) {
            SLC_DSWAP(&j, &a[j * lda], &one, &a[j], &lda);
        }
        f64 neg_one = -ONE;
        SLC_DGEMM("NoTranspose", "Transpose", &na, &na, &m, &neg_one, b, &ldb,
                  &dwork[kb1], &ldb1, &neg_one, a, &lda);

        i32 kw1 = na * na;
        i32 kw2 = kw1 + na;
        kw = kw2 + na;
        i32 ndim;
        tb01kd("Continuous", "Stability", "General", na, m, p, ZERO, a, lda, b, ldb,
               c, ldc, &ndim, dwork, na, &dwork[kw1], &dwork[kw2], &dwork[kw],
               ldwork - kw, &ierr);
        if (ierr != 0) {
            *info = 3;
            return;
        }

        if (ndim != *nr) {
            *info = 4;
            return;
        }
        tmp = (i32)dwork[kw] + kw;
        if (tmp > wrkopt) wrkopt = tmp;

        if (discr) {
            i32 ab04_info = ab04md('C', *nr, m, p, ONE, ONE, a, lda, b, ldb, c, ldc, d, ldd,
                                    iwork, dwork, ldwork);
            (void)ab04_info;
        }
    }

    iwork[0] = nminr;
    dwork[0] = (f64)wrkopt;
}
