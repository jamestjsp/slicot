/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 2025, slicot.c contributors
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <string.h>
#include <stdbool.h>

i32 ab09ix(const char* dico, const char* job, const char* fact,
           const char* ordsel, i32 n, i32 m, i32 p, i32* nr,
           f64 scalec, f64 scaleo,
           f64* a, i32 lda, f64* b, i32 ldb,
           f64* c, i32 ldc, f64* d, i32 ldd,
           f64* ti, i32 ldti, f64* t, i32 ldt,
           i32* nminr, f64* hsv, f64 tol1, f64 tol2,
           i32* iwork, f64* dwork, i32 ldwork, i32* iwarn) {

    const f64 ONE = 1.0;
    const f64 ZERO = 0.0;

    *iwarn = 0;

    bool discr = (dico[0] == 'D' || dico[0] == 'd');
    bool conti = (dico[0] == 'C' || dico[0] == 'c');
    bool bta = (job[0] == 'B' || job[0] == 'b' || job[0] == 'F' || job[0] == 'f');
    bool spa = (job[0] == 'S' || job[0] == 's' || job[0] == 'P' || job[0] == 'p');
    bool bal = (job[0] == 'B' || job[0] == 'b' || job[0] == 'S' || job[0] == 's');
    bool rsf = (fact[0] == 'S' || fact[0] == 's');
    bool fixord = (ordsel[0] == 'F' || ordsel[0] == 'f');

    i32 max_mp = (m > p) ? m : p;
    i32 lw = 2 * n * n + 5 * n;
    if (n * max_mp > lw) lw = n * max_mp;
    if (lw < 1) lw = 1;

    if (!conti && !discr) {
        return -1;
    }
    if (!bta && !spa) {
        return -2;
    }
    if (!rsf && !(fact[0] == 'N' || fact[0] == 'n')) {
        return -3;
    }
    if (!fixord && !(ordsel[0] == 'A' || ordsel[0] == 'a')) {
        return -4;
    }
    if (n < 0) {
        return -5;
    }
    if (m < 0) {
        return -6;
    }
    if (p < 0) {
        return -7;
    }
    if (fixord && (*nr < 0 || *nr > n)) {
        return -8;
    }
    if (scalec <= ZERO) {
        return -9;
    }
    if (scaleo <= ZERO) {
        return -10;
    }
    i32 max1n = (1 > n) ? 1 : n;
    if (lda < max1n) {
        return -12;
    }
    if (ldb < max1n) {
        return -14;
    }
    i32 max1p = (1 > p) ? 1 : p;
    if (ldc < max1p) {
        return -16;
    }
    if (ldd < 1 || (spa && ldd < p)) {
        return -18;
    }
    if (ldti < max1n) {
        return -20;
    }
    if (ldt < max1n) {
        return -22;
    }
    if (tol2 > ZERO && !fixord && tol2 > tol1) {
        return -26;
    }
    if (ldwork < lw) {
        return -29;
    }

    i32 minval = (n < m) ? n : m;
    if (p < minval) minval = p;
    if (minval == 0) {
        *nr = 0;
        *nminr = 0;
        dwork[0] = ONE;
        return 0;
    }

    i32 kv = 0;
    i32 ku = kv + n * n;
    i32 kw = ku + n * n;
    SLC_DLACPY("U", &n, &n, ti, &ldti, &dwork[kv], &n);

    i32 int1 = 1;
    for (i32 j = 0; j < n; j++) {
        i32 jp1 = j + 1;
        SLC_DTRMV("U", "N", "N", &jp1, t, &ldt, &ti[j * ldti], &int1);
    }

    i32 info_mb03ud = 0;
    i32 ldwork_mb03ud = ldwork - kw;
    mb03ud('V', 'V', n, ti, ldti, &dwork[ku], n, hsv, &dwork[kw], ldwork_mb03ud, &info_mb03ud);
    if (info_mb03ud != 0) {
        return 1;
    }
    i32 wrkopt = (i32)dwork[kw] + kw;

    f64 scale_factor = ONE / (scalec * scaleo);
    SLC_DSCAL(&n, &scale_factor, hsv, &int1);

    f64 toldef = (f64)n * SLC_DLAMCH("E");
    f64 atol = (tol2 > toldef * hsv[0]) ? tol2 : toldef * hsv[0];
    *nminr = n;
    while (*nminr > 0 && hsv[*nminr - 1] <= atol) {
        (*nminr)--;
    }

    if (fixord) {
        if (*nr > *nminr) {
            *nr = *nminr;
            *iwarn = 1;
        }

        if (*nr > 0 && *nr < *nminr) {
            f64 skp = hsv[*nr - 1];
            if (skp - hsv[*nr] <= toldef * skp) {
                *iwarn = 2;
                while (*nr > 0 && hsv[*nr - 1] - skp <= toldef * skp) {
                    (*nr)--;
                }
            }
        }
    } else {
        atol = (tol1 > atol) ? tol1 : atol;
        *nr = 0;
        for (i32 j = 0; j < *nminr; j++) {
            if (hsv[j] <= atol) break;
            (*nr)++;
        }
    }

    if (*nr == 0) {
        if (spa) {
            f64 rcond;
            ab09dd(dico, n, m, p, *nr, a, lda, b, ldb, c, ldc, d, ldd,
                   &rcond, iwork, dwork);
        }
        dwork[0] = (f64)wrkopt;
        return 0;
    }

    i32 nred;
    if (spa) {
        nred = *nminr;
    } else {
        nred = *nr;
    }
    i32 ns = nred - *nr;

    SLC_DTRMM("L", "U", "T", "N", &n, &nred, &ONE, t, &ldt, &dwork[ku], &n);

    ma02ad("F", nred, n, ti, ldti, t, ldt);
    SLC_DTRMM("L", "U", "N", "N", &n, &nred, &ONE, &dwork[kv], &n, t, &ldt);

    i32 ktau = kw;
    if (bal) {
        i32 ij = ku;
        for (i32 j = 0; j < *nr; j++) {
            f64 temp = ONE / sqrt(hsv[j]);
            SLC_DSCAL(&n, &temp, &t[j * ldt], &int1);
            SLC_DSCAL(&n, &temp, &dwork[ij], &int1);
            ij += n;
        }
    } else {
        i32 kw_qr = ktau + *nr;
        i32 ldw_qr = ldwork - kw_qr;
        i32 ierr;

        SLC_DGEQRF(&n, nr, t, &ldt, &dwork[ktau], &dwork[kw_qr], &ldw_qr, &ierr);
        SLC_DORGQR(&n, nr, nr, t, &ldt, &dwork[ktau], &dwork[kw_qr], &ldw_qr, &ierr);

        SLC_DGEQRF(&n, nr, &dwork[ku], &n, &dwork[ktau], &dwork[kw_qr], &ldw_qr, &ierr);
        i32 tmp = (i32)dwork[kw_qr] + kw_qr;
        if (tmp > wrkopt) wrkopt = tmp;
        SLC_DORGQR(&n, nr, nr, &dwork[ku], &n, &dwork[ktau], &dwork[kw_qr], &ldw_qr, &ierr);
        tmp = (i32)dwork[kw_qr] + kw_qr;
        if (tmp > wrkopt) wrkopt = tmp;
    }

    if (ns > 0) {
        i32 nr1 = *nr;
        i32 kw_qr = ktau + ns;
        i32 ldw_qr = ldwork - kw_qr;
        i32 ierr;

        SLC_DGEQRF(&n, &ns, &t[nr1 * ldt], &ldt, &dwork[ktau], &dwork[kw_qr], &ldw_qr, &ierr);
        SLC_DORGQR(&n, &ns, &ns, &t[nr1 * ldt], &ldt, &dwork[ktau], &dwork[kw_qr], &ldw_qr, &ierr);

        SLC_DGEQRF(&n, &ns, &dwork[ku + n * nr1], &n, &dwork[ktau], &dwork[kw_qr], &ldw_qr, &ierr);
        i32 tmp = (i32)dwork[kw_qr] + kw_qr;
        if (tmp > wrkopt) wrkopt = tmp;
        SLC_DORGQR(&n, &ns, &ns, &dwork[ku + n * nr1], &n, &dwork[ktau], &dwork[kw_qr], &ldw_qr, &ierr);
        tmp = (i32)dwork[kw_qr] + kw_qr;
        if (tmp > wrkopt) wrkopt = tmp;
    }

    ma02ad("F", n, nred, &dwork[ku], n, ti, ldti);

    if (!bal) {
        SLC_DGEMM("N", "N", nr, nr, &n, &ONE, ti, &ldti, t, &ldt, &ZERO, &dwork[ku], &n);
        SLC_DGETRF(nr, nr, &dwork[ku], &n, iwork, &info_mb03ud);
        SLC_DGETRS("N", nr, &n, &dwork[ku], &n, iwork, ti, &ldti, &info_mb03ud);

        if (ns > 0) {
            i32 nr1 = *nr;
            SLC_DGEMM("N", "N", &ns, &ns, &n, &ONE, &ti[nr1], &ldti, &t[nr1 * ldt], &ldt, &ZERO, &dwork[ku], &n);
            SLC_DGETRF(&ns, &ns, &dwork[ku], &n, iwork, &info_mb03ud);
            SLC_DGETRS("N", &ns, &n, &dwork[ku], &n, iwork, &ti[nr1], &ldti, &info_mb03ud);
        }
    }

    if (rsf) {
        i32 ij = 0;
        for (i32 j = 0; j < n; j++) {
            i32 k = (j + 2 > n) ? n : j + 2;
            SLC_DGEMV("N", &nred, &k, &ONE, ti, &ldti, &a[j * lda], &int1, &ZERO, &dwork[ij], &int1);
            ij += n;
        }
    } else {
        SLC_DGEMM("N", "N", &nred, &n, &n, &ONE, ti, &ldti, a, &lda, &ZERO, dwork, &n);
    }
    SLC_DGEMM("N", "N", &nred, &nred, &n, &ONE, dwork, &n, t, &ldt, &ZERO, a, &lda);

    SLC_DLACPY("F", &n, &m, b, &ldb, dwork, &n);
    SLC_DGEMM("N", "N", &nred, &m, &n, &ONE, ti, &ldti, dwork, &n, &ZERO, b, &ldb);

    SLC_DLACPY("F", &p, &n, c, &ldc, dwork, &p);
    SLC_DGEMM("N", "N", &p, &nred, &n, &ONE, dwork, &p, t, &ldt, &ZERO, c, &ldc);

    if (spa) {
        f64 rcond;
        i32 ierr = ab09dd(dico, nred, m, p, *nr, a, lda, b, ldb, c, ldc, d, ldd,
                          &rcond, iwork, dwork);
        (void)ierr;
    } else {
        *nminr = *nr;
    }

    dwork[0] = (f64)wrkopt;
    return 0;
}
