/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>

void ab09ax(
    const char* dico,
    const char* job,
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
    f64* t,
    i32 ldt,
    f64* ti,
    i32 ldti,
    f64 tol,
    i32* iwork,
    f64* dwork,
    i32 ldwork,
    i32* iwarn,
    i32* info
)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    *info = 0;
    *iwarn = 0;

    bool discr = (dico[0] == 'D' || dico[0] == 'd');
    bool conti = (dico[0] == 'C' || dico[0] == 'c');
    bool bal = (job[0] == 'B' || job[0] == 'b');
    bool bfree = (job[0] == 'N' || job[0] == 'n');
    bool fixord = (ordsel[0] == 'F' || ordsel[0] == 'f');
    bool autosel = (ordsel[0] == 'A' || ordsel[0] == 'a');

    if (!conti && !discr) {
        *info = -1;
    } else if (!bal && !bfree) {
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
    } else {
        i32 max1n = (1 > n) ? 1 : n;
        i32 max1p = (1 > p) ? 1 : p;
        if (lda < max1n) {
            *info = -9;
        } else if (ldb < max1n) {
            *info = -11;
        } else if (ldc < max1p) {
            *info = -13;
        } else if (ldt < max1n) {
            *info = -16;
        } else if (ldti < max1n) {
            *info = -18;
        } else {
            i32 max_nmp = n;
            if (m > max_nmp) max_nmp = m;
            if (p > max_nmp) max_nmp = p;
            i32 minwork = n * (max_nmp + 5) + (n * (n + 1)) / 2;
            if (minwork < 1) minwork = 1;
            if (ldwork < minwork) {
                *info = -22;
            }
        }
    }

    if (*info != 0) {
        i32 neginfo = -(*info);
        SLC_XERBLA("AB09AX", &neginfo);
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

    f64 rtol = (f64)n * SLC_DLAMCH("Epsilon");

    i32 max_nmp = n;
    if (m > max_nmp) max_nmp = m;
    if (p > max_nmp) max_nmp = p;

    i32 ku = 0;
    i32 ktau = ku + n * max_nmp;
    i32 kw = ktau + n;
    i32 ldw = ldwork - kw;

    SLC_DLACPY("Full", &n, &m, b, &ldb, &dwork[ku], &n);

    f64 scalec;
    i32 ierr = 0;
    sb03ou(discr, true, n, m, a, lda, &dwork[ku], n, &dwork[ktau], ti, ldti, &scalec, &dwork[kw], ldw, &ierr);
    if (ierr != 0) {
        *info = 1;
        return;
    }
    i32 wrkopt = (i32)dwork[kw] + kw;

    SLC_DLACPY("Full", &p, &n, c, &ldc, &dwork[ku], &p);

    f64 scaleo;
    sb03ou(discr, false, n, p, a, lda, &dwork[ku], p, &dwork[ktau], t, ldt, &scaleo, &dwork[kw], ldw, &ierr);
    i32 wrkopt2 = (i32)dwork[kw] + kw;
    if (wrkopt2 > wrkopt) wrkopt = wrkopt2;

    i32 kv = ktau;
    bool packed = (ldwork - kv < n * (n + 5));

    if (packed) {
        ma02dd("Pack", "Upper", n, ti, ldti, &dwork[kv]);
        kw = kv + (n * (n + 1)) / 2;
    } else {
        SLC_DLACPY("Upper", &n, &n, ti, &ldti, &dwork[kv], &n);
        kw = kv + n * n;
    }

    for (i32 j = 0; j < n; j++) {
        i32 jj = j + 1;
        SLC_DTRMV("Upper", "NoTranspose", "NonUnit", &jj, t, &ldt, &ti[0 + j * ldti], &(i32){1});
    }

    ldw = ldwork - kw;
    i32 mb03ud_info = 0;
    mb03ud('V', 'V', n, ti, ldti, &dwork[ku], n, hsv, &dwork[kw], ldw, &mb03ud_info);
    if (mb03ud_info != 0) {
        *info = 2;
        return;
    }
    wrkopt2 = (i32)dwork[kw] + kw;
    if (wrkopt2 > wrkopt) wrkopt = wrkopt2;

    f64 scale_factor = ONE / (scalec * scaleo);
    SLC_DSCAL(&n, &scale_factor, hsv, &(i32){1});

    f64 atol = rtol * hsv[0];
    if (fixord) {
        if (*nr > 0) {
            if (hsv[*nr - 1] <= atol) {
                *nr = 0;
                *iwarn = 1;
                fixord = false;
            }
        }
    } else {
        if (tol > atol) atol = tol;
        *nr = 0;
    }
    if (!fixord) {
        for (i32 j = 0; j < n; j++) {
            if (hsv[j] <= atol) break;
            (*nr)++;
        }
    }

    if (*nr == 0) {
        dwork[0] = (f64)wrkopt;
        return;
    }

    SLC_DTRMM("Left", "Upper", "Transpose", "NonUnit", &n, nr, &ONE, t, &ldt, &dwork[ku], &n);

    ma02ad("Full", *nr, n, ti, ldti, t, ldt);

    if (packed) {
        for (i32 j = 0; j < *nr; j++) {
            SLC_DTPMV("Upper", "NoTranspose", "NonUnit", &n, &dwork[kv], &t[0 + j * ldt], &(i32){1});
        }
    } else {
        SLC_DTRMM("Left", "Upper", "NoTranspose", "NonUnit", &n, nr, &ONE, &dwork[kv], &n, t, &ldt);
    }

    if (bal) {
        i32 ij = ku;
        for (i32 j = 0; j < *nr; j++) {
            f64 temp = ONE / sqrt(hsv[j]);
            SLC_DSCAL(&n, &temp, &t[0 + j * ldt], &(i32){1});
            SLC_DSCAL(&n, &temp, &dwork[ij], &(i32){1});
            ij += n;
        }
    } else {
        kw = ktau + *nr;
        ldw = ldwork - kw;

        SLC_DGEQRF(&n, nr, t, &ldt, &dwork[ktau], &dwork[kw], &ldw, &ierr);
        SLC_DORGQR(&n, nr, nr, t, &ldt, &dwork[ktau], &dwork[kw], &ldw, &ierr);

        SLC_DGEQRF(&n, nr, &dwork[ku], &n, &dwork[ktau], &dwork[kw], &ldw, &ierr);
        wrkopt2 = (i32)dwork[kw] + kw;
        if (wrkopt2 > wrkopt) wrkopt = wrkopt2;

        SLC_DORGQR(&n, nr, nr, &dwork[ku], &n, &dwork[ktau], &dwork[kw], &ldw, &ierr);
        wrkopt2 = (i32)dwork[kw] + kw;
        if (wrkopt2 > wrkopt) wrkopt = wrkopt2;
    }

    ma02ad("Full", n, *nr, &dwork[ku], n, ti, ldti);

    if (!bal) {
        SLC_DGEMM("NoTranspose", "NoTranspose", nr, nr, &n, &ONE, ti, &ldti, t, &ldt, &ZERO, &dwork[ku], &n);
        SLC_DGETRF(nr, nr, &dwork[ku], &n, iwork, &ierr);
        SLC_DGETRS("NoTranspose", nr, &n, &dwork[ku], &n, iwork, ti, &ldti, &ierr);
    }

    i32 ij = ku;
    for (i32 j = 0; j < n; j++) {
        i32 k = (j + 1 < n) ? j + 2 : n;
        SLC_DGEMV("NoTranspose", nr, &k, &ONE, ti, &ldti, &a[0 + j * lda], &(i32){1}, &ZERO, &dwork[ij], &(i32){1});
        ij += n;
    }
    SLC_DGEMM("NoTranspose", "NoTranspose", nr, nr, &n, &ONE, &dwork[ku], &n, t, &ldt, &ZERO, a, &lda);

    SLC_DLACPY("Full", &n, &m, b, &ldb, &dwork[ku], &n);
    SLC_DGEMM("NoTranspose", "NoTranspose", nr, &m, &n, &ONE, ti, &ldti, &dwork[ku], &n, &ZERO, b, &ldb);

    SLC_DLACPY("Full", &p, &n, c, &ldc, &dwork[ku], &p);
    SLC_DGEMM("NoTranspose", "NoTranspose", &p, nr, &n, &ONE, &dwork[ku], &p, t, &ldt, &ZERO, c, &ldc);

    dwork[0] = (f64)wrkopt;
}
