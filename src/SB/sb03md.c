/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>

static int select_dummy(const f64* reig, const f64* ieig)
{
    (void)reig;
    (void)ieig;
    return 0;
}

void sb03md(
    const char* dico,
    const char* job,
    const char* fact,
    const char* trana,
    const i32 n,
    f64* a,
    const i32 lda,
    f64* u,
    const i32 ldu,
    f64* c,
    const i32 ldc,
    f64* scale,
    f64* sep,
    f64* ferr,
    f64* wr,
    f64* wi,
    i32* iwork,
    f64* dwork,
    const i32 ldwork,
    i32* info
)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    bool cont = (dico[0] == 'C' || dico[0] == 'c');
    bool wantx = (job[0] == 'X' || job[0] == 'x');
    bool wantsp = (job[0] == 'S' || job[0] == 's');
    bool wantbh = (job[0] == 'B' || job[0] == 'b');
    bool nofact = (fact[0] == 'N' || fact[0] == 'n');
    bool nota = (trana[0] == 'N' || trana[0] == 'n');
    bool lquery = (ldwork == -1);

    i32 nn = n * n;
    i32 nn2 = 2 * nn;

    *info = 0;

    // Parameter validation
    if (!cont && !(dico[0] == 'D' || dico[0] == 'd')) {
        *info = -1;
    } else if (!wantbh && !wantsp && !wantx) {
        *info = -2;
    } else if (!nofact && !(fact[0] == 'F' || fact[0] == 'f')) {
        *info = -3;
    } else if (!nota && !(trana[0] == 'T' || trana[0] == 't') &&
               !(trana[0] == 'C' || trana[0] == 'c')) {
        *info = -4;
    } else if (n < 0) {
        *info = -5;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -7;
    } else if (ldu < (n > 1 ? n : 1)) {
        *info = -9;
    } else if ((wantsp && ldc < 1) || (!wantsp && ldc < (n > 1 ? n : 1))) {
        *info = -11;
    } else {
        // Compute minimum workspace
        i32 minwrk;
        if (wantx) {
            if (nofact) {
                minwrk = nn > 3*n ? nn : 3*n;
            } else if (cont) {
                minwrk = nn;
            } else {
                minwrk = nn > 2*n ? nn : 2*n;
            }
        } else {
            if (cont) {
                if (nofact) {
                    minwrk = nn2 > 3*n ? nn2 : 3*n;
                } else {
                    minwrk = nn2;
                }
            } else {
                minwrk = nn2 + 2*n;
            }
        }
        minwrk = minwrk > 1 ? minwrk : 1;

        if (lquery) {
            i32 lwa;
            if (nofact) {
                i32 n_val = n;
                i32 sdim;
                i32 bwork[1];
                SLC_DGEES("V", "N", select_dummy, &n_val, a, &lda, &sdim,
                          wr, wi, u, &ldu, dwork, &(i32){-1}, bwork, info);
                lwa = minwrk > (i32)dwork[0] ? minwrk : (i32)dwork[0];
            } else {
                lwa = minwrk;
            }
            dwork[0] = (f64)lwa;
            return;
        } else if (ldwork < minwrk) {
            *info = -19;
        }
    }

    if (*info != 0) {
        return;
    }

    // Quick return
    if (n == 0) {
        *scale = ONE;
        if (!wantx) *sep = ZERO;
        if (wantbh) *ferr = ZERO;
        dwork[0] = ONE;
        return;
    }

    i32 lwa = 0;
    i32 n_val = n;

    if (nofact) {
        // Compute Schur factorization
        i32 sdim;
        i32 bwork[1];
        SLC_DGEES("V", "N", select_dummy, &n_val, a, &lda, &sdim,
                  wr, wi, u, &ldu, dwork, &ldwork, bwork, info);
        if (*info > 0) return;
        lwa = (i32)dwork[0];
    }

    if (!wantsp) {
        // Transform RHS: C <- U' * C * U
        mb01rd("U", "T", n, n, ZERO, ONE, c, ldc, u, ldu, c, ldc, dwork, ldwork, info);

        // Copy upper triangle to lower
        for (i32 i = 1; i < n; i++) {
            i32 im1 = i;
            SLC_DCOPY(&im1, &c[i * ldc], &(i32){1}, &c[i], &ldc);
        }

        lwa = lwa > nn ? lwa : nn;

        // Solve transformed equation
        i32 ierr;
        if (cont) {
            sb03my(trana, n, a, lda, c, ldc, scale, &ierr);
        } else {
            sb03mx(trana, n, a, lda, c, ldc, scale, dwork, &ierr);
        }
        if (ierr > 0) *info = n + 1;

        // Transform back: C <- U * C * U'
        mb01rd("U", "N", n, n, ZERO, ONE, c, ldc, u, ldu, c, ldc, dwork, ldwork, &ierr);

        // Copy upper triangle to lower for symmetry
        for (i32 i = 1; i < n; i++) {
            i32 im1 = i;
            SLC_DCOPY(&im1, &c[i * ldc], &(i32){1}, &c[i], &ldc);
        }
    }

    if (!wantx) {
        // Estimate separation
        char notra[2];
        if (nota) {
            notra[0] = 'T';
        } else {
            notra[0] = 'N';
        }
        notra[1] = '\0';

        f64 est = ZERO;
        i32 kase = 0;
        i32 isave[3];
        f64 scalef = ONE;

        do {
            SLC_DLACN2(&nn, &dwork[nn], dwork, iwork, &est, &kase, isave);
            if (kase != 0) {
                i32 ierr;
                if (kase == 1) {
                    if (cont) {
                        sb03my(trana, n, a, lda, dwork, n, &scalef, &ierr);
                    } else {
                        sb03mx(trana, n, a, lda, dwork, n, &scalef, &dwork[nn2], &ierr);
                    }
                } else {
                    if (cont) {
                        sb03my(notra, n, a, lda, dwork, n, &scalef, &ierr);
                    } else {
                        sb03mx(notra, n, a, lda, dwork, n, &scalef, &dwork[nn2], &ierr);
                    }
                }
            }
        } while (kase != 0);

        *sep = (scalef > 0.0) ? scalef / est : 0.0;

        if (wantbh) {
            // Compute forward error bound
            f64 eps = SLC_DLAMCH("P");
            f64 anrm = SLC_DLANHS("F", &n_val, a, &lda, dwork);
            if (cont) {
                *ferr = eps * anrm / *sep;
            } else {
                *ferr = eps * anrm * anrm / *sep;
            }
        }
    }

    i32 minwrk;
    if (wantx) {
        if (nofact) {
            minwrk = nn > 3*n ? nn : 3*n;
        } else if (cont) {
            minwrk = nn;
        } else {
            minwrk = nn > 2*n ? nn : 2*n;
        }
    } else {
        if (cont) {
            if (nofact) {
                minwrk = nn2 > 3*n ? nn2 : 3*n;
            } else {
                minwrk = nn2;
            }
        } else {
            minwrk = nn2 + 2*n;
        }
    }
    minwrk = minwrk > 1 ? minwrk : 1;

    dwork[0] = (f64)(lwa > minwrk ? lwa : minwrk);
}
