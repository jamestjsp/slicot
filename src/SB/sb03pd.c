/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 *
 * SB03PD - Discrete-time Lyapunov equation solver with separation estimation
 *
 * Solves: op(A)' * X * op(A) - X = scale * C
 * and/or estimates sepd(op(A), op(A)')
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <ctype.h>
#include <math.h>

static int select_none(const f64* reig, const f64* ieig) {
    (void)reig;
    (void)ieig;
    return 0;
}

void sb03pd(
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
    f64* sepd,
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

    char job_c = (char)toupper((unsigned char)job[0]);
    char fact_c = (char)toupper((unsigned char)fact[0]);
    char trana_c = (char)toupper((unsigned char)trana[0]);

    bool wantx = (job_c == 'X');
    bool wantsp = (job_c == 'S');
    bool wantbh = (job_c == 'B');
    bool nofact = (fact_c == 'N');
    bool nota = (trana_c == 'N');

    *info = 0;

    if (!wantbh && !wantsp && !wantx) {
        *info = -1;
    } else if (!nofact && fact_c != 'F') {
        *info = -2;
    } else if (!nota && trana_c != 'T' && trana_c != 'C') {
        *info = -3;
    } else if (n < 0) {
        *info = -4;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -6;
    } else if (ldu < (n > 1 ? n : 1)) {
        *info = -8;
    } else if ((wantsp && ldc < 1) || (!wantsp && ldc < (n > 1 ? n : 1))) {
        *info = -10;
    }

    i32 minwrk;
    if (wantx) {
        if (nofact) {
            minwrk = (n * n > 3 * n) ? n * n : 3 * n;
        } else {
            minwrk = (n * n > 2 * n) ? n * n : 2 * n;
        }
    } else {
        minwrk = 2 * n * n + 2 * n;
    }
    if (ldwork < (minwrk > 1 ? minwrk : 1)) {
        *info = -18;
    }

    if (*info != 0) {
        i32 neginfo = -(*info);
        SLC_XERBLA("SB03PD", &neginfo);
        return;
    }

    if (n == 0) {
        *scale = ONE;
        if (wantbh) {
            *ferr = ZERO;
        }
        dwork[0] = ONE;
        return;
    }

    i32 lwa = 0;
    i32 one = 1;

    if (nofact) {
        i32 sdim;
        i32 bwork_dummy[1] = {0};

        SLC_DGEES("V", "N", select_none, &n, a, &lda, &sdim,
                  wr, wi, u, &ldu, dwork, &ldwork, bwork_dummy, info);
        if (*info > 0) {
            return;
        }
        lwa = (i32)dwork[0];
    }

    if (!wantsp) {
        mb01rd("U", "T", n, n, ZERO, ONE, c, ldc, u, ldu, c, ldc, dwork, ldwork, info);

        for (i32 i = 1; i < n; i++) {
            SLC_DCOPY(&i, &c[0 + i * ldc], &one, &c[i + 0 * ldc], &ldc);
        }

        i32 info_local = 0;
        sb03mx(trana, n, a, lda, c, ldc, scale, dwork, &info_local);
        if (info_local > 0) {
            *info = n + 1;
        }

        mb01rd("U", "N", n, n, ZERO, ONE, c, ldc, u, ldu, c, ldc, dwork, ldwork, &info_local);

        for (i32 i = 1; i < n; i++) {
            SLC_DCOPY(&i, &c[0 + i * ldc], &one, &c[i + 0 * ldc], &ldc);
        }
    }

    if (!wantx) {
        char notra = nota ? 'T' : 'N';

        f64 est = ZERO;
        i32 kase = 0;
        i32 isave[3] = {0, 0, 0};
        i32 n2 = n * n;
        f64 scalef = ONE;

        do {
            SLC_DLACN2(&n2, &dwork[n2], dwork, iwork, &est, &kase, isave);
            if (kase != 0) {
                i32 ierr = 0;
                if (kase == 1) {
                    sb03mx(trana, n, a, lda, dwork, n, &scalef, &dwork[2 * n2], &ierr);
                } else {
                    sb03mx(&notra, n, a, lda, dwork, n, &scalef, &dwork[2 * n2], &ierr);
                }
            }
        } while (kase != 0);

        *sepd = scalef / est;

        if (wantbh) {
            i32 nn = n;
            f64 anorm = SLC_DLANHS("F", &nn, a, &lda, dwork);
            f64 eps = SLC_DLAMCH("P");
            *ferr = eps * anorm * anorm / (*sepd);
        }
    }

    i32 optwork = (lwa > minwrk) ? lwa : minwrk;
    dwork[0] = (f64)optwork;
}
