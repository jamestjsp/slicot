/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>

f64 ab13bd(
    const char* dico,
    const char* jobn,
    const i32 n,
    const i32 m,
    const i32 p,
    f64* a,
    const i32 lda,
    f64* b,
    const i32 ldb,
    f64* c,
    const i32 ldc,
    f64* d,
    const i32 ldd,
    i32* nq,
    const f64 tol,
    f64* dwork,
    const i32 ldwork,
    i32* iwarn,
    i32* info
)
{
    const f64 ONE = 1.0;
    const f64 ZERO = 0.0;

    bool discr = (*dico == 'D' || *dico == 'd');
    *info = 0;
    *iwarn = 0;

    if (!(*dico == 'C' || *dico == 'c') && !discr) {
        *info = -1;
    } else if (!(*jobn == 'H' || *jobn == 'h') && !(*jobn == 'L' || *jobn == 'l')) {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    } else if (m < 0) {
        *info = -4;
    } else if (p < 0) {
        *info = -5;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -7;
    } else if (ldb < (n > 1 ? n : 1)) {
        *info = -9;
    } else if (ldc < (p > 1 ? p : 1)) {
        *info = -11;
    } else if (ldd < (p > 1 ? p : 1)) {
        *info = -13;
    } else {
        i32 mxnp = n > p ? n : p;
        i32 minnp = n < p ? n : p;
        i32 req1 = m * (n + m) + (n * (n + 5) > m * (m + 2) ? n * (n + 5) : m * (m + 2));
        req1 = req1 > 4 * p ? req1 : 4 * p;
        i32 req2 = n * (mxnp + 4) + minnp;
        i32 minwrk = req1 > req2 ? req1 : req2;
        minwrk = minwrk > 1 ? minwrk : 1;
        if (ldwork < minwrk) {
            *info = -17;
        }
    }

    if (*info != 0) {
        i32 neginfo = -(*info);
        SLC_XERBLA("AB13BD", &neginfo);
        return ZERO;
    }

    f64 s2norm = SLC_DLANGE("Frobenius", &p, &m, d, &ldd, dwork);
    if (!discr && s2norm != ZERO) {
        *info = 5;
        return ZERO;
    }

    i32 minn = m < n ? m : n;
    minn = minn < p ? minn : p;
    if (minn == 0) {
        *nq = 0;
        dwork[0] = ONE;
        return ZERO;
    }

    i32 kcr = 0;
    i32 kdr = kcr + m * n;
    i32 krw = kdr + m * m;

    i32 nr;
    sb08dd(dico, n, m, p, a, lda, b, ldb, c, ldc, d, ldd, nq, &nr,
           &dwork[kcr], m, &dwork[kdr], m, tol, &dwork[krw], ldwork - krw, iwarn, info);
    if (*info != 0) {
        return ZERO;
    }

    f64 wrkopt = dwork[krw] + (f64)krw;

    if ((*jobn == 'H' || *jobn == 'h') && nr > 0) {
        *info = 6;
        return ZERO;
    }

    if (*nq > 0) {
        i32 mxnp = *nq > p ? *nq : p;
        i32 ku = 0;
        i32 ktau = (*nq) * mxnp;
        krw = ktau + (*nq < p ? *nq : p);

        SLC_DLACPY("Full", &p, nq, c, &ldc, &dwork[ku], &mxnp);

        f64 scale;
        sb03ou(discr, false, *nq, p, a, lda, &dwork[ku], mxnp,
               &dwork[ktau], &dwork[ku], *nq, &scale, &dwork[krw], ldwork - krw, info);
        if (*info != 0) {
            if (*info == 1) {
                *info = 4;
            } else if (*info == 2) {
                *info = 3;
            }
            return ZERO;
        }

        f64 opt2 = dwork[krw] + (f64)krw;
        wrkopt = wrkopt > opt2 ? wrkopt : opt2;

        ktau = (*nq) * (*nq);
        SLC_DLACPY("Full", nq, &m, b, &ldb, &dwork[ktau], nq);
        SLC_DTRMM("Left", "Upper", "N", "NonUnit", nq, &m, &ONE, &dwork[ku], nq,
                  &dwork[ktau], nq);

        if (nr > 0) {
            s2norm = SLC_DLANGE("Frobenius", &p, &m, d, &ldd, dwork);
        }

        f64 bx_norm = SLC_DLANGE("Frobenius", nq, &m, &dwork[ktau], nq, &dwork[krw]) / scale;
        s2norm = SLC_DLAPY2(&s2norm, &bx_norm);
    }

    dwork[0] = wrkopt;

    return s2norm;
}
