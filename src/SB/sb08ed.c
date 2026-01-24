/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

/*
 * SB08ED - Left coprime factorization with prescribed stability degree
 *
 * Purpose:
 *   To construct, for a given system G = (A,B,C,D), an output
 *   injection matrix H and an orthogonal transformation matrix Z, such
 *   that the systems
 *
 *        Q = (Z'*(A+H*C)*Z, Z'*(B+H*D), C*Z, D)
 *   and
 *        R = (Z'*(A+H*C)*Z, Z'*H, C*Z, I)
 *
 *   provide a stable left coprime factorization of G in the form
 *                  -1
 *             G = R  * Q,
 *
 *   where G, Q and R are the corresponding transfer-function matrices.
 */

#include "slicot.h"
#include "slicot_blas.h"

void sb08ed(
    const char* dico,
    const i32 n,
    const i32 m,
    const i32 p,
    const f64* alpha,
    f64* a,
    const i32 lda,
    f64* b,
    const i32 ldb,
    f64* c,
    const i32 ldc,
    f64* d,
    const i32 ldd,
    i32* nq,
    i32* nr,
    f64* br,
    const i32 ldbr,
    f64* dr,
    const i32 lddr,
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
    *iwarn = 0;
    *info = 0;

    if (!(*dico == 'C' || *dico == 'c') && !discr) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (m < 0) {
        *info = -3;
    } else if (p < 0) {
        *info = -4;
    } else if ((discr && (alpha[0] < ZERO || alpha[0] >= ONE ||
                          alpha[1] < ZERO || alpha[1] >= ONE)) ||
               (!discr && (alpha[0] >= ZERO || alpha[1] >= ZERO))) {
        *info = -5;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -7;
    } else if (ldb < (n > 1 ? n : 1)) {
        *info = -9;
    } else if (ldc < 1 || (n > 0 && ldc < (m > p ? m : p))) {
        *info = -11;
    } else if (ldd < (1 > (m > p ? m : p) ? 1 : (m > p ? m : p))) {
        *info = -13;
    } else if (ldbr < (n > 1 ? n : 1)) {
        *info = -17;
    } else if (lddr < (p > 1 ? p : 1)) {
        *info = -19;
    } else {
        i32 min1 = n * (n + 5);
        i32 min2 = 5 * p;
        i32 min3 = 4 * m;
        i32 minwrk = min1 > min2 ? min1 : min2;
        minwrk = minwrk > min3 ? minwrk : min3;
        minwrk = n * p + minwrk;
        minwrk = minwrk > 1 ? minwrk : 1;
        if (ldwork < minwrk) {
            *info = -22;
        }
    }

    if (*info != 0) {
        i32 neginfo = -(*info);
        SLC_XERBLA("SB08ED", &neginfo);
        return;
    }

    i32 minp = (n < p) ? n : p;
    if (minp == 0) {
        *nq = 0;
        *nr = 0;
        dwork[0] = ONE;
        SLC_DLASET("Full", &p, &p, &ZERO, &ONE, dr, &lddr);
        return;
    }

    ab07md('D', n, m, p, a, lda, b, ldb, c, ldc, d, ldd);

    i32 kbr = 0;
    i32 kw = kbr + p * n;
    i32 ldw_remaining = ldwork - kw;

    sb08fd(dico, n, p, m, alpha, a, lda, b, ldb, c, ldc, d, ldd,
           nq, nr, &dwork[kbr], p, dr, lddr, tol, &dwork[kw],
           ldw_remaining, iwarn, info);

    if (*info == 0) {
        i32 nqm1 = *nq > 0 ? *nq - 1 : 0;
        tb01xd("D", *nq, p, m, nqm1, nqm1, a, lda, b, ldb, c, ldc, d, ldd, info);

        ma02ad("Full", p, *nq, &dwork[kbr], p, br, ldbr);
        ma02bd('L', *nq, p, br, ldbr);
    }

    dwork[0] = dwork[kw] + (f64)kw;
}
