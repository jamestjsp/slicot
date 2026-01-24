/*
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * SB08CD - Left coprime factorization with inner denominator.
 *
 * Constructs, for a given system G = (A,B,C,D), an output injection matrix H,
 * an orthogonal transformation matrix Z, and a gain matrix V, such that
 *
 *     Q = (Z'*(A+H*C)*Z, Z'*(B+H*D), V*C*Z, V*D)
 * and
 *     R = (Z'*(A+H*C)*Z, Z'*H, V*C*Z, V)
 *
 * provide a stable left coprime factorization of G in the form G = R^{-1} * Q.
 *
 * Uses the right coprime factorization algorithm applied to G'.
 */

#include "slicot.h"
#include "slicot_blas.h"

void sb08cd(
    const char* dico,
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
    i32* nr,
    f64* br,
    const i32 ldbr,
    f64* dr,
    const i32 lddr,
    const f64 tol,
    f64* dwork,
    const i32 ldwork,
    i32* iwarn,
    i32* info)
{
    const f64 ONE = 1.0;
    const f64 ZERO = 0.0;

    *iwarn = 0;
    *info = 0;

    i32 max1n = (n > 1) ? n : 1;
    i32 maxmp = (m > p) ? m : p;
    i32 max1mp = (maxmp > 1) ? maxmp : 1;
    i32 max1p = (p > 1) ? p : 1;

    if (!(*dico == 'C' || *dico == 'c') && !(*dico == 'D' || *dico == 'd')) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (m < 0) {
        *info = -3;
    } else if (p < 0) {
        *info = -4;
    } else if (lda < max1n) {
        *info = -6;
    } else if (ldb < max1n) {
        *info = -8;
    } else if ((n > 0 && ldc < max1mp) || (n == 0 && ldc < 1)) {
        *info = -10;
    } else if (ldd < max1mp) {
        *info = -12;
    } else if (ldbr < max1n) {
        *info = -16;
    } else if (lddr < max1p) {
        *info = -18;
    } else {
        i32 minwrk1 = n * (n + 5);
        i32 minwrk2 = p * (p + 2);
        i32 minwrk3 = 4 * p;
        i32 minwrk4 = 4 * m;
        i32 minwrk = minwrk1;
        if (minwrk2 > minwrk) minwrk = minwrk2;
        if (minwrk3 > minwrk) minwrk = minwrk3;
        if (minwrk4 > minwrk) minwrk = minwrk4;
        i32 ldwork_req = p * n + (minwrk > 1 ? minwrk : 1);
        if (ldwork < ldwork_req) {
            *info = -21;
        }
    }

    if (*info != 0) {
        i32 neginfo = -(*info);
        SLC_XERBLA("SB08CD", &neginfo);
        return;
    }

    i32 minnp = (n < p) ? n : p;
    if (minnp == 0) {
        *nq = 0;
        *nr = 0;
        dwork[0] = ONE;
        SLC_DLASET("Full", &p, &p, &ZERO, &ONE, dr, &lddr);
        return;
    }

    i32 ab_info = ab07md('D', n, m, p, a, lda, b, ldb, c, ldc, d, ldd);
    if (ab_info != 0) {
        *info = ab_info;
        return;
    }

    i32 kbr = 0;
    i32 kw = kbr + p * n;

    sb08dd(dico, n, p, m, a, lda, b, ldb, c, ldc, d, ldd,
           nq, nr, &dwork[kbr], p, dr, lddr, tol, &dwork[kw],
           ldwork - kw, iwarn, info);

    if (*info == 0) {
        i32 nq_val = *nq;
        i32 kl = (nq_val > 1) ? nq_val - 1 : 0;
        i32 ku = (nq_val > 1) ? nq_val - 1 : 0;

        tb01xd("D", nq_val, p, m, kl, ku, a, lda, b, ldb, c, ldc, d, ldd, info);

        ma02ad("Full", p, nq_val, &dwork[kbr], p, br, ldbr);

        ma02bd('L', nq_val, p, br, ldbr);

        i32 one = 1;
        for (i32 i = 1; i < p; i++) {
            SLC_DSWAP(&i, &dr[i], &lddr, &dr[i * lddr], &one);
        }
    }

    dwork[0] = dwork[kw] + (f64)kw;
}
