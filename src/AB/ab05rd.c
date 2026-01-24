/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 2025, slicot.c contributors
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <ctype.h>

void ab05rd(const char* fbtype, const char* jobd, i32 n, i32 m, i32 p,
            i32 mv, i32 pz, f64 alpha, f64 beta,
            f64* a, i32 lda, f64* b, i32 ldb,
            f64* c, i32 ldc, f64* d, i32 ldd,
            const f64* f, i32 ldf, const f64* k, i32 ldk,
            const f64* g, i32 ldg, const f64* h, i32 ldh,
            f64* rcond, f64* bc, i32 ldbc, f64* cc, i32 ldcc,
            f64* dc, i32 lddc, i32* iwork, f64* dwork, i32 ldwork, i32* info) {

    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    char fbtype_upper = (char)toupper((unsigned char)fbtype[0]);
    char jobd_upper = (char)toupper((unsigned char)jobd[0]);

    bool unitf = (fbtype_upper == 'I');
    bool outpf = (fbtype_upper == 'O');
    bool ljobd = (jobd_upper == 'D');

    *info = 0;

    i32 max1n = (1 > n) ? 1 : n;
    i32 max1p = (1 > p) ? 1 : p;
    i32 max1m = (1 > m) ? 1 : m;
    i32 max1pz = (1 > pz) ? 1 : pz;

    if (!unitf && !outpf) {
        *info = -1;
    } else if (!ljobd && jobd_upper != 'Z') {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    } else if (m < 0) {
        *info = -4;
    } else if (p < 0 || (unitf && p != m)) {
        *info = -5;
    } else if (mv < 0) {
        *info = -6;
    } else if (pz < 0) {
        *info = -7;
    } else if (lda < max1n) {
        *info = -11;
    } else if (ldb < max1n) {
        *info = -13;
    } else if ((n > 0 && ldc < max1p) || (n == 0 && ldc < 1)) {
        *info = -15;
    } else if ((ljobd && ldd < max1p) || (!ljobd && ldd < 1)) {
        *info = -17;
    } else if ((outpf && alpha != ZERO && ldf < max1m) ||
               ((unitf || alpha == ZERO) && ldf < 1)) {
        *info = -19;
    } else if ((beta != ZERO && ldk < max1m) || (beta == ZERO && ldk < 1)) {
        *info = -21;
    } else if (ldg < max1m) {
        *info = -23;
    } else if (ldh < max1pz) {
        *info = -25;
    } else if (ldbc < max1n) {
        *info = -28;
    } else if ((n > 0 && ldcc < max1pz) || (n == 0 && ldcc < 1)) {
        *info = -30;
    } else if ((ljobd && lddc < max1pz) || (!ljobd && lddc < 1)) {
        *info = -32;
    } else {
        i32 pmv = p * mv;
        i32 pp4p = p * p + 4 * p;
        i32 wspace;
        if (ljobd) {
            wspace = max1m;
            if (pmv > wspace) wspace = pmv;
            if (pp4p > wspace) wspace = pp4p;
            if (wspace < 1) wspace = 1;
        } else {
            wspace = max1m;
            if (wspace < 1) wspace = 1;
        }
        if (ldwork < wspace) {
            *info = -35;
        }
    }

    if (*info != 0) {
        return;
    }

    i32 minmp = (m < p) ? m : p;
    i32 minmvpz = (mv < pz) ? mv : pz;
    i32 maxnminmp = (n > minmp) ? n : minmp;
    i32 maxnminmvpz = (n > minmvpz) ? n : minmvpz;
    if (maxnminmp == 0 || maxnminmvpz == 0) {
        *rcond = ONE;
        return;
    }

    ab05sd(fbtype, jobd, n, m, p, alpha, a, lda, b, ldb, c, ldc, d, ldd,
           f, ldf, rcond, iwork, dwork, ldwork, info);
    if (*info != 0) {
        return;
    }

    if (beta != ZERO && n > 0) {
        SLC_DGEMM("N", "N", &n, &n, &m, &beta, b, &ldb, k, &ldk, &ONE, a, &lda);
        if (ljobd) {
            SLC_DGEMM("N", "N", &p, &n, &m, &beta, d, &ldd, k, &ldk, &ONE, c, &ldc);
        }
    }

    SLC_DGEMM("N", "N", &n, &mv, &m, &ONE, b, &ldb, g, &ldg, &ZERO, bc, &ldbc);

    if (n > 0) {
        SLC_DGEMM("N", "N", &pz, &n, &p, &ONE, h, &ldh, c, &ldc, &ZERO, cc, &ldcc);
    }

    if (ljobd) {
        i32 ldwp = (1 > p) ? 1 : p;
        SLC_DGEMM("N", "N", &p, &mv, &m, &ONE, d, &ldd, g, &ldg, &ZERO, dwork, &ldwp);
        SLC_DGEMM("N", "N", &pz, &mv, &p, &ONE, h, &ldh, dwork, &ldwp, &ZERO, dc, &lddc);
    }
}
