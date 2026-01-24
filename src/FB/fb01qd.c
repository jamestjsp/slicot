/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

/*
 * FB01QD - Time-varying square root covariance Kalman filter (dense matrices)
 *
 * Performs one recursion of the square root covariance filter algorithm:
 *
 *  |  R^{1/2}    C*S_{i-1}    0       |     | (RINOV)^{1/2}   0    0 |
 *  |    i         i                   |     |      i              |
 *  |                          1/2  |T=|                          |
 *  |   0        A*S_{i-1}  B*Q^    |   |     AK_i        S_i   0 |
 *  |             i          i     |   |                          |
 *
 * where T is an orthogonal transformation triangularizing the pre-array.
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <ctype.h>

void fb01qd(const char* jobk, const char* multbq,
            i32 n, i32 m, i32 p,
            f64* s, i32 lds,
            const f64* a, i32 lda,
            const f64* b, i32 ldb,
            const f64* q, i32 ldq,
            const f64* c, i32 ldc,
            f64* r, i32 ldr,
            f64* k, i32 ldk,
            f64 tol,
            i32* iwork,
            f64* dwork, i32 ldwork,
            i32* info)
{
    const f64 one = 1.0;
    const f64 two = 2.0;

    i32 pn = p + n;
    i32 n1 = (n > 1) ? n : 1;
    *info = 0;

    bool ljobk = (toupper((unsigned char)jobk[0]) == 'K');
    bool lmultb = (toupper((unsigned char)multbq[0]) == 'P');

    if (!ljobk && toupper((unsigned char)jobk[0]) != 'N') {
        *info = -1;
    } else if (!lmultb && toupper((unsigned char)multbq[0]) != 'N') {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    } else if (m < 0) {
        *info = -4;
    } else if (p < 0) {
        *info = -5;
    } else if (lds < n1) {
        *info = -7;
    } else if (lda < n1) {
        *info = -9;
    } else if (ldb < n1) {
        *info = -11;
    } else if (ldq < 1 || (!lmultb && ldq < m)) {
        *info = -13;
    } else if (ldc < ((p > 1) ? p : 1)) {
        *info = -15;
    } else if (ldr < ((p > 1) ? p : 1)) {
        *info = -17;
    } else if (ldk < n1) {
        *info = -19;
    } else {
        i32 ldwork_min;
        i32 val1 = pn * n + 2 * p;
        i32 val2 = n * (n + m + 2);
        if (ljobk) {
            i32 val3 = 3 * p;
            ldwork_min = (val1 > val2) ? val1 : val2;
            ldwork_min = (ldwork_min > val3) ? ldwork_min : val3;
            ldwork_min = (ldwork_min > 2) ? ldwork_min : 2;
        } else {
            ldwork_min = (val1 > val2) ? val1 : val2;
            ldwork_min = (ldwork_min > 1) ? ldwork_min : 1;
        }
        if (ldwork < ldwork_min) {
            *info = -23;
        }
    }

    if (*info != 0) {
        return;
    }

    if (n == 0) {
        if (ljobk) {
            dwork[0] = two;
            dwork[1] = one;
        } else {
            dwork[0] = one;
        }
        return;
    }

    // Construction of pre-array in DWORK
    // Store A*S and C*S in (1,1) and (2,1) blocks of DWORK

    // Copy A to DWORK[0..pn*n-1], rows 0..n-1
    SLC_DLACPY("Full", &n, &n, (f64*)a, &lda, dwork, &pn);

    // Copy C to DWORK[n..], rows n..n+p-1
    SLC_DLACPY("Full", &p, &n, (f64*)c, &ldc, dwork + n, &pn);

    // Multiply [A; C] by S (lower triangular) from right
    SLC_DTRMM("Right", "Lower", "No transpose", "Non-unit", &pn, &n,
              &one, s, &lds, dwork, &pn);

    // Step 1: annihilate C*S using MB04LD
    // MB04LD('Full', P, N, N, R, LDR, DWORK(N+1), PN, DWORK, PN, K, LDK, TAU, WORK)
    i32 itau = pn * n;
    i32 jwork = itau + p;

    mb04ld('F', p, n, n, r, ldr, dwork + n, pn, dwork, pn, k, ldk,
           dwork + itau, dwork + jwork);

    i32 wrkopt = pn * n + 2 * p;

    // Adjust workspace - copy A*S to contiguous N x N block
    // Original DWORK has leading dimension PN, we need leading dimension N
    // SLC_DLACPY("Full", &n, &n, dwork, &pn, dwork, &n) handles overlap incorrectly
    for (i32 j = 0; j < n; j++) {
        for (i32 i = 0; i < n; i++) {
            dwork[i + j * n] = dwork[i + j * pn];
        }
    }

    i32 i12 = n * n;

    // Store B*Q in (1,2) block of DWORK starting at i12
    SLC_DLACPY("Full", &n, &m, (f64*)b, &ldb, dwork + i12, &n);

    if (!lmultb) {
        // Multiply B by Q (lower triangular) from right
        SLC_DTRMM("Right", "Lower", "No transpose", "Non-unit", &n, &m,
                  &one, (f64*)q, &ldq, dwork + i12, &n);
    }

    i32 tmp = n * (n + m);
    wrkopt = (wrkopt > tmp) ? wrkopt : tmp;

    // Step 2: LQ triangularization of [A*S  B*Q] (modified at step 1)
    itau = n * (n + m);
    jwork = itau + n;

    i32 nm = n + m;
    i32 ldwork_remain = ldwork - jwork;
    SLC_DGELQF(&n, &nm, dwork, &n, dwork + itau, dwork + jwork, &ldwork_remain, info);

    tmp = (i32)dwork[jwork] + jwork;
    wrkopt = (wrkopt > tmp) ? wrkopt : tmp;

    // Output S (lower triangular part of LQ result)
    SLC_DLACPY("Lower", &n, &n, dwork, &n, s, &lds);

    if (ljobk) {
        // Compute K = AK * (RINOV)^{-1/2}
        // K already contains AK from MB04LD
        // R now contains (RINOV)^{1/2} (lower triangular)
        f64 rcond;
        mb02od("Right", "Lower", "No transpose", "Non-unit", "1-norm",
               n, p, one, r, ldr, k, ldk, &rcond, tol, iwork, dwork, info);

        if (*info == 0) {
            tmp = 3 * p;
            wrkopt = (wrkopt > tmp) ? wrkopt : tmp;
            dwork[1] = rcond;
        }
    }

    dwork[0] = (f64)wrkopt;
}
