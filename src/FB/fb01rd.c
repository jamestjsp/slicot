/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <ctype.h>


void fb01rd(const char* jobk, const char* multbq,
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
    const f64 one = 1.0, two = 2.0;

    i32 pn = p + n;
    i32 n1 = (n > 1) ? n : 1;
    *info = 0;

    bool ljobk = (toupper((unsigned char)jobk[0]) == 'K');
    bool lmultb = (toupper((unsigned char)multbq[0]) == 'P');

    // Parameter validation
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
        i32 val1 = pn * n + n;
        i32 val2 = pn * n + 2 * p;
        i32 val3 = n * (n + m + 2);
        if (ljobk) {
            ldwork_min = (val1 > val2) ? val1 : val2;
            ldwork_min = (ldwork_min > val3) ? ldwork_min : val3;
            ldwork_min = (ldwork_min > 3 * p) ? ldwork_min : (3 * p);
            ldwork_min = (ldwork_min > 2) ? ldwork_min : 2;
        } else {
            ldwork_min = (val1 > val2) ? val1 : val2;
            ldwork_min = (ldwork_min > val3) ? ldwork_min : val3;
            ldwork_min = (ldwork_min > 1) ? ldwork_min : 1;
        }
        if (ldwork < ldwork_min) {
            *info = -23;
        }
    }

    if (*info != 0) {
        return;
    }

    // Quick return if n == 0
    if (n == 0) {
        if (ljobk) {
            dwork[0] = two;
            dwork[1] = one;
        } else {
            dwork[0] = one;
        }
        return;
    }

    // Construction of the needed part of the pre-array in DWORK.
    // To save workspace, only the blocks (1,3), (2,2), and (2,3) will be
    // constructed.
    //
    // Storing C x S and A x S in the (1,1) and (2,1) blocks of DWORK,
    // respectively. The lower trapezoidal structure of [ C' A' ]' is
    // fully exploited.
    //
    // If P <= N, use partition:
    //   [ C1  0  ] [ S1  0  ]
    //   [ A1  A3 ] [ S2  S3 ],
    //   [ A2  A4 ]
    // where C1, S1, A2 are P-by-P, A1 and S2 are (N-P)-by-P, etc.
    //
    // If P > N, use partition:
    //   [ C1 ]
    //   [ C2 ] [ S ],
    //   [ A  ]

    // Workspace: need (P+N)*N.
    i32 min_np = (n < p) ? n : p;
    SLC_DLACPY("Lower", &p, &min_np, (f64*)c, &ldc, dwork, &pn);
    SLC_DLACPY("Full", &n, &min_np, (f64*)a, &lda, dwork + p, &pn);

    if (n > p) {
        i32 n_minus_p = n - p;
        SLC_DLACPY("Lower", &n, &n_minus_p, (f64*)a + p * lda, &lda,
                   dwork + p * pn + p, &pn);
    }

    // Compute [C1 0; A1 A3] x S or C1 x S as a product of lower triangular matrices.
    // Workspace: need (P+N+1)*N.
    i32 ii = 0;
    i32 pl = n * pn;  // 0-based offset for temp work
    i32 wrkopt = pl + n;

    for (i32 i = 0; i < n; i++) {
        i32 len = n - i;
        SLC_DCOPY(&len, s + i + i * lds, &(i32){1}, dwork + pl, &(i32){1});
        SLC_DTRMV("Lower", "No transpose", "Non-unit", &len,
                  dwork + ii, &pn, dwork + pl, &(i32){1});
        SLC_DCOPY(&len, dwork + pl, &(i32){1}, dwork + ii, &(i32){1});
        ii = ii + pn + 1;
    }

    // Compute [ A2  A4 ] x S.
    SLC_DTRMM("Right", "Lower", "No transpose", "Non-unit", &p, &n,
              &one, s, &lds, dwork + n, &pn);

    // Triangularization (2 steps).
    // Step 1: annihilate the matrix C x S (hence C1 x S1, if P <= N).
    // Workspace: need (N+P)*N + 2*P.
    i32 itau = pl;          // 0-based
    i32 jwork = itau + p;   // 0-based

    mb04ld('L', p, n, n, r, ldr, dwork, pn, dwork + p, pn,
           k, ldk, dwork + itau, dwork + jwork);
    wrkopt = (wrkopt > pn * n + 2 * p) ? wrkopt : (pn * n + 2 * p);

    // Now, the workspace for C x S is no longer needed.
    // Adjust the leading dimension of DWORK, to save space for the
    // following computations, and make room for B x Q.
    // SLC_DLACPY("Full", &n, &n, dwork + p, &pn, dwork, &n);
    for (i32 j = 0; j < n; j++) {
        for (i32 i = 0; i < n; i++) {
            dwork[i + j * n] = dwork[i + p + j * pn];
        }
    }

    // DO 20 I = N*( N - 1 ) + 1, 1, -N
    //    CALL DCOPY( N, DWORK(I), 1, DWORK(I+N*M), 1 )
    for (i32 i = n * (n - 1); i >= 0; i -= n) {
        SLC_DCOPY(&n, dwork + i, &(i32){1}, dwork + i + n * m, &(i32){1});
    }

    // Storing B x Q in the (1,1) block of DWORK.
    // Workspace: need N*(M+N).
    SLC_DLACPY("Full", &n, &m, (f64*)b, &ldb, dwork, &n);
    if (!lmultb) {
        SLC_DTRMM("Right", "Lower", "No transpose", "Non-unit", &n, &m,
                  &one, (f64*)q, &ldq, dwork, &n);
    }

    // Step 2: LQ triangularization of the matrix [ B x Q  A x S ], where
    // A x S was modified at Step 1.
    // Workspace: need   N*(N+M+2);
    //            prefer N*(N+M+1)+(P+1)*NB.
    itau = n * (m + n);     // 0-based
    jwork = itau + n;       // 0-based

    i32 nm = m + n;
    i32 nrows_zero = (n - p - 1 > 0) ? (n - p - 1) : 0;
    i32 ldwork_remaining = ldwork - jwork;

    mb04jd(n, nm, nrows_zero, 0, dwork, n, dwork, n,
           dwork + itau, dwork + jwork, ldwork_remaining, info);

    if (*info == 0) {
        i32 optimal = (i32)dwork[jwork] + jwork;
        wrkopt = (wrkopt > optimal) ? wrkopt : optimal;
    }

    // Output S and K (if needed) and set the optimal workspace
    // dimension (and the reciprocal of the condition number estimate).
    SLC_DLACPY("Lower", &n, &n, dwork, &n, s, &lds);

    if (ljobk) {
        // Compute K.
        // Workspace: need 3*P.
        f64 rcond = 0.0;
        mb02od("Right", "Lower", "No transpose", "Non-unit", "1-norm",
               n, p, one, r, ldr, k, ldk, &rcond, tol, iwork, dwork, info);
        if (*info == 0) {
            wrkopt = (wrkopt > 3 * p) ? wrkopt : (3 * p);
            dwork[1] = rcond;
        }
    }

    dwork[0] = (f64)wrkopt;
}
