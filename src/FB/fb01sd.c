/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <ctype.h>
#include <string.h>


void fb01sd(const char* jobx, const char* multab, const char* multrc,
            i32 n, i32 m, i32 p,
            f64* sinv, i32 ldsinv,
            const f64* ainv, i32 ldainv,
            const f64* b, i32 ldb,
            const f64* rinv, i32 ldrinv,
            const f64* c, i32 ldc,
            f64* qinv, i32 ldqinv,
            f64* x, const f64* rinvy, const f64* z, f64* e,
            f64 tol,
            i32* iwork,
            f64* dwork, i32 ldwork,
            i32* info)
{
    const f64 zero = 0.0, one = 1.0, two = 2.0;

    i32 np = n + p;
    i32 n1 = (n > 1) ? n : 1;
    i32 m1 = (m > 1) ? m : 1;
    *info = 0;

    bool ljobx = (toupper((unsigned char)jobx[0]) == 'X');
    bool lmulta = (toupper((unsigned char)multab[0]) == 'P');
    bool lmultr = (toupper((unsigned char)multrc[0]) == 'P');

    if (!ljobx && toupper((unsigned char)jobx[0]) != 'N') {
        *info = -1;
    } else if (!lmulta && toupper((unsigned char)multab[0]) != 'N') {
        *info = -2;
    } else if (!lmultr && toupper((unsigned char)multrc[0]) != 'N') {
        *info = -3;
    } else if (n < 0) {
        *info = -4;
    } else if (m < 0) {
        *info = -5;
    } else if (p < 0) {
        *info = -6;
    } else if (ldsinv < n1) {
        *info = -8;
    } else if (ldainv < n1) {
        *info = -10;
    } else if (ldb < n1) {
        *info = -12;
    } else if (ldrinv < 1 || (!lmultr && ldrinv < p)) {
        *info = -14;
    } else if (ldc < ((p > 1) ? p : 1)) {
        *info = -16;
    } else if (ldqinv < m1) {
        *info = -18;
    } else {
        i32 ldwork_min;
        if (ljobx) {
            i32 val1 = n*(n + 2*m) + 3*m;
            i32 val2 = np*(n + 1) + 2*n;
            i32 val3 = 3*n;
            ldwork_min = val1 > val2 ? val1 : val2;
            ldwork_min = ldwork_min > val3 ? ldwork_min : val3;
            ldwork_min = ldwork_min > 2 ? ldwork_min : 2;
        } else {
            i32 val1 = n*(n + 2*m) + 3*m;
            i32 val2 = np*(n + 1) + 2*n;
            ldwork_min = val1 > val2 ? val1 : val2;
            ldwork_min = ldwork_min > 1 ? ldwork_min : 1;
        }
        if (ldwork < ldwork_min) {
            *info = -26;
        }
    }

    if (*info != 0) {
        return;
    }

    // Quick return
    if (n == 0 || (n == 0 && p == 0)) {
        if (p > 0) {
            SLC_DCOPY(&p, (f64*)rinvy, &(i32){1}, e, &(i32){1});
        }
        if (ljobx) {
            dwork[0] = two;
            dwork[1] = one;
        } else {
            dwork[0] = one;
        }
        return;
    }

    // Workspace layout (all 1-based indices converted to 0-based):
    // DWORK[0..N*N-1]: SINV*AINV (N x N)
    // DWORK[I21..I21+N*M-1]: SINV*AINV*B (N x M)
    // DWORK[I13..]: Z transformed
    // DWORK[I12..]: C block from MB04KD

    i32 ldw = n1;
    i32 i21 = n*n;  // 0-based offset for block (2,1)

    // Copy AINV to workspace
    SLC_DLACPY("Full", &n, &n, (f64*)ainv, &ldainv, dwork, &ldw);

    // If MULTAB='N', compute AINV*B, else just copy B
    if (lmulta) {
        SLC_DLACPY("Full", &n, &m, (f64*)b, &ldb, dwork + i21, &ldw);
    } else {
        SLC_DGEMM("N", "N", &n, &m, &n, &one, dwork, &ldw,
                  (f64*)b, &ldb, &zero, dwork + i21, &ldw);
    }

    // Compute SINV * [AINV | AINV*B] (i.e., multiply N columns of SINV with first N+M columns)
    i32 nm = n + m;
    SLC_DTRMM("Left", "Upper", "No transpose", "Non-unit", &n, &nm,
              &one, sinv, &ldsinv, dwork, &ldw);

    // Store QINV * Z in (1,3) block
    i32 i13 = n*(n + m);  // 0-based offset
    SLC_DCOPY(&m, (f64*)z, &(i32){1}, dwork + i13, &(i32){1});
    SLC_DTRMV("Upper", "No transpose", "Non-unit", &m, qinv, &ldqinv,
              dwork + i13, &(i32){1});

    // Computing SINV * X in X
    SLC_DTRMV("Upper", "No transpose", "Non-unit", &n, sinv, &ldsinv, x, &(i32){1});

    // Step 1: annihilate SINV*AINV*B using MB04KD
    // MB04KD('F', M, N, N, QINV, LDQINV, DWORK(I21), LDW, DWORK, LDW, DWORK(I12), M1, TAU, DWORK(JWORK))
    i32 i12 = i13 + m;     // 0-based
    i32 itau = i12 + m*n;  // 0-based
    i32 jwork = itau + m;  // 0-based

    mb04kd('F', m, n, n, qinv, ldqinv, dwork + i21, ldw,
           dwork, ldw, dwork + i12, m1, dwork + itau, dwork + jwork);

    i32 wrkopt = n*(n + 2*m) + 3*m;
    wrkopt = wrkopt > 1 ? wrkopt : 1;

    if (n == 0) {
        SLC_DCOPY(&p, (f64*)rinvy, &(i32){1}, e, &(i32){1});
        if (ljobx) {
            dwork[1] = one;
        }
        dwork[0] = (f64)wrkopt;
        return;
    }

    // Apply transformations to last column of pre-array
    // Loop DO 10 I = 1, M
    i32 ij = i21;  // 0-based, points to DWORK(I21)
    for (i32 i = 0; i < m; i++) {
        // dot = DDOT(N, DWORK(IJ), 1, X, 1)
        f64 dot_result = SLC_DDOT(&n, dwork + ij, &(i32){1}, x, &(i32){1});
        // scale = -TAU[i] * (DWORK(I13+i) + dot)
        f64 scale = -dwork[itau + i] * (dwork[i13 + i] + dot_result);
        // DAXPY(N, scale, DWORK(IJ), 1, X, 1)
        SLC_DAXPY(&n, &scale, dwork + ij, &(i32){1}, x, &(i32){1});
        ij += n;
    }

    // Copy updated X to DWORK(I21)
    SLC_DCOPY(&n, x, &(i32){1}, dwork + i21, &(i32){1});
    ldw = (np > 1) ? np : 1;

    // Rearrange workspace: move blocks to prepare for step 2
    // DO 30 I = N + 1, 1, -1
    //    DO 20 IJ = N, 1, -1
    //       DWORK(NP*(I-1)+IJ) = DWORK(N*(I-1)+IJ)
    for (i32 i = n; i >= 0; i--) {
        for (i32 ij_idx = n - 1; ij_idx >= 0; ij_idx--) {
            dwork[np*i + ij_idx] = dwork[n1*i + ij_idx];
        }
    }

    // Copy RINV*C in (2,1) block: DWORK(N+1) in 1-based -> DWORK[n] in 0-based
    SLC_DLACPY("Full", &p, &n, (f64*)c, &ldc, dwork + n, &ldw);
    if (!lmultr) {
        SLC_DTRMM("Left", "Upper", "No transpose", "Non-unit", &p, &n,
                  &one, (f64*)rinv, &ldrinv, dwork + n, &ldw);
    }

    // Copy RINVY to (2,2) block
    i21 = np*n;  // 0-based, new I21 position
    i32 i23 = i21 + n;  // 0-based
    SLC_DCOPY(&p, (f64*)rinvy, &(i32){1}, dwork + i23, &(i32){1});
    wrkopt = wrkopt > np*(n + 1) ? wrkopt : np*(n + 1);

    // Step 2: QR factorization
    itau = i21 + np;    // 0-based
    jwork = itau + n;   // 0-based

    i32 ldwork_remaining = ldwork - jwork;
    SLC_DGEQRF(&np, &n, dwork, &ldw, dwork + itau, dwork + jwork, &ldwork_remaining, info);
    if (*info == 0) {
        i32 optimal = (i32)dwork[jwork] + jwork;
        wrkopt = wrkopt > optimal ? wrkopt : optimal;
    }

    // Apply Householder to last column
    ldwork_remaining = ldwork - jwork;
    SLC_DORMQR("Left", "Transpose", &np, &(i32){1}, &n, dwork, &ldw, dwork + itau,
               dwork + i21, &ldw, dwork + jwork, &ldwork_remaining, info);
    if (*info == 0) {
        i32 optimal = (i32)dwork[jwork] + jwork;
        wrkopt = wrkopt > optimal ? wrkopt : optimal;
    }

    // Output SINV, X, E
    SLC_DLACPY("Upper", &n, &n, dwork, &ldw, sinv, &ldsinv);
    SLC_DCOPY(&n, dwork + i21, &(i32){1}, x, &(i32){1});
    SLC_DCOPY(&p, dwork + i23, &(i32){1}, e, &(i32){1});

    if (ljobx) {
        // Compute X by solving SINV * X_new = X_current
        f64 rcond = 0.0;
        mb02od("Left", "Upper", "No transpose", "Non-unit", "1-norm",
               n, 1, one, sinv, ldsinv, x, n, &rcond, tol, iwork, dwork, info);
        if (*info == 0) {
            wrkopt = wrkopt > 3*n ? wrkopt : 3*n;
            dwork[1] = rcond;
        }
    }

    dwork[0] = (f64)wrkopt;
}
