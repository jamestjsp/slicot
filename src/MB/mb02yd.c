/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <stdlib.h>

void mb02yd(
    const char* cond,
    const i32 n,
    f64* r,
    const i32 ldr,
    const i32* ipvt,
    const f64* diag,
    const f64* qtb,
    i32* rank,
    f64* x,
    const f64 tol,
    f64* dwork,
    const i32 ldwork,
    i32* info
)
{
    f64 zero = 0.0;
    f64 svlmax = 0.0;
    i32 int1 = 1;

    f64 cs, sn, temp, qtbpj, toldef;
    i32 i, j, k, l;
    bool econd, ncond, ucond;
    f64 dum[3];

    econd = (*cond == 'E' || *cond == 'e');
    ncond = (*cond == 'N' || *cond == 'n');
    ucond = (*cond == 'U' || *cond == 'u');
    *info = 0;

    if (!econd && !ncond && !ucond) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (ldr < (n > 1 ? n : 1)) {
        *info = -4;
    } else if (ucond && (*rank < 0 || *rank > n)) {
        *info = -8;
    } else if (ldwork < 2*n || (econd && ldwork < 4*n)) {
        *info = -12;
    }

    // Validate ipvt array contains valid 1-based indices [1, n]
    for (j = 0; j < n; j++) {
        if (ipvt[j] < 1 || ipvt[j] > n) {
            *info = -5;
            break;
        }
    }

    if (*info != 0) {
        return;
    }

    if (n == 0) {
        if (!ucond) {
            *rank = 0;
        }
        return;
    }

    // Copy R and Q'*b to preserve input and initialize S
    // Save diagonal elements of R in X
    for (j = 0; j < n; j++) {
        x[j] = r[j + j*ldr];
        for (i = j; i < n; i++) {
            r[i + j*ldr] = r[j + i*ldr];
        }
    }

    SLC_DCOPY(&n, qtb, &int1, &dwork[n], &int1);

    // Eliminate diagonal matrix D using Givens rotations
    for (j = 0; j < n; j++) {
        // Prepare row of D to be eliminated
        // Locate diagonal element using P from QR factorization
        l = ipvt[j] - 1;  // Convert 1-based to 0-based

        if (diag[l] != zero) {
            qtbpj = zero;
            dwork[j] = diag[l];

            for (k = j + 1; k < n; k++) {
                dwork[k] = zero;
            }

            // Transformations modify single element of Q'*b beyond first n
            for (k = j; k < n; k++) {
                // Determine Givens rotation to eliminate element in current row of D
                if (dwork[k] != zero) {
                    SLC_DLARTG(&r[k + k*ldr], &dwork[k], &cs, &sn, &temp);

                    // Compute modified diagonal element of R and modified (Q'*b, 0)
                    // Accumulate transformation in row of S
                    temp = cs * dwork[n + k] + sn * qtbpj;
                    qtbpj = -sn * dwork[n + k] + cs * qtbpj;
                    dwork[n + k] = temp;

                    i32 nk1 = n - k;
                    SLC_DROT(&nk1, &r[k + k*ldr], &int1, &dwork[k], &int1, &cs, &sn);
                }
            }
        }

        // Store diagonal element of S and restore R diagonal if not ECOND
        dwork[j] = r[j + j*ldr];
        if (!econd) {
            r[j + j*ldr] = x[j];
        }
    }

    // Solve triangular system for z
    // If system is singular, obtain least squares solution
    if (econd) {
        toldef = tol;
        if (toldef <= zero) {
            // Use default tolerance
            toldef = (f64)n * SLC_DLAMCH("Epsilon");
        }

        // Interchange strict upper and lower triangular parts of R
        for (j = 1; j < n; j++) {
            i32 jm1 = j - 1;
            SLC_DSWAP(&jm1, &r[0 + j*ldr], &int1, &r[j + 0*ldr], &ldr);
        }

        // Estimate reciprocal condition number of S and set rank
        mb03od("No QR", n, n, r, ldr, (i32*)ipvt, toldef, svlmax, dwork, rank,
               dum, &dwork[2*n], ldwork - 2*n, info);

        r[0 + 0*ldr] = x[0];

        // Restore strict upper and lower triangular parts of R
        for (j = 1; j < n; j++) {
            i32 jm1 = j - 1;
            SLC_DSWAP(&jm1, &r[0 + j*ldr], &int1, &r[j + 0*ldr], &ldr);
            r[j + j*ldr] = x[j];
        }

    } else if (ncond) {
        // Determine rank(S) by checking zero diagonal entries
        *rank = n;
        for (j = 0; j < n; j++) {
            if (dwork[j] == zero && *rank == n) {
                *rank = j;
            }
        }
    }

    // Zero out components beyond rank
    dum[0] = zero;
    if (*rank < n) {
        i32 nrank = n - *rank;
        i32 int0 = 0;
        SLC_DCOPY(&nrank, dum, &int0, &dwork[n + *rank], &int1);
    }

    // Solve S*z = c using back substitution
    for (j = *rank - 1; j >= 0; j--) {
        temp = zero;
        for (i = j + 1; i < *rank; i++) {
            temp += r[i + j*ldr] * dwork[n + i];
        }
        dwork[n + j] = (dwork[n + j] - temp) / dwork[j];
    }

    // Permute components of z back to components of x
    for (j = 0; j < n; j++) {
        l = ipvt[j] - 1;  // Convert 1-based to 0-based
        x[l] = dwork[n + j];
    }
}
