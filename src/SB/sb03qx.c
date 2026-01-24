/**
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * SB03QX - Estimate forward error bound for continuous-time Lyapunov equation
 *
 * Estimates forward error bound for the solution X of:
 *     op(A)' * X + X * op(A) = C
 * where op(A) = A or A' and C is symmetric.
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <ctype.h>
#include <stdbool.h>

void sb03qx(
    const char* trana,
    const char* uplo,
    const char* lyapun,
    const i32 n,
    const f64 xanorm,
    const f64* t,
    const i32 ldt,
    const f64* u,
    const i32 ldu,
    f64* r,
    const i32 ldr,
    f64* ferr,
    i32* iwork,
    f64* dwork,
    const i32 ldwork,
    i32* info
)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const f64 HALF = 0.5;

    char trana_c = (char)toupper((unsigned char)trana[0]);
    char uplo_c = (char)toupper((unsigned char)uplo[0]);
    char lyapun_c = (char)toupper((unsigned char)lyapun[0]);

    bool notrna = (trana_c == 'N');
    bool update = (lyapun_c == 'O');

    i32 nn = n * n;
    *info = 0;

    if (!notrna && trana_c != 'T' && trana_c != 'C') {
        *info = -1;
    } else if (uplo_c != 'U' && uplo_c != 'L') {
        *info = -2;
    } else if (!update && lyapun_c != 'R') {
        *info = -3;
    } else if (n < 0) {
        *info = -4;
    } else if (xanorm < ZERO) {
        *info = -5;
    } else if (ldt < (n > 1 ? n : 1)) {
        *info = -7;
    } else if (ldu < 1 || (update && ldu < n)) {
        *info = -9;
    } else if (ldr < (n > 1 ? n : 1)) {
        *info = -11;
    } else if (ldwork < 2 * nn) {
        *info = -15;
    }

    if (*info != 0) {
        i32 neginfo = -(*info);
        SLC_XERBLA("SB03QX", &neginfo);
        return;
    }

    *ferr = ZERO;
    if (n == 0 || xanorm == ZERO) {
        return;
    }

    i32 itmp = nn;  // 0-based offset: DWORK(ITMP) in Fortran = dwork[itmp] in C

    char tranat = notrna ? 'T' : 'N';

    // Fill in the remaining triangle of the symmetric residual matrix
    ma02ed(uplo_c, n, r, ldr);

    i32 kase = 0;
    i32 isave[3] = {0, 0, 0};
    f64 est = ZERO;
    f64 scale = ONE;

    // REPEAT loop for 1-norm estimation
    while (1) {
        i32 nn_arg = nn;
        SLC_DLACN2(&nn_arg, &dwork[itmp], dwork, iwork, &est, &kase, isave);

        if (kase == 0) {
            break;
        }

        // Select the triangular part of symmetric matrix to be used
        char uplow;
        bool lower;
        {
            i32 n_arg = n;
            f64 upper_norm = SLC_DLANSY("1", "U", &n_arg, dwork, &n_arg, &dwork[itmp]);
            f64 lower_norm = SLC_DLANSY("1", "L", &n_arg, dwork, &n_arg, &dwork[itmp]);
            if (upper_norm >= lower_norm) {
                uplow = 'U';
                lower = false;
            } else {
                uplow = 'L';
                lower = true;
            }
        }

        if (kase == 2) {
            // Scale the selected triangular part by the residual matrix
            i32 ij = 0;
            if (lower) {
                for (i32 j = 0; j < n; j++) {
                    for (i32 i = j; i < n; i++) {
                        dwork[ij] = dwork[ij] * r[i + j * ldr];
                        ij++;
                    }
                    ij += j + 1;  // Skip to next column start
                }
            } else {
                for (i32 j = 0; j < n; j++) {
                    for (i32 i = 0; i <= j; i++) {
                        dwork[ij] = dwork[ij] * r[i + j * ldr];
                        ij++;
                    }
                    ij += n - j - 1;  // Skip to next column start
                }
            }
        }

        if (update) {
            // Transform the right-hand side: RHS := U' * RHS * U
            i32 info2;
            mb01ru(&uplow, "T", n, n, ZERO, ONE, dwork, n, u, ldu, dwork, n, &dwork[itmp], nn, &info2);
            i32 inc = n + 1;
            SLC_DSCAL(&n, &HALF, dwork, &inc);
        }

        // Fill in the remaining triangle of the symmetric matrix
        ma02ed(uplow, n, dwork, n);

        i32 info2;
        if (kase == 2) {
            // Solve op(T)' * Y + Y * op(T) = scale * RHS
            sb03my(trana, n, t, ldt, dwork, n, &scale, &info2);
        } else {
            // Solve op(T) * W + W * op(T)' = scale * RHS
            sb03my(&tranat, n, t, ldt, dwork, n, &scale, &info2);
        }

        if (info2 > 0) {
            *info = n + 1;
        }

        if (update) {
            // Transform back: Z := U * Z * U'
            mb01ru(&uplow, "N", n, n, ZERO, ONE, dwork, n, u, ldu, dwork, n, &dwork[itmp], nn, &info2);
            i32 inc = n + 1;
            SLC_DSCAL(&n, &HALF, dwork, &inc);
        }

        if (kase == 1) {
            // Scale the selected triangular part by the residual matrix
            i32 ij = 0;
            if (lower) {
                for (i32 j = 0; j < n; j++) {
                    for (i32 i = j; i < n; i++) {
                        dwork[ij] = dwork[ij] * r[i + j * ldr];
                        ij++;
                    }
                    ij += j + 1;
                }
            } else {
                for (i32 j = 0; j < n; j++) {
                    for (i32 i = 0; i <= j; i++) {
                        dwork[ij] = dwork[ij] * r[i + j * ldr];
                        ij++;
                    }
                    ij += n - j - 1;
                }
            }
        }

        // Fill in the remaining triangle of the symmetric matrix
        ma02ed(uplow, n, dwork, n);
    }

    // Compute the estimate of the relative error
    f64 temp = xanorm * scale;
    if (temp > est) {
        *ferr = est / temp;
    } else {
        *ferr = ONE;
    }
}
