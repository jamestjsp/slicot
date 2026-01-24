/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

/*
 * BB03AD - Benchmark examples for (generalized) continuous-time Lyapunov equations
 *
 * Purpose:
 *   Generate benchmark examples of (generalized) continuous-time Lyapunov equations
 *       A^T X E + E^T X A = Y
 *
 *   In some examples, the right hand side has the form Y = -B^T B
 *   and the solution can be represented as X = U^T U.
 *
 *   E, A, Y, X, and U are real N-by-N matrices, and B is M-by-N.
 *   Note that E can be the identity matrix. For some examples, B, X, or U
 *   are not provided.
 *
 *   This routine is an implementation of the benchmark library
 *   CTLEX (Version 1.0) described in [1].
 *
 * References:
 *   [1] D. Kressner, V. Mehrmann, and T. Penzl.
 *       CTLEX - a Collection of Benchmark Examples for Continuous-Time
 *       Lyapunov Equations. SLICOT Working Note 1999-6, 1999.
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <string.h>
#include <ctype.h>

#define ZERO 0.0
#define ONE 1.0
#define TWO 2.0
#define FOUR 4.0

static bool lsame_local(char a, char b) {
    return toupper((unsigned char)a) == toupper((unsigned char)b);
}

void bb03ad(const char* def, const i32* nr, f64* dpar, i32* ipar,
            bool* vec, i32* n, i32* m,
            f64* e, const i32 lde, f64* a, const i32 lda,
            f64* y, const i32 ldy, f64* b, const i32 ldb,
            f64* x, const i32 ldx, f64* u, const i32 ldu,
            char* note, f64* dwork, const i32 ldwork, i32* info)
{
    i32 i, j, k;
    f64 temp, ttp1, ttm1, twobyn;
    i32 int1 = 1;
    f64 dbl0 = ZERO, dbl1 = ONE;

    static const bool vecdef[8] = {true, true, false, true, true, false, false, false};

    *info = 0;
    for (i = 0; i < 8; i++) {
        vec[i] = vecdef[i];
    }

    if (nr[0] == 4) {
        if (!(lsame_local(def[0], 'D') || lsame_local(def[0], 'N'))) {
            *info = -1;
            return;
        }

        if (nr[1] == 1) {
            strncpy(note, "CTLEX: Example 4.1", 70);
            if (lsame_local(def[0], 'D')) {
                ipar[0] = 10;
                dpar[0] = 1.5;
                dpar[1] = 1.5;
            }
            if ((dpar[0] <= ONE) || (dpar[1] <= ONE)) *info = -3;
            if (ipar[0] < 2) *info = -4;
            *n = ipar[0];
            *m = 1;
            if (lde < *n) *info = -9;
            if (lda < *n) *info = -11;
            if (ldy < *n) *info = -13;
            if (ldb < *m) *info = -15;
            if (ldx < *n) *info = -17;
            if (ldwork < (*n) * 2) *info = -22;
            if (*info != 0) return;

            vec[5] = true;
            vec[6] = true;
            twobyn = TWO / (f64)(*n);
            SLC_DLASET("A", n, n, &dbl0, &dbl1, e, &lde);
            SLC_DLASET("A", n, n, &dbl0, &dbl0, a, &lda);
            SLC_DLASET("A", n, n, &dbl0, &dbl0, y, &ldy);
            SLC_DLASET("A", m, n, &dbl0, &dbl0, b, &ldb);
            SLC_DLASET("A", n, n, &dbl0, &dbl0, x, &ldx);

            for (j = 0; j < *n; j++) {
                temp = pow(dpar[0], (f64)j);
                a[j + j * lda] = -temp;
                dwork[j] = ONE;
                for (i = 0; i < *n; i++) {
                    x[i + j * ldx] = (f64)((i + 1) * (j + 1)) / (temp + pow(dpar[0], (f64)i));
                }
            }

            f64 neg_twobyn = -twobyn;
            SLC_DGEMV("T", n, n, &dbl1, a, &lda, dwork, &int1, &dbl0, &dwork[*n], &int1);
            SLC_DGER(n, n, &neg_twobyn, dwork, &int1, &dwork[*n], &int1, a, &lda);

            SLC_DGEMV("N", n, n, &dbl1, a, &lda, dwork, &int1, &dbl0, &dwork[*n], &int1);
            SLC_DGER(n, n, &neg_twobyn, &dwork[*n], &int1, dwork, &int1, a, &lda);

            SLC_DGEMV("T", n, n, &dbl1, x, &ldx, dwork, &int1, &dbl0, &dwork[*n], &int1);
            SLC_DGER(n, n, &neg_twobyn, dwork, &int1, &dwork[*n], &int1, x, &ldx);

            SLC_DGEMV("N", n, n, &dbl1, x, &ldx, dwork, &int1, &dbl0, &dwork[*n], &int1);
            SLC_DGER(n, n, &neg_twobyn, &dwork[*n], &int1, dwork, &int1, x, &ldx);

            for (j = 0; j < *n; j++) {
                b[0 + j * ldb] = (f64)(j - *n) / pow(dpar[1], (f64)j);
                for (i = 0; i < *n; i++) {
                    x[i + j * ldx] = x[i + j * ldx] / pow(dpar[1], (f64)(i + j));
                    a[i + j * lda] = a[i + j * lda] * pow(dpar[1], (f64)(i - j));
                }
                dwork[j] = ONE - TWO * (j % 2);
            }

            SLC_DGEMV("T", n, n, &dbl1, a, &lda, dwork, &int1, &dbl0, &dwork[*n], &int1);
            SLC_DGER(n, n, &neg_twobyn, dwork, &int1, &dwork[*n], &int1, a, &lda);

            SLC_DGEMV("N", n, n, &dbl1, a, &lda, dwork, &int1, &dbl0, &dwork[*n], &int1);
            SLC_DGER(n, n, &neg_twobyn, &dwork[*n], &int1, dwork, &int1, a, &lda);

            SLC_DGEMV("T", n, n, &dbl1, x, &ldx, dwork, &int1, &dbl0, &dwork[*n], &int1);
            SLC_DGER(n, n, &neg_twobyn, dwork, &int1, &dwork[*n], &int1, x, &ldx);

            SLC_DGEMV("N", n, n, &dbl1, x, &ldx, dwork, &int1, &dbl0, &dwork[*n], &int1);
            SLC_DGER(n, n, &neg_twobyn, &dwork[*n], &int1, dwork, &int1, x, &ldx);

            f64 dot = SLC_DDOT(n, b, &ldb, dwork, &int1);
            f64 factor = -twobyn * dot;
            SLC_DAXPY(n, &factor, dwork, &int1, b, &ldb);

            f64 neg_one = -ONE;
            SLC_DGER(n, n, &neg_one, b, &ldb, b, &ldb, y, &ldy);

        } else if (nr[1] == 2) {
            strncpy(note, "CTLEX: Example 4.2", 70);
            if (lsame_local(def[0], 'D')) {
                ipar[0] = 10;
                dpar[0] = -0.5;
                dpar[1] = 1.5;
            }
            if ((dpar[0] >= ZERO) || (dpar[1] <= ONE)) *info = -3;
            if (ipar[0] < 2) *info = -4;
            *n = ipar[0];
            *m = 1;
            if (lde < *n) *info = -9;
            if (lda < *n) *info = -11;
            if (ldy < *n) *info = -13;
            if (ldb < *m) *info = -15;
            if (ldwork < (*n) * 2) *info = -22;
            if (*info != 0) return;

            vec[5] = true;
            twobyn = TWO / (f64)(*n);
            SLC_DLASET("A", n, n, &dbl0, &dbl1, e, &lde);
            f64 lambda = dpar[0];
            SLC_DLASET("A", n, n, &dbl0, &lambda, a, &lda);
            SLC_DLASET("A", n, n, &dbl0, &dbl0, y, &ldy);
            f64 neg_twobyn = -twobyn;
            f64 b_diag = ONE - twobyn;
            SLC_DLASET("A", m, n, &neg_twobyn, &b_diag, b, &ldb);

            for (i = 0; i < *n - 1; i++) {
                dwork[i] = ONE;
                a[i + (i + 1) * lda] = ONE;
            }
            dwork[*n - 1] = ONE;

            SLC_DGEMV("T", n, n, &dbl1, a, &lda, dwork, &int1, &dbl0, &dwork[*n], &int1);
            SLC_DGER(n, n, &neg_twobyn, dwork, &int1, &dwork[*n], &int1, a, &lda);

            SLC_DGEMV("N", n, n, &dbl1, a, &lda, dwork, &int1, &dbl0, &dwork[*n], &int1);
            SLC_DGER(n, n, &neg_twobyn, &dwork[*n], &int1, dwork, &int1, a, &lda);

            for (j = 0; j < *n; j++) {
                b[0 + j * ldb] = b[0 + j * ldb] / pow(dpar[1], (f64)j);
                for (i = 0; i < *n; i++) {
                    a[i + j * lda] = a[i + j * lda] * pow(dpar[1], (f64)(i - j));
                }
                dwork[j] = ONE - TWO * (j % 2);
            }

            SLC_DGEMV("T", n, n, &dbl1, a, &lda, dwork, &int1, &dbl0, &dwork[*n], &int1);
            SLC_DGER(n, n, &neg_twobyn, dwork, &int1, &dwork[*n], &int1, a, &lda);

            SLC_DGEMV("N", n, n, &dbl1, a, &lda, dwork, &int1, &dbl0, &dwork[*n], &int1);
            SLC_DGER(n, n, &neg_twobyn, &dwork[*n], &int1, dwork, &int1, a, &lda);

            f64 dot = SLC_DDOT(n, b, &ldb, dwork, &int1);
            f64 factor = -twobyn * dot;
            SLC_DAXPY(n, &factor, dwork, &int1, b, &ldb);

            f64 neg_one = -ONE;
            SLC_DGER(n, n, &neg_one, b, &ldb, b, &ldb, y, &ldy);

        } else if (nr[1] == 3) {
            strncpy(note, "CTLEX: Example 4.3", 70);
            if (lsame_local(def[0], 'D')) {
                ipar[0] = 10;
                dpar[0] = 10.0;
            }
            if (dpar[0] < ZERO) *info = -3;
            if (ipar[0] < 2) *info = -4;
            *n = ipar[0];
            *m = 0;
            if (lde < *n) *info = -9;
            if (lda < *n) *info = -11;
            if (ldy < *n) *info = -13;
            if (ldx < *n) *info = -17;
            if (*info != 0) return;

            vec[2] = true;
            vec[6] = true;
            temp = pow(TWO, -dpar[0]);
            SLC_DLASET("U", n, n, &dbl0, &dbl0, e, &lde);
            SLC_DLASET("L", n, n, &temp, &dbl1, e, &lde);
            SLC_DLASET("L", n, n, &dbl0, &dbl0, a, &lda);
            SLC_DLASET("U", n, n, &dbl1, &dbl0, a, &lda);
            SLC_DLASET("A", n, n, &dbl1, &dbl1, x, &ldx);

            for (i = 0; i < *n; i++) {
                a[i + i * lda] = (f64)i + temp;
            }

            y[0 + 0 * ldy] = TWO * temp + TWO * (f64)(*n - 1) * temp * temp;
            ttp1 = TWO * (f64)(*n + 1) * temp + TWO - temp * temp;
            ttm1 = TWO * (f64)(*n - 1) * temp + TWO - temp * temp;

            for (i = 1; i < *n; i++) {
                y[i + 0 * ldy] = y[0 + 0 * ldy] + (f64)i * ttm1;
            }

            for (j = 1; j < *n; j++) {
                for (i = 0; i < *n; i++) {
                    y[i + j * ldy] = y[i + 0 * ldy] + (f64)j * (ttp1 - FOUR * (f64)(i + 1) * temp);
                }
            }

        } else if (nr[1] == 4) {
            strncpy(note, "CTLEX: Example 4.4", 70);
            if (lsame_local(def[0], 'D')) {
                ipar[0] = 10;
                dpar[0] = 1.5;
            }
            if (dpar[0] < ONE) *info = -3;
            if (ipar[0] < 1) *info = -4;
            *n = ipar[0] * 3;
            *m = 1;
            if (lde < *n) *info = -9;
            if (lda < *n) *info = -11;
            if (ldy < *n) *info = -13;
            if (ldb < *m) *info = -15;
            if (*info != 0) return;

            vec[2] = true;
            vec[5] = true;
            SLC_DLASET("A", n, n, &dbl0, &dbl0, e, &lde);
            SLC_DLASET("A", n, n, &dbl0, &dbl0, a, &lda);

            for (i = 0; i < ipar[0]; i++) {
                temp = -pow(dpar[0], (f64)(i + 1));
                for (j = 0; j < i; j++) {
                    for (k = 0; k < 3; k++) {
                        a[(*n - (i + 1) * 3 + 2) + ((j + 1) * 3 - 1 - k) * lda] = temp;
                        a[(*n - (i + 1) * 3 + 1) + ((j + 1) * 3 - 1 - k) * lda] = TWO * temp;
                    }
                }
                a[(*n - (i + 1) * 3 + 2) + ((i + 1) * 3 - 3) * lda] = temp;
                a[(*n - (i + 1) * 3 + 1) + ((i + 1) * 3 - 3) * lda] = TWO * temp;
                a[(*n - (i + 1) * 3 + 1) + ((i + 1) * 3 - 2) * lda] = TWO * temp;
                a[(*n - (i + 1) * 3 + 1) + ((i + 1) * 3 - 1) * lda] = temp;
                a[(*n - (i + 1) * 3 + 0) + ((i + 1) * 3 - 1) * lda] = temp;
            }

            for (j = 0; j < *n; j++) {
                if (j > 0) {
                    SLC_DAXPY(n, &dbl1, &a[(j - 1) + 0 * lda], &lda, &a[j + 0 * lda], &lda);
                }
                b[0 + j * ldb] = (f64)(j + 1);
                for (i = 0; i < *n; i++) {
                    i32 min_ij = (i + 1) < (j + 1) ? (i + 1) : (j + 1);
                    e[i + (*n - j - 1) * lde] = (f64)min_ij;
                    y[i + j * ldy] = -(f64)((i + 1) * (j + 1));
                }
            }

        } else {
            *info = -2;
        }
    } else {
        *info = -2;
    }
}
