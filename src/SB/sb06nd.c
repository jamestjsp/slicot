/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"

void sb06nd(
    const i32 n,
    const i32 m,
    const i32 kmax,
    f64* a,
    const i32 lda,
    f64* b,
    const i32 ldb,
    const i32* kstair,
    f64* u,
    const i32 ldu,
    f64* f,
    const i32 ldf,
    f64* dwork,
    i32* info
)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const f64 NEGONE = -1.0;

    i32 int1 = 1;
    i32 j, j0, jcur, jkcur, jmkcur, kcur, kk, kmin, kstep, mkcur, ncont;

    *info = 0;

    // Parameter validation
    if (n < 0) {
        *info = -1;
    } else if (m < 0) {
        *info = -2;
    } else if (kmax < 0 || kmax > n) {
        *info = -3;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -5;
    } else if (ldb < (n > 1 ? n : 1)) {
        *info = -7;
    } else if (ldu < (n > 1 ? n : 1)) {
        *info = -10;
    } else if (ldf < (m > 1 ? m : 1)) {
        *info = -12;
    } else {
        ncont = 0;
        for (kk = 0; kk < kmax; kk++) {
            ncont += kstair[kk];
        }
        if (ncont > n) {
            *info = -8;
        }
    }

    if (*info != 0) {
        return;
    }

    // Quick return
    if (n == 0 || m == 0) {
        return;
    }

    // Main algorithm
    for (kmin = 1; kmin <= kmax; kmin++) {
        jcur = ncont;  // 1-based conceptual index
        kstep = kmax - kmin;

        // Triangularize bottom part of A (if kstep > 0)
        for (kk = kmax; kk >= kmax - kstep + 1; kk--) {
            kcur = kstair[kk - 1];  // kstair is 0-based in C

            // Construct Ukk and store in Fkk
            for (j = 1; j <= kcur; j++) {
                jmkcur = jcur - kcur;

                // DCOPY: Copy A(jcur, jmkcur:jmkcur+kcur-1) column-wise to F(1:kcur, jcur)
                // Fortran: A(JCUR, JMKCUR) with increment LDA -> column elements
                // This copies the row A(jcur, jmkcur) ... A(jcur, jmkcur+kcur-1)
                // In column-major: A[jcur-1 + (jmkcur-1)*lda], A[jcur-1 + jmkcur*lda], ...
                // with increment lda (accessing columns)
                SLC_DCOPY(&kcur, &a[(jcur - 1) + (jmkcur - 1) * lda], &lda,
                          &f[(jcur - 1) * ldf], &int1);

                // DLARFG: Generate elementary reflector
                // On input: F(1:kcur, jcur), A(jcur, jcur)
                // On output: tau in DWORK(jcur)
                i32 kcur_p1 = kcur + 1;
                SLC_DLARFG(&kcur_p1, &a[(jcur - 1) + (jcur - 1) * lda],
                           &f[(jcur - 1) * ldf], &int1, &dwork[jcur - 1]);

                // DLASET: Zero out A(jcur, jmkcur:jmkcur+kcur-1)
                i32 one = 1;
                SLC_DLASET("F", &one, &kcur, &ZERO, &ZERO,
                           &a[(jcur - 1) + (jmkcur - 1) * lda], &lda);

                // DLATZM: Backmultiply A and U with Ukk
                // Apply to A: rows 1:jcur-1, columns jcur and jmkcur:jmkcur+kcur-1
                i32 jcur_m1 = jcur - 1;
                i32 n_rows = n;
                SLC_DLATZM("R", &jcur_m1, &kcur_p1, &f[(jcur - 1) * ldf], &int1,
                           &dwork[jcur - 1], &a[(jcur - 1) * lda],
                           &a[(jmkcur - 1) * lda], &lda, dwork);

                // Apply to U: all rows, columns jcur and jmkcur:jmkcur+kcur-1
                SLC_DLATZM("R", &n_rows, &kcur_p1, &f[(jcur - 1) * ldf], &int1,
                           &dwork[jcur - 1], &u[(jcur - 1) * ldu],
                           &u[(jmkcur - 1) * ldu], &ldu, &dwork[n]);

                jcur--;
            }
        }

        // Eliminate diagonal block Aii by feedback Fi
        kcur = kstair[kmin - 1];
        j0 = jcur - kcur + 1;  // 1-based
        mkcur = m - kcur + 1;  // 1-based index in F

        // DLACPY: Copy A(j0:j0+kcur-1, j0:j0+kcur-1) to F(mkcur:m, j0:j0+kcur-1)
        SLC_DLACPY("F", &kcur, &kcur, &a[(j0 - 1) + (j0 - 1) * lda], &lda,
                   &f[(mkcur - 1) + (j0 - 1) * ldf], &ldf);

        // DTRSM: Solve for Fi (left triangular solve with -1 multiplier)
        // F(mkcur:m, j0:j0+kcur-1) = -B(j0:j0+kcur-1, mkcur:m)^(-1) * F(mkcur:m, j0:j0+kcur-1)
        SLC_DTRSM("L", "U", "N", "N", &kcur, &kcur, &NEGONE,
                  &b[(j0 - 1) + (mkcur - 1) * ldb], &ldb,
                  &f[(mkcur - 1) + (j0 - 1) * ldf], &ldf);

        // DGEMM: Add B * Fi to A (if j0 > 1)
        if (j0 > 1) {
            i32 j0_m1 = j0 - 1;
            SLC_DGEMM("N", "N", &j0_m1, &kcur, &kcur, &ONE,
                      &b[(mkcur - 1) * ldb], &ldb,
                      &f[(mkcur - 1) + (j0 - 1) * ldf], &ldf,
                      &ONE, &a[(j0 - 1) * lda], &lda);
        }

        // DLASET: Zero out diagonal block A(j0:j0+kcur-1, j0:j0+kcur-1)
        SLC_DLASET("F", &kcur, &kcur, &ZERO, &ZERO,
                   &a[(j0 - 1) + (j0 - 1) * lda], &lda);

        // DLASET: Zero out F(1:m-kcur, j0:j0+kcur-1)
        i32 m_minus_kcur = m - kcur;
        if (m_minus_kcur > 0) {
            SLC_DLASET("F", &m_minus_kcur, &kcur, &ZERO, &ZERO,
                       &f[(j0 - 1) * ldf], &ldf);
        }

        if (kstep != 0) {
            jkcur = ncont;

            // Premultiply A with Ukk
            for (kk = kmax; kk >= kmax - kstep + 1; kk--) {
                kcur = kstair[kk - 1];
                jcur = jkcur - kcur;

                for (j = 1; j <= kcur; j++) {
                    i32 kcur_p1 = kcur + 1;
                    i32 n_minus_jcur_p1 = n - jcur + 1;
                    SLC_DLATZM("L", &kcur_p1, &n_minus_jcur_p1, &f[(jkcur - 1) * ldf], &int1,
                               &dwork[jkcur - 1], &a[(jkcur - 1) + (jcur - 1) * lda],
                               &a[(jcur - 1) + (jcur - 1) * lda], &lda, &dwork[n]);
                    jcur--;
                    jkcur--;
                }
            }

            // Premultiply B with Ukk
            jcur = jcur + kcur;
            jkcur = jcur + kcur;

            for (j = m; j >= m - kcur + 1; j--) {
                i32 kcur_p1 = kcur + 1;
                i32 m_minus_j_p1 = m - j + 1;
                SLC_DLATZM("L", &kcur_p1, &m_minus_j_p1, &f[(jkcur - 1) * ldf], &int1,
                           &dwork[jkcur - 1], &b[(jkcur - 1) + (j - 1) * ldb],
                           &b[(jcur - 1) + (j - 1) * ldb], &ldb, &dwork[n]);
                jcur--;
                jkcur--;
            }
        }
    }

    // Zero uncontrollable part of F
    if (ncont != n) {
        i32 n_minus_ncont = n - ncont;
        i32 m_val = m;
        SLC_DLASET("F", &m_val, &n_minus_ncont, &ZERO, &ZERO,
                   &f[ncont * ldf], &ldf);
    }
}
