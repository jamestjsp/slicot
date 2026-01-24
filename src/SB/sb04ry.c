/**
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * SB04RY - Solve a system of equations in Hessenberg form with one right-hand side
 *
 * Solves (I + LAMBDA * A) * x = d for upper/lower Hessenberg A using QR
 * decomposition with Givens rotations.
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>

void sb04ry(
    const char* rc,
    const char* ul,
    const i32 m,
    const f64* a,
    const i32 lda,
    const f64 lambda,
    f64* d,
    const f64 tol,
    i32* iwork,
    f64* dwork,
    const i32 lddwor,
    i32* info)
{
    const f64 one = 1.0;
    const f64 zero = 0.0;
    i32 int1 = 1;

    *info = 0;

    if (m == 0)
        return;

    bool lulu = (*ul == 'U' || *ul == 'u');
    bool lrow = (*rc == 'R' || *rc == 'r');
    char trans;
    f64 c, s, r, rcond;
    i32 j, mj, j1, len;

    if (lulu) {
        // Upper Hessenberg: copy and scale A, then add identity
        // Fortran: DO J = 1, M; DCOPY(MIN(J+1,M), A(1,J), ..., DWORK(1,J), ...)
        for (j = 1; j <= m; j++) {
            len = (j + 1 < m) ? j + 1 : m;  // min(j+1, m)
            // A(1,J) -> a[(1-1) + (J-1)*lda] = a[(j-1)*lda]
            // DWORK(1,J) -> dwork[(1-1) + (J-1)*lddwor] = dwork[(j-1)*lddwor]
            SLC_DCOPY(&len, &a[(j-1) * lda], &int1, &dwork[(j-1) * lddwor], &int1);
            SLC_DSCAL(&len, &lambda, &dwork[(j-1) * lddwor], &int1);
            // DWORK(J,J) -> dwork[(J-1) + (J-1)*lddwor]
            dwork[(j-1) + (j-1) * lddwor] += one;
        }

        if (lrow) {
            trans = 'N';
            // Row transformations for upper Hessenberg
            // Fortran: DO J = 1, M-1; MJ = M - J
            for (j = 1; j <= m - 1; j++) {
                mj = m - j;
                // DWORK(J+1, J) -> dwork[(J+1-1) + (J-1)*lddwor] = dwork[j + (j-1)*lddwor]
                if (dwork[j + (j-1) * lddwor] != zero) {
                    // DLARTG(DWORK(J,J), DWORK(J+1,J), ...)
                    // DWORK(J,J) -> dwork[(j-1) + (j-1)*lddwor]
                    // DWORK(J+1,J) -> dwork[j + (j-1)*lddwor]
                    SLC_DLARTG(&dwork[(j-1) + (j-1) * lddwor], &dwork[j + (j-1) * lddwor], &c, &s, &r);
                    dwork[(j-1) + (j-1) * lddwor] = r;
                    dwork[j + (j-1) * lddwor] = zero;
                    // DROT(MJ, DWORK(J,J+1), LDDWOR, DWORK(J+1,J+1), LDDWOR, C, S)
                    // DWORK(J,J+1) -> dwork[(j-1) + j*lddwor]
                    // DWORK(J+1,J+1) -> dwork[j + j*lddwor]
                    SLC_DROT(&mj, &dwork[(j-1) + j * lddwor], &lddwor,
                             &dwork[j + j * lddwor], &lddwor, &c, &s);
                    // DROT(1, D(J), 1, D(J+1), 1, C, S)
                    // D(J) -> d[j-1], D(J+1) -> d[j]
                    SLC_DROT(&int1, &d[j-1], &int1, &d[j], &int1, &c, &s);
                }
            }
        } else {
            trans = 'T';
            // Column transformations for upper Hessenberg
            // Fortran: DO J = 1, M-1; MJ = M - J
            for (j = 1; j <= m - 1; j++) {
                mj = m - j;
                // DWORK(MJ+1, MJ) -> dwork[(MJ+1-1) + (MJ-1)*lddwor] = dwork[mj + (mj-1)*lddwor]
                if (dwork[mj + (mj-1) * lddwor] != zero) {
                    // DLARTG(DWORK(MJ+1,MJ+1), DWORK(MJ+1,MJ), ...)
                    // DWORK(MJ+1,MJ+1) -> dwork[mj + mj*lddwor]
                    // DWORK(MJ+1,MJ) -> dwork[mj + (mj-1)*lddwor]
                    SLC_DLARTG(&dwork[mj + mj * lddwor], &dwork[mj + (mj-1) * lddwor], &c, &s, &r);
                    dwork[mj + mj * lddwor] = r;
                    dwork[mj + (mj-1) * lddwor] = zero;
                    // DROT(MJ, DWORK(1,MJ+1), 1, DWORK(1,MJ), 1, C, S)
                    // DWORK(1,MJ+1) -> dwork[0 + mj*lddwor]
                    // DWORK(1,MJ) -> dwork[0 + (mj-1)*lddwor]
                    SLC_DROT(&mj, &dwork[mj * lddwor], &int1,
                             &dwork[(mj-1) * lddwor], &int1, &c, &s);
                    // DROT(1, D(MJ+1), 1, D(MJ), 1, C, S)
                    // D(MJ+1) -> d[mj], D(MJ) -> d[mj-1]
                    SLC_DROT(&int1, &d[mj], &int1, &d[mj-1], &int1, &c, &s);
                }
            }
        }
    } else {
        // Lower Hessenberg: copy and scale A, then add identity
        // Fortran: DO J = 1, M; J1 = MAX(J-1, 1); DCOPY(M-J1+1, A(J1,J), ..., DWORK(J1,J), ...)
        for (j = 1; j <= m; j++) {
            j1 = (j - 1 > 1) ? j - 1 : 1;  // max(j-1, 1)
            len = m - j1 + 1;
            // A(J1,J) -> a[(j1-1) + (j-1)*lda]
            // DWORK(J1,J) -> dwork[(j1-1) + (j-1)*lddwor]
            SLC_DCOPY(&len, &a[(j1-1) + (j-1) * lda], &int1, &dwork[(j1-1) + (j-1) * lddwor], &int1);
            SLC_DSCAL(&len, &lambda, &dwork[(j1-1) + (j-1) * lddwor], &int1);
            // DWORK(J,J) -> dwork[(j-1) + (j-1)*lddwor]
            dwork[(j-1) + (j-1) * lddwor] += one;
        }

        if (lrow) {
            trans = 'N';
            // Row transformations for lower Hessenberg
            // Fortran: DO J = 1, M-1; MJ = M - J
            for (j = 1; j <= m - 1; j++) {
                mj = m - j;
                // DWORK(MJ, MJ+1) -> dwork[(mj-1) + mj*lddwor]
                if (dwork[(mj-1) + mj * lddwor] != zero) {
                    // DLARTG(DWORK(MJ+1,MJ+1), DWORK(MJ,MJ+1), ...)
                    // DWORK(MJ+1,MJ+1) -> dwork[mj + mj*lddwor]
                    // DWORK(MJ,MJ+1) -> dwork[(mj-1) + mj*lddwor]
                    SLC_DLARTG(&dwork[mj + mj * lddwor], &dwork[(mj-1) + mj * lddwor], &c, &s, &r);
                    dwork[mj + mj * lddwor] = r;
                    dwork[(mj-1) + mj * lddwor] = zero;
                    // DROT(MJ, DWORK(MJ+1,1), LDDWOR, DWORK(MJ,1), LDDWOR, C, S)
                    // DWORK(MJ+1,1) -> dwork[mj + 0*lddwor] = dwork[mj]
                    // DWORK(MJ,1) -> dwork[(mj-1) + 0*lddwor] = dwork[mj-1]
                    SLC_DROT(&mj, &dwork[mj], &lddwor,
                             &dwork[mj-1], &lddwor, &c, &s);
                    // DROT(1, D(MJ+1), 1, D(MJ), 1, C, S)
                    // D(MJ+1) -> d[mj], D(MJ) -> d[mj-1]
                    SLC_DROT(&int1, &d[mj], &int1, &d[mj-1], &int1, &c, &s);
                }
            }
        } else {
            trans = 'T';
            // Column transformations for lower Hessenberg
            // Fortran: DO J = 1, M-1; MJ = M - J
            for (j = 1; j <= m - 1; j++) {
                mj = m - j;
                // DWORK(J, J+1) -> dwork[(j-1) + j*lddwor]
                if (dwork[(j-1) + j * lddwor] != zero) {
                    // DLARTG(DWORK(J,J), DWORK(J,J+1), ...)
                    // DWORK(J,J) -> dwork[(j-1) + (j-1)*lddwor]
                    // DWORK(J,J+1) -> dwork[(j-1) + j*lddwor]
                    SLC_DLARTG(&dwork[(j-1) + (j-1) * lddwor], &dwork[(j-1) + j * lddwor], &c, &s, &r);
                    dwork[(j-1) + (j-1) * lddwor] = r;
                    dwork[(j-1) + j * lddwor] = zero;
                    // DROT(MJ, DWORK(J+1,J), 1, DWORK(J+1,J+1), 1, C, S)
                    // DWORK(J+1,J) -> dwork[j + (j-1)*lddwor]
                    // DWORK(J+1,J+1) -> dwork[j + j*lddwor]
                    SLC_DROT(&mj, &dwork[j + (j-1) * lddwor], &int1,
                             &dwork[j + j * lddwor], &int1, &c, &s);
                    // DROT(1, D(J), 1, D(J+1), 1, C, S)
                    // D(J) -> d[j-1], D(J+1) -> d[j]
                    SLC_DROT(&int1, &d[j-1], &int1, &d[j], &int1, &c, &s);
                }
            }
        }
    }

    // Estimate condition number
    // DTRCON('1-norm', UL, 'Non-unit', M, DWORK, LDDWOR, RCOND, DWORK(1,M+1), IWORK, INFO)
    // DWORK(1,M+1) -> dwork[0 + m*lddwor] = dwork[m*lddwor]
    SLC_DTRCON("1", ul, "N", &m, dwork, &lddwor, &rcond, &dwork[m * lddwor], iwork, info);
    if (rcond <= tol) {
        *info = 1;
    } else {
        // Solve triangular system
        // DTRSV(UL, TRANS, 'Non-unit', M, DWORK, LDDWOR, D, 1)
        char trans_str[2] = {trans, '\0'};
        SLC_DTRSV(ul, trans_str, "N", &m, dwork, &lddwor, d, &int1);
    }
}
