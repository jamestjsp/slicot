/**
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * SB04RX - Solve quasi-Hessenberg system with two right-hand sides
 *
 * Solves a system of equations in quasi-Hessenberg form (Hessenberg form
 * plus two consecutive offdiagonals) with two right-hand sides via QR
 * decomposition with Givens rotations.
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <string.h>

// Helper macros for Fortran 1-based indexing into DWORK(row, col) stored column-major
// DWORK(i,j) in Fortran (1-based) = dwork[(j-1)*lddwor + (i-1)] in C (0-based)
// DW(r,c) where r,c are Fortran 1-based indices
#define DW(r, c) dwork[((c)-1)*lddwor + ((r)-1)]

void sb04rx(
    const char* rc,
    const char* ul,
    const i32 m,
    const f64* a,
    const i32 lda,
    const f64 lambd1,
    const f64 lambd2,
    const f64 lambd3,
    const f64 lambd4,
    f64* d,
    const f64 tol,
    i32* iwork,
    f64* dwork,
    const i32 lddwor,
    i32* info)
{
    const f64 one = 1.0;
    const f64 zero = 0.0;

    char trans_buf[2] = {0};
    i32 j, j1, j2, m2, mj, ml;
    f64 c_val, r_val, s_val, rcond;
    i32 int1 = 1;
    i32 int2 = 2;

    *info = 0;

    if (m == 0) {
        return;
    }

    m2 = m * 2;

    bool is_upper = (ul[0] == 'U' || ul[0] == 'u');
    bool is_row = (rc[0] == 'R' || rc[0] == 'r');

    if (is_upper) {
        // Upper Hessenberg matrix setup
        for (j = 1; j <= m; j++) {  // Fortran 1-based loop
            j2 = j * 2;
            ml = (m < j + 1) ? m : (j + 1);  // MIN(M, J+1)

            // Clear two columns J2-1 and J2
            SLC_DLASET("Full", &m2, &int2, &zero, &zero, &DW(1, j2-1), &lddwor);

            // Copy A(1:ML, J) to DWORK columns with stride 2
            // A(i,j) in Fortran (1-based) = a[(j-1)*lda + (i-1)] in C
            for (i32 i = 1; i <= ml; i++) {
                f64 aij = a[(j-1)*lda + (i-1)];
                DW(2*i-1, j2-1) = aij * lambd1;
                DW(2*i, j2-1) = aij * lambd3;
                DW(2*i-1, j2) = aij * lambd2;
                DW(2*i, j2) = aij * lambd4;
            }

            // Add identity to diagonal
            DW(j2-1, j2-1) += one;
            DW(j2, j2) += one;
        }

        if (is_row) {
            trans_buf[0] = 'N';

            // Row transformations for upper Hessenberg
            for (j = 1; j <= m2 - 1; j++) {  // Fortran 1-based J = 1, M2-1
                mj = m2 - j;

                // Extra elimination for odd j and j < m2-2
                if ((j % 2) == 1 && j < m2 - 2) {
                    if (DW(j+3, j) != zero) {
                        SLC_DLARTG(&DW(j+2, j), &DW(j+3, j), &c_val, &s_val, &r_val);
                        DW(j+2, j) = r_val;
                        DW(j+3, j) = zero;
                        SLC_DROT(&mj, &DW(j+2, j+1), &lddwor, &DW(j+3, j+1), &lddwor, &c_val, &s_val);
                        // D is 0-based: D(J+2) in Fortran = d[j+1] in C
                        SLC_DROT(&int1, &d[j+1], &int1, &d[j+2], &int1, &c_val, &s_val);
                    }
                }

                if (j < m2 - 1) {
                    if (DW(j+2, j) != zero) {
                        SLC_DLARTG(&DW(j+1, j), &DW(j+2, j), &c_val, &s_val, &r_val);
                        DW(j+1, j) = r_val;
                        DW(j+2, j) = zero;
                        SLC_DROT(&mj, &DW(j+1, j+1), &lddwor, &DW(j+2, j+1), &lddwor, &c_val, &s_val);
                        // D(J+1) in Fortran = d[j] in C
                        SLC_DROT(&int1, &d[j], &int1, &d[j+1], &int1, &c_val, &s_val);
                    }
                }

                if (DW(j+1, j) != zero) {
                    SLC_DLARTG(&DW(j, j), &DW(j+1, j), &c_val, &s_val, &r_val);
                    DW(j, j) = r_val;
                    DW(j+1, j) = zero;
                    SLC_DROT(&mj, &DW(j, j+1), &lddwor, &DW(j+1, j+1), &lddwor, &c_val, &s_val);
                    // D(J) in Fortran = d[j-1] in C
                    SLC_DROT(&int1, &d[j-1], &int1, &d[j], &int1, &c_val, &s_val);
                }
            }
        } else {
            trans_buf[0] = 'T';

            // Column transformations for upper Hessenberg
            for (j = 1; j <= m2 - 1; j++) {
                mj = m2 - j;

                if ((j % 2) == 1 && j < m2 - 2) {
                    // DWORK(MJ+1, MJ-2) in Fortran
                    if (DW(mj+1, mj-2) != zero) {
                        SLC_DLARTG(&DW(mj+1, mj-1), &DW(mj+1, mj-2), &c_val, &s_val, &r_val);
                        DW(mj+1, mj-1) = r_val;
                        DW(mj+1, mj-2) = zero;
                        // DROT(MJ, DWORK(1, MJ-1), 1, DWORK(1, MJ-2), 1, ...)
                        SLC_DROT(&mj, &DW(1, mj-1), &int1, &DW(1, mj-2), &int1, &c_val, &s_val);
                        // D(MJ-1), D(MJ-2) in Fortran = d[mj-2], d[mj-3] in C
                        SLC_DROT(&int1, &d[mj-2], &int1, &d[mj-3], &int1, &c_val, &s_val);
                    }
                }

                if (j < m2 - 1) {
                    if (DW(mj+1, mj-1) != zero) {
                        SLC_DLARTG(&DW(mj+1, mj), &DW(mj+1, mj-1), &c_val, &s_val, &r_val);
                        DW(mj+1, mj) = r_val;
                        DW(mj+1, mj-1) = zero;
                        SLC_DROT(&mj, &DW(1, mj), &int1, &DW(1, mj-1), &int1, &c_val, &s_val);
                        // D(MJ), D(MJ-1) in Fortran = d[mj-1], d[mj-2] in C
                        SLC_DROT(&int1, &d[mj-1], &int1, &d[mj-2], &int1, &c_val, &s_val);
                    }
                }

                if (DW(mj+1, mj) != zero) {
                    SLC_DLARTG(&DW(mj+1, mj+1), &DW(mj+1, mj), &c_val, &s_val, &r_val);
                    DW(mj+1, mj+1) = r_val;
                    DW(mj+1, mj) = zero;
                    SLC_DROT(&mj, &DW(1, mj+1), &int1, &DW(1, mj), &int1, &c_val, &s_val);
                    // D(MJ+1), D(MJ) in Fortran = d[mj], d[mj-1] in C
                    SLC_DROT(&int1, &d[mj], &int1, &d[mj-1], &int1, &c_val, &s_val);
                }
            }
        }
    } else {
        // Lower Hessenberg matrix setup
        for (j = 1; j <= m; j++) {
            j2 = j * 2;
            j1 = (j > 1) ? (j - 1) : 1;  // MAX(J-1, 1)
            ml = m - j + 2;  // M - J + 2
            if (ml > m) ml = m;

            SLC_DLASET("Full", &m2, &int2, &zero, &zero, &DW(1, j2-1), &lddwor);

            // Copy A(J1:J1+ML-1, J) to DWORK with stride 2
            // Starting row in DWORK: J1*2-1
            for (i32 i = 0; i < ml; i++) {
                f64 aij = a[(j-1)*lda + (j1-1) + i];
                i32 row = j1*2 - 1 + 2*i;
                DW(row, j2-1) = aij * lambd1;
                DW(row+1, j2-1) = aij * lambd3;
                DW(row, j2) = aij * lambd2;
                DW(row+1, j2) = aij * lambd4;
            }

            DW(j2-1, j2-1) += one;
            DW(j2, j2) += one;
        }

        if (is_row) {
            trans_buf[0] = 'N';

            // Row transformations for lower Hessenberg
            for (j = 1; j <= m2 - 1; j++) {
                mj = m2 - j;

                if ((j % 2) == 1 && j < m2 - 2) {
                    // DWORK(MJ-2, MJ+1) in Fortran
                    if (DW(mj-2, mj+1) != zero) {
                        SLC_DLARTG(&DW(mj-1, mj+1), &DW(mj-2, mj+1), &c_val, &s_val, &r_val);
                        DW(mj-1, mj+1) = r_val;
                        DW(mj-2, mj+1) = zero;
                        // DROT(MJ, DWORK(MJ-1, 1), LDDWOR, DWORK(MJ-2, 1), LDDWOR, ...)
                        SLC_DROT(&mj, &DW(mj-1, 1), &lddwor, &DW(mj-2, 1), &lddwor, &c_val, &s_val);
                        // D(MJ-1), D(MJ-2) in Fortran = d[mj-2], d[mj-3] in C
                        SLC_DROT(&int1, &d[mj-2], &int1, &d[mj-3], &int1, &c_val, &s_val);
                    }
                }

                if (j < m2 - 1) {
                    if (DW(mj-1, mj+1) != zero) {
                        SLC_DLARTG(&DW(mj, mj+1), &DW(mj-1, mj+1), &c_val, &s_val, &r_val);
                        DW(mj, mj+1) = r_val;
                        DW(mj-1, mj+1) = zero;
                        SLC_DROT(&mj, &DW(mj, 1), &lddwor, &DW(mj-1, 1), &lddwor, &c_val, &s_val);
                        // D(MJ), D(MJ-1) in Fortran = d[mj-1], d[mj-2] in C
                        SLC_DROT(&int1, &d[mj-1], &int1, &d[mj-2], &int1, &c_val, &s_val);
                    }
                }

                if (DW(mj, mj+1) != zero) {
                    SLC_DLARTG(&DW(mj+1, mj+1), &DW(mj, mj+1), &c_val, &s_val, &r_val);
                    DW(mj+1, mj+1) = r_val;
                    DW(mj, mj+1) = zero;
                    SLC_DROT(&mj, &DW(mj+1, 1), &lddwor, &DW(mj, 1), &lddwor, &c_val, &s_val);
                    // D(MJ+1), D(MJ) in Fortran = d[mj], d[mj-1] in C
                    SLC_DROT(&int1, &d[mj], &int1, &d[mj-1], &int1, &c_val, &s_val);
                }
            }
        } else {
            trans_buf[0] = 'T';

            // Column transformations for lower Hessenberg
            for (j = 1; j <= m2 - 1; j++) {
                mj = m2 - j;

                if ((j % 2) == 1 && j < m2 - 2) {
                    // DWORK(J, J+3) in Fortran
                    if (DW(j, j+3) != zero) {
                        SLC_DLARTG(&DW(j, j+2), &DW(j, j+3), &c_val, &s_val, &r_val);
                        DW(j, j+2) = r_val;
                        DW(j, j+3) = zero;
                        // DROT(MJ, DWORK(J+1, J+2), 1, DWORK(J+1, J+3), 1, ...)
                        SLC_DROT(&mj, &DW(j+1, j+2), &int1, &DW(j+1, j+3), &int1, &c_val, &s_val);
                        // D(J+2), D(J+3) in Fortran = d[j+1], d[j+2] in C
                        SLC_DROT(&int1, &d[j+1], &int1, &d[j+2], &int1, &c_val, &s_val);
                    }
                }

                if (j < m2 - 1) {
                    if (DW(j, j+2) != zero) {
                        SLC_DLARTG(&DW(j, j+1), &DW(j, j+2), &c_val, &s_val, &r_val);
                        DW(j, j+1) = r_val;
                        DW(j, j+2) = zero;
                        SLC_DROT(&mj, &DW(j+1, j+1), &int1, &DW(j+1, j+2), &int1, &c_val, &s_val);
                        // D(J+1), D(J+2) in Fortran = d[j], d[j+1] in C
                        SLC_DROT(&int1, &d[j], &int1, &d[j+1], &int1, &c_val, &s_val);
                    }
                }

                if (DW(j, j+1) != zero) {
                    SLC_DLARTG(&DW(j, j), &DW(j, j+1), &c_val, &s_val, &r_val);
                    DW(j, j) = r_val;
                    DW(j, j+1) = zero;
                    SLC_DROT(&mj, &DW(j+1, j), &int1, &DW(j+1, j+1), &int1, &c_val, &s_val);
                    // D(J), D(J+1) in Fortran = d[j-1], d[j] in C
                    SLC_DROT(&int1, &d[j-1], &int1, &d[j], &int1, &c_val, &s_val);
                }
            }
        }
    }

    // Estimate condition number and solve
    SLC_DTRCON("1-norm", ul, "Non-unit", &m2, dwork, &lddwor, &rcond,
               &dwork[m2 * lddwor], iwork, info);

    if (rcond <= tol) {
        *info = 1;
    } else {
        SLC_DTRSV(ul, trans_buf, "Non-unit", &m2, dwork, &lddwor, d, &int1);
    }
}

#undef DW
