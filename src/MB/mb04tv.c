// SPDX-License-Identifier: BSD-3-Clause

#include "slicot.h"
#include "slicot_blas.h"

void mb04tv(bool updatz, i32 n, i32 nra, i32 nca, i32 ifira, i32 ifica,
            f64 *a, i32 lda, f64 *e, i32 lde, f64 *z, i32 ldz, i32 *info) {
    *info = 0;

    if (n <= 0 || nra <= 0 || nca <= 0) {
        return;
    }

    i32 ifira1 = ifira - 1;  // Helper for E row count (0-based: 0 to ifira-2)
    i32 jpvt = ifica + nca;  // Initial pivot column (1-based, will decrement)
    i32 int1 = 1;

    // Process rows from bottom to top
    // Fortran: DO I = IFIRA1 + NRA, IFIRA, -1
    // i is 1-based row index in A
    for (i32 i = ifira1 + nra; i >= ifira; i--) {
        jpvt--;  // Decrement pivot column

        // Eliminate elements left of diagonal in row i
        // Fortran: DO J = JPVT - 1, IFICA, -1
        for (i32 j = jpvt - 1; j >= ifica; j--) {
            f64 sc, ss;

            // Generate Givens rotation to annihilate A(i,j) using A(i,jpvt)
            // C indices: row i-1, cols jpvt-1 and j-1
            i32 i_idx = i - 1;      // 0-based row
            i32 jpvt_idx = jpvt - 1; // 0-based pivot column
            i32 j_idx = j - 1;       // 0-based column to annihilate

            SLC_DROTG(&a[i_idx + jpvt_idx * lda], &a[i_idx + j_idx * lda], &sc, &ss);

            // Apply rotation to rows 1:i-1 of columns jpvt and j in A
            // Fortran: CALL DROT(I-1, A(1,JPVT), 1, A(1,J), 1, SC, SS)
            i32 nrot = i - 1;  // Number of rows to rotate (rows 1 to i-1)
            if (nrot > 0) {
                SLC_DROT(&nrot, &a[jpvt_idx * lda], &int1,
                         &a[j_idx * lda], &int1, &sc, &ss);
            }

            // Set annihilated element to zero
            a[i_idx + j_idx * lda] = 0.0;

            // Apply same rotation to E matrix (rows 1:ifira-1)
            // Fortran: CALL DROT(IFIRA1, E(1,JPVT), 1, E(1,J), 1, SC, SS)
            if (ifira1 > 0) {
                SLC_DROT(&ifira1, &e[jpvt_idx * lde], &int1,
                         &e[j_idx * lde], &int1, &sc, &ss);
            }

            // Update Z if requested
            // Fortran: IF(UPDATZ) CALL DROT(N, Z(1,JPVT), 1, Z(1,J), 1, SC, SS)
            if (updatz) {
                SLC_DROT(&n, &z[jpvt_idx * ldz], &int1,
                         &z[j_idx * ldz], &int1, &sc, &ss);
            }
        }
    }
}
