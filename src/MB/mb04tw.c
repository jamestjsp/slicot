// SPDX-License-Identifier: BSD-3-Clause

#include "slicot.h"
#include "slicot_blas.h"

void mb04tw(bool updatq, i32 m, i32 n, i32 nre, i32 nce, i32 ifire, i32 ifice,
            i32 ifica, f64 *a, i32 lda, f64 *e, i32 lde, f64 *q, i32 ldq,
            i32 *info) {
    *info = 0;

    if (m <= 0 || n <= 0 || nre <= 0 || nce <= 0) {
        return;
    }

    i32 ipvt = ifire - 1;  // Initial pivot row (1-based), increments in outer loop
    i32 int1 = 1;

    // Process columns from left to right
    // Fortran: DO J = IFICE, IFICE + NCE - 1
    for (i32 j = ifice; j <= ifice + nce - 1; j++) {
        ipvt++;  // Increment pivot row

        // Eliminate elements below diagonal in column j
        // Fortran: DO I = IPVT + 1, IFIRE + NRE - 1
        for (i32 i = ipvt + 1; i <= ifire + nre - 1; i++) {
            f64 sc, ss;

            // 0-based indices
            i32 ipvt_idx = ipvt - 1;  // Pivot row
            i32 i_idx = i - 1;        // Row to annihilate
            i32 j_idx = j - 1;        // Current column

            // Generate Givens rotation to annihilate E(i,j) using E(ipvt,j)
            SLC_DROTG(&e[ipvt_idx + j_idx * lde], &e[i_idx + j_idx * lde], &sc, &ss);

            // Apply rotation to rows ipvt and i in E, columns j+1:n
            // Fortran: CALL DROT(N-J, E(IPVT,J+1), LDE, E(I,J+1), LDE, SC, SS)
            i32 nrot_e = n - j;  // Number of columns (j+1 to n, 1-based)
            if (nrot_e > 0) {
                SLC_DROT(&nrot_e, &e[ipvt_idx + j * lde], &lde,
                         &e[i_idx + j * lde], &lde, &sc, &ss);
            }

            // Set annihilated element to zero
            e[i_idx + j_idx * lde] = 0.0;

            // Apply same rotation to A matrix (columns ifica:n)
            // Fortran: CALL DROT(N-IFICA+1, A(IPVT,IFICA), LDA, A(I,IFICA), LDA, SC, SS)
            i32 nrot_a = n - ifica + 1;
            i32 ifica_idx = ifica - 1;  // 0-based
            SLC_DROT(&nrot_a, &a[ipvt_idx + ifica_idx * lda], &lda,
                     &a[i_idx + ifica_idx * lda], &lda, &sc, &ss);

            // Update Q if requested
            // Fortran: IF(UPDATQ) CALL DROT(M, Q(1,IPVT), 1, Q(1,I), 1, SC, SS)
            if (updatq) {
                SLC_DROT(&m, &q[ipvt_idx * ldq], &int1,
                         &q[i_idx * ldq], &int1, &sc, &ss);
            }
        }
    }
}
