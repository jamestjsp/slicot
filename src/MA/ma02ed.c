/**
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * MA02ED - Store by symmetry the upper or lower triangle of a symmetric matrix
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <ctype.h>

void ma02ed(const char uplo, i32 n, f64 *a, i32 lda) {
    char uplo_upper = toupper((unsigned char)uplo);
    i32 inc_one = 1;

    if (uplo_upper == 'L') {
        // Construct upper triangle from lower triangle
        for (i32 j = 1; j < n; j++) {
            // Copy A(j,0:j-1) to A(0:j-1,j)
            // DCOPY(J-1, A(J,1), LDA, A(1,J), 1)
            i32 len = j;
            SLC_DCOPY(&len, &a[j + 0*lda], &lda, &a[0 + j*lda], &inc_one);
        }
    } else if (uplo_upper == 'U') {
        // Construct lower triangle from upper triangle
        for (i32 j = 1; j < n; j++) {
            // Copy A(0:j-1,j) to A(j,0:j-1)
            // DCOPY(J-1, A(1,J), 1, A(J,1), LDA)
            i32 len = j;
            SLC_DCOPY(&len, &a[0 + j*lda], &inc_one, &a[j + 0*lda], &lda);
        }
    }
}
