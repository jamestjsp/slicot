/**
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * MA02ES - Store by skew-symmetry the upper or lower triangle of a skew-symmetric matrix
 */

#include "slicot.h"
#include <ctype.h>

void ma02es(const char uplo, i32 n, f64 *a, i32 lda) {
    char uplo_upper = toupper((unsigned char)uplo);
    const f64 ZERO = 0.0;

    if (uplo_upper == 'L') {
        for (i32 i = 0; i < n; i++) {
            a[i + i*lda] = ZERO;
            for (i32 j = 1; j < n; j++) {
                a[i + j*lda] = -a[j + i*lda];
            }
        }
    } else if (uplo_upper == 'U') {
        for (i32 i = 0; i < n; i++) {
            a[i + i*lda] = ZERO;
            for (i32 j = 1; j < n; j++) {
                a[j + i*lda] = -a[i + j*lda];
            }
        }
    }
}
