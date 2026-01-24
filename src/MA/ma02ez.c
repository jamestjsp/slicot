/**
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * MA02EZ - Store by (skew-)symmetry the upper or lower triangle of a
 *          (skew-)symmetric/Hermitian complex matrix
 */

#include "slicot.h"
#include <ctype.h>

void ma02ez(const char uplo, const char trans, const char skew,
            i32 n, c128 *a, i32 lda) {
    char uplo_upper = toupper((unsigned char)uplo);
    char trans_upper = toupper((unsigned char)trans);
    char skew_upper = toupper((unsigned char)skew);

    if (uplo_upper == 'L') {
        if (trans_upper == 'T') {
            if (skew_upper == 'S') {
                for (i32 i = 0; i < n; i++) {
                    for (i32 j = i + 1; j < n; j++) {
                        a[i + j*lda] = -a[j + i*lda];
                    }
                }
            } else {
                for (i32 i = 0; i < n; i++) {
                    for (i32 j = i + 1; j < n; j++) {
                        a[i + j*lda] = a[j + i*lda];
                    }
                }
            }
        } else {
            if (skew_upper == 'G') {
                for (i32 i = 0; i < n; i++) {
                    for (i32 j = i; j < n; j++) {
                        a[i + j*lda] = conj(a[j + i*lda]);
                    }
                }
            } else if (skew_upper == 'N') {
                for (i32 i = 0; i < n; i++) {
                    a[i + i*lda] = creal(a[i + i*lda]);
                    for (i32 j = i + 1; j < n; j++) {
                        a[i + j*lda] = conj(a[j + i*lda]);
                    }
                }
            } else {
                for (i32 i = 0; i < n; i++) {
                    a[i + i*lda] = cimag(a[i + i*lda]) * I;
                    for (i32 j = i + 1; j < n; j++) {
                        a[i + j*lda] = -conj(a[j + i*lda]);
                    }
                }
            }
        }
    } else if (uplo_upper == 'U') {
        if (trans_upper == 'T') {
            if (skew_upper == 'S') {
                for (i32 i = 0; i < n; i++) {
                    for (i32 j = i + 1; j < n; j++) {
                        a[j + i*lda] = -a[i + j*lda];
                    }
                }
            } else {
                for (i32 i = 0; i < n; i++) {
                    for (i32 j = i + 1; j < n; j++) {
                        a[j + i*lda] = a[i + j*lda];
                    }
                }
            }
        } else {
            if (skew_upper == 'G') {
                for (i32 i = 0; i < n; i++) {
                    for (i32 j = i; j < n; j++) {
                        a[j + i*lda] = conj(a[i + j*lda]);
                    }
                }
            } else if (skew_upper == 'N') {
                for (i32 i = 0; i < n; i++) {
                    a[i + i*lda] = creal(a[i + i*lda]);
                    for (i32 j = i + 1; j < n; j++) {
                        a[j + i*lda] = conj(a[i + j*lda]);
                    }
                }
            } else {
                for (i32 i = 0; i < n; i++) {
                    a[i + i*lda] = cimag(a[i + i*lda]) * I;
                    for (i32 j = i + 1; j < n; j++) {
                        a[j + i*lda] = -conj(a[i + j*lda]);
                    }
                }
            }
        }
    }
}
