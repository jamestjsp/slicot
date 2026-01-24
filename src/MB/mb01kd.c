/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <stdbool.h>

void mb01kd(const char* uplo_str, const char* trans_str, i32 n, i32 k,
            f64 alpha, const f64* a, i32 lda, const f64* b, i32 ldb,
            f64 beta, f64* c, i32 ldc, i32* info)
{
    const f64 zero = 0.0;
    const f64 one = 1.0;

    char uplo = uplo_str[0];
    char trans = trans_str[0];
    bool lup = (uplo == 'U' || uplo == 'u');
    bool ltran = (trans == 'T' || trans == 't' || trans == 'C' || trans == 'c');

    *info = 0;

    if (!lup && uplo != 'L' && uplo != 'l') {
        *info = -1;
    } else if (!ltran && trans != 'N' && trans != 'n') {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    } else if (k < 0) {
        *info = -4;
    } else if ((!ltran && lda < n) || lda < 1 || (ltran && lda < k)) {
        *info = -7;
    } else if ((!ltran && ldb < n) || ldb < 1 || (ltran && ldb < k)) {
        *info = -9;
    } else if (ldc < (n > 1 ? n : 1)) {
        *info = -12;
    }

    if (*info != 0) {
        return;
    }

    if ((n <= 1) || (((alpha == zero) || (k == 0)) && (beta == one))) {
        return;
    }

    if (alpha == zero) {
        if (lup) {
            if (beta == zero) {
                for (i32 j = 1; j < n; j++) {
                    for (i32 i = 0; i < j; i++) {
                        c[i + j * ldc] = zero;
                    }
                }
            } else {
                for (i32 j = 1; j < n; j++) {
                    for (i32 i = 0; i < j; i++) {
                        c[i + j * ldc] = beta * c[i + j * ldc];
                    }
                }
            }
        } else {
            if (beta == zero) {
                for (i32 j = 0; j < n - 1; j++) {
                    for (i32 i = j + 1; i < n; i++) {
                        c[i + j * ldc] = zero;
                    }
                }
            } else {
                for (i32 j = 0; j < n - 1; j++) {
                    for (i32 i = j + 1; i < n; i++) {
                        c[i + j * ldc] = beta * c[i + j * ldc];
                    }
                }
            }
        }
        return;
    }

    if (!ltran) {
        if (lup) {
            for (i32 j = 1; j < n; j++) {
                if (beta == zero) {
                    for (i32 i = 0; i < j; i++) {
                        c[i + j * ldc] = zero;
                    }
                } else if (beta != one) {
                    for (i32 i = 0; i < j; i++) {
                        c[i + j * ldc] = beta * c[i + j * ldc];
                    }
                }
                for (i32 l = 0; l < k; l++) {
                    if ((a[j + l * lda] != zero) || (b[j + l * ldb] != zero)) {
                        f64 temp1 = alpha * b[j + l * ldb];
                        f64 temp2 = alpha * a[j + l * lda];
                        for (i32 i = 0; i < j; i++) {
                            c[i + j * ldc] = c[i + j * ldc] +
                                             a[i + l * lda] * temp1 -
                                             b[i + l * ldb] * temp2;
                        }
                    }
                }
            }
        } else {
            for (i32 j = 0; j < n - 1; j++) {
                if (beta == zero) {
                    for (i32 i = j + 1; i < n; i++) {
                        c[i + j * ldc] = zero;
                    }
                } else if (beta != one) {
                    for (i32 i = j + 1; i < n; i++) {
                        c[i + j * ldc] = beta * c[i + j * ldc];
                    }
                }
                for (i32 l = 0; l < k; l++) {
                    if ((a[j + l * lda] != zero) || (b[j + l * ldb] != zero)) {
                        f64 temp1 = alpha * b[j + l * ldb];
                        f64 temp2 = alpha * a[j + l * lda];
                        for (i32 i = j + 1; i < n; i++) {
                            c[i + j * ldc] = c[i + j * ldc] +
                                             a[i + l * lda] * temp1 -
                                             b[i + l * ldb] * temp2;
                        }
                    }
                }
            }
        }
    } else {
        if (lup) {
            for (i32 j = 1; j < n; j++) {
                for (i32 i = 0; i < j; i++) {
                    f64 temp1 = zero;
                    f64 temp2 = zero;
                    for (i32 l = 0; l < k; l++) {
                        temp1 = temp1 + a[l + i * lda] * b[l + j * ldb];
                        temp2 = temp2 + b[l + i * ldb] * a[l + j * lda];
                    }
                    if (beta == zero) {
                        c[i + j * ldc] = alpha * temp1 - alpha * temp2;
                    } else {
                        c[i + j * ldc] = beta * c[i + j * ldc] +
                                         alpha * temp1 - alpha * temp2;
                    }
                }
            }
        } else {
            for (i32 j = 0; j < n - 1; j++) {
                for (i32 i = j + 1; i < n; i++) {
                    f64 temp1 = zero;
                    f64 temp2 = zero;
                    for (i32 l = 0; l < k; l++) {
                        temp1 = temp1 + a[l + i * lda] * b[l + j * ldb];
                        temp2 = temp2 + b[l + i * ldb] * a[l + j * lda];
                    }
                    if (beta == zero) {
                        c[i + j * ldc] = alpha * temp1 - alpha * temp2;
                    } else {
                        c[i + j * ldc] = beta * c[i + j * ldc] +
                                         alpha * temp1 - alpha * temp2;
                    }
                }
            }
        }
    }
}
