/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <stdbool.h>

static inline i32 max_i32(i32 a, i32 b) { return a > b ? a : b; }

void mb01oh(const char* uplo_str, const char* trans_str, i32 n,
            f64 alpha, f64 beta, f64* r, i32 ldr,
            const f64* h, i32 ldh, const f64* a, i32 lda, i32* info)
{
    const f64 zero = 0.0;
    const f64 one = 1.0;
    const f64 two = 2.0;

    char uplo = uplo_str[0];
    char trans = trans_str[0];
    bool upper = (uplo == 'U' || uplo == 'u');
    bool ltrans = (trans == 'T' || trans == 't' || trans == 'C' || trans == 'c');

    *info = 0;

    if (!upper && uplo != 'L' && uplo != 'l') {
        *info = -1;
    } else if (!ltrans && trans != 'N' && trans != 'n') {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    } else if (ldr < max_i32(1, n)) {
        *info = -7;
    } else if (ldh < max_i32(1, n)) {
        *info = -9;
    } else if (lda < max_i32(1, n)) {
        *info = -11;
    }

    if (*info != 0) {
        return;
    }

    if (n == 0 || (beta == zero && alpha == one)) {
        return;
    }

    if (beta == zero) {
        if (alpha == zero) {
            if (upper) {
                for (i32 j = 0; j < n; j++) {
                    for (i32 i = 0; i <= j; i++) {
                        r[i + j * ldr] = zero;
                    }
                }
            } else {
                for (i32 j = 0; j < n; j++) {
                    for (i32 i = j; i < n; i++) {
                        r[i + j * ldr] = zero;
                    }
                }
            }
        } else if (alpha != one) {
            if (upper) {
                for (i32 j = 0; j < n; j++) {
                    for (i32 i = 0; i <= j; i++) {
                        r[i + j * ldr] *= alpha;
                    }
                }
            } else {
                for (i32 j = 0; j < n; j++) {
                    for (i32 i = j; i < n; i++) {
                        r[i + j * ldr] *= alpha;
                    }
                }
            }
        }
        return;
    }

    if (!ltrans) {
        if (upper) {
            for (i32 j = 0; j < n; j++) {
                if (alpha == zero) {
                    for (i32 i = 0; i <= j; i++) {
                        r[i + j * ldr] = zero;
                    }
                } else if (alpha != one) {
                    for (i32 i = 0; i <= j; i++) {
                        r[i + j * ldr] *= alpha;
                    }
                }

                i32 i_start = max_i32(0, j - 1);
                for (i32 ii = i_start; ii < n; ii++) {
                    f64 scale_a = beta * a[j + ii * lda];
                    f64 scale_h = beta * h[j + ii * ldh];
                    for (i32 i = 0; i <= j; i++) {
                        r[i + j * ldr] += scale_a * h[i + ii * ldh];
                        r[i + j * ldr] += scale_h * a[i + ii * lda];
                    }
                }
            }
        } else {
            for (i32 j = 0; j < n; j++) {
                if (alpha == zero) {
                    for (i32 i = j; i < n; i++) {
                        r[i + j * ldr] = zero;
                    }
                } else if (alpha != one) {
                    for (i32 i = j; i < n; i++) {
                        r[i + j * ldr] *= alpha;
                    }
                }

                i32 i_start = max_i32(0, j - 1);
                for (i32 ii = i_start; ii < n - 1; ii++) {
                    i32 len = ii - j + 2;
                    if (len > n - j) len = n - j;
                    f64 scale_a = beta * a[j + ii * lda];
                    f64 scale_h = beta * h[j + ii * ldh];
                    for (i32 i = j; i < j + len; i++) {
                        r[i + j * ldr] += scale_a * h[i + ii * ldh];
                        r[i + j * ldr] += scale_h * a[i + ii * lda];
                    }
                }

                i32 last_col = n - 1;
                f64 scale_a_last = beta * a[j + last_col * lda];
                f64 scale_h_last = beta * h[j + last_col * ldh];
                for (i32 i = j; i < n; i++) {
                    r[i + j * ldr] += scale_a_last * h[i + last_col * ldh];
                    r[i + j * ldr] += scale_h_last * a[i + last_col * lda];
                }
            }
        }
    } else {
        f64 beta2 = two * beta;

        if (upper) {
            for (i32 j = 0; j < n - 1; j++) {
                for (i32 i = 0; i <= j; i++) {
                    f64 sum1 = zero;
                    i32 len1 = i + 2;
                    if (len1 > n) len1 = n;
                    for (i32 l = 0; l < len1; l++) {
                        sum1 += h[l + i * ldh] * a[l + j * lda];
                    }
                    f64 sum2 = zero;
                    i32 len2 = i + 2;
                    if (len2 > n) len2 = n;
                    for (i32 l = 0; l < len2; l++) {
                        sum2 += a[l + i * lda] * h[l + j * ldh];
                    }
                    f64 temp = beta * (sum1 + sum2);

                    if (alpha == zero) {
                        r[i + j * ldr] = temp;
                    } else {
                        r[i + j * ldr] = alpha * r[i + j * ldr] + temp;
                    }
                }
            }

            for (i32 i = 0; i < n - 1; i++) {
                f64 sum1 = zero;
                i32 len1 = i + 2;
                if (len1 > n) len1 = n;
                for (i32 l = 0; l < len1; l++) {
                    sum1 += h[l + i * ldh] * a[l + (n - 1) * lda];
                }
                f64 sum2 = zero;
                for (i32 l = 0; l < len1; l++) {
                    sum2 += a[l + i * lda] * h[l + (n - 1) * ldh];
                }
                f64 temp = beta * (sum1 + sum2);

                if (alpha == zero) {
                    r[i + (n - 1) * ldr] = temp;
                } else {
                    r[i + (n - 1) * ldr] = alpha * r[i + (n - 1) * ldr] + temp;
                }
            }

            f64 diag_sum = zero;
            for (i32 l = 0; l < n; l++) {
                diag_sum += h[l + (n - 1) * ldh] * a[l + (n - 1) * lda];
            }
            f64 temp = beta2 * diag_sum;

            if (alpha == zero) {
                r[(n - 1) + (n - 1) * ldr] = temp;
            } else {
                r[(n - 1) + (n - 1) * ldr] = alpha * r[(n - 1) + (n - 1) * ldr] + temp;
            }
        } else {
            for (i32 j = 0; j < n - 1; j++) {
                i32 j1 = j + 1;

                for (i32 i = j; i < n; i++) {
                    f64 sum1 = zero;
                    for (i32 l = 0; l <= j1 && l < n; l++) {
                        sum1 += h[l + i * ldh] * a[l + j * lda];
                    }
                    f64 sum2 = zero;
                    for (i32 l = 0; l <= j1 && l < n; l++) {
                        sum2 += a[l + i * lda] * h[l + j * ldh];
                    }
                    f64 temp = beta * (sum1 + sum2);

                    if (alpha == zero) {
                        r[i + j * ldr] = temp;
                    } else {
                        r[i + j * ldr] = alpha * r[i + j * ldr] + temp;
                    }
                }
            }

            f64 diag_sum = zero;
            for (i32 l = 0; l < n; l++) {
                diag_sum += h[l + (n - 1) * ldh] * a[l + (n - 1) * lda];
            }
            f64 temp = beta2 * diag_sum;

            if (alpha == zero) {
                r[(n - 1) + (n - 1) * ldr] = temp;
            } else {
                r[(n - 1) + (n - 1) * ldr] = alpha * r[(n - 1) + (n - 1) * ldr] + temp;
            }
        }
    }
}
