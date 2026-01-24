/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <stdbool.h>

static inline i32 max_i32(i32 a, i32 b) { return a > b ? a : b; }
static inline i32 min_i32(i32 a, i32 b) { return a < b ? a : b; }

void mb01oe(const char* uplo_str, const char* trans_str, i32 n,
            f64 alpha, f64 beta, f64* r, i32 ldr,
            const f64* h, i32 ldh, const f64* e, i32 lde, i32* info)
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
    } else if (lde < max_i32(1, n)) {
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
            f64 beta2 = two * beta;

            if (alpha == zero) {
                r[0] = zero;
            } else if (alpha != one) {
                r[0] *= alpha;
            }

            f64 dot_val = 0.0;
            for (i32 k = 0; k < n; k++) {
                dot_val += h[0 + k * ldh] * e[0 + k * lde];
            }
            r[0] += beta2 * dot_val;

            for (i32 j = 1; j < n; j++) {
                if (alpha == zero) {
                    for (i32 i = 0; i <= j; i++) {
                        r[i + j * ldr] = zero;
                    }
                } else if (alpha != one) {
                    for (i32 i = 0; i <= j; i++) {
                        r[i + j * ldr] *= alpha;
                    }
                }

                f64 h_jm1_jm1 = h[j + (j - 1) * ldh];
                f64 scale = beta * h_jm1_jm1;
                for (i32 i = 0; i < j; i++) {
                    r[i + j * ldr] += scale * e[i + (j - 1) * lde];
                }

                for (i32 ii = j; ii < n; ii++) {
                    f64 e_j_ii = e[j + ii * lde];
                    f64 h_j_ii = h[j + ii * ldh];
                    f64 scale_e = beta * e_j_ii;
                    f64 scale_h = beta * h_j_ii;
                    for (i32 i = 0; i <= j; i++) {
                        r[i + j * ldr] += scale_e * h[i + ii * ldh];
                        r[i + j * ldr] += scale_h * e[i + ii * lde];
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

                for (i32 ii = max_i32(0, j - 1); ii < n - 1; ii++) {
                    i32 len1 = min_i32(ii + 2, n) - j;
                    f64 scale_e = beta * e[j + ii * lde];
                    for (i32 i = j; i < j + len1; i++) {
                        r[i + j * ldr] += scale_e * h[i + ii * ldh];
                    }

                    i32 len2 = ii - j + 1;
                    f64 scale_h = beta * h[j + ii * ldh];
                    for (i32 i = j; i < j + len2; i++) {
                        r[i + j * ldr] += scale_h * e[i + ii * lde];
                    }
                }

                i32 last_col = n - 1;
                f64 scale_e_last = beta * e[j + last_col * lde];
                f64 scale_h_last = beta * h[j + last_col * ldh];
                for (i32 i = j; i < n; i++) {
                    r[i + j * ldr] += scale_e_last * h[i + last_col * ldh];
                    r[i + j * ldr] += scale_h_last * e[i + last_col * lde];
                }
            }
        }
    } else {
        f64 beta2 = two * beta;

        if (upper) {
            for (i32 j = 0; j < n; j++) {
                for (i32 i = 0; i < j; i++) {
                    f64 sum1 = 0.0;
                    for (i32 l = 0; l <= i + 1 && l < n; l++) {
                        sum1 += h[l + i * ldh] * e[l + j * lde];
                    }
                    f64 sum2 = 0.0;
                    for (i32 l = 0; l <= i; l++) {
                        sum2 += e[l + i * lde] * h[l + j * ldh];
                    }
                    f64 temp = beta * (sum1 + sum2);

                    if (alpha == zero) {
                        r[i + j * ldr] = temp;
                    } else {
                        r[i + j * ldr] = alpha * r[i + j * ldr] + temp;
                    }
                }

                f64 diag_sum = 0.0;
                for (i32 l = 0; l <= j; l++) {
                    diag_sum += h[l + j * ldh] * e[l + j * lde];
                }
                f64 temp = beta2 * diag_sum;

                if (alpha == zero) {
                    r[j + j * ldr] = temp;
                } else {
                    r[j + j * ldr] = alpha * r[j + j * ldr] + temp;
                }
            }
        } else {
            for (i32 j = 0; j < n; j++) {
                f64 diag_sum = 0.0;
                for (i32 l = 0; l <= j; l++) {
                    diag_sum += h[l + j * ldh] * e[l + j * lde];
                }
                f64 temp = beta2 * diag_sum;

                if (alpha == zero) {
                    r[j + j * ldr] = temp;
                } else {
                    r[j + j * ldr] = alpha * r[j + j * ldr] + temp;
                }

                i32 j1 = j + 1;
                for (i32 i = j1; i < n; i++) {
                    f64 sum1 = 0.0;
                    for (i32 l = 0; l <= j; l++) {
                        sum1 += h[l + i * ldh] * e[l + j * lde];
                    }
                    f64 sum2 = 0.0;
                    for (i32 l = 0; l <= j1 && l < n; l++) {
                        sum2 += e[l + i * lde] * h[l + j * ldh];
                    }
                    temp = beta * (sum1 + sum2);

                    if (alpha == zero) {
                        r[i + j * ldr] = temp;
                    } else {
                        r[i + j * ldr] = alpha * r[i + j * ldr] + temp;
                    }
                }
            }
        }
    }
}
