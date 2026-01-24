/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <stdbool.h>

static inline i32 min_i32(i32 a, i32 b) { return a < b ? a : b; }
static inline i32 max_i32(i32 a, i32 b) { return a > b ? a : b; }

void mb01oc(const char* uplo_str, const char* trans_str, i32 n,
            f64 alpha, f64 beta, f64* r, i32 ldr,
            const f64* h, i32 ldh, const f64* x, i32 ldx, i32* info)
{
    const f64 zero = 0.0;
    const f64 one = 1.0;

    char uplo = uplo_str[0];
    char trans = trans_str[0];
    bool luplo = (uplo == 'U' || uplo == 'u');
    bool ltrans = (trans == 'T' || trans == 't' || trans == 'C' || trans == 'c');

    *info = 0;

    if (!luplo && uplo != 'L' && uplo != 'l') {
        *info = -1;
    } else if (!ltrans && trans != 'N' && trans != 'n') {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    } else if (ldr < max_i32(1, n)) {
        *info = -7;
    } else if (ldh < max_i32(1, n)) {
        *info = -9;
    } else if (ldx < max_i32(1, n)) {
        *info = -11;
    }

    if (*info != 0) {
        return;
    }

    if (n == 0) {
        return;
    }

    if (beta == zero) {
        if (alpha == zero) {
            if (luplo) {
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
            if (luplo) {
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
        if (luplo) {
            for (i32 j = 1; j <= n; j++) {
                if (alpha == zero) {
                    for (i32 i = 1; i <= j; i++) {
                        r[(i-1) + (j-1) * ldr] = zero;
                    }
                } else if (alpha != one) {
                    for (i32 i = 1; i <= j; i++) {
                        r[(i-1) + (j-1) * ldr] *= alpha;
                    }
                }
                i32 ii = max_i32(1, j - 1);
                for (i32 l = 1; l <= n; l++) {
                    f64 temp1;
                    if (l <= j) {
                        temp1 = x[(l-1) + (j-1) * ldx];
                    } else {
                        temp1 = x[(j-1) + (l-1) * ldx];
                    }
                    if (temp1 != zero) {
                        i32 axpy_n = min_i32(l + 1, j);
                        f64 axpy_alpha = beta * temp1;
                        for (i32 i = 1; i <= axpy_n; i++) {
                            r[(i-1) + (j-1) * ldr] += axpy_alpha * h[(i-1) + (l-1) * ldh];
                        }
                    }
                    if (l >= ii) {
                        f64 temp2 = h[(j-1) + (l-1) * ldh];
                        if (temp2 != zero) {
                            temp2 *= beta;
                            for (i32 i = 1; i <= ii; i++) {
                                r[(i-1) + (j-1) * ldr] += temp2 * x[(i-1) + (l-1) * ldx];
                            }
                            if (j > 1) {
                                r[(j-1) + (j-1) * ldr] += temp1 * temp2;
                            }
                        }
                    }
                }
            }
        } else {
            for (i32 j = 1; j <= n; j++) {
                if (alpha == zero) {
                    for (i32 i = j; i <= n; i++) {
                        r[(i-1) + (j-1) * ldr] = zero;
                    }
                } else if (alpha != one) {
                    for (i32 i = j; i <= n; i++) {
                        r[(i-1) + (j-1) * ldr] *= alpha;
                    }
                }
                for (i32 l = max_i32(1, j - 1); l <= n; l++) {
                    i32 ii = min_i32(l + 1, n);
                    f64 temp2 = beta * h[(j-1) + (l-1) * ldh];
                    f64 temp1;
                    if (l >= j) {
                        temp1 = beta * x[(l-1) + (j-1) * ldx];
                    } else {
                        temp1 = beta * x[(j-1) + (l-1) * ldx];
                    }
                    for (i32 i = j; i <= ii; i++) {
                        r[(i-1) + (j-1) * ldr] += temp1 * h[(i-1) + (l-1) * ldh];
                    }
                    for (i32 i = l; i <= n; i++) {
                        r[(i-1) + (j-1) * ldr] += temp2 * x[(i-1) + (l-1) * ldx];
                    }
                    if (l > j) {
                        for (i32 i = j; i <= l - 1; i++) {
                            r[(i-1) + (j-1) * ldr] += temp2 * x[(l-1) + (i-1) * ldx];
                        }
                    }
                }
            }
        }
    } else {
        if (luplo) {
            for (i32 j = 1; j <= n; j++) {
                if (alpha == zero) {
                    for (i32 i = 1; i <= j; i++) {
                        r[(i-1) + (j-1) * ldr] = zero;
                    }
                } else if (alpha != one) {
                    for (i32 i = 1; i <= j; i++) {
                        r[(i-1) + (j-1) * ldr] *= alpha;
                    }
                }
                for (i32 i = 1; i <= j; i++) {
                    for (i32 l = 1; l <= min_i32(j + 1, n); l++) {
                        f64 temp1, temp2;
                        if (l <= j) {
                            temp1 = x[(l-1) + (j-1) * ldx];
                            if (l <= i) {
                                temp2 = x[(l-1) + (i-1) * ldx];
                            } else {
                                temp2 = x[(i-1) + (l-1) * ldx];
                            }
                        } else {
                            temp1 = x[(j-1) + (l-1) * ldx];
                            temp2 = x[(i-1) + (l-1) * ldx];
                        }
                        if (l <= min_i32(i + 1, n)) {
                            r[(i-1) + (j-1) * ldr] += beta * temp1 * h[(l-1) + (i-1) * ldh];
                        }
                        r[(i-1) + (j-1) * ldr] += beta * temp2 * h[(l-1) + (j-1) * ldh];
                    }
                }
            }
        } else {
            for (i32 j = 1; j <= n; j++) {
                if (alpha == zero) {
                    for (i32 i = j; i <= n; i++) {
                        r[(i-1) + (j-1) * ldr] = zero;
                    }
                } else if (alpha != one) {
                    for (i32 i = j; i <= n; i++) {
                        r[(i-1) + (j-1) * ldr] *= alpha;
                    }
                }
                for (i32 i = j; i <= n; i++) {
                    for (i32 l = 1; l <= min_i32(i + 1, n); l++) {
                        f64 temp1, temp2;
                        if (l >= i) {
                            temp1 = x[(l-1) + (j-1) * ldx];
                            temp2 = x[(l-1) + (i-1) * ldx];
                        } else {
                            if (l >= j) {
                                temp1 = x[(l-1) + (j-1) * ldx];
                            } else {
                                temp1 = x[(j-1) + (l-1) * ldx];
                            }
                            temp2 = x[(i-1) + (l-1) * ldx];
                        }
                        r[(i-1) + (j-1) * ldr] += beta * temp1 * h[(l-1) + (i-1) * ldh];
                        if (l <= min_i32(j + 1, n)) {
                            r[(i-1) + (j-1) * ldr] += beta * temp2 * h[(l-1) + (j-1) * ldh];
                        }
                    }
                }
            }
        }
    }
}
