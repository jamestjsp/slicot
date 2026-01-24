/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <stdbool.h>

static inline i32 max_i32(i32 a, i32 b) { return a > b ? a : b; }

void mb01ot(const char* uplo_str, const char* trans_str, i32 n,
            f64 alpha, f64 beta, f64* r, i32 ldr,
            const f64* e, i32 lde, const f64* t, i32 ldt, i32* info)
{
    const f64 zero = 0.0;
    const f64 one = 1.0;
    i32 inc1 = 1;

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
    } else if (lde < max_i32(1, n)) {
        *info = -9;
    } else if (ldt < max_i32(1, n)) {
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
            SLC_DLASET(uplo_str, &n, &n, &zero, &zero, r, &ldr);
        } else {
            if (alpha != one) {
                i32 kl = 0, ku = 0;
                SLC_DLASCL(uplo_str, &kl, &ku, &one, &alpha, &n, &n, r, &ldr, info);
            }
        }
        return;
    }

    f64 tmp_val = zero;
    i32 inc0 = 0;

    if (!ltrans) {
        if (upper) {
            for (i32 jf = 1; jf <= n; jf++) {
                i32 j = jf - 1;

                if (alpha == zero) {
                    tmp_val = zero;
                    SLC_DCOPY(&jf, &tmp_val, &inc0, &r[j * ldr], &inc1);
                } else if (alpha != one) {
                    SLC_DSCAL(&jf, &alpha, &r[j * ldr], &inc1);
                }

                for (i32 if_ = jf; if_ <= n; if_++) {
                    i32 i = if_ - 1;
                    f64 bt_ji = beta * t[j + i * ldt];
                    f64 be_ji = beta * e[j + i * lde];
                    SLC_DAXPY(&jf, &bt_ji, &e[i * lde], &inc1, &r[j * ldr], &inc1);
                    SLC_DAXPY(&jf, &be_ji, &t[i * ldt], &inc1, &r[j * ldr], &inc1);
                }
            }
        } else {
            for (i32 jf = 1; jf <= n; jf++) {
                i32 j = jf - 1;

                if (alpha == zero) {
                    tmp_val = zero;
                    SLC_DCOPY(&jf, &tmp_val, &inc0, &r[j], &ldr);
                } else if (alpha != one) {
                    SLC_DSCAL(&jf, &alpha, &r[j], &ldr);
                }

                for (i32 if_ = 1; if_ <= jf; if_++) {
                    i32 i = if_ - 1;
                    f64 bt_ij = beta * t[i + j * ldt];
                    f64 be_ij = beta * e[i + j * lde];
                    SLC_DAXPY(&if_, &bt_ij, &e[j * lde], &inc1, &r[i], &ldr);
                    SLC_DAXPY(&if_, &be_ij, &t[j * ldt], &inc1, &r[i], &ldr);
                }
            }
        }
    } else {
        if (upper) {
            for (i32 jf = 1; jf <= n; jf++) {
                i32 j = jf - 1;

                for (i32 if_ = 1; if_ <= jf; if_++) {
                    i32 i = if_ - 1;
                    f64 temp = beta * (SLC_DDOT(&if_, &e[i * lde], &inc1, &t[j * ldt], &inc1) +
                                       SLC_DDOT(&if_, &t[i * ldt], &inc1, &e[j * lde], &inc1));
                    if (alpha == zero) {
                        r[i + j * ldr] = temp;
                    } else {
                        r[i + j * ldr] = alpha * r[i + j * ldr] + temp;
                    }
                }
            }
        } else {
            for (i32 jf = 1; jf <= n; jf++) {
                i32 j = jf - 1;

                for (i32 if_ = jf; if_ <= n; if_++) {
                    i32 i = if_ - 1;
                    f64 temp = beta * (SLC_DDOT(&jf, &e[i * lde], &inc1, &t[j * ldt], &inc1) +
                                       SLC_DDOT(&jf, &t[i * ldt], &inc1, &e[j * lde], &inc1));
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
