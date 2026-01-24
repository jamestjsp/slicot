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

void mb01os(const char* uplo_str, const char* trans_str, i32 n,
            const f64* h, i32 ldh, const f64* x, i32 ldx,
            f64* p, i32 ldp, i32* info)
{
    const f64 zero = 0.0;
    const f64 one = 1.0;
    i32 inc1 = 1;

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
    } else if (ldh < max_i32(1, n)) {
        *info = -5;
    } else if (ldx < max_i32(1, n)) {
        *info = -7;
    } else if (ldp < max_i32(1, n)) {
        *info = -9;
    }

    if (*info != 0) {
        return;
    }

    if (n == 0) {
        return;
    }

    if (!ltrans) {
        if (luplo) {
            for (i32 jf = 1; jf <= n - 1; jf++) {
                i32 j = jf - 1;
                SLC_DCOPY(&jf, &x[j * ldx], &inc1, &p[j * ldp], &inc1);
                SLC_DTRMV("U", "N", "N", &jf, h, &ldh, &p[j * ldp], &inc1);
                for (i32 i = jf; i < n; i++) {
                    p[i + j * ldp] = zero;
                }

                for (i32 if_ = 2; if_ <= jf + 1; if_++) {
                    i32 i = if_ - 1;
                    p[i + j * ldp] += h[i + (i - 1) * ldh] * x[(i - 1) + j * ldx];
                }

                for (i32 if_ = jf + 2; if_ <= n; if_++) {
                    f64 xji1 = x[j + (if_ - 2) * ldx];
                    SLC_DAXPY(&if_, &xji1, &h[(if_ - 2) * ldh], &inc1, &p[j * ldp], &inc1);
                }

                f64 xjn = x[j + (n - 1) * ldx];
                SLC_DAXPY(&n, &xjn, &h[(n - 1) * ldh], &inc1, &p[j * ldp], &inc1);
            }

            SLC_DCOPY(&n, &x[(n - 1) * ldx], &inc1, &p[(n - 1) * ldp], &inc1);
            SLC_DTRMV("U", "N", "N", &n, h, &ldh, &p[(n - 1) * ldp], &inc1);

            for (i32 if_ = 2; if_ <= n; if_++) {
                i32 i = if_ - 1;
                p[i + (n - 1) * ldp] += h[i + (i - 1) * ldh] * x[(i - 1) + (n - 1) * ldx];
            }
        } else {
            SLC_DCOPY(&n, x, &inc1, p, &inc1);
            SLC_DTRMV("U", "N", "N", &n, h, &ldh, p, &inc1);

            for (i32 if_ = 2; if_ <= n; if_++) {
                i32 i = if_ - 1;
                p[i] += h[i + (i - 1) * ldh] * x[i - 1];
            }

            for (i32 jf = 2; jf <= n; jf++) {
                i32 j = jf - 1;
                i32 jm1 = jf - 1;
                for (i32 if_ = 1; if_ < jf; if_++) {
                    i32 i = if_ - 1;
                    p[i + j * ldp] = x[j + i * ldx];
                }
                SLC_DTRMV("U", "N", "N", &jm1, h, &ldh, &p[j * ldp], &inc1);
                p[j + j * ldp] = zero;

                for (i32 if_ = 2; if_ <= jf; if_++) {
                    i32 i = if_ - 1;
                    p[i + j * ldp] += h[i + (i - 1) * ldh] * x[j + (i - 1) * ldx];
                }

                f64 temp = p[j + j * ldp];
                i32 gemv_m = jm1;
                i32 gemv_n = n - j;
                SLC_DGEMV("N", &gemv_m, &gemv_n, &one, &h[j * ldh], &ldh,
                          &x[j + j * ldx], &inc1, &one, &p[j * ldp], &inc1);

                i32 nj1 = n - j;
                SLC_DCOPY(&nj1, &x[j + j * ldx], &inc1, &p[j + j * ldp], &inc1);
                SLC_DTRMV("U", "N", "N", &nj1, &h[j + j * ldh], &ldh, &p[j + j * ldp], &inc1);
                p[j + j * ldp] += temp;

                for (i32 if_ = jf + 1; if_ <= n; if_++) {
                    i32 i = if_ - 1;
                    p[i + j * ldp] += h[i + (i - 1) * ldh] * x[(i - 1) + j * ldx];
                }
            }
        }
    } else {
        if (luplo) {
            for (i32 jf = 1; jf <= n - 2; jf++) {
                i32 j = jf - 1;
                i32 j3 = min_i32(jf + 3, n);
                i32 jp1 = jf + 1;

                SLC_DCOPY(&jp1, &h[j * ldh], &inc1, &p[j * ldp], &inc1);
                SLC_DTRMV("U", "N", "N", &jp1, x, &ldx, &p[j * ldp], &inc1);

                SLC_DCOPY(&jp1, &h[j * ldh], &inc1, &p[1 + (j + 1) * ldp], &inc1);
                SLC_DTRMV("U", "T", "N", &jp1, &x[ldx], &ldx, &p[1 + (j + 1) * ldp], &inc1);

                i32 jlen = jf;
                SLC_DAXPY(&jlen, &one, &p[1 + (j + 1) * ldp], &inc1, &p[1 + j * ldp], &inc1);

                p[jf + 1 + j * ldp] = SLC_DDOT(&jp1, &x[(jf + 1) * ldx], &inc1, &h[j * ldh], &inc1);

                i32 gemv_n2 = n - j3;
                if (gemv_n2 > 0) {
                    SLC_DGEMV("T", &jp1, &gemv_n2, &one, &x[(j3 - 1) * ldx], &ldx,
                              &h[j * ldh], &inc1, &one, &p[(j3 - 1) + j * ldp], &inc1);
                }

                p[(n - 1) + j * ldp] = SLC_DDOT(&jp1, &x[(n - 1) * ldx], &inc1, &h[j * ldh], &inc1);
            }

            if (n == 1) {
                p[0] = x[0] * h[0];
            } else {
                SLC_DCOPY(&n, &h[(n - 2) * ldh], &inc1, &p[(n - 2) * ldp], &inc1);
                SLC_DCOPY(&n, &h[(n - 1) * ldh], &inc1, &p[(n - 1) * ldp], &inc1);
                i32 two = 2;
                SLC_DTRMM("L", "U", "N", "N", &n, &two, &one, x, &ldx, &p[(n - 2) * ldp], &ldp);

                for (i32 if_ = 2; if_ <= n; if_++) {
                    i32 i = if_ - 1;
                    i32 im1 = if_ - 1;
                    SLC_DGEMV("T", &im1, &two, &one, &h[(n - 2) * ldh], &ldh,
                              &x[i * ldx], &inc1, &one, &p[i + (n - 2) * ldp], &ldp);
                }
            }
        } else {
            for (i32 jf = 1; jf <= n - 1; jf++) {
                i32 j = jf - 1;
                SLC_DCOPY(&jf, &h[j * ldh], &inc1, &p[j * ldp], &inc1);
                SLC_DTRMV("L", "N", "N", &jf, x, &ldx, &p[j * ldp], &inc1);

                SLC_DCOPY(&jf, &h[1 + j * ldh], &inc1, &p[(j + 1) * ldp], &inc1);
                SLC_DTRMV("L", "T", "N", &jf, &x[1], &ldx, &p[(j + 1) * ldp], &inc1);

                SLC_DAXPY(&jf, &one, &p[(j + 1) * ldp], &inc1, &p[j * ldp], &inc1);

                i32 gemv_m = n - jf;
                i32 gemv_n = jf + 1;
                SLC_DGEMV("N", &gemv_m, &gemv_n, &one, &x[jf], &ldx,
                          &h[j * ldh], &inc1, &zero, &p[jf + j * ldp], &inc1);
            }

            SLC_DCOPY(&n, &h[(n - 1) * ldh], &inc1, &p[(n - 1) * ldp], &inc1);
            SLC_DTRMV("L", "N", "N", &n, x, &ldx, &p[(n - 1) * ldp], &inc1);

            for (i32 if_ = 1; if_ <= n - 1; if_++) {
                i32 i = if_ - 1;
                i32 ni = n - if_;
                p[i + (n - 1) * ldp] += SLC_DDOT(&ni, &x[if_ + i * ldx], &inc1, &h[if_ + (n - 1) * ldh], &inc1);
            }
        }
    }
}
