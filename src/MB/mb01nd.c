/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include <ctype.h>

void mb01nd(const char uplo, const i32 n, const f64 alpha,
            const f64* x, const i32 incx, const f64* y, const i32 incy,
            f64* a, const i32 lda, i32* info)
{
    const f64 ZERO = 0.0;

    *info = 0;
    char uplo_upper = (char)toupper((unsigned char)uplo);

    if (uplo_upper != 'U' && uplo_upper != 'L') {
        *info = 1;
        return;
    }
    if (n < 0) {
        *info = 2;
        return;
    }
    if (incx == 0) {
        *info = 5;
        return;
    }
    if (incy == 0) {
        *info = 7;
        return;
    }
    i32 lda_min = n > 1 ? n : 1;
    if (lda < lda_min) {
        *info = 9;
        return;
    }

    if (n == 0 || alpha == ZERO) {
        return;
    }

    i32 kx, ky;
    if (incx > 0) {
        kx = 0;
    } else {
        kx = -(n - 1) * incx;
    }
    if (incy > 0) {
        ky = 0;
    } else {
        ky = -(n - 1) * incy;
    }

    if (uplo_upper == 'U') {
        if (incx == 1 && incy == 1) {
            for (i32 j = 1; j < n; j++) {
                if (x[j] != ZERO || y[j] != ZERO) {
                    f64 temp1 = alpha * y[j];
                    f64 temp2 = alpha * x[j];
                    for (i32 i = 0; i < j; i++) {
                        a[i + j * lda] = a[i + j * lda] + x[i] * temp1 - y[i] * temp2;
                    }
                }
            }
        } else {
            i32 jx = kx + incx;
            i32 jy = ky + incy;
            for (i32 j = 1; j < n; j++) {
                if (x[jx] != ZERO || y[jy] != ZERO) {
                    f64 temp1 = alpha * y[jy];
                    f64 temp2 = alpha * x[jx];
                    i32 ix = kx;
                    i32 iy = ky;
                    for (i32 i = 0; i < j; i++) {
                        a[i + j * lda] = a[i + j * lda] + x[ix] * temp1 - y[iy] * temp2;
                        ix += incx;
                        iy += incy;
                    }
                }
                jx += incx;
                jy += incy;
            }
        }
    } else {
        if (incx == 1 && incy == 1) {
            for (i32 j = 0; j < n - 1; j++) {
                if (x[j] != ZERO || y[j] != ZERO) {
                    f64 temp1 = alpha * y[j];
                    f64 temp2 = alpha * x[j];
                    for (i32 i = j + 1; i < n; i++) {
                        a[i + j * lda] = a[i + j * lda] + x[i] * temp1 - y[i] * temp2;
                    }
                }
            }
        } else {
            i32 jx = kx;
            i32 jy = ky;
            for (i32 j = 0; j < n - 1; j++) {
                if (x[jx] != ZERO || y[jy] != ZERO) {
                    f64 temp1 = alpha * y[jy];
                    f64 temp2 = alpha * x[jx];
                    i32 ix = jx;
                    i32 iy = jy;
                    for (i32 i = j + 1; i < n; i++) {
                        ix += incx;
                        iy += incy;
                        a[i + j * lda] = a[i + j * lda] + x[ix] * temp1 - y[iy] * temp2;
                    }
                }
                jx += incx;
                jy += incy;
            }
        }
    }
}
