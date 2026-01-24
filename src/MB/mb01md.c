/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include <ctype.h>
#include <math.h>

void mb01md(const char uplo, const i32 n, const f64 alpha,
            const f64* a, const i32 lda, const f64* x, const i32 incx,
            const f64 beta, f64* y, const i32 incy, i32* info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

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
    i32 lda_min = n > 1 ? n : 1;
    if (lda < lda_min) {
        *info = 5;
        return;
    }
    if (incx == 0) {
        *info = 7;
        return;
    }
    if (incy == 0) {
        *info = 10;
        return;
    }

    if (n == 0 || (alpha == ZERO && beta == ONE)) {
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

    if (beta != ONE) {
        if (incy == 1) {
            if (beta == ZERO) {
                for (i32 i = 0; i < n; i++) {
                    y[i] = ZERO;
                }
            } else {
                for (i32 i = 0; i < n; i++) {
                    y[i] = beta * y[i];
                }
            }
        } else {
            i32 iy = ky;
            if (beta == ZERO) {
                for (i32 i = 0; i < n; i++) {
                    y[iy] = ZERO;
                    iy += incy;
                }
            } else {
                for (i32 i = 0; i < n; i++) {
                    y[iy] = beta * y[iy];
                    iy += incy;
                }
            }
        }
    }

    if (alpha == ZERO) {
        return;
    }

    if (uplo_upper == 'U') {
        if (incx == 1 && incy == 1) {
            for (i32 j = 1; j < n; j++) {
                f64 temp1 = alpha * x[j];
                f64 temp2 = ZERO;
                for (i32 i = 0; i < j; i++) {
                    y[i] = y[i] + temp1 * a[i + j * lda];
                    temp2 = temp2 + a[i + j * lda] * x[i];
                }
                y[j] = y[j] - alpha * temp2;
            }
        } else {
            i32 jx = kx + incx;
            i32 jy = ky + incy;
            for (i32 j = 1; j < n; j++) {
                f64 temp1 = alpha * x[jx];
                f64 temp2 = ZERO;
                i32 ix = kx;
                i32 iy = ky;
                for (i32 i = 0; i < j; i++) {
                    y[iy] = y[iy] + temp1 * a[i + j * lda];
                    temp2 = temp2 + a[i + j * lda] * x[ix];
                    ix += incx;
                    iy += incy;
                }
                y[jy] = y[jy] - alpha * temp2;
                jx += incx;
                jy += incy;
            }
        }
    } else {
        if (incx == 1 && incy == 1) {
            for (i32 j = 0; j < n - 1; j++) {
                f64 temp1 = alpha * x[j];
                f64 temp2 = ZERO;
                for (i32 i = j + 1; i < n; i++) {
                    y[i] = y[i] + temp1 * a[i + j * lda];
                    temp2 = temp2 + a[i + j * lda] * x[i];
                }
                y[j] = y[j] - alpha * temp2;
            }
        } else {
            i32 jx = kx;
            i32 jy = ky;
            for (i32 j = 0; j < n - 1; j++) {
                f64 temp1 = alpha * x[jx];
                f64 temp2 = ZERO;
                i32 ix = jx;
                i32 iy = jy;
                for (i32 i = j + 1; i < n; i++) {
                    ix += incx;
                    iy += incy;
                    y[iy] = y[iy] + temp1 * a[i + j * lda];
                    temp2 = temp2 + a[i + j * lda] * x[ix];
                }
                y[jy] = y[jy] - alpha * temp2;
                jx += incx;
                jy += incy;
            }
        }
    }
}
