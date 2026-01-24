/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"

void mb03ry(
    const i32 m,
    const i32 n,
    const f64 pmax,
    const f64* a,
    const i32 lda,
    const f64* b,
    const i32 ldb,
    f64* c,
    const i32 ldc,
    i32* info
)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const i32 MINUS_ONE = -1;
    const i32 FALSE_VAL = 0;
    const i32 INC1 = 1;

    i32 dk, dl, i, j, k, kk, kk1, l, ll, lm1;
    f64 pnorm, scale;
    f64 p[4];
    i32 ierr;

    *info = 0;

    l = 0;
    while (l < n) {
        lm1 = l;
        dl = 1;
        if (l < n - 1) {
            if (b[(l + 1) + l * ldb] != ZERO) {
                dl = 2;
            }
        }
        ll = lm1 + dl - 1;

        if (lm1 > 0) {
            if (dl == 2) {
                SLC_DGEMM("N", "N", &m, &dl, &lm1, &(f64){-ONE}, c, &ldc,
                          &b[l * ldb], &ldb, &ONE, &c[l * ldc], &ldc);
            } else {
                SLC_DGEMV("N", &m, &lm1, &(f64){-ONE}, c, &ldc, &b[l * ldb],
                          &INC1, &ONE, &c[l * ldc], &INC1);
            }
        }

        kk = m - 1;
        while (kk >= 0) {
            kk1 = kk + 1;
            dk = 1;
            if (kk > 0) {
                if (a[kk + (kk - 1) * lda] != ZERO) {
                    dk = 2;
                }
            }
            k = kk1 - dk;

            if (k < m - 1) {
                i32 len = m - kk - 1;
                for (j = l; j <= ll; j++) {
                    for (i = k; i <= kk; i++) {
                        c[i + j * ldc] += SLC_DDOT(&len, &a[i + kk1 * lda], &lda, &c[kk1 + j * ldc], &INC1);
                    }
                }
            }

            SLC_DLASY2(&FALSE_VAL, &FALSE_VAL, &MINUS_ONE, &dk, &dl,
                       &a[k + k * lda], &lda, &b[l + l * ldb], &ldb,
                       &c[k + l * ldc], &ldc, &scale, p, &dk, &pnorm, &ierr);

            if (scale != ONE || pnorm > pmax) {
                *info = 1;
                return;
            }

            c[k + l * ldc] = -p[0];
            if (dl == 1) {
                if (dk == 2) {
                    c[kk + l * ldc] = -p[1];
                }
            } else {
                if (dk == 1) {
                    c[k + ll * ldc] = -p[1];
                } else {
                    c[kk + l * ldc] = -p[1];
                    c[k + ll * ldc] = -p[2];
                    c[kk + ll * ldc] = -p[3];
                }
            }

            kk = kk - dk;
        }

        l = l + dl;
    }
}
