/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"

void tf01qd(const i32 nc, const i32 nb, const i32 n, const i32* iord,
            const f64* ar, const f64* ma, f64* h, const i32 ldh, i32* info)
{
    *info = 0;

    i32 nc_min = (nc > 1) ? nc : 1;

    if (nc < 0) {
        *info = -1;
    } else if (nb < 0) {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    } else if (ldh < nc_min) {
        *info = -8;
    }

    if (*info != 0) {
        return;
    }

    if (nc == 0 || nb == 0 || n == 0) {
        return;
    }

    i32 nl = 0;
    i32 k = 0;

    for (i32 i = 0; i < nc; i++) {
        for (i32 j = 0; j < nb; j++) {
            i32 nord = iord[k];

            h[i + j * ldh] = ma[nl];
            i32 jk = j;

            for (i32 ki = 1; ki < nord && jk + nb < n * nb; ki++) {
                jk += nb;
                f64 dot_sum = 0.0;
                for (i32 p = 0; p < ki; p++) {
                    i32 col_idx = jk - (p + 1) * nb;
                    if (col_idx >= 0) {
                        dot_sum += ar[nl + p] * h[i + col_idx * ldh];
                    }
                }
                h[i + jk * ldh] = ma[nl + ki] - dot_sum;
            }

            for (i32 jj = j; jj < j + (n - nord) * nb && jk + nb < n * nb; jj += nb) {
                jk += nb;
                f64 dot_sum = 0.0;
                for (i32 p = 0; p < nord; p++) {
                    i32 col_idx = jk - (p + 1) * nb;
                    if (col_idx >= 0) {
                        dot_sum += ar[nl + p] * h[i + col_idx * ldh];
                    }
                }
                h[i + jk * ldh] = -dot_sum;
            }

            nl += nord;
            k++;
        }
    }
}
