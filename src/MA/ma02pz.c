/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

/*
 * MA02PZ - Count zero rows and zero columns of a complex matrix
 *
 * Purpose:
 *   To compute the number of zero rows and zero columns of a complex matrix.
 */

#include "slicot.h"

void ma02pz(const i32 m, const i32 n, const c128* a, const i32 lda,
            i32* nzr, i32* nzc)
{
    const c128 ZERO = 0.0 + 0.0 * I;

    *nzc = 0;
    *nzr = 0;

    if (m <= 0 || n <= 0) {
        return;
    }

    for (i32 i = 0; i < n; i++) {
        bool is_zero_col = true;
        for (i32 j = 0; j < m; j++) {
            if (a[j + i * lda] != ZERO) {
                is_zero_col = false;
                break;
            }
        }
        if (is_zero_col) {
            (*nzc)++;
        }
    }

    for (i32 i = 0; i < m; i++) {
        bool is_zero_row = true;
        for (i32 j = 0; j < n; j++) {
            if (a[i + j * lda] != ZERO) {
                is_zero_row = false;
                break;
            }
        }
        if (is_zero_row) {
            (*nzr)++;
        }
    }
}
