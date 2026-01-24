/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

/*
 * MA02AD - Matrix transposition
 *
 * Purpose:
 *   To transpose all or part of a two-dimensional matrix A into
 *   another matrix B.
 */

#include "slicot.h"

void ma02ad(const char* job, const i32 m, const i32 n,
            const f64* a, const i32 lda,
            f64* b, const i32 ldb)
{
    if (job[0] == 'U' || job[0] == 'u') {
        /* Upper triangular/trapezoidal part */
        for (i32 j = 0; j < n; j++) {
            i32 imax = (j < m) ? j : m - 1;
            for (i32 i = 0; i <= imax; i++) {
                b[j + i * ldb] = a[i + j * lda];
            }
        }
    } else if (job[0] == 'L' || job[0] == 'l') {
        /* Lower triangular/trapezoidal part */
        for (i32 j = 0; j < n; j++) {
            for (i32 i = j; i < m; i++) {
                b[j + i * ldb] = a[i + j * lda];
            }
        }
    } else {
        /* Full matrix */
        for (i32 j = 0; j < n; j++) {
            for (i32 i = 0; i < m; i++) {
                b[j + i * ldb] = a[i + j * lda];
            }
        }
    }
}
