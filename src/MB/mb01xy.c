/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"

void mb01xy(
    const char* uplo,
    const i32 n,
    f64* a,
    const i32 lda,
    i32* info
)
{
    const f64 one = 1.0;
    const i32 inc1 = 1;

    *info = 0;
    bool upper = (*uplo == 'U' || *uplo == 'u');

    if (!upper && *uplo != 'L' && *uplo != 'l') {
        *info = -1;
        return;
    }
    if (n < 0) {
        *info = -2;
        return;
    }
    i32 max_1_n = (n > 1) ? n : 1;
    if (lda < max_1_n) {
        *info = -4;
        return;
    }

    if (n == 0) {
        return;
    }

    if (upper) {
        a[(n-1) + (n-1)*lda] = SLC_DDOT(&n, &a[(n-1)*lda], &inc1, &a[(n-1)*lda], &inc1);

        for (i32 i = n - 2; i >= 1; i--) {
            f64 aii = a[i + i*lda];
            i32 i1 = i + 1;
            a[i + i*lda] = SLC_DDOT(&i1, &a[i*lda], &inc1, &a[i*lda], &inc1);

            i32 rows = i;
            i32 cols = n - i - 1;
            if (cols > 0 && rows > 0) {
                SLC_DGEMV("T", &rows, &cols, &one, &a[(i+1)*lda], &lda,
                          &a[i*lda], &inc1, &aii, &a[i + (i+1)*lda], &lda);
            }
        }

        if (n > 1) {
            f64 aii = a[0];
            SLC_DSCAL(&n, &aii, a, &lda);
        }
    } else {
        a[(n-1) + (n-1)*lda] = SLC_DDOT(&n, &a[n-1], &lda, &a[n-1], &lda);

        for (i32 i = n - 2; i >= 1; i--) {
            f64 aii = a[i + i*lda];
            i32 i1 = i + 1;
            a[i + i*lda] = SLC_DDOT(&i1, &a[i], &lda, &a[i], &lda);

            i32 rows = n - i - 1;
            i32 cols = i;
            if (rows > 0 && cols > 0) {
                SLC_DGEMV("N", &rows, &cols, &one, &a[i+1], &lda,
                          &a[i], &lda, &aii, &a[(i+1) + i*lda], &inc1);
            }
        }

        if (n > 1) {
            f64 aii = a[0];
            SLC_DSCAL(&n, &aii, a, &inc1);
        }
    }
}
