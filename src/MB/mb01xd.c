/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"

void mb01xd(
    const char* uplo,
    const i32 n,
    f64* a,
    const i32 lda,
    i32* info
)
{
    const f64 one = 1.0;

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

    const i32 ispec = 1;
    const i32 neg1 = -1;
    i32 nb = SLC_ILAENV(&ispec, "DLAUUM", uplo, &n, &neg1, &neg1, &neg1);

    if (nb <= 1 || nb >= n) {
        mb01xy(uplo, n, a, lda, info);
    } else {
        if (upper) {
            for (i32 i = n; i >= 1; i -= nb) {
                i32 ib = (nb < i) ? nb : i;
                i32 ii = i - ib;  // 0-based start of block (Fortran II-1)

                if (i < n) {
                    i32 n_i = n - i;
                    SLC_DTRMM("L", "U", "T", "N", &ib, &n_i, &one,
                              &a[ii + ii*lda], &lda, &a[ii + i*lda], &lda);

                    i32 rows = ii;  // i - ib in Fortran = ii in 0-based
                    if (rows > 0) {
                        SLC_DGEMM("T", "N", &ib, &n_i, &rows, &one,
                                  &a[ii*lda], &lda, &a[i*lda], &lda,
                                  &one, &a[ii + i*lda], &lda);
                    }
                }

                mb01xy("U", ib, &a[ii + ii*lda], lda, info);

                if (ii > 0) {
                    SLC_DSYRK("U", "T", &ib, &ii, &one,
                              &a[ii*lda], &lda, &one, &a[ii + ii*lda], &lda);
                }
            }
        } else {
            for (i32 i = n; i >= 1; i -= nb) {
                i32 ib = (nb < i) ? nb : i;
                i32 ii = i - ib;  // 0-based start of block

                if (i < n) {
                    i32 n_i = n - i;
                    SLC_DTRMM("R", "L", "T", "N", &n_i, &ib, &one,
                              &a[ii + ii*lda], &lda, &a[i + ii*lda], &lda);

                    i32 cols = ii;
                    if (cols > 0) {
                        SLC_DGEMM("N", "T", &n_i, &ib, &cols, &one,
                                  &a[i], &lda, &a[ii], &lda,
                                  &one, &a[i + ii*lda], &lda);
                    }
                }

                mb01xy("L", ib, &a[ii + ii*lda], lda, info);

                if (ii > 0) {
                    SLC_DSYRK("L", "N", &ib, &ii, &one,
                              &a[ii], &lda, &one, &a[ii + ii*lda], &lda);
                }
            }
        }
    }
}
