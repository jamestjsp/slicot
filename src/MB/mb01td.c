/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"

void mb01td(const i32 n, const f64* a, const i32 lda,
            f64* b, const i32 ldb, f64* dwork, i32* info)
{
    const f64 zero = 0.0;
    const f64 one = 1.0;

    *info = 0;

    if (n < 0) {
        *info = -1;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -3;
    } else if (ldb < (n > 1 ? n : 1)) {
        *info = -5;
    }

    if (*info != 0) {
        return;
    }

    if (n == 0) {
        return;
    }

    if (n == 1) {
        b[0] = a[0] * b[0];
        return;
    }

    for (i32 i = 0; i < n - 1; i++) {
        if (a[(i + 1) + i * lda] == zero) {
            if (b[(i + 1) + i * ldb] != zero) {
                *info = 1;
                return;
            }
        } else if (i < n - 2) {
            if (a[(i + 2) + (i + 1) * lda] != zero) {
                *info = 1;
                return;
            }
        }
    }

    for (i32 j = 0; j < n; j++) {
        i32 jmin = (j + 1) < n ? (j + 1) : (n - 1);
        jmin += 1;
        i32 jmnm = jmin < n ? jmin : (n - 1);

        for (i32 i = 0; i < jmnm; i++) {
            dwork[i] = a[(i + 1) + i * lda] * b[i + j * ldb];
        }

        SLC_DTRMV("U", "N", "N", &jmin, a, &lda, &b[j * ldb], &(i32){1});
        SLC_DAXPY(&jmnm, &one, dwork, &(i32){1}, &b[1 + j * ldb], &(i32){1});
    }
}
