/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include <ctype.h>

void mb01ss(const char jobs, const char uplo, i32 n,
            f64* a, i32 lda, const f64* d)
{
    if (n == 0) {
        return;
    }

    char jobs_upper = (char)toupper((unsigned char)jobs);
    char uplo_upper = (char)toupper((unsigned char)uplo);
    bool upper = (uplo_upper == 'U');

    if (jobs_upper == 'D') {
        if (upper) {
            for (i32 j = 0; j < n; j++) {
                f64 dj = d[j];
                for (i32 i = 0; i <= j; i++) {
                    a[i + j * lda] = dj * d[i] * a[i + j * lda];
                }
            }
        } else {
            for (i32 j = 0; j < n; j++) {
                f64 dj = d[j];
                for (i32 i = j; i < n; i++) {
                    a[i + j * lda] = dj * d[i] * a[i + j * lda];
                }
            }
        }
    } else {
        if (upper) {
            for (i32 j = 0; j < n; j++) {
                f64 dj_inv = 1.0 / d[j];
                for (i32 i = 0; i <= j; i++) {
                    a[i + j * lda] = (dj_inv / d[i]) * a[i + j * lda];
                }
            }
        } else {
            for (i32 j = 0; j < n; j++) {
                f64 dj_inv = 1.0 / d[j];
                for (i32 i = j; i < n; i++) {
                    a[i + j * lda] = (dj_inv / d[i]) * a[i + j * lda];
                }
            }
        }
    }
}
