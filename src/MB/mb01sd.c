/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include <ctype.h>

void mb01sd(const char jobs, const i32 m, const i32 n,
            f64* a, const i32 lda, const f64* r, const f64* c)
{
    if (m == 0 || n == 0) {
        return;
    }

    char jobs_upper = (char)toupper((unsigned char)jobs);

    if (jobs_upper == 'C') {
        for (i32 j = 0; j < n; j++) {
            f64 cj = c[j];
            for (i32 i = 0; i < m; i++) {
                a[i + j * lda] = cj * a[i + j * lda];
            }
        }
    } else if (jobs_upper == 'R') {
        for (i32 j = 0; j < n; j++) {
            for (i32 i = 0; i < m; i++) {
                a[i + j * lda] = r[i] * a[i + j * lda];
            }
        }
    } else if (jobs_upper == 'B') {
        for (i32 j = 0; j < n; j++) {
            f64 cj = c[j];
            for (i32 i = 0; i < m; i++) {
                a[i + j * lda] = cj * r[i] * a[i + j * lda];
            }
        }
    }
}
