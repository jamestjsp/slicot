/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"

void mb05oy(
    const char* job,
    const i32 n,
    const i32 low,
    const i32 igh,
    f64* a,
    const i32 lda,
    const f64* scale,
    i32* info
)
{
    const f64 ONE = 1.0;
    const i32 int1 = 1;

    char job_ch = *job;
    bool job_n = (job_ch == 'N' || job_ch == 'n');
    bool job_p = (job_ch == 'P' || job_ch == 'p');
    bool job_s = (job_ch == 'S' || job_ch == 's');
    bool job_b = (job_ch == 'B' || job_ch == 'b');

    *info = 0;

    if (!job_n && !job_p && !job_s && !job_b) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (low < 1 || low > (n > 1 ? n : 1)) {
        *info = -3;
    } else if (igh < (low < n ? low : n) || igh > n) {
        *info = -4;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -6;
    }

    if (*info != 0) {
        return;
    }

    if (n == 0 || job_n) {
        return;
    }

    if (!job_p && igh != low) {
        for (i32 i = low - 1; i < igh; i++) {
            f64 s = scale[i];
            SLC_DSCAL(&n, &s, &a[i], &lda);
        }

        for (i32 j = low - 1; j < igh; j++) {
            f64 s = ONE / scale[j];
            SLC_DSCAL(&n, &s, &a[j * lda], &int1);
        }
    }

    if (!job_s) {
        for (i32 ii = 1; ii <= n; ii++) {
            i32 i = ii;
            if (i < low || i > igh) {
                if (i < low) {
                    i = low - ii;
                }
                i32 k = (i32)scale[i - 1];
                if (k != i) {
                    SLC_DSWAP(&n, &a[i - 1], &lda, &a[k - 1], &lda);
                    SLC_DSWAP(&n, &a[(i - 1) * lda], &int1, &a[(k - 1) * lda], &int1);
                }
            }
        }
    }
}
