/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"

bool ma02hd(const char *job, i32 m, i32 n, f64 diag, const f64 *a, i32 lda)
{
    const f64 ZERO = 0.0;

    i32 minmn = m < n ? m : n;
    if (minmn == 0) {
        return false;
    }

    bool is_upper = (job[0] == 'U' || job[0] == 'u');
    bool is_lower = (job[0] == 'L' || job[0] == 'l');

    if (is_upper) {
        for (i32 j = 0; j < n; j++) {
            i32 upper_limit = j < m ? j : m;
            for (i32 i = 0; i < upper_limit; i++) {
                if (a[i + j * lda] != ZERO) {
                    return false;
                }
            }
            if (j < m) {
                if (a[j + j * lda] != diag) {
                    return false;
                }
            }
        }
    } else if (is_lower) {
        for (i32 j = 0; j < minmn; j++) {
            if (a[j + j * lda] != diag) {
                return false;
            }
            if (j < m - 1) {
                for (i32 i = j + 1; i < m; i++) {
                    if (a[i + j * lda] != ZERO) {
                        return false;
                    }
                }
            }
        }
    } else {
        for (i32 j = 0; j < n; j++) {
            i32 upper_limit = j < m ? j : m;
            for (i32 i = 0; i < upper_limit; i++) {
                if (a[i + j * lda] != ZERO) {
                    return false;
                }
            }
            if (j < m) {
                if (a[j + j * lda] != diag) {
                    return false;
                }
            }
            if (j < m - 1) {
                for (i32 i = j + 1; i < m; i++) {
                    if (a[i + j * lda] != ZERO) {
                        return false;
                    }
                }
            }
        }
    }

    return true;
}
