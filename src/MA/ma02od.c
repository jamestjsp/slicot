/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"

i32 ma02od(const char *skew, i32 m, const f64 *a, i32 lda,
           const f64 *de, i32 ldde)
{
    const f64 ZERO = 0.0;
    i32 nz = 0;

    if (m <= 0) {
        return 0;
    }

    bool isham = (skew[0] == 'H' || skew[0] == 'h');

    i32 i = 0;
    while (i < m) {
        bool row_is_zero = true;

        for (i32 j = 0; j < m && row_is_zero; j++) {
            if (a[j + i * lda] != ZERO) {
                row_is_zero = false;
            }
        }

        if (row_is_zero) {
            for (i32 j = 0; j < i && row_is_zero; j++) {
                if (de[i + j * ldde] != ZERO) {
                    row_is_zero = false;
                }
            }
        }

        if (row_is_zero && isham) {
            if (de[i + i * ldde] != ZERO) {
                row_is_zero = false;
            }
        }

        if (row_is_zero) {
            for (i32 j = i + 1; j < m && row_is_zero; j++) {
                if (de[j + i * ldde] != ZERO) {
                    row_is_zero = false;
                }
            }
        }

        if (row_is_zero) {
            nz++;
        }
        i++;
    }

    i = 0;
    while (i < m) {
        bool row_is_zero = true;

        for (i32 j = 0; j < m && row_is_zero; j++) {
            if (a[i + j * lda] != ZERO) {
                row_is_zero = false;
            }
        }

        if (row_is_zero) {
            for (i32 j = 0; j < i && row_is_zero; j++) {
                if (de[j + (i + 1) * ldde] != ZERO) {
                    row_is_zero = false;
                }
            }
        }

        if (row_is_zero && isham) {
            if (de[i + (i + 1) * ldde] != ZERO) {
                row_is_zero = false;
            }
        }

        if (row_is_zero) {
            for (i32 j = i + 1; j < m && row_is_zero; j++) {
                if (de[i + (j + 1) * ldde] != ZERO) {
                    row_is_zero = false;
                }
            }
        }

        if (row_is_zero) {
            nz++;
        }
        i++;
    }

    return nz;
}
