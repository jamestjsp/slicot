/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"

i32 ma02oz(const char *skew, i32 m, const c128 *a, i32 lda,
           const c128 *de, i32 ldde)
{
    const f64 ZERO = 0.0;
    const c128 CZERO = 0.0 + 0.0 * I;
    i32 nz = 0;

    if (m <= 0) {
        return 0;
    }

    bool isskew = (skew[0] == 'S' || skew[0] == 's');

    i32 i = 0;
    while (i < m) {
        bool row_is_zero = true;

        for (i32 j = 0; j < m && row_is_zero; j++) {
            if (a[j + i * lda] != CZERO) {
                row_is_zero = false;
            }
        }

        if (row_is_zero) {
            for (i32 j = 0; j < i && row_is_zero; j++) {
                if (de[i + j * ldde] != CZERO) {
                    row_is_zero = false;
                }
            }
        }

        if (row_is_zero) {
            if (isskew) {
                if (cimag(de[i + i * ldde]) != ZERO) {
                    row_is_zero = false;
                }
            } else {
                if (creal(de[i + i * ldde]) != ZERO) {
                    row_is_zero = false;
                }
            }
        }

        if (row_is_zero) {
            for (i32 j = i + 1; j < m && row_is_zero; j++) {
                if (de[j + i * ldde] != CZERO) {
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
            if (a[i + j * lda] != CZERO) {
                row_is_zero = false;
            }
        }

        if (row_is_zero) {
            for (i32 j = 0; j < i && row_is_zero; j++) {
                if (de[j + (i + 1) * ldde] != CZERO) {
                    row_is_zero = false;
                }
            }
        }

        if (row_is_zero) {
            if (isskew) {
                if (cimag(de[i + (i + 1) * ldde]) != ZERO) {
                    row_is_zero = false;
                }
            } else {
                if (creal(de[i + (i + 1) * ldde]) != ZERO) {
                    row_is_zero = false;
                }
            }
        }

        if (row_is_zero) {
            for (i32 j = i + 1; j < m && row_is_zero; j++) {
                if (de[i + (j + 1) * ldde] != CZERO) {
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
