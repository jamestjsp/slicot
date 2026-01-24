/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

/*
 * MA02AZ - Complex matrix (conjugate) transposition
 *
 * Purpose:
 *   To (conjugate) transpose all or part of a two-dimensional complex
 *   matrix A into another matrix B.
 */

#include "slicot.h"

void ma02az(const char* trans, const char* job, const i32 m, const i32 n,
            const c128* a, const i32 lda, c128* b, const i32 ldb)
{
    if (trans[0] == 'T' || trans[0] == 't') {
        if (job[0] == 'U' || job[0] == 'u') {
            for (i32 j = 0; j < n; j++) {
                i32 imax = (j < m) ? j : m - 1;
                for (i32 i = 0; i <= imax; i++) {
                    b[j + i * ldb] = a[i + j * lda];
                }
            }
        } else if (job[0] == 'L' || job[0] == 'l') {
            for (i32 j = 0; j < n; j++) {
                for (i32 i = j; i < m; i++) {
                    b[j + i * ldb] = a[i + j * lda];
                }
            }
        } else {
            for (i32 j = 0; j < n; j++) {
                for (i32 i = 0; i < m; i++) {
                    b[j + i * ldb] = a[i + j * lda];
                }
            }
        }
    } else {
        if (job[0] == 'U' || job[0] == 'u') {
            for (i32 j = 0; j < n; j++) {
                i32 imax = (j < m) ? j : m - 1;
                for (i32 i = 0; i <= imax; i++) {
                    b[j + i * ldb] = conj(a[i + j * lda]);
                }
            }
        } else if (job[0] == 'L' || job[0] == 'l') {
            for (i32 j = 0; j < n; j++) {
                for (i32 i = j; i < m; i++) {
                    b[j + i * ldb] = conj(a[i + j * lda]);
                }
            }
        } else {
            for (i32 j = 0; j < n; j++) {
                for (i32 i = 0; i < m; i++) {
                    b[j + i * ldb] = conj(a[i + j * lda]);
                }
            }
        }
    }
}
