/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

/*
 * MA02DD - Pack/unpack upper or lower triangle of symmetric matrix
 *
 * Purpose:
 *   To pack/unpack the upper or lower triangle of a symmetric matrix.
 *   The packed matrix is stored column-wise in the one-dimensional
 *   array AP.
 */

#include "slicot.h"
#include "slicot_blas.h"

void ma02dd(const char* job, const char* uplo, const i32 n,
            f64* a, const i32 lda, f64* ap)
{
    bool luplo = (uplo[0] == 'L' || uplo[0] == 'l');
    i32 ij = 0;
    i32 one = 1;

    if (job[0] == 'P' || job[0] == 'p') {
        if (luplo) {
            /* Pack the lower triangle of A */
            for (i32 j = 0; j < n; j++) {
                i32 len = n - j;
                SLC_DCOPY(&len, &a[j + j * lda], &one, &ap[ij], &one);
                ij += len;
            }
        } else {
            /* Pack the upper triangle of A */
            for (i32 j = 0; j < n; j++) {
                i32 len = j + 1;
                SLC_DCOPY(&len, &a[0 + j * lda], &one, &ap[ij], &one);
                ij += len;
            }
        }
    } else {
        if (luplo) {
            /* Unpack the lower triangle of A */
            for (i32 j = 0; j < n; j++) {
                i32 len = n - j;
                SLC_DCOPY(&len, &ap[ij], &one, &a[j + j * lda], &one);
                ij += len;
            }
        } else {
            /* Unpack the upper triangle of A */
            for (i32 j = 0; j < n; j++) {
                i32 len = j + 1;
                SLC_DCOPY(&len, &ap[ij], &one, &a[0 + j * lda], &one);
                ij += len;
            }
        }
    }
}
