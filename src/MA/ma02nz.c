/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <ctype.h>

void ma02nz(const char *uplo, const char *trans, const char *skew,
            i32 n, i32 k, i32 l, c128 *a, i32 lda)
{
    if (n == 0 || k == 0 || k == l) {
        return;
    }

    i32 k0 = k - 1;
    i32 l0 = l - 1;

    c128 t = a[k0 + k0*lda];
    a[k0 + k0*lda] = a[l0 + l0*lda];
    a[l0 + l0*lda] = t;

    char uplo_upper = toupper((unsigned char)uplo[0]);
    char trans_upper = toupper((unsigned char)trans[0]);
    char skew_upper = toupper((unsigned char)skew[0]);

    if (uplo_upper == 'L') {
        i32 ione = 1;
        SLC_ZSWAP(&k0, &a[k0], &lda, &a[l0], &lda);

        if (trans_upper == 'T') {
            if (skew_upper == 'N') {
                i32 len = l - k - 1;
                if (len > 0) {
                    SLC_ZSWAP(&len, &a[k0+1 + k0*lda], &ione, &a[l0 + (k0+1)*lda], &lda);
                }
            } else {
                a[l0 + k0*lda] = -a[l0 + k0*lda];
                for (i32 i = k0+1; i < l0; i++) {
                    t = -a[l0 + i*lda];
                    a[l0 + i*lda] = -a[i + k0*lda];
                    a[i + k0*lda] = t;
                }
            }
        } else {
            if (skew_upper == 'N') {
                a[l0 + k0*lda] = conj(a[l0 + k0*lda]);
                for (i32 i = k0+1; i < l0; i++) {
                    t = conj(a[l0 + i*lda]);
                    a[l0 + i*lda] = conj(a[i + k0*lda]);
                    a[i + k0*lda] = t;
                }
            } else {
                a[l0 + k0*lda] = -creal(a[l0 + k0*lda]) + cimag(a[l0 + k0*lda])*I;
                for (i32 i = k0+1; i < l0; i++) {
                    t = -creal(a[l0 + i*lda]) + cimag(a[l0 + i*lda])*I;
                    a[l0 + i*lda] = -creal(a[i + k0*lda]) + cimag(a[i + k0*lda])*I;
                    a[i + k0*lda] = t;
                }
            }
        }

        i32 tail_len = n - l;
        if (tail_len > 0) {
            SLC_ZSWAP(&tail_len, &a[l0+1 + k0*lda], &ione, &a[l0+1 + l0*lda], &ione);
        }

    } else if (uplo_upper == 'U') {
        i32 ione = 1;
        SLC_ZSWAP(&k0, &a[k0*lda], &ione, &a[l0*lda], &ione);

        if (trans_upper == 'T') {
            if (skew_upper == 'N') {
                i32 len = l - k - 1;
                if (len > 0) {
                    SLC_ZSWAP(&len, &a[k0 + (k0+1)*lda], &lda, &a[(k0+1) + l0*lda], &ione);
                }
            } else {
                a[k0 + l0*lda] = -a[k0 + l0*lda];
                for (i32 i = k0+1; i < l0; i++) {
                    t = -a[i + l0*lda];
                    a[i + l0*lda] = -a[k0 + i*lda];
                    a[k0 + i*lda] = t;
                }
            }
        } else {
            if (skew_upper == 'N') {
                a[k0 + l0*lda] = conj(a[k0 + l0*lda]);
                for (i32 i = k0+1; i < l0; i++) {
                    t = conj(a[i + l0*lda]);
                    a[i + l0*lda] = conj(a[k0 + i*lda]);
                    a[k0 + i*lda] = t;
                }
            } else {
                a[k0 + l0*lda] = -creal(a[k0 + l0*lda]) + cimag(a[k0 + l0*lda])*I;
                for (i32 i = k0+1; i < l0; i++) {
                    t = -creal(a[i + l0*lda]) + cimag(a[i + l0*lda])*I;
                    a[i + l0*lda] = -creal(a[k0 + i*lda]) + cimag(a[k0 + i*lda])*I;
                    a[k0 + i*lda] = t;
                }
            }
        }

        i32 tail_len = n - l;
        if (tail_len > 0) {
            SLC_ZSWAP(&tail_len, &a[k0 + (l0+1)*lda], &lda, &a[l0 + (l0+1)*lda], &lda);
        }
    }
}
