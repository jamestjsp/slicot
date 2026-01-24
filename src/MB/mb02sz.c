/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>

i32 slicot_mb02sz(i32 n, c128* h, i32 ldh, i32* ipiv) {
    i32 info = 0;

    if (n < 0) {
        info = -1;
    } else if (ldh < (n > 1 ? n : 1)) {
        info = -3;
    }

    if (info != 0) {
        i32 xinfo = -info;
        SLC_XERBLA("MB02SZ", &xinfo);
        return info;
    }

    if (n == 0) {
        return 0;
    }

    for (i32 j = 0; j < n; j++) {
        i32 jp = j;
        if (j < n - 1) {
            f64 cabs_jj = fabs(creal(h[j + j * ldh])) + fabs(cimag(h[j + j * ldh]));
            f64 cabs_j1j = fabs(creal(h[j + 1 + j * ldh])) + fabs(cimag(h[j + 1 + j * ldh]));
            if (cabs_j1j > cabs_jj) {
                jp = j + 1;
            }
        }
        ipiv[j] = jp + 1;  // 1-based index for SLICOT compatibility

        if (h[jp + j * ldh] != 0.0) {
            if (jp != j) {
                i32 len = n - j;
                SLC_ZSWAP(&len, &h[j + j * ldh], &ldh, &h[jp + j * ldh], &ldh);
            }

            if (j < n - 1) {
                h[j + 1 + j * ldh] = h[j + 1 + j * ldh] / h[j + j * ldh];
            }
        } else if (info == 0) {
            info = j + 1;
        }

        if (j < n - 1) {
            i32 len = n - j - 1;
            c128 neg_mult = -h[j + 1 + j * ldh];
            SLC_ZAXPY(&len, &neg_mult, &h[j + (j + 1) * ldh], &ldh,
                      &h[j + 1 + (j + 1) * ldh], &ldh);
        }
    }

    return info;
}
