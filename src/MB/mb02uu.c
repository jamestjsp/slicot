/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>

void mb02uu(i32 n, const f64* a, i32 lda, f64* rhs, const i32* ipiv,
            const i32* jpiv, f64* scale)
{
    const f64 ONE = 1.0;
    const f64 TWO = 2.0;

    f64 eps = SLC_DLAMCH("Precision");
    f64 smlnum = SLC_DLAMCH("Safe minimum") / eps;
    f64 bignum = ONE / smlnum;
    SLC_DLABAD(&smlnum, &bignum);

    for (i32 i = 0; i < n - 1; i++) {
        i32 ip = ipiv[i] - 1;
        if (ip >= 0 && ip < n && ip != i) {
            f64 temp = rhs[i];
            rhs[i] = rhs[ip];
            rhs[ip] = temp;
        }
    }

    for (i32 i = 0; i < n - 1; i++) {
        i32 len = n - i - 1;
        f64 al = -rhs[i];
        i32 incx = 1;
        i32 incy = 1;
        SLC_DAXPY(&len, &al, &a[i + 1 + i * lda], &incx, &rhs[i + 1], &incy);
    }

    f64 factor = TWO * (f64)n;
    i32 i = 0;
    bool need_scale = false;

    while (i < n) {
        if ((factor * smlnum) * fabs(rhs[i]) > fabs(a[i + i * lda])) {
            need_scale = true;
            break;
        }
        i++;
    }

    if (!need_scale) {
        *scale = ONE;
    } else {
        i32 nn = n;
        i32 inc = 1;
        i32 idx = SLC_IDAMAX(&nn, rhs, &inc);
        *scale = (ONE / factor) / fabs(rhs[idx - 1]);
        SLC_DSCAL(&nn, scale, rhs, &inc);
    }

    for (i32 i = n - 1; i >= 0; i--) {
        f64 temp = ONE / a[i + i * lda];
        rhs[i] = rhs[i] * temp;
        for (i32 j = i + 1; j < n; j++) {
            rhs[i] = rhs[i] - rhs[j] * (a[i + j * lda] * temp);
        }
    }

    for (i32 i = n - 2; i >= 0; i--) {
        i32 ip = jpiv[i] - 1;
        if (ip >= 0 && ip < n && ip != i) {
            f64 temp = rhs[i];
            rhs[i] = rhs[ip];
            rhs[ip] = temp;
        }
    }
}
