/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>

void mb02uv(i32 n, f64* a, i32 lda, i32* ipiv, i32* jpiv, i32* info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    *info = 0;

    f64 eps = SLC_DLAMCH("Precision");
    f64 smlnum = SLC_DLAMCH("Safe minimum") / eps;
    f64 bignum = ONE / smlnum;
    SLC_DLABAD(&smlnum, &bignum);

    i32 ipv = 0;
    i32 jpv = 0;
    f64 xmax = ZERO;

    for (i32 jp = 0; jp < n; jp++) {
        for (i32 ip = 0; ip < n; ip++) {
            if (fabs(a[ip + jp * lda]) > xmax) {
                xmax = fabs(a[ip + jp * lda]);
                ipv = ip;
                jpv = jp;
            }
        }
    }

    f64 smin = fmax(eps * xmax, smlnum);

    if (ipv != 0) {
        i32 nn = n;
        i32 incx = lda;
        SLC_DSWAP(&nn, &a[ipv + 0 * lda], &incx, &a[0 + 0 * lda], &incx);
    }
    ipiv[0] = ipv + 1;

    if (jpv != 0) {
        i32 nn = n;
        i32 incx = 1;
        SLC_DSWAP(&nn, &a[0 + jpv * lda], &incx, &a[0 + 0 * lda], &incx);
    }
    jpiv[0] = jpv + 1;

    if (fabs(a[0 + 0 * lda]) < smin) {
        *info = 1;
        a[0 + 0 * lda] = smin;
    }

    if (n > 1) {
        f64 scale = ONE / a[0 + 0 * lda];
        i32 len = n - 1;
        i32 inc = 1;
        SLC_DSCAL(&len, &scale, &a[1 + 0 * lda], &inc);

        i32 mm = n - 1;
        i32 nn = n - 1;
        f64 al = -ONE;
        i32 incx = 1;
        i32 incy = lda;
        SLC_DGER(&mm, &nn, &al, &a[1 + 0 * lda], &incx, &a[0 + 1 * lda], &incy, &a[1 + 1 * lda], &lda);
    }

    for (i32 i = 1; i < n - 1; i++) {
        ipv = i;
        jpv = i;
        xmax = ZERO;

        for (i32 jp = i; jp < n; jp++) {
            for (i32 ip = i; ip < n; ip++) {
                if (fabs(a[ip + jp * lda]) > xmax) {
                    xmax = fabs(a[ip + jp * lda]);
                    ipv = ip;
                    jpv = jp;
                }
            }
        }

        if (ipv != i) {
            i32 nn = n;
            i32 incx = lda;
            SLC_DSWAP(&nn, &a[ipv + 0 * lda], &incx, &a[i + 0 * lda], &incx);
        }
        ipiv[i] = ipv + 1;

        if (jpv != i) {
            i32 nn = n;
            i32 incx = 1;
            SLC_DSWAP(&nn, &a[0 + jpv * lda], &incx, &a[0 + i * lda], &incx);
        }
        jpiv[i] = jpv + 1;

        if (fabs(a[i + i * lda]) < smin) {
            *info = i + 1;
            a[i + i * lda] = smin;
        }

        f64 scale = ONE / a[i + i * lda];
        i32 len = n - i - 1;
        i32 inc = 1;
        SLC_DSCAL(&len, &scale, &a[i + 1 + i * lda], &inc);

        i32 mm = n - i - 1;
        i32 nn = n - i - 1;
        f64 al = -ONE;
        i32 incx = 1;
        i32 incy = lda;
        SLC_DGER(&mm, &nn, &al, &a[i + 1 + i * lda], &incx, &a[i + (i + 1) * lda], &incy, &a[i + 1 + (i + 1) * lda], &lda);
    }

    if (n > 0 && fabs(a[n - 1 + (n - 1) * lda]) < smin) {
        *info = n;
        a[n - 1 + (n - 1) * lda] = smin;
    }

    if (n > 0) {
        ipiv[n - 1] = n;
        jpiv[n - 1] = n;
    }
}
