/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"

void tf01md(const i32 n, const i32 m, const i32 p, const i32 ny,
            const f64* a, const i32 lda, const f64* b, const i32 ldb,
            const f64* c, const i32 ldc, const f64* d, const i32 ldd,
            const f64* u, const i32 ldu, f64* x, f64* y, const i32 ldy,
            f64* dwork, i32* info)
{
    const f64 zero = 0.0;
    const f64 one = 1.0;
    const i32 int1 = 1;
    i32 ik;

    *info = 0;

    if (n < 0) {
        *info = -1;
    } else if (m < 0) {
        *info = -2;
    } else if (p < 0) {
        *info = -3;
    } else if (ny < 0) {
        *info = -4;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -6;
    } else if (ldb < (n > 1 ? n : 1)) {
        *info = -8;
    } else if (ldc < (p > 1 ? p : 1)) {
        *info = -10;
    } else if (ldd < (p > 1 ? p : 1)) {
        *info = -12;
    } else if (ldu < (m > 1 ? m : 1)) {
        *info = -14;
    } else if (ldy < (p > 1 ? p : 1)) {
        *info = -17;
    }

    if (*info != 0) {
        return;
    }

    i32 minpny = (p < ny ? p : ny);
    if (minpny == 0) {
        return;
    }

    if (n == 0) {
        if (m == 0) {
            SLC_DLASET("Full", &p, &ny, &zero, &zero, y, &ldy);
        } else {
            SLC_DGEMM("No transpose", "No transpose", &p, &ny, &m, &one,
                      d, &ldd, u, &ldu, &zero, y, &ldy);
        }
        return;
    }

    for (ik = 0; ik < ny; ik++) {
        // y(:,ik) = C * x
        SLC_DGEMV("No transpose", &p, &n, &one, c, &ldc, x, &int1,
                  &zero, &y[ik * ldy], &int1);

        // dwork = A * x
        SLC_DGEMV("No transpose", &n, &n, &one, a, &lda, x, &int1,
                  &zero, dwork, &int1);

        // dwork = dwork + B * u(:,ik) = A*x + B*u
        SLC_DGEMV("No transpose", &n, &m, &one, b, &ldb, &u[ik * ldu], &int1,
                  &one, dwork, &int1);

        // x = dwork (x(k+1) = A*x(k) + B*u(k))
        SLC_DCOPY(&n, dwork, &int1, x, &int1);
    }

    // Y = Y + D * U
    SLC_DGEMM("No transpose", "No transpose", &p, &ny, &m, &one, d, &ldd,
              u, &ldu, &one, y, &ldy);
}
