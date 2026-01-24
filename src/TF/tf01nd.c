/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"

void tf01nd(const char* uplo, const i32 n, const i32 m, const i32 p, const i32 ny,
            const f64* a, const i32 lda, const f64* b, const i32 ldb,
            const f64* c, const i32 ldc, const f64* d, const i32 ldd,
            const f64* u, const i32 ldu, f64* x, f64* y, const i32 ldy,
            f64* dwork, i32* info)
{
    const f64 zero = 0.0;
    const f64 one = 1.0;
    const i32 int1 = 1;

    bool luplo = (uplo[0] == 'U' || uplo[0] == 'u');
    bool is_lower = (uplo[0] == 'L' || uplo[0] == 'l');

    *info = 0;

    if (!luplo && !is_lower) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (m < 0) {
        *info = -3;
    } else if (p < 0) {
        *info = -4;
    } else if (ny < 0) {
        *info = -5;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -7;
    } else if (ldb < (n > 1 ? n : 1)) {
        *info = -9;
    } else if (ldc < (p > 1 ? p : 1)) {
        *info = -11;
    } else if (ldd < (p > 1 ? p : 1)) {
        *info = -13;
    } else if (ldu < (m > 1 ? m : 1)) {
        *info = -15;
    } else if (ldy < (p > 1 ? p : 1)) {
        *info = -18;
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

    SLC_DCOPY(&n, x, &int1, dwork, &int1);

    for (i32 ik = 0; ik < ny; ik++) {
        // y(:,ik) = C * dwork (where dwork contains current x)
        SLC_DGEMV("No transpose", &p, &n, &one, c, &ldc, dwork, &int1,
                  &zero, &y[ik * ldy], &int1);

        // dwork = triangular part of A * dwork (using DTRMV)
        SLC_DTRMV(uplo, "No transpose", "Non-unit", &n, a, &lda, dwork, &int1);

        // Add contribution from subdiagonal (upper Hessenberg) or superdiagonal (lower Hessenberg)
        if (luplo) {
            // Upper Hessenberg: add A(i,i-1)*x(i-1) to dwork(i) for i=2..n
            for (i32 i = 1; i < n; i++) {
                dwork[i] = dwork[i] + a[i + (i-1)*lda] * x[i-1];
            }
        } else {
            // Lower Hessenberg: add A(i,i+1)*x(i+1) to dwork(i) for i=1..n-1
            for (i32 i = 0; i < n - 1; i++) {
                dwork[i] = dwork[i] + a[i + (i+1)*lda] * x[i+1];
            }
        }

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
