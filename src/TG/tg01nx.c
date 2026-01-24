/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <stdbool.h>

void tg01nx(
    const char* jobt,
    const i32 n, const i32 m, const i32 p, const i32 ndim,
    f64* a, const i32 lda,
    f64* e, const i32 lde,
    f64* b, const i32 ldb,
    f64* c, const i32 ldc,
    f64* q, const i32 ldq,
    f64* z, const i32 ldz,
    i32* iwork,
    i32* info
)
{
    const f64 one = 1.0;
    const f64 zero = 0.0;
    i32 int1 = 1;

    bool trinv;
    f64 dif, scale;
    i32 i, n1, n2;
    f64 dum[1];

    *info = 0;
    trinv = (jobt[0] == 'I' || jobt[0] == 'i');

    if (jobt[0] != 'D' && jobt[0] != 'd' && !trinv) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (m < 0) {
        *info = -3;
    } else if (p < 0) {
        *info = -4;
    } else if (ndim < 0 || ndim > n) {
        *info = -5;
    } else if (lda < (1 > n ? 1 : n)) {
        *info = -7;
    } else if (lde < (1 > n ? 1 : n)) {
        *info = -9;
    } else if (ldb < (1 > n ? 1 : n)) {
        *info = -11;
    } else if (ldc < (1 > p ? 1 : p)) {
        *info = -13;
    } else if (ldq < (1 > n ? 1 : n)) {
        *info = -15;
    } else if (ldz < (1 > n ? 1 : n)) {
        *info = -17;
    }

    if (*info != 0) {
        return;
    }

    if (n == 0) {
        return;
    }

    if (trinv) {
        for (i = 1; i < n; i++) {
            i32 len = i;
            SLC_DSWAP(&len, &z[i * ldz], &int1, &z[i], &ldz);
        }

        for (i = 1; i < n; i++) {
            i32 len = i;
            SLC_DSWAP(&len, &q[i * ldq], &int1, &q[i], &ldq);
        }
    }

    n1 = ndim;
    n2 = n - ndim;

    if (n1 > 0 && n2 > 0) {
        i32 ijob = 0;
        i32 lddum = 1;
        SLC_DTGSYL("N", &ijob, &n1, &n2,
                   a, &lda, &a[n1 + n1 * lda], &lda, &a[n1 * lda], &lda,
                   e, &lde, &e[n1 + n1 * lde], &lde, &e[n1 * lde], &lde,
                   &scale, &dif, dum, &lddum, iwork, info);

        if (*info != 0) {
            *info = 1;
            return;
        }

        if (scale > 0.0) {
            scale = one / scale;
        }

        SLC_DGEMM("N", "N", &n1, &m, &n2, &scale,
                  &e[n1 * lde], &lde, &b[n1], &ldb, &one, b, &ldb);

        f64 neg_scale = -scale;
        SLC_DGEMM("N", "N", &p, &n2, &n1, &neg_scale,
                  c, &ldc, &a[n1 * lda], &lda, &one, &c[n1 * ldc], &ldc);

        if (trinv) {
            SLC_DGEMM("N", "N", &n, &n2, &n1, &neg_scale,
                      q, &ldq, &e[n1 * lde], &lde, &one, &q[n1 * ldq], &ldq);

            SLC_DGEMM("N", "N", &n1, &n, &n2, &scale,
                      &a[n1 * lda], &lda, &z[n1], &ldz, &one, z, &ldz);
        } else {
            SLC_DGEMM("N", "N", &n1, &n, &n2, &scale,
                      &e[n1 * lde], &lde, &q[n1], &ldq, &one, q, &ldq);

            SLC_DGEMM("N", "N", &n, &n2, &n1, &neg_scale,
                      z, &ldz, &a[n1 * lda], &lda, &one, &z[n1 * ldz], &ldz);
        }

        SLC_DLASET("F", &n1, &n2, &zero, &zero, &a[n1 * lda], &lda);
        SLC_DLASET("F", &n1, &n2, &zero, &zero, &e[n1 * lde], &lde);
    }
}
