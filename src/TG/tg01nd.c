/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot/tg.h"
#include "slicot_blas.h"

#include <stdbool.h>

void tg01nd(
    const char* job, const char* jobt,
    const i32 n, const i32 m, const i32 p,
    f64* a, const i32 lda,
    f64* e, const i32 lde,
    f64* b, const i32 ldb,
    f64* c, const i32 ldc,
    f64* alphar, f64* alphai, f64* beta,
    f64* q, const i32 ldq,
    f64* z, const i32 ldz,
    i32* nf, i32* nd, i32* niblck, i32* iblck,
    const f64 tol,
    i32* iwork, f64* dwork, const i32 ldwork,
    i32* info
)
{
    const f64 one = 1.0;
    const f64 zero = 0.0;

    bool lquery, trinf, trinv;
    f64 dif, scale;
    i32 i, minwrk, n1, n11, n2, wrkopt;
    i32 int1 = 1;

    *info = 0;
    trinf = (job[0] == 'I' || job[0] == 'i');
    trinv = (jobt[0] == 'I' || jobt[0] == 'i');

    if (!(job[0] == 'F' || job[0] == 'f') && !trinf) {
        *info = -1;
    } else if (!(jobt[0] == 'D' || jobt[0] == 'd') && !trinv) {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    } else if (m < 0) {
        *info = -4;
    } else if (p < 0) {
        *info = -5;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -7;
    } else if (lde < (n > 1 ? n : 1)) {
        *info = -9;
    } else if (ldb < (n > 1 ? n : 1)) {
        *info = -11;
    } else if (ldc < (p > 1 ? p : 1)) {
        *info = -13;
    } else if (ldq < (n > 1 ? n : 1)) {
        *info = -18;
    } else if (ldz < (n > 1 ? n : 1)) {
        *info = -20;
    } else if (tol >= one) {
        *info = -25;
    } else {
        lquery = (ldwork == -1);
        if (n == 0) {
            minwrk = 1;
        } else {
            minwrk = 4 * n;
        }

        if (lquery) {
            i32 tg01md_info;
            tg01md(job, n, m, p, a, lda, e, lde, b, ldb, c, ldc,
                   alphar, alphai, beta, q, ldq, z, ldz, nf, nd,
                   niblck, iblck, tol, iwork, dwork, -1, &tg01md_info);
            wrkopt = minwrk > (i32)dwork[0] ? minwrk : (i32)dwork[0];
        } else if (ldwork < minwrk) {
            *info = -28;
        }
    }

    if (*info != 0) {
        i32 neg_info = -(*info);
        SLC_XERBLA("TG01ND", &neg_info);
        return;
    } else if (lquery) {
        dwork[0] = (f64)wrkopt;
        return;
    }

    if (n == 0) {
        *nf = 0;
        *nd = 0;
        *niblck = 0;
        dwork[0] = one;
        return;
    }

    i32 tg01md_info;
    tg01md(job, n, m, p, a, lda, e, lde, b, ldb, c, ldc,
           alphar, alphai, beta, q, ldq, z, ldz, nf, nd,
           niblck, iblck, tol, iwork, dwork, ldwork, &tg01md_info);

    if (tg01md_info != 0) {
        *info = tg01md_info;
        return;
    }
    wrkopt = minwrk > (i32)dwork[0] ? minwrk : (i32)dwork[0];

    if (trinv) {
        for (i = 1; i < n; i++) {
            i32 im1 = i;
            SLC_DSWAP(&im1, &z[i * ldz], &int1, &z[i], &ldz);
        }
    } else {
        for (i = 1; i < n; i++) {
            i32 im1 = i;
            SLC_DSWAP(&im1, &q[i * ldq], &int1, &q[i], &ldq);
        }
    }

    if (trinf) {
        n1 = n - *nf;
        n2 = *nf;
    } else {
        n1 = *nf;
        n2 = n - *nf;
    }
    n11 = n1 < n ? n1 : n - 1;
    if (n11 < 0) n11 = 0;

    if (n1 > 0 && n2 > 0) {
        i32 dtgsyl_info;
        i32 ijob = 0;
        SLC_DTGSYL("N", &ijob, &n1, &n2, a, &lda, &a[n11 + n11 * lda], &lda,
                   &a[n11 * lda], &lda, e, &lde, &e[n11 + n11 * lde], &lde,
                   &e[n11 * lde], &lde, &scale, &dif, dwork, &ldwork, iwork,
                   &dtgsyl_info);
        if (dtgsyl_info != 0) {
            *info = 3;
            return;
        }

        if (scale > 0) {
            scale = one / scale;
        }

        SLC_DGEMM("N", "N", &n1, &m, &n2, &scale, &e[n11 * lde], &lde,
                  &b[n11], &ldb, &one, b, &ldb);

        f64 neg_scale = -scale;
        SLC_DGEMM("N", "N", &p, &n2, &n1, &neg_scale, c, &ldc, &a[n11 * lda],
                  &lda, &one, &c[n11 * ldc], &ldc);

        if (trinv) {
            SLC_DGEMM("N", "N", &n, &n2, &n1, &neg_scale, q, &ldq, &e[n11 * lde],
                      &lde, &one, &q[n11 * ldq], &ldq);

            SLC_DGEMM("N", "N", &n1, &n, &n2, &scale, &a[n11 * lda], &lda,
                      &z[n11], &ldz, &one, z, &ldz);
        } else {
            SLC_DGEMM("N", "N", &n1, &n, &n2, &scale, &e[n11 * lde], &lde,
                      &q[n11], &ldq, &one, q, &ldq);

            SLC_DGEMM("N", "N", &n, &n2, &n1, &neg_scale, z, &ldz, &a[n11 * lda],
                      &lda, &one, &z[n11 * ldz], &ldz);
        }

        SLC_DLASET("F", &n1, &n2, &zero, &zero, &a[n11 * lda], &lda);
        SLC_DLASET("F", &n1, &n2, &zero, &zero, &e[n11 * lde], &lde);
    }

    dwork[0] = (f64)wrkopt;
}
