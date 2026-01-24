/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <stdbool.h>

static int delctg_dummy(const f64* par1, const f64* par2, const f64* par3)
{
    (void)par1;
    (void)par2;
    (void)par3;
    return 1;
}

void tg01wd(
    const i32 n, const i32 m, const i32 p,
    f64* a, const i32 lda,
    f64* e, const i32 lde,
    f64* b, const i32 ldb,
    f64* c, const i32 ldc,
    f64* q, const i32 ldq,
    f64* z, const i32 ldz,
    f64* alphar, f64* alphai, f64* beta,
    f64* dwork, const i32 ldwork,
    i32* info
)
{
    const f64 zero = 0.0;
    const f64 one = 1.0;

    bool blas3, block;
    i32 bl, chunk, i, j, maxwrk, sdim;
    i32 int1 = 1;
    i32 bwork_dummy = 0;

    *info = 0;

    i32 max_1_n = (1 > n) ? 1 : n;
    i32 max_1_p = (1 > p) ? 1 : p;

    if (n < 0) {
        *info = -1;
    } else if (m < 0) {
        *info = -2;
    } else if (p < 0) {
        *info = -3;
    } else if (lda < max_1_n) {
        *info = -5;
    } else if (lde < max_1_n) {
        *info = -7;
    } else if (ldb < max_1_n) {
        *info = -9;
    } else if (ldc < max_1_p) {
        *info = -11;
    } else if (ldq < max_1_n) {
        *info = -13;
    } else if (ldz < max_1_n) {
        *info = -15;
    } else if (ldwork < 8 * n + 16) {
        *info = -20;
    }

    if (*info != 0) {
        return;
    }

    if (n == 0) {
        dwork[0] = one;
        return;
    }

    SLC_DGGES("V", "V", "N", delctg_dummy, &n,
              a, &lda, e, &lde, &sdim, alphar, alphai, beta,
              q, &ldq, z, &ldz, dwork, &ldwork, &bwork_dummy, info);
    if (*info != 0) {
        return;
    }
    maxwrk = (i32)dwork[0];

    chunk = ldwork / n;
    block = (m > 1);
    blas3 = (chunk >= m) && block;

    if (blas3) {
        SLC_DLACPY("F", &n, &m, b, &ldb, dwork, &n);
        SLC_DGEMM("T", "N", &n, &m, &n, &one, q, &ldq,
                  dwork, &n, &zero, b, &ldb);
    } else if (block) {
        for (j = 0; j < m; j += chunk) {
            bl = (m - j < chunk) ? m - j : chunk;
            if (bl <= 0) break;
            SLC_DLACPY("F", &n, &bl, &b[j * ldb], &ldb, dwork, &n);
            SLC_DGEMM("T", "N", &n, &bl, &n, &one, q, &ldq,
                      dwork, &n, &zero, &b[j * ldb], &ldb);
        }
    } else {
        if (m > 0) {
            SLC_DCOPY(&n, b, &int1, dwork, &int1);
            SLC_DGEMV("T", &n, &n, &one, q, &ldq, dwork, &int1, &zero, b, &int1);
        }
    }
    i32 nm = n * m;
    if (nm > maxwrk) maxwrk = nm;

    block = (p > 1);
    blas3 = (chunk >= p) && block;

    if (blas3) {
        SLC_DLACPY("F", &p, &n, c, &ldc, dwork, &p);
        SLC_DGEMM("N", "N", &p, &n, &n, &one,
                  dwork, &p, z, &ldz, &zero, c, &ldc);
    } else if (block) {
        for (i = 0; i < p; i += chunk) {
            bl = (p - i < chunk) ? p - i : chunk;
            if (bl <= 0) break;
            SLC_DLACPY("F", &bl, &n, &c[i], &ldc, dwork, &bl);
            SLC_DGEMM("N", "N", &bl, &n, &n, &one,
                      dwork, &bl, z, &ldz, &zero, &c[i], &ldc);
        }
    } else {
        if (p > 0) {
            SLC_DCOPY(&n, c, &ldc, dwork, &int1);
            SLC_DGEMV("T", &n, &n, &one, z, &ldz, dwork, &int1, &zero, c, &ldc);
        }
    }
    i32 pn = p * n;
    if (pn > maxwrk) maxwrk = pn;

    dwork[0] = (f64)maxwrk;
}
