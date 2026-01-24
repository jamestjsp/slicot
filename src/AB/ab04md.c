/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 2025, slicot.c contributors
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <ctype.h>

i32 ab04md(char type, i32 n, i32 m, i32 p, f64 alpha, f64 beta,
           f64* a, i32 lda, f64* b, i32 ldb, f64* c, i32 ldc, f64* d, i32 ldd,
           i32* iwork, f64* dwork, i32 ldwork) {

    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const f64 TWO = 2.0;

    char type_upper = (char)toupper((unsigned char)type);
    bool ltype = (type_upper == 'D');

    if (type_upper != 'D' && type_upper != 'C') {
        return -1;
    }
    if (n < 0) {
        return -2;
    }
    if (m < 0) {
        return -3;
    }
    if (p < 0) {
        return -4;
    }
    if (alpha == ZERO) {
        return -5;
    }
    if (beta == ZERO) {
        return -6;
    }
    i32 max1n = (1 > n) ? 1 : n;
    if (lda < max1n) {
        return -8;
    }
    if (ldb < max1n) {
        return -10;
    }
    i32 max1p = (1 > p) ? 1 : p;
    if (ldc < max1p) {
        return -12;
    }
    if (ldd < max1p) {
        return -14;
    }
    if (ldwork < max1n) {
        return -17;
    }

    i32 maxnmp = n;
    if (m > maxnmp) maxnmp = m;
    if (p > maxnmp) maxnmp = p;
    if (maxnmp == 0) {
        return 0;
    }

    f64 palpha, pbeta;
    if (ltype) {
        palpha = alpha;
        pbeta = beta;
    } else {
        palpha = -beta;
        pbeta = -alpha;
    }

    f64 ab2 = palpha * pbeta * TWO;
    f64 sqrab2 = sqrt(fabs(ab2));
    if (palpha < 0.0) {
        sqrab2 = -sqrab2;
    }

    for (i32 i = 0; i < n; i++) {
        a[i + i * lda] += palpha;
    }

    i32 info = 0;
    SLC_DGETRF(&n, &n, a, &lda, iwork, &info);

    if (info != 0) {
        return ltype ? 1 : 2;
    }

    i32 one_int = 1;
    f64 neg_one = -ONE;

    SLC_DGETRS("N", &n, &m, a, &lda, iwork, b, &ldb, &info);

    SLC_DGEMM("N", "N", &p, &m, &n, &neg_one, c, &ldc, b, &ldb, &ONE, d, &ldd);

    i32 zero_int = 0;
    SLC_DLASCL("G", &zero_int, &zero_int, &ONE, &sqrab2, &n, &m, b, &ldb, &info);

    SLC_DTRSM("R", "U", "N", "N", &p, &n, &sqrab2, a, &lda, c, &ldc);

    SLC_DTRSM("R", "L", "N", "U", &p, &n, &ONE, a, &lda, c, &ldc);

    for (i32 i = n - 2; i >= 0; i--) {
        i32 ip = iwork[i] - 1;
        if (ip < 0 || ip >= n) continue;
        if (ip != i) {
            SLC_DSWAP(&p, &c[i * ldc], &one_int, &c[ip * ldc], &one_int);
        }
    }

    SLC_DGETRI(&n, a, &lda, iwork, dwork, &ldwork, &info);

    f64 neg_ab2 = -ab2;
    for (i32 j = 0; j < n; j++) {
        SLC_DSCAL(&n, &neg_ab2, &a[j * lda], &one_int);
        a[j + j * lda] += pbeta;
    }

    return 0;
}
