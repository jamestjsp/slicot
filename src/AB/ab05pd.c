/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 2025, slicot.c contributors
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <ctype.h>

i32 ab05pd(char over, i32 n1, i32 m, i32 p, i32 n2, f64 alpha,
           const f64* a1, i32 lda1, const f64* b1, i32 ldb1,
           const f64* c1, i32 ldc1, const f64* d1, i32 ldd1,
           const f64* a2, i32 lda2, const f64* b2, i32 ldb2,
           const f64* c2, i32 ldc2, const f64* d2, i32 ldd2,
           i32* n, f64* a, i32 lda, f64* b, i32 ldb,
           f64* c, i32 ldc, f64* d, i32 ldd) {

    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    char over_upper = (char)toupper((unsigned char)over);
    bool lover = (over_upper == 'O');

    *n = n1 + n2;

    if (over_upper != 'N' && over_upper != 'O') {
        return -1;
    }
    if (n1 < 0) {
        return -2;
    }
    if (m < 0) {
        return -3;
    }
    if (p < 0) {
        return -4;
    }
    if (n2 < 0) {
        return -5;
    }

    i32 max1n1 = (1 > n1) ? 1 : n1;
    if (lda1 < max1n1) {
        return -8;
    }
    if (ldb1 < max1n1) {
        return -10;
    }

    i32 max1p = (1 > p) ? 1 : p;
    if ((n1 > 0 && ldc1 < max1p) || (n1 == 0 && ldc1 < 1)) {
        return -12;
    }
    if (ldd1 < max1p) {
        return -14;
    }

    i32 max1n2 = (1 > n2) ? 1 : n2;
    if (lda2 < max1n2) {
        return -16;
    }
    if (ldb2 < max1n2) {
        return -18;
    }

    if ((n2 > 0 && ldc2 < max1p) || (n2 == 0 && ldc2 < 1)) {
        return -20;
    }
    if (ldd2 < max1p) {
        return -22;
    }

    i32 max1n = (1 > *n) ? 1 : *n;
    if (lda < max1n) {
        return -25;
    }
    if (ldb < max1n) {
        return -27;
    }

    if ((*n > 0 && ldc < max1p) || (*n == 0 && ldc < 1)) {
        return -29;
    }
    if (ldd < max1p) {
        return -31;
    }

    i32 minmp = (m < p) ? m : p;
    i32 maxnminmp = *n;
    if (minmp < maxnminmp) maxnminmp = minmp;
    if (maxnminmp == 0) {
        return 0;
    }

    i32 n1p1 = n1;

    /*
     *                       ( A1   0  )
     *     Construct     A = (         ) .
     *                       ( 0    A2 )
     */
    if (lover && lda1 <= lda) {
        if (lda1 < lda) {
            for (i32 j = n1 - 1; j >= 0; j--) {
                for (i32 i = n1 - 1; i >= 0; i--) {
                    a[i + j * lda] = a1[i + j * lda1];
                }
            }
        }
    } else {
        SLC_DLACPY("F", &n1, &n1, a1, &lda1, a, &lda);
    }

    if (n2 > 0) {
        SLC_DLASET("F", &n1, &n2, &ZERO, &ZERO, &a[n1p1 * lda], &lda);
        SLC_DLASET("F", &n2, &n1, &ZERO, &ZERO, &a[n1p1], &lda);
        SLC_DLACPY("F", &n2, &n2, a2, &lda2, &a[n1p1 + n1p1 * lda], &lda);
    }

    /*
     *                        ( B1 )
     *     Construct      B = (    ) .
     *                        ( B2 )
     */
    if (lover && ldb1 <= ldb) {
        if (ldb1 < ldb) {
            for (i32 j = m - 1; j >= 0; j--) {
                for (i32 i = n1 - 1; i >= 0; i--) {
                    b[i + j * ldb] = b1[i + j * ldb1];
                }
            }
        }
    } else {
        SLC_DLACPY("F", &n1, &m, b1, &ldb1, b, &ldb);
    }

    if (n2 > 0) {
        SLC_DLACPY("F", &n2, &m, b2, &ldb2, &b[n1p1], &ldb);
    }

    /*
     *     Construct      C = ( C1 alpha*C2 ) .
     */
    if (lover && ldc1 <= ldc) {
        if (ldc1 < ldc) {
            for (i32 j = n1 - 1; j >= 0; j--) {
                for (i32 i = p - 1; i >= 0; i--) {
                    c[i + j * ldc] = c1[i + j * ldc1];
                }
            }
        }
    } else {
        SLC_DLACPY("F", &p, &n1, c1, &ldc1, c, &ldc);
    }

    if (n2 > 0) {
        SLC_DLACPY("F", &p, &n2, c2, &ldc2, &c[n1p1 * ldc], &ldc);
        if (alpha != ONE) {
            i32 zero_int = 0;
            i32 info = 0;
            SLC_DLASCL("G", &zero_int, &zero_int, &ONE, &alpha, &p, &n2,
                       &c[n1p1 * ldc], &ldc, &info);
        }
    }

    /*
     *     Construct       D = D1 + alpha*D2 .
     */
    if (lover && ldd1 <= ldd) {
        if (ldd1 < ldd) {
            for (i32 j = m - 1; j >= 0; j--) {
                for (i32 i = p - 1; i >= 0; i--) {
                    d[i + j * ldd] = d1[i + j * ldd1];
                }
            }
        }
    } else {
        SLC_DLACPY("F", &p, &m, d1, &ldd1, d, &ldd);
    }

    i32 one_int = 1;
    for (i32 j = 0; j < m; j++) {
        SLC_DAXPY(&p, &alpha, &d2[j * ldd2], &one_int, &d[j * ldd], &one_int);
    }

    return 0;
}
