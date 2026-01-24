/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 2025, slicot.c contributors
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <ctype.h>
#include <stdlib.h>

i32 ab05nd(char over, i32 n1, i32 m1, i32 p1, i32 n2, f64 alpha,
           const f64* a1, i32 lda1, const f64* b1, i32 ldb1,
           const f64* c1, i32 ldc1, const f64* d1, i32 ldd1,
           const f64* a2, i32 lda2, const f64* b2, i32 ldb2,
           const f64* c2, i32 ldc2, const f64* d2, i32 ldd2,
           i32* n, f64* a, i32 lda, f64* b, i32 ldb,
           f64* c, i32 ldc, f64* d, i32 ldd,
           i32* iwork, f64* dwork, i32 ldwork) {

    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    char over_upper = (char)toupper((unsigned char)over);
    bool lover = (over_upper == 'O');

    bool alias_a = (a1 == a);
    bool alias_b = (b1 == b);
    bool alias_c = (c1 == c);
    bool alias_d = (d1 == d);

    *n = n1 + n2;
    i32 info = 0;

    i32 ldwm1 = (1 > m1) ? 1 : m1;

    if (over_upper != 'N' && over_upper != 'O') {
        return -1;
    }
    if (n1 < 0) {
        return -2;
    }
    if (m1 < 0) {
        return -3;
    }
    if (p1 < 0) {
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

    i32 max1p1 = (1 > p1) ? 1 : p1;
    if ((n1 > 0 && ldc1 < max1p1) || (n1 == 0 && ldc1 < 1)) {
        return -12;
    }
    if (ldd1 < max1p1) {
        return -14;
    }

    i32 max1n2 = (1 > n2) ? 1 : n2;
    if (lda2 < max1n2) {
        return -16;
    }
    if (ldb2 < max1n2) {
        return -18;
    }

    if ((n2 > 0 && ldc2 < ldwm1) || (n2 == 0 && ldc2 < 1)) {
        return -20;
    }
    if (ldd2 < ldwm1) {
        return -22;
    }

    i32 max1n = (1 > *n) ? 1 : *n;
    if (lda < max1n) {
        return -25;
    }
    if (ldb < max1n) {
        return -27;
    }

    if ((*n > 0 && ldc < max1p1) || (*n == 0 && ldc < 1)) {
        return -29;
    }
    if (ldd < max1p1) {
        return -31;
    }

    i32 ldw = p1 * p1;
    if (m1 * m1 > ldw) ldw = m1 * m1;
    if (n1 * p1 > ldw) ldw = n1 * p1;
    if (lover) {
        if (m1 > (*n) * n2) {
            i32 ldw2 = m1 * (m1 + 1);
            if (ldw2 > ldw) ldw = ldw2;
        }
        ldw = n1 * p1 + ldw;
    }
    if (ldw < 1) ldw = 1;
    if (ldwork < ldw) {
        return -34;
    }

    i32 min_m1_p1 = (m1 < p1) ? m1 : p1;
    if (*n == 0 || min_m1_p1 == 0) {
        return 0;
    }

    i32 ldwsave = ldw;

    if (p1 > 0) {
        SLC_DLASET("F", &p1, &p1, &ZERO, &ONE, dwork, &p1);

        SLC_DGEMM("N", "N", &p1, &p1, &m1, &alpha,
                  d1, &ldd1, d2, &ldd2, &ONE, dwork, &p1);

        SLC_DGETRF(&p1, &p1, dwork, &p1, iwork, &info);
        if (info != 0) {
            return info;
        }

        if (lover && alias_d && ldd1 <= ldd) {
            if (ldd1 < ldd) {
                for (i32 j = m1 - 1; j >= 0; j--) {
                    for (i32 i = p1 - 1; i >= 0; i--) {
                        d[i + j * ldd] = d1[i + j * ldd1];
                    }
                }
            }
        } else {
            SLC_DLACPY("F", &p1, &m1, d1, &ldd1, d, &ldd);
        }

        SLC_DGETRS("N", &p1, &m1, dwork, &p1, iwork, d, &ldd, &info);

        if (n1 > 0) {
            if (lover && alias_c) {
                ldw = ldwsave - p1 * n1 + 1;
                SLC_DLACPY("F", &p1, &n1, c1, &ldc1, &dwork[ldw - 1], &p1);

                if (ldc1 != ldc) {
                    SLC_DLACPY("F", &p1, &n1, &dwork[ldw - 1], &p1, c, &ldc);
                }
            } else {
                SLC_DLACPY("F", &p1, &n1, c1, &ldc1, c, &ldc);
            }

            SLC_DGETRS("N", &p1, &n1, dwork, &p1, iwork, c, &ldc, &info);
        }

        SLC_DLASET("F", &m1, &m1, &ZERO, &ONE, dwork, &ldwm1);
        f64 neg_alpha = -alpha;
        SLC_DGEMM("N", "N", &m1, &m1, &p1, &neg_alpha,
                  d2, &ldd2, d, &ldd, &ONE, dwork, &ldwm1);

    } else {
        SLC_DLASET("F", &m1, &m1, &ZERO, &ONE, dwork, &ldwm1);
    }

    if (lover && alias_a && lda1 <= lda) {
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

    if (n1 > 0 && m1 > 0) {
        if (lover && alias_b) {
            if (n1 * m1 <= (*n) * n2) {
                SLC_DLACPY("F", &n1, &m1, b1, &ldb1, &a[n1 * lda], &n1);
                SLC_DGEMM("N", "N", &n1, &m1, &m1, &ONE,
                          &a[n1 * lda], &n1, dwork, &ldwm1, &ZERO, b, &ldb);
            } else if (ldb1 < ldb) {
                for (i32 j = m1 - 1; j >= 0; j--) {
                    for (i32 i = n1 - 1; i >= 0; i--) {
                        b[i + j * ldb] = b1[i + j * ldb1];
                    }
                }

                if (m1 <= (*n) * n2) {
                    i32 int_one = 1;
                    for (i32 j = 0; j < n1; j++) {
                        SLC_DCOPY(&m1, &b[j], &ldb, &a[n1 * lda], &int_one);
                        SLC_DGEMV("T", &m1, &m1, &ONE, dwork, &ldwm1,
                                  &a[n1 * lda], &int_one, &ZERO, &b[j], &ldb);
                    }
                } else {
                    i32 int_one = 1;
                    for (i32 j = 0; j < n1; j++) {
                        SLC_DCOPY(&m1, &b[j], &ldb, &dwork[m1 * m1], &int_one);
                        SLC_DGEMV("T", &m1, &m1, &ONE, dwork, &ldwm1,
                                  &dwork[m1 * m1], &int_one, &ZERO, &b[j], &ldb);
                    }
                }
            } else if (m1 <= (*n) * n2) {
                i32 int_one = 1;
                for (i32 j = 0; j < n1; j++) {
                    SLC_DCOPY(&m1, &b1[j], &ldb1, &a[n1 * lda], &int_one);
                    SLC_DGEMV("T", &m1, &m1, &ONE, dwork, &ldwm1,
                              &a[n1 * lda], &int_one, &ZERO, &b[j], &ldb);
                }
            } else {
                i32 int_one = 1;
                for (i32 j = 0; j < n1; j++) {
                    SLC_DCOPY(&m1, &b1[j], &ldb1, &dwork[m1 * m1], &int_one);
                    SLC_DGEMV("T", &m1, &m1, &ONE, dwork, &ldwm1,
                              &dwork[m1 * m1], &int_one, &ZERO, &b[j], &ldb);
                }
            }
        } else {
            SLC_DGEMM("N", "N", &n1, &m1, &m1, &ONE,
                      b1, &ldb1, dwork, &ldwm1, &ZERO, b, &ldb);
        }
    }

    if (n2 > 0) {
        if (p1 > 0) {
            SLC_DGEMM("N", "N", &n2, &m1, &p1, &ONE,
                      b2, &ldb2, d, &ldd, &ZERO, &b[n1], &ldb);

            f64 neg_alpha = -alpha;
            SLC_DGEMM("N", "N", &p1, &n2, &m1, &neg_alpha,
                      d, &ldd, c2, &ldc2, &ZERO, &c[n1 * ldc], &ldc);
        } else if (m1 > 0) {
            SLC_DLASET("F", &n2, &m1, &ZERO, &ZERO, &b[n1], &ldb);
        }
    }

    if (n1 > 0 && p1 > 0) {
        f64 neg_alpha = -alpha;
        SLC_DGEMM("N", "N", &n1, &p1, &m1, &neg_alpha,
                  b, &ldb, d2, &ldd2, &ZERO, dwork, &n1);

        if (lover && alias_c) {
            SLC_DGEMM("N", "N", &n1, &n1, &p1, &ONE,
                      dwork, &n1, &dwork[ldw - 1], &p1, &ONE, a, &lda);
        } else {
            SLC_DGEMM("N", "N", &n1, &n1, &p1, &ONE,
                      dwork, &n1, c1, &ldc1, &ONE, a, &lda);
        }
    }

    if (n2 > 0) {
        SLC_DLACPY("F", &n2, &n2, a2, &lda2, &a[(n1) + (n1) * lda], &lda);

        if (m1 > 0) {
            f64 neg_alpha = -alpha;
            SLC_DGEMM("N", "N", &n2, &n2, &m1, &neg_alpha,
                      &b[n1], &ldb, c2, &ldc2, &ONE, &a[(n1) + (n1) * lda], &lda);
        }

        SLC_DGEMM("N", "N", &n2, &n1, &p1, &ONE,
                  b2, &ldb2, c, &ldc, &ZERO, &a[n1], &lda);

        f64 neg_alpha = -alpha;
        SLC_DGEMM("N", "N", &n1, &n2, &m1, &neg_alpha,
                  b, &ldb, c2, &ldc2, &ZERO, &a[n1 * lda], &lda);
    }

    return 0;
}
