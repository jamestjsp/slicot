/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 2025, slicot.c contributors
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <ctype.h>

i32 ab05md(char uplo, char over, i32 n1, i32 m1, i32 p1, i32 n2, i32 p2,
           const f64* a1, i32 lda1, const f64* b1, i32 ldb1,
           const f64* c1, i32 ldc1, const f64* d1, i32 ldd1,
           const f64* a2, i32 lda2, const f64* b2, i32 ldb2,
           const f64* c2, i32 ldc2, const f64* d2, i32 ldd2,
           i32* n, f64* a, i32 lda, f64* b, i32 ldb,
           f64* c, i32 ldc, f64* d, i32 ldd,
           f64* dwork, i32 ldwork) {

    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    char uplo_upper = (char)toupper((unsigned char)uplo);
    char over_upper = (char)toupper((unsigned char)over);

    bool luplo = (uplo_upper == 'L');
    bool lover = (over_upper == 'O');

    *n = n1 + n2;

    if (uplo_upper != 'L' && uplo_upper != 'U') {
        return -1;
    }
    if (over_upper != 'N' && over_upper != 'O') {
        return -2;
    }
    if (n1 < 0) {
        return -3;
    }
    if (m1 < 0) {
        return -4;
    }
    if (p1 < 0) {
        return -5;
    }
    if (n2 < 0) {
        return -6;
    }
    if (p2 < 0) {
        return -7;
    }

    i32 max1n1 = (1 > n1) ? 1 : n1;
    if (lda1 < max1n1) {
        return -9;
    }
    if (ldb1 < max1n1) {
        return -11;
    }

    i32 max1p1 = (1 > p1) ? 1 : p1;
    if ((n1 > 0 && ldc1 < max1p1) || (n1 == 0 && ldc1 < 1)) {
        return -13;
    }
    if (ldd1 < max1p1) {
        return -15;
    }

    i32 max1n2 = (1 > n2) ? 1 : n2;
    if (lda2 < max1n2) {
        return -17;
    }
    if (ldb2 < max1n2) {
        return -19;
    }

    i32 max1p2 = (1 > p2) ? 1 : p2;
    if ((n2 > 0 && ldc2 < max1p2) || (n2 == 0 && ldc2 < 1)) {
        return -21;
    }
    if (ldd2 < max1p2) {
        return -23;
    }

    i32 max1n = (1 > *n) ? 1 : *n;
    if (lda < max1n) {
        return -26;
    }
    if (ldb < max1n) {
        return -28;
    }
    if ((*n > 0 && ldc < max1p2) || (*n == 0 && ldc < 1)) {
        return -30;
    }
    if (ldd < max1p2) {
        return -32;
    }

    i32 maxdim = n1;
    if (m1 > maxdim) maxdim = m1;
    if (n2 > maxdim) maxdim = n2;
    if (p2 > maxdim) maxdim = p2;
    i32 minwork = (lover) ? ((1 > p1 * maxdim) ? 1 : p1 * maxdim) : 1;
    if (ldwork < minwork) {
        return -34;
    }

    i32 min_m1_p2 = (m1 < p2) ? m1 : p2;
    if (*n == 0 || min_m1_p2 == 0) {
        return 0;
    }

    i32 i1, i2;
    if (luplo) {
        i1 = 0;
        i2 = (n1 < *n) ? n1 : (*n - 1);
    } else {
        i1 = (n2 < *n) ? n2 : (*n - 1);
        i2 = 0;
    }

    i32 ldwn2 = (1 > n2) ? 1 : n2;
    i32 ldwp1 = (1 > p1) ? 1 : p1;
    i32 ldwp2 = (1 > p2) ? 1 : p2;

    if (luplo) {
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
            SLC_DLACPY("F", &n2, &n2, a2, &lda2, &a[i2 + i2 * lda], &lda);
        }
    } else {
        if (lover && lda2 <= lda) {
            if (lda2 < lda) {
                for (i32 j = n2 - 1; j >= 0; j--) {
                    for (i32 i = n2 - 1; i >= 0; i--) {
                        a[i + j * lda] = a2[i + j * lda2];
                    }
                }
            }
        } else {
            SLC_DLACPY("F", &n2, &n2, a2, &lda2, a, &lda);
        }
        if (n1 > 0) {
            SLC_DLACPY("F", &n1, &n1, a1, &lda1, &a[i1 + i1 * lda], &lda);
        }
    }

    if (n1 > 0 && n2 > 0) {
        SLC_DLASET("F", &n1, &n2, &ZERO, &ZERO, &a[i1 + i2 * lda], &lda);
        SLC_DGEMM("N", "N", &n2, &n1, &p1, &ONE, b2, &ldb2, c1, &ldc1, &ZERO, &a[i2 + i1 * lda], &lda);
    }

    if (luplo) {
        if (lover && ldb1 <= ldb) {
            if (ldb1 < ldb) {
                for (i32 j = m1 - 1; j >= 0; j--) {
                    for (i32 i = n1 - 1; i >= 0; i--) {
                        b[i + j * ldb] = b1[i + j * ldb1];
                    }
                }
            }
        } else {
            SLC_DLACPY("F", &n1, &m1, b1, &ldb1, b, &ldb);
        }

        if (n2 > 0 && m1 > 0) {
            SLC_DGEMM("N", "N", &n2, &m1, &p1, &ONE, b2, &ldb2, d1, &ldd1, &ZERO, &b[i2], &ldb);
        }

        if (n1 > 0) {
            if (lover) {
                SLC_DLACPY("F", &p1, &n1, c1, &ldc1, dwork, &ldwp1);
                SLC_DGEMM("N", "N", &p2, &n1, &p1, &ONE, d2, &ldd2, dwork, &ldwp1, &ZERO, c, &ldc);
            } else {
                SLC_DGEMM("N", "N", &p2, &n1, &p1, &ONE, d2, &ldd2, c1, &ldc1, &ZERO, c, &ldc);
            }
        }

        if (p2 > 0 && n2 > 0) {
            SLC_DLACPY("F", &p2, &n2, c2, &ldc2, &c[i2 * ldc], &ldc);
        }

        if (lover) {
            SLC_DLACPY("F", &p1, &m1, d1, &ldd1, dwork, &ldwp1);
            SLC_DGEMM("N", "N", &p2, &m1, &p1, &ONE, d2, &ldd2, dwork, &ldwp1, &ZERO, d, &ldd);
        } else {
            SLC_DGEMM("N", "N", &p2, &m1, &p1, &ONE, d2, &ldd2, d1, &ldd1, &ZERO, d, &ldd);
        }
    } else {
        if (lover) {
            SLC_DLACPY("F", &n2, &p1, b2, &ldb2, dwork, &ldwn2);
            if (n2 > 0 && m1 > 0) {
                SLC_DGEMM("N", "N", &n2, &m1, &p1, &ONE, dwork, &ldwn2, d1, &ldd1, &ZERO, &b[i2], &ldb);
            }
        } else {
            SLC_DGEMM("N", "N", &n2, &m1, &p1, &ONE, b2, &ldb2, d1, &ldd1, &ZERO, b, &ldb);
        }

        if (n1 > 0 && m1 > 0) {
            SLC_DLACPY("F", &n1, &m1, b1, &ldb1, &b[i1], &ldb);
        }

        if (lover && ldc2 <= ldc) {
            if (ldc2 < ldc) {
                for (i32 j = n2 - 1; j >= 0; j--) {
                    for (i32 i = p2 - 1; i >= 0; i--) {
                        c[i + j * ldc] = c2[i + j * ldc2];
                    }
                }
            }
        } else {
            SLC_DLACPY("F", &p2, &n2, c2, &ldc2, c, &ldc);
        }

        if (p2 > 0 && n1 > 0) {
            SLC_DGEMM("N", "N", &p2, &n1, &p1, &ONE, d2, &ldd2, c1, &ldc1, &ZERO, &c[i1 * ldc], &ldc);
        }

        if (lover) {
            SLC_DLACPY("F", &p2, &p1, d2, &ldd2, dwork, &ldwp2);
            SLC_DGEMM("N", "N", &p2, &m1, &p1, &ONE, dwork, &ldwp2, d1, &ldd1, &ZERO, d, &ldd);
        } else {
            SLC_DGEMM("N", "N", &p2, &m1, &p1, &ONE, d2, &ldd2, d1, &ldd1, &ZERO, d, &ldd);
        }
    }

    return 0;
}
