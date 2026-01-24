/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 2025, slicot.c contributors
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <ctype.h>

i32 ab05od(char over, i32 n1, i32 m1, i32 p1, i32 n2, i32 m2, f64 alpha,
           const f64* a1, i32 lda1, const f64* b1, i32 ldb1,
           const f64* c1, i32 ldc1, const f64* d1, i32 ldd1,
           const f64* a2, i32 lda2, const f64* b2, i32 ldb2,
           const f64* c2, i32 ldc2, const f64* d2, i32 ldd2,
           i32* n, i32* m, f64* a, i32 lda, f64* b, i32 ldb,
           f64* c, i32 ldc, f64* d, i32 ldd) {

    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    char over_upper = (char)toupper((unsigned char)over);
    bool lover = (over_upper == 'O');

    *n = n1 + n2;
    *m = m1 + m2;

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
    if (m2 < 0) {
        return -6;
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

    if ((n2 > 0 && ldc2 < max1p1) || (n2 == 0 && ldc2 < 1)) {
        return -21;
    }
    if (ldd2 < max1p1) {
        return -23;
    }

    i32 max1n = (1 > *n) ? 1 : *n;
    if (lda < max1n) {
        return -27;
    }
    if (ldb < max1n) {
        return -29;
    }

    if ((*n > 0 && ldc < max1p1) || (*n == 0 && ldc < 1)) {
        return -31;
    }
    if (ldd < max1p1) {
        return -33;
    }

    i32 min_m_p1 = (*m < p1) ? *m : p1;
    if (*n == 0 || min_m_p1 == 0) {
        if (p1 > 0 && *m > 0) {
            SLC_DLACPY("F", &p1, &m1, d1, &ldd1, d, &ldd);
            if (m2 > 0) {
                SLC_DLACPY("F", &p1, &m2, d2, &ldd2, &d[m1 * ldd], &ldd);
                if (alpha != ONE) {
                    SLC_DLASCL("G", &(i32){0}, &(i32){0}, &ONE, &alpha, &p1, &m2,
                               &d[m1 * ldd], &ldd, &(i32){0});
                }
            }
        }
        return 0;
    }

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
        SLC_DLACPY("F", &n2, &n2, a2, &lda2, &a[(n1) + (n1) * lda], &lda);
        SLC_DLASET("F", &n1, &n2, &ZERO, &ZERO, &a[n1 * lda], &lda);
        SLC_DLASET("F", &n2, &n1, &ZERO, &ZERO, &a[n1], &lda);
    }

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

    if (m2 > 0) {
        if (n2 > 0) {
            SLC_DLACPY("F", &n2, &m2, b2, &ldb2, &b[n1 + m1 * ldb], &ldb);
        }
        SLC_DLASET("F", &n1, &m2, &ZERO, &ZERO, &b[m1 * ldb], &ldb);
    }
    if (n2 > 0) {
        SLC_DLASET("F", &n2, &m1, &ZERO, &ZERO, &b[n1], &ldb);
    }

    if (lover && ldc1 <= ldc) {
        if (ldc1 < ldc) {
            for (i32 j = n1 - 1; j >= 0; j--) {
                for (i32 i = p1 - 1; i >= 0; i--) {
                    c[i + j * ldc] = c1[i + j * ldc1];
                }
            }
        }
    } else {
        SLC_DLACPY("F", &p1, &n1, c1, &ldc1, c, &ldc);
    }

    if (n2 > 0) {
        SLC_DLACPY("F", &p1, &n2, c2, &ldc2, &c[n1 * ldc], &ldc);
        if (alpha != ONE) {
            SLC_DLASCL("G", &(i32){0}, &(i32){0}, &ONE, &alpha, &p1, &n2,
                       &c[n1 * ldc], &ldc, &(i32){0});
        }
    }

    if (lover && ldd1 <= ldd) {
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

    if (m2 > 0) {
        SLC_DLACPY("F", &p1, &m2, d2, &ldd2, &d[m1 * ldd], &ldd);
        if (alpha != ONE) {
            SLC_DLASCL("G", &(i32){0}, &(i32){0}, &ONE, &alpha, &p1, &m2,
                       &d[m1 * ldd], &ldd, &(i32){0});
        }
    }

    return 0;
}
