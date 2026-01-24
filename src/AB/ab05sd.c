/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 2025, slicot.c contributors
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <ctype.h>

void ab05sd(const char* fbtype, const char* jobd, i32 n, i32 m, i32 p,
            f64 alpha, f64* a, i32 lda, f64* b, i32 ldb,
            f64* c, i32 ldc, f64* d, i32 ldd,
            const f64* f, i32 ldf, f64* rcond,
            i32* iwork, f64* dwork, i32 ldwork, i32* info) {

    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    char fbtype_upper = (char)toupper((unsigned char)fbtype[0]);
    char jobd_upper = (char)toupper((unsigned char)jobd[0]);

    bool unitf = (fbtype_upper == 'I');
    bool outpf = (fbtype_upper == 'O');
    bool ljobd = (jobd_upper == 'D');

    i32 ldwn = (1 > n) ? 1 : n;
    i32 ldwp = (1 > p) ? 1 : p;

    *info = 0;

    if (!unitf && !outpf) {
        *info = -1;
    } else if (!ljobd && jobd_upper != 'Z') {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    } else if (m < 0) {
        *info = -4;
    } else if (p < 0 || (unitf && p != m)) {
        *info = -5;
    } else if (lda < ldwn) {
        *info = -8;
    } else if (ldb < ldwn) {
        *info = -10;
    } else if ((n > 0 && ldc < ldwp) || (n == 0 && ldc < 1)) {
        *info = -12;
    } else if ((ljobd && ldd < ldwp) || (!ljobd && ldd < 1)) {
        *info = -14;
    } else if ((outpf && alpha != ZERO && ldf < ((1 > m) ? 1 : m)) ||
               ((unitf || alpha == ZERO) && ldf < 1)) {
        *info = -16;
    } else {
        i32 pp4p = p * p + 4 * p;
        i32 max1m = (1 > m) ? 1 : m;
        i32 wspace;
        if (ljobd) {
            wspace = (max1m > pp4p) ? max1m : pp4p;
            wspace = (1 > wspace) ? 1 : wspace;
        } else {
            wspace = max1m;
            wspace = (1 > wspace) ? 1 : wspace;
        }
        if (ldwork < wspace) {
            *info = -20;
        }
    }

    if (*info != 0) {
        return;
    }

    *rcond = ONE;
    i32 minmp = (m < p) ? m : p;
    i32 maxnminmp = (n > minmp) ? n : minmp;
    if (maxnminmp == 0 || alpha == ZERO) {
        return;
    }

    i32 int0 = 0;
    i32 int1 = 1;
    f64 neg_alpha = -alpha;
    f64 neg_one = -ONE;

    if (ljobd) {
        i32 iw = p * p;

        if (unitf) {
            SLC_DLACPY("F", &p, &p, d, &ldd, dwork, &ldwp);
            if (alpha != neg_one) {
                i32 scale_info = 0;
                SLC_DLASCL("G", &int0, &int0, &ONE, &neg_alpha, &p, &p, dwork, &ldwp, &scale_info);
            }
        } else {
            SLC_DGEMM("N", "N", &p, &p, &m, &neg_alpha, d, &ldd, f, &ldf, &ZERO, dwork, &ldwp);
        }

        f64 dummy = ONE;
        i32 pp1 = p + 1;
        SLC_DAXPY(&p, &ONE, &dummy, &int0, dwork, &pp1);

        f64 enorm = SLC_DLANGE("1", &p, &p, dwork, &ldwp, &dwork[iw]);
        SLC_DGETRF(&p, &p, dwork, &ldwp, iwork, info);
        if (*info > 0) {
            *rcond = ZERO;
            *info = 1;
            return;
        }

        SLC_DGECON("1", &p, dwork, &ldwp, &enorm, rcond, &dwork[iw], &iwork[p], info);
        f64 eps = SLC_DLAMCH("E");
        if (*rcond <= eps) {
            *info = 1;
            return;
        }

        if (n > 0) {
            SLC_DGETRS("N", &p, &n, dwork, &ldwp, iwork, c, &ldc, info);
        }
        SLC_DGETRS("N", &p, &m, dwork, &ldwp, iwork, d, &ldd, info);
    }

    if (n == 0) {
        return;
    }

    if (unitf) {
        SLC_DGEMM("N", "N", &n, &n, &m, &alpha, b, &ldb, c, &ldc, &ONE, a, &lda);

        if (ljobd) {
            i32 nm = n * m;
            if (ldwork < nm) {
                for (i32 i = 0; i < n; i++) {
                    SLC_DCOPY(&p, &b[i], &ldb, dwork, &int1);
                    SLC_DGEMV("T", &p, &p, &alpha, d, &ldd, dwork, &int1, &ONE, &b[i], &ldb);
                }
            } else {
                SLC_DLACPY("F", &n, &m, b, &ldb, dwork, &ldwn);
                SLC_DGEMM("N", "N", &n, &p, &m, &alpha, dwork, &ldwn, d, &ldd, &ONE, b, &ldb);
            }
        }
    } else {
        i32 np = n * p;
        if (ldwork < np) {
            for (i32 i = 0; i < n; i++) {
                SLC_DGEMV("N", &m, &p, &alpha, f, &ldf, &c[i * ldc], &int1, &ZERO, dwork, &int1);
                SLC_DGEMV("N", &n, &m, &ONE, b, &ldb, dwork, &int1, &ONE, &a[i * lda], &int1);
            }

            if (ljobd) {
                for (i32 i = 0; i < n; i++) {
                    SLC_DGEMV("T", &m, &p, &alpha, f, &ldf, &b[i], &ldb, &ZERO, dwork, &int1);
                    SLC_DGEMV("T", &p, &m, &ONE, d, &ldd, dwork, &int1, &ONE, &b[i], &ldb);
                }
            }
        } else {
            SLC_DGEMM("N", "N", &n, &p, &m, &alpha, b, &ldb, f, &ldf, &ZERO, dwork, &ldwn);
            SLC_DGEMM("N", "N", &n, &n, &p, &ONE, dwork, &ldwn, c, &ldc, &ONE, a, &lda);
            if (ljobd) {
                SLC_DGEMM("N", "N", &n, &m, &p, &ONE, dwork, &ldwn, d, &ldd, &ONE, b, &ldb);
            }
        }
    }
}
