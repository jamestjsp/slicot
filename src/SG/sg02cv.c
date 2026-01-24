/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <stdbool.h>

static inline i32 max_i32(i32 a, i32 b) { return a > b ? a : b; }

static inline bool lsame(char a, char b) {
    if (a >= 'a' && a <= 'z') a -= 32;
    if (b >= 'a' && b <= 'z') b -= 32;
    return a == b;
}

void sg02cv(
    const char* dico, const char* job, const char* jobe,
    const char* uplo, const char* trans,
    const i32 n,
    f64* a, const i32 lda,
    const f64* e, const i32 lde,
    f64* x, const i32 ldx,
    f64* r, const i32 ldr,
    f64* norms,
    f64* dwork, const i32 ldwork,
    i32* info
)
{
    const f64 zero = 0.0, one = 1.0;
    const char* ntrans;
    bool discr, ljobe, ljobn, ljobr, ltrans, luplo, unite;
    i32 j, minwrk, nn, optwrk;
    i32 int1 = 1;
    f64 dmone = -1.0;

    *info = 0;

    discr  = lsame(dico[0], 'D');
    ljobn  = lsame(job[0], 'N') || lsame(job[0], 'B');
    ljobr  = lsame(job[0], 'R');
    ljobe  = lsame(jobe[0], 'G');
    luplo  = lsame(uplo[0], 'U');
    ltrans = lsame(trans[0], 'T') || lsame(trans[0], 'C');
    unite  = !ljobe;

    if (!discr && !lsame(dico[0], 'C')) {
        *info = -1;
    } else if (!ljobn && !ljobr) {
        *info = -2;
    } else if (unite && !lsame(jobe[0], 'I')) {
        *info = -3;
    } else if (!luplo && !lsame(uplo[0], 'L')) {
        *info = -4;
    } else if (!ltrans && !lsame(trans[0], 'N')) {
        *info = -5;
    } else if (n < 0) {
        *info = -6;
    } else if (lda < max_i32(1, n)) {
        *info = -8;
    } else if (lde < 1 || (ljobe && lde < n)) {
        *info = -10;
    } else if (ldx < max_i32(1, n)) {
        *info = -12;
    } else if (ldr < max_i32(1, n)) {
        *info = -14;
    } else {
        nn = n * n;

        if (ljobn) {
            if (discr) {
                minwrk = 2 * nn;
            } else {
                minwrk = nn;
            }
        } else if (!discr && unite) {
            minwrk = 0;
        } else {
            minwrk = nn;
        }

        optwrk = minwrk;

        if (ldwork == -2) {
            dwork[0] = (f64)max_i32(1, minwrk);
            return;
        } else if (ldwork == -1) {
            dwork[0] = (f64)max_i32(1, optwrk);
            return;
        }

        if (ldwork < minwrk) {
            *info = -17;
            dwork[0] = (f64)max_i32(1, minwrk);
        }
    }

    if (*info != 0) {
        i32 neg_info = -(*info);
        SLC_XERBLA("SG02CV", &neg_info);
        return;
    }

    if (n == 0) {
        if (!ljobr) {
            norms[0] = zero;
            if (discr && ljobe) {
                norms[1] = zero;
            }
        }
        return;
    }

    if (ltrans) {
        ntrans = "N";
    } else {
        ntrans = "T";
    }

    if (ljobr) {
        if (discr) {
            mb01rh(uplo, ntrans, n, one, one, r, ldr, a, lda, x, ldx, dwork, ldwork, info);

            if (unite) {
                if (luplo) {
                    for (j = 0; j < n; j++) {
                        i32 jj = j + 1;
                        SLC_DAXPY(&jj, &dmone, &x[j * ldx], &int1, &r[j * ldr], &int1);
                    }
                } else {
                    for (j = 0; j < n; j++) {
                        i32 cnt = n - j;
                        SLC_DAXPY(&cnt, &dmone, &x[j + j * ldx], &int1, &r[j + j * ldr], &int1);
                    }
                }
            } else {
                mb01rt(uplo, ntrans, n, one, -one, r, ldr, e, lde, x, ldx, dwork, ldwork, info);
            }
        } else {
            if (ljobe) {
                mb01od(uplo, ntrans, n, one, one, r, ldr, a, lda, x, ldx, e, lde, dwork, nn, info);
            } else {
                mb01oc(uplo, ntrans, n, one, one, r, ldr, a, lda, x, ldx, info);
            }
        }
    } else {
        if (discr) {
            if (ljobe) {
                mb01rt(uplo, ntrans, n, zero, one, dwork, n, e, lde, x, ldx, &dwork[nn], ldwork, info);

                norms[1] = SLC_DLANSY("F", uplo, &n, dwork, &n, dwork);

                if (luplo) {
                    for (j = 0; j < n; j++) {
                        i32 jj = j + 1;
                        SLC_DAXPY(&jj, &dmone, &dwork[j * n], &int1, &r[j * ldr], &int1);
                    }
                } else {
                    for (j = 0; j < n; j++) {
                        i32 cnt = n - j;
                        SLC_DAXPY(&cnt, &dmone, &dwork[j + j * n], &int1, &r[j + j * ldr], &int1);
                    }
                }
            } else {
                if (luplo) {
                    for (j = 0; j < n; j++) {
                        i32 jj = j + 1;
                        SLC_DAXPY(&jj, &dmone, &x[j * ldx], &int1, &r[j * ldr], &int1);
                    }
                } else {
                    for (j = 0; j < n; j++) {
                        i32 cnt = n - j;
                        SLC_DAXPY(&cnt, &dmone, &x[j + j * ldx], &int1, &r[j + j * ldr], &int1);
                    }
                }
            }

            mb01rh(uplo, ntrans, n, zero, one, dwork, n, a, lda, x, ldx, &dwork[nn], nn, info);

            norms[0] = SLC_DLANSY("F", uplo, &n, dwork, &n, dwork);

            if (luplo) {
                f64 dp1 = 1.0;
                for (j = 0; j < n; j++) {
                    i32 jj = j + 1;
                    SLC_DAXPY(&jj, &dp1, &dwork[j * n], &int1, &r[j * ldr], &int1);
                }
            } else {
                f64 dp1 = 1.0;
                for (j = 0; j < n; j++) {
                    i32 cnt = n - j;
                    SLC_DAXPY(&cnt, &dp1, &dwork[j + j * n], &int1, &r[j + j * ldr], &int1);
                }
            }
        } else {
            if (ljobe) {
                mb01oo(uplo, ntrans, n, a, lda, x, ldx, e, lde, dwork, n, info);
            } else {
                mb01os(uplo, ntrans, n, a, lda, x, ldx, dwork, n, info);
            }

            norms[0] = SLC_DLANGE("F", &n, &n, dwork, &n, dwork);

            if (luplo) {
                f64 dp1 = 1.0;
                for (j = 0; j < n; j++) {
                    i32 jj = j + 1;
                    SLC_DAXPY(&jj, &dp1, &dwork[j * n], &int1, &r[j * ldr], &int1);
                    SLC_DAXPY(&jj, &dp1, &dwork[j], &n, &r[j * ldr], &int1);
                }
            } else {
                f64 dp1 = 1.0;
                for (j = 0; j < n; j++) {
                    i32 cnt = n - j;
                    SLC_DAXPY(&cnt, &dp1, &dwork[j + j * n], &int1, &r[j + j * ldr], &int1);
                    SLC_DAXPY(&cnt, &dp1, &dwork[j + j * n], &n, &r[j + j * ldr], &int1);
                }
            }
        }
    }

    dwork[0] = (f64)max_i32(1, optwrk);
}
