/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <ctype.h>
#include <math.h>

void mb01rd(const char* uplo, const char* trans, i32 m, i32 n,
            f64 alpha, f64 beta, f64* r, i32 ldr,
            const f64* a, i32 lda, f64* x, i32 ldx,
            f64* dwork, i32 ldwork, i32* info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const f64 HALF = 0.5;

    char uplo_c = (char)toupper((unsigned char)uplo[0]);
    char trans_c = (char)toupper((unsigned char)trans[0]);

    bool luplo = (uplo_c == 'U');
    bool ltrans = (trans_c == 'T' || trans_c == 'C');

    i32 nrowa = ltrans ? n : m;
    i32 ldw = (m > 1) ? m : 1;

    *info = 0;
    if (!luplo && uplo_c != 'L') {
        *info = -1;
    } else if (!ltrans && trans_c != 'N') {
        *info = -2;
    } else if (m < 0) {
        *info = -3;
    } else if (n < 0) {
        *info = -4;
    } else if (ldr < ldw) {
        *info = -8;
    } else if (lda < (nrowa > 1 ? nrowa : 1)) {
        *info = -10;
    } else if (ldx < (n > 1 ? n : 1)) {
        *info = -12;
    } else if ((beta != ZERO && ldwork < (m * n > 1 ? m * n : 1)) ||
               (beta == ZERO && ldwork < 1)) {
        *info = -14;
    }

    if (*info != 0) {
        return;
    }

    for (i32 j = 0; j < n; j++) {
        x[j + j * ldx] *= HALF;
    }

    if (m == 0) {
        return;
    }

    if (beta == ZERO || n == 0) {
        if (alpha == ZERO) {
            if (luplo) {
                for (i32 j = 0; j < m; j++) {
                    for (i32 i = 0; i <= j; i++) {
                        r[i + j * ldr] = ZERO;
                    }
                }
            } else {
                for (i32 j = 0; j < m; j++) {
                    for (i32 i = j; i < m; i++) {
                        r[i + j * ldr] = ZERO;
                    }
                }
            }
        } else if (alpha != ONE) {
            if (luplo) {
                for (i32 j = 0; j < m; j++) {
                    for (i32 i = 0; i <= j; i++) {
                        r[i + j * ldr] *= alpha;
                    }
                }
            } else {
                for (i32 j = 0; j < m; j++) {
                    for (i32 i = j; i < m; i++) {
                        r[i + j * ldr] *= alpha;
                    }
                }
            }
        }
        return;
    }

    if (ltrans) {
        i32 jwork = 0;
        for (i32 j = 0; j < n; j++) {
            for (i32 i = 0; i < m; i++) {
                dwork[jwork + i] = a[j + i * lda];
            }
            jwork += ldw;
        }
    } else {
        for (i32 j = 0; j < n; j++) {
            for (i32 i = 0; i < m; i++) {
                dwork[i + j * ldw] = a[i + j * lda];
            }
        }
    }

    {
        char side = 'R';
        char ul = uplo_c;
        char tr = 'N';
        char diag = 'N';
        i32 mm = m;
        i32 nn = n;
        f64 al = beta;
        SLC_DTRMM(&side, &ul, &tr, &diag, &mm, &nn, &al, x, &ldx, dwork, &ldw);
    }

    if (alpha != ZERO) {
        if (m > 1) {
            if (luplo) {
                for (i32 j = 0; j < m - 1; j++) {
                    for (i32 i = j + 1; i < m; i++) {
                        r[i + j * ldr] = ZERO;
                    }
                }
            } else {
                for (i32 j = 1; j < m; j++) {
                    for (i32 i = 0; i < j; i++) {
                        r[i + j * ldr] = ZERO;
                    }
                }
            }
        }
        for (i32 j = 0; j < m; j++) {
            r[j + j * ldr] *= HALF;
        }
    }

    {
        char tra = 'N';
        char trb = ltrans ? 'N' : 'T';
        i32 mm = m;
        i32 nn = m;
        i32 kk = n;
        f64 al = ONE;
        f64 be = alpha;
        SLC_DGEMM(&tra, &trb, &mm, &nn, &kk, &al, dwork, &ldw, a, &lda, &be, r, &ldr);
    }

    if (luplo) {
        for (i32 j = 0; j < m; j++) {
            i32 incx = ldr;
            i32 incy = 1;
            i32 len = j + 1;
            SLC_DAXPY(&len, &ONE, &r[j + 0 * ldr], &incx, &r[0 + j * ldr], &incy);
        }
    } else {
        for (i32 j = 0; j < m; j++) {
            i32 incx = 1;
            i32 incy = ldr;
            i32 len = j + 1;
            SLC_DAXPY(&len, &ONE, &r[0 + j * ldr], &incx, &r[j + 0 * ldr], &incy);
        }
    }
}
