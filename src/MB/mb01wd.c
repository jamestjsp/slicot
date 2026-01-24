/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"

void mb01wd(
    const char* dico,
    const char* uplo,
    const char* trans,
    const char* hess,
    const i32 n,
    const f64 alpha,
    const f64 beta,
    f64* r,
    const i32 ldr,
    f64* a,
    const i32 lda,
    const f64* t,
    const i32 ldt,
    i32* info
)
{
    const f64 zero = 0.0;
    const f64 one = 1.0;

    *info = 0;

    bool discr = (*dico == 'D' || *dico == 'd');
    bool upper = (*uplo == 'U' || *uplo == 'u');
    bool transp = (*trans == 'T' || *trans == 't' || *trans == 'C' || *trans == 'c');
    bool reduc = (*hess == 'H' || *hess == 'h');

    if (!discr && *dico != 'C' && *dico != 'c') {
        *info = -1;
    } else if (!upper && *uplo != 'L' && *uplo != 'l') {
        *info = -2;
    } else if (!transp && *trans != 'N' && *trans != 'n') {
        *info = -3;
    } else if (!reduc && *hess != 'F' && *hess != 'f') {
        *info = -4;
    } else if (n < 0) {
        *info = -5;
    } else if (ldr < (n > 1 ? n : 1)) {
        *info = -9;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -11;
    } else if (ldt < (n > 1 ? n : 1)) {
        *info = -13;
    }

    if (*info != 0) {
        return;
    }

    if (n == 0) {
        return;
    }

    if (alpha == zero) {
        if (beta == zero) {
            SLC_DLASET(uplo, &n, &n, &zero, &zero, r, &ldr);
        } else {
            if (beta != one) {
                i32 zero_int = 0;
                i32 info2;
                SLC_DLASCL(uplo, &zero_int, &zero_int, &one, &beta, &n, &n, r, &ldr, &info2);
            }
        }
        return;
    }

    const char* side;
    const char* negtra;
    if (transp) {
        side = "R";
        negtra = "N";
    } else {
        side = "L";
        negtra = "T";
    }

    i32 info2;
    i32 l_val = 1;

    if (reduc && n > 2) {
        mb01zd(side, uplo, "N", "N", n, n, l_val, one, t, ldt, a, lda, &info2);
    } else {
        SLC_DTRMM(side, uplo, "N", "N", &n, &n, &one, t, &ldt, a, &lda);
    }

    if (!discr) {
        if (reduc && n > 2) {
            mb01zd(side, uplo, "T", "N", n, n, l_val, alpha, t, ldt, a, lda, &info2);
        } else {
            SLC_DTRMM(side, uplo, "T", "N", &n, &n, &alpha, t, &ldt, a, &lda);
        }

        if (upper) {
            if (beta == zero) {
                for (i32 j = 0; j < n; j++) {
                    for (i32 i = 0; i <= j; i++) {
                        r[i + j * ldr] = a[i + j * lda] + a[j + i * lda];
                    }
                }
            } else {
                for (i32 j = 0; j < n; j++) {
                    for (i32 i = 0; i <= j; i++) {
                        r[i + j * ldr] = a[i + j * lda] + a[j + i * lda] + beta * r[i + j * ldr];
                    }
                }
            }
        } else {
            if (beta == zero) {
                for (i32 j = 0; j < n; j++) {
                    for (i32 i = j; i < n; i++) {
                        r[i + j * ldr] = a[i + j * lda] + a[j + i * lda];
                    }
                }
            } else {
                for (i32 j = 0; j < n; j++) {
                    for (i32 i = j; i < n; i++) {
                        r[i + j * ldr] = a[i + j * lda] + a[j + i * lda] + beta * r[i + j * ldr];
                    }
                }
            }
        }
    } else {
        i32 l_zero = 0;
        if (reduc && n > 2) {
            mb01yd(uplo, negtra, n, n, l_val, alpha, beta, a, lda, r, ldr, &info2);
        } else {
            SLC_DSYRK(uplo, negtra, &n, &n, &alpha, a, &lda, &beta, r, &ldr);
        }

        f64 neg_alpha = -alpha;
        mb01yd(uplo, negtra, n, n, l_zero, neg_alpha, one, t, ldt, r, ldr, &info2);
    }
}
