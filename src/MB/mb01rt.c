/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <stdbool.h>

static inline i32 max_i32(i32 a, i32 b) { return a > b ? a : b; }

void mb01rt(const char* uplo_str, const char* trans_str, i32 n,
            f64 alpha, f64 beta, f64* r, i32 ldr,
            const f64* e, i32 lde, f64* x, i32 ldx,
            f64* dwork, i32 ldwork, i32* info)
{
    const f64 zero = 0.0;
    const f64 one = 1.0;
    const f64 two = 2.0;
    const f64 half = 0.5;
    i32 inc1 = 1;

    char uplo = uplo_str[0];
    char trans = trans_str[0];
    bool upper = (uplo == 'U' || uplo == 'u');
    bool ltrans = (trans == 'T' || trans == 't' || trans == 'C' || trans == 'c');

    *info = 0;

    if (!upper && uplo != 'L' && uplo != 'l') {
        *info = -1;
    } else if (!ltrans && trans != 'N' && trans != 'n') {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    } else if (ldr < max_i32(1, n)) {
        *info = -7;
    } else if (lde < max_i32(1, n)) {
        *info = -9;
    } else if (ldx < max_i32(1, n)) {
        *info = -11;
    } else if ((beta != zero && ldwork < n * n) ||
               (beta == zero && ldwork < 0)) {
        *info = -13;
    }

    if (*info != 0) {
        return;
    }

    if (n == 0) {
        return;
    }

    if (beta == zero) {
        if (alpha == zero) {
            SLC_DLASET(uplo_str, &n, &n, &zero, &zero, r, &ldr);
        } else {
            if (alpha != one) {
                i32 kl = 0, ku = 0;
                SLC_DLASCL(uplo_str, &kl, &ku, &one, &alpha, &n, &n, r, &ldr, info);
            }
        }
        return;
    }

    SLC_DSCAL(&n, &half, x, &(i32){ldx + 1});

    if (!ltrans) {
        if (upper) {
            for (i32 jf = 1; jf <= n; jf++) {
                i32 j = jf - 1;
                SLC_DCOPY(&jf, &x[j * ldx], &inc1, &dwork[j * n], &inc1);
                SLC_DTRMV("Upper", "NoTran", "NoDiag", &jf, e, &lde,
                          &dwork[j * n], &inc1);
            }
        } else {
            for (i32 jf = 1; jf <= n; jf++) {
                i32 j = jf - 1;
                SLC_DCOPY(&jf, &x[j], &ldx, &dwork[j * n], &inc1);
                SLC_DTRMV("Upper", "NoTran", "NoDiag", &jf, e, &lde,
                          &dwork[j * n], &inc1);
            }
        }
    } else {
        if (upper) {
            for (i32 jf = 1; jf <= n; jf++) {
                i32 j = jf - 1;
                SLC_DCOPY(&jf, &e[j * lde], &inc1, &dwork[j * n], &inc1);
                SLC_DTRMV("Upper", "NoTran", "NoDiag", &jf, x, &ldx,
                          &dwork[j * n], &inc1);
            }
        } else {
            for (i32 jf = 1; jf <= n; jf++) {
                i32 j = jf - 1;
                SLC_DCOPY(&jf, &e[j * lde], &inc1, &dwork[j * n], &inc1);
                SLC_DTRMV("Upper", "Tran", "NoDiag", &jf, x, &ldx,
                          &dwork[j * n], &inc1);
            }
        }
    }

    SLC_DSCAL(&n, &two, x, &(i32){ldx + 1});

    mb01ot(uplo_str, trans_str, n, alpha, beta, r, ldr, e, lde, dwork, n, info);
}
