/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <stdbool.h>

static inline i32 max_i32(i32 a, i32 b) { return a > b ? a : b; }

void mb01oo(const char* uplo_str, const char* trans_str, i32 n,
            const f64* h, i32 ldh, const f64* x, i32 ldx,
            const f64* e, i32 lde, f64* p, i32 ldp, i32* info)
{
    const f64 one = 1.0;

    char uplo = uplo_str[0];
    char trans = trans_str[0];
    bool luplo = (uplo == 'U' || uplo == 'u');
    bool ltrans = (trans == 'T' || trans == 't' || trans == 'C' || trans == 'c');

    *info = 0;

    if (!luplo && uplo != 'L' && uplo != 'l') {
        *info = -1;
    } else if (!ltrans && trans != 'N' && trans != 'n') {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    } else if (ldh < max_i32(1, n)) {
        *info = -5;
    } else if (ldx < max_i32(1, n)) {
        *info = -7;
    } else if (lde < max_i32(1, n)) {
        *info = -9;
    } else if (ldp < max_i32(1, n)) {
        *info = -11;
    }

    if (*info != 0) {
        return;
    }

    if (n == 0) {
        return;
    }

    char side;
    if (ltrans) {
        side = 'L';
    } else {
        side = 'R';
    }

    mb01os(uplo_str, trans_str, n, h, ldh, x, ldx, p, ldp, info);

    SLC_DTRMM(&side, "U", "T", "N", &n, &n, &one, e, &lde, p, &ldp);
}
