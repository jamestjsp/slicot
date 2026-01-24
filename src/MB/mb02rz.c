/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <ctype.h>

i32 slicot_mb02rz(char trans, i32 n, i32 nrhs, const c128* h, i32 ldh,
                  const i32* ipiv, c128* b, i32 ldb) {
    i32 info = 0;
    char trans_up = (char)toupper((unsigned char)trans);
    bool notran = (trans_up == 'N');

    if (!notran && trans_up != 'T' && trans_up != 'C') {
        info = -1;
    } else if (n < 0) {
        info = -2;
    } else if (nrhs < 0) {
        info = -3;
    } else if (ldh < (n > 1 ? n : 1)) {
        info = -5;
    } else if (ldb < (n > 1 ? n : 1)) {
        info = -8;
    }

    if (info != 0) {
        i32 xinfo = -info;
        SLC_XERBLA("MB02RZ", &xinfo);
        return info;
    }

    if (n == 0 || nrhs == 0) {
        return 0;
    }

    c128 one = 1.0;

    if (notran) {
        for (i32 j = 0; j < n - 1; j++) {
            i32 jp = ipiv[j] - 1;  // Convert to 0-based
            if (jp != j) {
                SLC_ZSWAP(&nrhs, &b[jp], &ldb, &b[j], &ldb);
            }
            c128 neg_mult = -h[j + 1 + j * ldh];
            SLC_ZAXPY(&nrhs, &neg_mult, &b[j], &ldb, &b[j + 1], &ldb);
        }

        SLC_ZTRSM("Left", "Upper", "No transpose", "Non-unit", &n, &nrhs,
                  &one, h, &ldh, b, &ldb);
    } else if (trans_up == 'T') {
        SLC_ZTRSM("Left", "Upper", "Transpose", "Non-unit", &n, &nrhs,
                  &one, h, &ldh, b, &ldb);

        for (i32 j = n - 2; j >= 0; j--) {
            c128 neg_mult = -h[j + 1 + j * ldh];
            SLC_ZAXPY(&nrhs, &neg_mult, &b[j + 1], &ldb, &b[j], &ldb);
            i32 jp = ipiv[j] - 1;  // Convert to 0-based
            if (jp != j) {
                SLC_ZSWAP(&nrhs, &b[jp], &ldb, &b[j], &ldb);
            }
        }
    } else {
        SLC_ZTRSM("Left", "Upper", "Conjugate transpose", "Non-unit", &n, &nrhs,
                  &one, h, &ldh, b, &ldb);

        for (i32 j = n - 2; j >= 0; j--) {
            c128 neg_mult = -conj(h[j + 1 + j * ldh]);
            SLC_ZAXPY(&nrhs, &neg_mult, &b[j + 1], &ldb, &b[j], &ldb);
            i32 jp = ipiv[j] - 1;  // Convert to 0-based
            if (jp != j) {
                SLC_ZSWAP(&nrhs, &b[jp], &ldb, &b[j], &ldb);
            }
        }
    }

    return info;
}
