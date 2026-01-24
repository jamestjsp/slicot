/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <stdbool.h>

static inline i32 max_i32(i32 a, i32 b) { return a > b ? a : b; }

void mb01rh(const char* uplo_str, const char* trans_str, i32 n,
            f64 alpha, f64 beta, f64* r, i32 ldr,
            f64* h, i32 ldh, f64* x, i32 ldx,
            f64* dwork, i32 ldwork, i32* info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const f64 TWO = 2.0;
    const f64 HALF = 0.5;

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
    } else if (ldr < max_i32(1, n)) {
        *info = -7;
    } else if (ldh < max_i32(1, n)) {
        *info = -9;
    } else if (ldx < max_i32(1, n)) {
        *info = -11;
    } else if ((beta != ZERO && ldwork < n * n) || (beta == ZERO && ldwork < 0)) {
        *info = -13;
    }

    if (*info != 0) {
        return;
    }

    if (n == 0) {
        return;
    }

    if (beta == ZERO) {
        if (alpha == ZERO) {
            if (luplo) {
                for (i32 j = 0; j < n; j++) {
                    for (i32 i = 0; i <= j; i++) {
                        r[i + j * ldr] = ZERO;
                    }
                }
            } else {
                for (i32 j = 0; j < n; j++) {
                    for (i32 i = j; i < n; i++) {
                        r[i + j * ldr] = ZERO;
                    }
                }
            }
        } else if (alpha != ONE) {
            if (luplo) {
                for (i32 j = 0; j < n; j++) {
                    for (i32 i = 0; i <= j; i++) {
                        r[i + j * ldr] *= alpha;
                    }
                }
            } else {
                for (i32 j = 0; j < n; j++) {
                    for (i32 i = j; i < n; i++) {
                        r[i + j * ldr] *= alpha;
                    }
                }
            }
        }
        return;
    }

    for (i32 j = 0; j < n; j++) {
        x[j + j * ldx] *= HALF;
    }

    if (!ltrans) {
        if (n > 2) {
            for (i32 i = 0; i < n - 2; i++) {
                f64 temp = h[(i + 2) + 0 * ldh];
                h[(i + 2) + 0 * ldh] = h[(i + 2) + (i + 1) * ldh];
                h[(i + 2) + (i + 1) * ldh] = temp;
            }
        }

        if (luplo) {
            for (i32 j = 0; j < n - 1; j++) {
                i32 j1 = j + 1;
                for (i32 i = 0; i <= j; i++) {
                    dwork[i + j * n] = x[i + j * ldx];
                }
                {
                    char ul = 'U';
                    char tr = 'N';
                    char dg = 'N';
                    i32 nn = j + 1;
                    SLC_DTRMV(&ul, &tr, &dg, &nn, h, &ldh, &dwork[j * n], &(i32){1});
                }
                for (i32 i = 1; i <= j; i++) {
                    dwork[i + j * n] += h[i + 0 * ldh] * x[(i - 1) + j * ldx];
                }
                dwork[j1 + j * n] = h[j1 + 0 * ldh] * x[j + j * ldx];
            }

            for (i32 i = 0; i < n; i++) {
                dwork[i + (n - 1) * n] = x[i + (n - 1) * ldx];
            }
            {
                char ul = 'U';
                char tr = 'N';
                char dg = 'N';
                SLC_DTRMV(&ul, &tr, &dg, &n, h, &ldh, &dwork[(n - 1) * n], &(i32){1});
            }
            for (i32 i = 1; i < n; i++) {
                dwork[i + (n - 1) * n] += h[i + 0 * ldh] * x[(i - 1) + (n - 1) * ldx];
            }

        } else {
            for (i32 j = 0; j < n - 1; j++) {
                i32 j1 = j + 1;
                for (i32 i = 0; i <= j; i++) {
                    dwork[i + j * n] = x[j + i * ldx];
                }
                {
                    char ul = 'U';
                    char tr = 'N';
                    char dg = 'N';
                    i32 nn = j + 1;
                    SLC_DTRMV(&ul, &tr, &dg, &nn, h, &ldh, &dwork[j * n], &(i32){1});
                }
                for (i32 i = 1; i <= j; i++) {
                    dwork[i + j * n] += h[i + 0 * ldh] * x[j + (i - 1) * ldx];
                }
                dwork[j1 + j * n] = h[j1 + 0 * ldh] * x[j + j * ldx];
            }

            for (i32 i = 0; i < n; i++) {
                dwork[i + (n - 1) * n] = x[(n - 1) + i * ldx];
            }
            {
                char ul = 'U';
                char tr = 'N';
                char dg = 'N';
                SLC_DTRMV(&ul, &tr, &dg, &n, h, &ldh, &dwork[(n - 1) * n], &(i32){1});
            }
            for (i32 i = 1; i < n; i++) {
                dwork[i + (n - 1) * n] += h[i + 0 * ldh] * x[(n - 1) + (i - 1) * ldx];
            }
        }

        if (n > 2) {
            for (i32 i = 0; i < n - 2; i++) {
                f64 temp = h[(i + 2) + 0 * ldh];
                h[(i + 2) + 0 * ldh] = h[(i + 2) + (i + 1) * ldh];
                h[(i + 2) + (i + 1) * ldh] = temp;
            }
        }

    } else {
        if (luplo) {
            for (i32 j = 0; j < n - 1; j++) {
                i32 j1 = j + 1;
                for (i32 i = 0; i <= j; i++) {
                    dwork[i + j * n] = h[i + j * ldh];
                }
                {
                    char ul = 'U';
                    char tr = 'N';
                    char dg = 'N';
                    i32 nn = j + 1;
                    SLC_DTRMV(&ul, &tr, &dg, &nn, x, &ldx, &dwork[j * n], &(i32){1});
                }
                {
                    i32 nn = j + 1;
                    f64 scale = h[j1 + j * ldh];
                    i32 inc = 1;
                    SLC_DAXPY(&nn, &scale, &x[0 + j1 * ldx], &inc, &dwork[j * n], &inc);
                }
                dwork[j1 + j * n] = h[j1 + j * ldh] * x[j1 + j1 * ldx];
            }

            for (i32 i = 0; i < n; i++) {
                dwork[i + (n - 1) * n] = h[i + (n - 1) * ldh];
            }
            {
                char ul = 'U';
                char tr = 'N';
                char dg = 'N';
                SLC_DTRMV(&ul, &tr, &dg, &n, x, &ldx, &dwork[(n - 1) * n], &(i32){1});
            }

        } else {
            for (i32 j = 0; j < n - 1; j++) {
                i32 j1 = j + 1;
                for (i32 i = 0; i <= j; i++) {
                    dwork[i + j * n] = h[i + j * ldh];
                }
                {
                    char ul = 'L';
                    char tr = 'T';
                    char dg = 'N';
                    i32 nn = j + 1;
                    SLC_DTRMV(&ul, &tr, &dg, &nn, x, &ldx, &dwork[j * n], &(i32){1});
                }
                {
                    i32 nn = j + 1;
                    f64 scale = h[j1 + j * ldh];
                    i32 inc1 = ldx;
                    i32 inc2 = 1;
                    SLC_DAXPY(&nn, &scale, &x[j1 + 0 * ldx], &inc1, &dwork[j * n], &inc2);
                }
                dwork[j1 + j * n] = h[j1 + j * ldh] * x[j1 + j1 * ldx];
            }

            for (i32 i = 0; i < n; i++) {
                dwork[i + (n - 1) * n] = h[i + (n - 1) * ldh];
            }
            {
                char ul = 'L';
                char tr = 'T';
                char dg = 'N';
                SLC_DTRMV(&ul, &tr, &dg, &n, x, &ldx, &dwork[(n - 1) * n], &(i32){1});
            }
        }
    }

    for (i32 j = 0; j < n; j++) {
        x[j + j * ldx] *= TWO;
    }

    mb01oh(uplo_str, trans_str, n, alpha, beta, r, ldr, h, ldh, dwork, n, info);
}
