/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <ctype.h>

i32 slicot_mb02tz(char norm, i32 n, f64 hnorm, const c128* h, i32 ldh,
                  const i32* ipiv, f64* rcond, f64* dwork, c128* zwork) {
    i32 info = 0;
    char norm_up = (char)toupper((unsigned char)norm);
    bool onenrm = (norm_up == '1' || norm_up == 'O');

    if (!onenrm && norm_up != 'I') {
        info = -1;
    } else if (n < 0) {
        info = -2;
    } else if (hnorm < 0.0) {
        info = -3;
    } else if (ldh < (n > 1 ? n : 1)) {
        info = -5;
    }

    if (info != 0) {
        i32 xinfo = -info;
        SLC_XERBLA("MB02TZ", &xinfo);
        return info;
    }

    *rcond = 0.0;
    if (n == 0) {
        *rcond = 1.0;
        return 0;
    } else if (hnorm == 0.0) {
        return 0;
    }

    f64 smlnum = SLC_DLAMCH("Safe minimum");
    f64 hinvnm = 0.0;
    char normin = 'N';
    i32 kase1 = onenrm ? 1 : 2;
    i32 kase = 0;
    i32 inc1 = 1;

    c128* v = &zwork[n];
    c128* x = &zwork[0];

    while (1) {
        SLC_ZLACON(&n, v, x, &hinvnm, &kase);
        if (kase == 0) {
            break;
        }

        f64 scale;
        if (kase == kase1) {
            for (i32 j = 0; j < n - 1; j++) {
                i32 jp = ipiv[j] - 1;  // Convert to 0-based
                c128 t = x[jp];
                if (jp != j) {
                    x[jp] = x[j];
                    x[j] = t;
                }
                x[j + 1] = x[j + 1] - t * h[j + 1 + j * ldh];
            }

            SLC_ZLATRS("Upper", "No transpose", "Non-unit", &normin, &n,
                       h, &ldh, x, &scale, dwork, &info);
        } else {
            SLC_ZLATRS("Upper", "Conjugate transpose", "Non-unit", &normin, &n,
                       h, &ldh, x, &scale, dwork, &info);

            for (i32 j = n - 2; j >= 0; j--) {
                x[j] = x[j] - conj(h[j + 1 + j * ldh]) * x[j + 1];
                i32 jp = ipiv[j] - 1;  // Convert to 0-based
                if (jp != j) {
                    c128 t = x[jp];
                    x[jp] = x[j];
                    x[j] = t;
                }
            }
        }

        normin = 'Y';
        if (scale != 1.0) {
            i32 ix = SLC_IZAMAX(&n, x, &inc1) - 1;  // Convert to 0-based
            f64 cabs_ix = fabs(creal(x[ix])) + fabs(cimag(x[ix]));
            if (scale < cabs_ix * smlnum || scale == 0.0) {
                return 0;
            }
            SLC_ZDRSCL(&n, &scale, x, &inc1);
        }
    }

    if (hinvnm != 0.0) {
        *rcond = (1.0 / hinvnm) / hnorm;
    }

    return 0;
}
