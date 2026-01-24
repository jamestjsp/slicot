/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 *
 * MB04RV - Complex generalized Sylvester equation solver
 *
 * Solves:
 *     A * R - L * B = scale * C
 *     D * R - L * E = scale * F
 *
 * where A, B, D, E are complex upper triangular (generalized Schur form).
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <complex.h>
#include <math.h>

static inline f64 cabs1(c128 z) {
    return fabs(creal(z)) + fabs(cimag(z));
}

void mb04rv(i32 m, i32 n, f64 pmax,
            const c128 *a, i32 lda, const c128 *b, i32 ldb,
            c128 *c, i32 ldc, const c128 *d, i32 ldd,
            const c128 *e, i32 lde, c128 *f, i32 ldf,
            f64 *scale, i32 *info) {

    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const i32 LDZ = 2;
    const i32 int1 = 1;

    *info = 0;
    i32 ierr = 0;

    *scale = ONE;
    f64 scaloc = ONE;

    if (m == 0 || n == 0) {
        return;
    }

    c128 z[4], zs[4];
    c128 rhs[2];
    i32 ipiv[2], jpiv[2];

    for (i32 j = 0; j < n; j++) {
        for (i32 i = m - 1; i >= 0; i--) {
            z[0 + 0*LDZ] =  a[i + i*lda];
            z[1 + 0*LDZ] =  d[i + i*ldd];
            z[0 + 1*LDZ] = -b[j + j*ldb];
            z[1 + 1*LDZ] = -e[j + j*lde];

            SLC_ZLACPY("F", &LDZ, &LDZ, z, &LDZ, zs, &LDZ);

            rhs[0] = c[i + j*ldc];
            rhs[1] = f[i + j*ldf];

            SLC_ZGETC2(&LDZ, z, &LDZ, ipiv, jpiv, &ierr);
            if (ierr > 0) {
                *info = 2;

                for (i32 l = 0; l < LDZ; l++) {
                    i32 ix = SLC_IZAMAX(&LDZ, &zs[l + 0*LDZ], &LDZ);
                    ix--;  // IZAMAX returns 1-based
                    if (ix < 0 || ix >= LDZ) {
                        *info = 1;
                        return;
                    }
                    f64 sc = cabs1(zs[l + ix*LDZ]);
                    if (sc == ZERO) {
                        *info = 1;
                        return;
                    } else if (sc != ONE) {
                        f64 inv_sc = ONE / sc;
                        SLC_ZDSCAL(&LDZ, &inv_sc, &zs[l + 0*LDZ], &LDZ);
                        rhs[l] = rhs[l] / sc;
                    }
                }

                SLC_ZGETC2(&LDZ, zs, &LDZ, ipiv, jpiv, &ierr);
                if (ierr == 0) {
                    *info = 0;
                }
                if (*info > 0) {
                    return;
                }
                SLC_ZLACPY("F", &LDZ, &LDZ, zs, &LDZ, z, &LDZ);
            }

            SLC_ZGESC2(&LDZ, z, &LDZ, rhs, ipiv, jpiv, &scaloc);

            f64 arhs1 = cabs1(rhs[0]);
            f64 arhs2 = cabs1(rhs[1]);
            *scale = (*scale) * scaloc;
            f64 max_arhs = (arhs1 > arhs2) ? arhs1 : arhs2;
            if (max_arhs * (*scale) > pmax) {
                *info = 1;
                return;
            }

            if (scaloc != ONE) {
                for (i32 k = 0; k < n; k++) {
                    SLC_ZDSCAL(&m, &scaloc, &c[0 + k*ldc], &int1);
                    SLC_ZDSCAL(&m, &scaloc, &f[0 + k*ldf], &int1);
                }
            }

            c[i + j*ldc] = rhs[0];
            f[i + j*ldf] = rhs[1];

            if (i > 0) {
                c128 alpha = -rhs[0];
                SLC_ZAXPY(&i, &alpha, &a[0 + i*lda], &int1, &c[0 + j*ldc], &int1);
                SLC_ZAXPY(&i, &alpha, &d[0 + i*ldd], &int1, &f[0 + j*ldf], &int1);
            }

            if (j < n - 1) {
                i32 nj = n - j - 1;
                SLC_ZAXPY(&nj, &rhs[1], &b[j + (j+1)*ldb], &ldb, &c[i + (j+1)*ldc], &ldc);
                SLC_ZAXPY(&nj, &rhs[1], &e[j + (j+1)*lde], &lde, &f[i + (j+1)*ldf], &ldf);
            }
        }
    }
}
