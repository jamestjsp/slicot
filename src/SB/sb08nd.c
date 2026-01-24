/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>

/**
 * @brief Spectral factorization of polynomials (discrete-time case)
 *
 * Computes a real polynomial E(z) such that:
 *   (a) E(1/z) * E(z) = A(1/z) * A(z)
 *   (b) E(z) is stable (all zeros have modulus <= 1)
 *
 * Input polynomial may be supplied as A(z) or B(z) = A(1/z) * A(z).
 */
void sb08nd(
    const char* acona,
    const i32 da,
    f64* a,
    f64* res,
    f64* e,
    f64* dwork,
    const i32 ldwork,
    i32* info)
{
    const f64 ZERO = 0.0;
    const f64 HALF = 0.5;
    const f64 ONE = 1.0;
    const f64 TWO = 2.0;

    *info = 0;
    bool lacona = (acona[0] == 'A' || acona[0] == 'a');

    if (!lacona && acona[0] != 'B' && acona[0] != 'b') {
        *info = -1;
    } else if (da < 0) {
        *info = -2;
    } else if (ldwork < 5 * da + 5) {
        *info = -7;
    }

    if (*info != 0) {
        return;
    }

    i32 int1 = 1;
    i32 nc = da + 1;
    f64 w;

    if (!lacona) {
        if (a[0] <= ZERO) {
            *info = 2;
            return;
        }
        SLC_DCOPY(&nc, a, &int1, e, &int1);
    } else {
        sb08ny(da, a, e, &w);
    }

    i32 lalpha = 0;
    i32 lro = lalpha + nc;
    i32 leta = lro + nc;
    i32 llambda = leta + nc;
    i32 lq = llambda + nc;

    f64 a0 = e[0];
    f64 sa0 = sqrt(a0);
    f64 s = ZERO;

    for (i32 j = 0; j < nc; j++) {
        w = e[j];
        a[j] = w;
        w = w / sa0;
        e[j] = w;
        dwork[lq + j] = w;
        s = s + w * w;
    }

    f64 res0 = s - a0;

    i32 i = 0;
    bool conv = false;
    bool hurwtz = true;

    while (i < 30 && !conv && hurwtz) {
        i++;
        SLC_DCOPY(&nc, a, &int1, &dwork[leta], &int1);
        SLC_DSCAL(&nc, &TWO, &dwork[leta], &int1);
        SLC_DCOPY(&nc, &dwork[lq], &int1, &dwork[lalpha], &int1);

        i32 k = 1;

        while (k <= da && hurwtz) {
            i32 nck = nc - k;

            for (i32 jj = 0; jj <= nck; jj++) {
                dwork[lro + jj] = dwork[lalpha + nck - jj];
            }

            w = dwork[lalpha + nck] / dwork[lro + nck];
            if (fabs(w) >= ONE) {
                hurwtz = false;
            }
            if (hurwtz) {
                dwork[llambda + k - 1] = w;
                f64 negw = -w;
                i32 nck1 = nck;
                SLC_DAXPY(&nck1, &negw, &dwork[lro], &int1, &dwork[lalpha], &int1);

                w = dwork[leta + nck] / dwork[lalpha];
                dwork[leta + nck] = w;

                i32 nck_m1 = nck - 1;
                if (nck_m1 > 0) {
                    f64 negw2 = -w;
                    i32 negint1 = -1;
                    SLC_DAXPY(&nck_m1, &negw2, &dwork[lalpha + 1], &negint1, &dwork[leta + 1], &int1);
                }
                k++;
            }
        }

        if (hurwtz) {
            SLC_DCOPY(&nc, &dwork[lq], &int1, e, &int1);

            f64 tolq;
            sb08ny(da, e, &dwork[lq], &tolq);

            f64 negone = -ONE;
            SLC_DAXPY(&nc, &negone, a, &int1, &dwork[lq], &int1);

            i32 imax = SLC_IDAMAX(&nc, &dwork[lq], &int1);
            *res = fabs(dwork[lq + imax - 1]);
            conv = (*res < tolq) || (res0 < ZERO);

            if (!conv) {
                dwork[leta] = HALF * dwork[leta] / dwork[lalpha];

                for (k = da; k >= 1; k--) {
                    i32 nck1 = nc - k + 1;

                    for (i32 jj = 0; jj < nck1; jj++) {
                        dwork[lro + jj] = dwork[leta + nck1 - 1 - jj];
                    }

                    w = dwork[llambda + k - 1];
                    f64 negw = -w;
                    SLC_DAXPY(&nck1, &negw, &dwork[lro], &int1, &dwork[leta], &int1);
                }

                s = ZERO;
                for (i32 j = 0; j <= da; j++) {
                    w = HALF * (dwork[leta + j] + e[j]);
                    dwork[lq + j] = w;
                    s = s + w * w;
                }

                res0 = s - a0;

                conv = dwork[lq] > e[0];
            }
        }
    }

    SLC_DSWAP(&nc, e, &int1, dwork, &(i32){-1});
    SLC_DSWAP(&nc, dwork, &int1, e, &int1);

    if (!conv) {
        if (hurwtz) {
            *info = 3;
        } else if (i == 1) {
            *info = 2;
        } else {
            *info = 4;
        }
    }
}
