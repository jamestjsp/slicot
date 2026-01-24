/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>

void sb08md(
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

    if (!lacona) {
        SLC_DCOPY(&(i32){da + 1}, a, &int1, e, &int1);
    } else {
        f64 w = ZERO;
        sb08my(da, a, e, &w);
    }

    i32 da1 = da + 1;

    while (da1 >= 1 && e[da1 - 1] == ZERO) {
        da1--;
    }

    da1 = da1 - 1;
    if (da1 < 0) {
        *info = 1;
        return;
    }

    i32 i0 = 1;
    while (e[i0 - 1] == ZERO) {
        i0++;
    }

    i0 = i0 - 1;
    f64 signi0 = ONE;
    if (i0 != 0) {
        if (i0 % 2 != 0) {
            signi0 = -ONE;
        }

        for (i32 i = 1; i <= da1 - i0 + 1; i++) {
            e[i - 1] = signi0 * e[i + i0 - 1];
        }

        da1 = da1 - i0;
    }

    f64 signi = (da1 % 2 == 0) ? ONE : -ONE;
    i32 nc = da1 + 1;

    if (e[0] < ZERO || (e[nc - 1] * signi) < ZERO) {
        *info = 2;
        return;
    }

    if (da1 == 0) {
        f64 sqrte0 = sqrt(e[0]);
        f64 b0 = e[0];
        for (i32 j = 0; j <= da; j++) {
            e[j] = ZERO;
            a[j] = ZERO;
        }
        e[i0] = sqrte0;
        a[i0] = signi0 * b0;
        *res = ZERO;
        return;
    }

    f64 eps = SLC_DLAMCH("Epsilon");
    f64 si = ONE / SLC_DLAMCH("Safe minimum");

    i32 lq = 0;
    i32 lay = lq + nc;
    i32 lambda = lay + nc;
    i32 lphi = lambda + nc;
    i32 ldif = lphi + nc;

    f64 a0 = e[0];
    i32 binc = 1;

    f64 mu = pow(a0 / fabs(e[nc - 1]), ONE / (f64)da1);
    f64 muj = ONE;

    for (i32 j = 0; j < nc; j++) {
        f64 w = e[j] * muj / a0;
        a[j] = w;
        e[j] = (f64)binc;
        dwork[lq + j] = (f64)binc;
        muj = muj * mu;
        binc = binc * (nc - j - 1) / (j + 1);
    }

    bool conv = false;
    bool stable = true;

    i32 iter = 0;

    while (iter < 30 && !conv && stable) {
        iter++;
        SLC_DCOPY(&nc, a, &int1, &dwork[lay], &int1);
        SLC_DCOPY(&nc, &dwork[lq], &int1, &dwork[lphi], &int1);

        i32 m = da1 / 2;
        i32 layend = lay + da1;
        i32 lphend = lphi + da1;
        f64 xda = a[nc - 1] / dwork[lq + da1];

        for (i32 k = 1; k <= m; k++) {
            dwork[lay + k] = dwork[lay + k] - dwork[lphi + 2 * k];
            dwork[layend - k] = dwork[layend - k] - dwork[lphend - 2 * k] * xda;
        }

        i32 k = 1;
        while (k <= da1 - 2 && stable) {
            if (dwork[lphi + k] <= ZERO) {
                stable = false;
            }
            if (stable) {
                f64 w = dwork[lphi + k - 1] / dwork[lphi + k];
                dwork[lambda + k] = w;
                i32 nax = (da1 - k) / 2;
                f64 negw = -w;
                SLC_DAXPY(&nax, &negw, &dwork[lphi + k + 2], &(i32){2}, &dwork[lphi + k + 1], &(i32){2});

                w = dwork[lay + k] / dwork[lphi + k];
                dwork[lay + k] = w;
                negw = -w;
                SLC_DAXPY(&nax, &negw, &dwork[lphi + k + 2], &(i32){2}, &dwork[lay + k + 1], &int1);
                k++;
            }
        }

        if (dwork[lphi + da1 - 1] <= ZERO) {
            stable = false;
        } else {
            dwork[lay + da1 - 1] = dwork[lay + da1 - 1] / dwork[lphi + da1 - 1];
        }

        if (stable) {
            for (i32 k = da1 - 2; k >= 1; k--) {
                f64 w = dwork[lambda + k];
                i32 nax = (da1 - k) / 2;
                f64 negw = -w;
                SLC_DAXPY(&nax, &negw, &dwork[lay + k + 1], &(i32){2}, &dwork[lay + k], &(i32){2});
            }

            dwork[lay + da1] = xda;

            SLC_DCOPY(&nc, &dwork[lq], &int1, e, &int1);
            f64 simin1 = si;
            si = dwork[lq];
            f64 signj = -ONE;

            for (i32 j = 1; j <= da1; j++) {
                f64 w = HALF * (dwork[lq + j] + signj * dwork[lay + j]);
                dwork[lq + j] = w;
                si = si + w;
                signj = -signj;
            }

            f64 tolphi = eps;
            sb08my(da1, e, &dwork[ldif], &tolphi);
            f64 negone = -ONE;
            SLC_DAXPY(&nc, &negone, a, &int1, &dwork[ldif], &int1);

            i32 imax = SLC_IDAMAX(&nc, &dwork[ldif], &int1);
            *res = fabs(dwork[ldif + imax - 1]);

            if (si > simin1 || *res < tolphi) {
                conv = true;
            }
        }
    }

    mu = ONE / mu;
    f64 sqrta0 = sqrt(a0);
    f64 sqrtmu = sqrt(mu);
    muj = ONE;
    f64 sqrtmj = ONE;

    for (i32 j = 0; j < nc; j++) {
        e[j] = e[j] * sqrta0 * sqrtmj;
        a[j] = a[j] * a0 * muj;
        muj = muj * mu;
        sqrtmj = sqrtmj * sqrtmu;
    }

    if (i0 != 0) {
        for (i32 j = nc; j >= 1; j--) {
            e[i0 + j - 1] = e[j - 1];
            a[i0 + j - 1] = signi0 * a[j - 1];
        }

        for (i32 j = 0; j < i0; j++) {
            e[j] = ZERO;
            a[j] = ZERO;
        }
    }

    if (!conv) {
        if (stable) {
            *info = 3;
        } else {
            *info = 4;
        }
    }
}
