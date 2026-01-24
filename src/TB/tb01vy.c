/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>

void tb01vy(const char* apply, i32 n, i32 m, i32 l, const f64* theta,
            i32 ltheta, f64* a, i32 lda, f64* b, i32 ldb, f64* c, i32 ldc,
            f64* d, i32 ldd, f64* x0, f64* dwork, i32 ldwork, i32* info)
{
    const f64 zero = 0.0;
    const f64 one = 1.0;
    const f64 half = 0.5;
    const i32 int1 = 1;
    const i32 int0 = 0;

    i32 i, j, k, in, ca, jwork, ldca;
    f64 factor, ri, ti, tobypi;
    bool lapply;

    lapply = (apply[0] == 'A' || apply[0] == 'a');

    *info = 0;

    if (!lapply && !(apply[0] == 'N' || apply[0] == 'n')) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (m < 0) {
        *info = -3;
    } else if (l < 0) {
        *info = -4;
    } else if (ltheta < n * (l + m + 1) + l * m) {
        *info = -6;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -8;
    } else if (ldb < (n > 1 ? n : 1)) {
        *info = -10;
    } else if (ldc < (l > 1 ? l : 1)) {
        *info = -12;
    } else if (ldd < (l > 1 ? l : 1)) {
        *info = -14;
    } else if (ldwork < n * (n + l + 1)) {
        *info = -17;
    }

    if (*info != 0) {
        return;
    }

    if (n == 0 && m == 0 && l == 0) {
        return;
    }
    if (n > 0 || m > 0 || l > 0) {
        if (n == 0 || m == 0 || l == 0) {
            i32 max_nml = n;
            if (m > max_nml) max_nml = m;
            if (l > max_nml) max_nml = l;
            if (max_nml == 0) return;
        }
    }

    if (m > 0) {
        SLC_DLACPY("F", &n, &m, &theta[n*l], &n, b, &ldb);
        SLC_DLACPY("F", &l, &m, &theta[n*(l+m)], &l, d, &ldd);
    }

    if (n == 0) {
        return;
    } else if (l == 0) {
        SLC_DCOPY(&n, &theta[n*m], &int1, x0, &int1);
        return;
    }

    ldca = n + l;
    ca = 0;
    jwork = ca + n * ldca;
    tobypi = half / atan(one);

    dwork[ca] = zero;
    i32 len = n * (l + n);
    SLC_DCOPY(&len, &dwork[ca], &int0, &dwork[ca], &int1);

    dwork[ca + l] = one;
    i32 stride = ldca + 1;
    SLC_DCOPY(&n, &dwork[ca + l], &int0, &dwork[ca + l], &stride);

    for (i = n - 1; i >= 0; i--) {
        SLC_DCOPY(&l, &theta[i*l], &int1, c, &int1);

        ti = SLC_DNRM2(&l, c, &int1);

        if (lapply && ti != zero) {
            factor = tobypi * atan(ti) / ti;
            SLC_DSCAL(&l, &factor, c, &int1);
            ti = ti * factor;
        }

        ri = sqrt((one - ti) * (one + ti));

        f64 neg_one = -one;
        i32 offset_ca = ca + n - i - 1;
        SLC_DGEMV("T", &l, &n, &neg_one, &dwork[offset_ca], &ldca, c, &int1,
                  &zero, &dwork[jwork], &int1);

        if (ti > zero) {
            f64 coeff = (one - ri) / (ti * ti);
            SLC_DGER(&l, &n, &coeff, c, &int1, &dwork[jwork], &int1,
                     &dwork[offset_ca], &ldca);
        } else {
            SLC_DGER(&l, &n, &half, c, &int1, &dwork[jwork], &int1,
                     &dwork[offset_ca], &ldca);
        }

        SLC_DGER(&l, &n, &one, c, &int1, &dwork[offset_ca + l], &ldca,
                 &dwork[offset_ca], &ldca);

        SLC_DAXPY(&n, &ri, &dwork[offset_ca + l], &ldca, &dwork[jwork], &int1);

        for (j = 0; j < n; j++) {
            in = ca + n - i - 1 + j * ldca;
            for (k = in + l - 1; k >= in; k--) {
                dwork[k + 1] = dwork[k];
            }
            dwork[in] = dwork[jwork + j];
        }
    }

    for (i = 0; i < n; i++) {
        i32 offset = ca + i * ldca;
        SLC_DCOPY(&l, &dwork[offset], &int1, &c[i*ldc], &int1);
        SLC_DCOPY(&n, &dwork[offset + l], &int1, &a[i*lda], &int1);
    }

    SLC_DCOPY(&n, &theta[n*(l+m) + l*m], &int1, x0, &int1);
}
