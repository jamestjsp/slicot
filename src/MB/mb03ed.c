/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>

void mb03ed(i32 n, f64 prec, const f64 *a, i32 lda, const f64 *b, i32 ldb,
            f64 *d, i32 ldd, f64 *q1, i32 ldq1, f64 *q2, i32 ldq2,
            f64 *q3, i32 ldq3, f64 *dwork, i32 ldwork, i32 *info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    bool compg;
    i32 idum, ievs, iwrk;
    f64 a11, a22, b11, b22, d12, d21, co, si, tmp;
    i32 bwork[4];

    *info = 0;

    if (n == 4) {
        f64 dum = ZERO;
        i32 int1 = 1;
        i32 int0 = 0;
        i32 int16 = 16;

        SLC_DCOPY(&int16, &dum, &int0, dwork, &int1);

        dwork[0] = b[0];
        dwork[4] = b[ldb];
        dwork[5] = b[1 + ldb];
        dwork[10] = b[2 + 2 * ldb];
        dwork[14] = b[2 + 3 * ldb];
        dwork[15] = b[3 + 3 * ldb];

        i32 int2 = 2;
        i32 int4 = 4;

        SLC_DTRMM("L", "U", "N", "N", &int2, &int4, &ONE, a, &lda, dwork, &n);

        SLC_DTRMM("L", "U", "N", "N", &int2, &int2, &ONE, &a[2 + 2 * lda], &lda,
                  &dwork[10], &n);

        ievs = n * n;
        iwrk = ievs + 3 * n;

        i32 lwork_dgges = ldwork - iwrk;

        SLC_DGGES("V", "V", "S", sb02ow, &n, d, &ldd, dwork, &n, &idum,
                  &dwork[ievs], &dwork[ievs + n], &dwork[ievs + 2 * n],
                  q3, &ldq3, q1, &ldq1, &dwork[iwrk], &lwork_dgges, bwork, info);

        if (*info != 0) {
            if (*info >= 1 && *info <= 4) {
                *info = 1;
                return;
            } else if (*info != 6) {
                *info = 2;
                return;
            } else {
                *info = 0;
            }
        }

        SLC_DLACPY("F", &n, &n, q1, &ldq1, q2, &ldq2);

        SLC_DTRMM("L", "U", "N", "N", &int2, &int4, &ONE, b, &ldb, q2, &ldq2);

        SLC_DTRMM("L", "U", "N", "N", &int2, &int4, &ONE, &b[2 + 2 * ldb], &ldb,
                  &q2[2], &ldq2);

        i32 info_qr = 0;
        SLC_DGEQR2(&n, &n, q2, &ldq2, dwork, &dwork[n], &info_qr);

        i32 info_org = 0;
        SLC_DORG2R(&n, &n, &n, q2, &ldq2, dwork, &dwork[n], &info_org);

    } else {
        a11 = fabs(a[0]);
        a22 = fabs(a[1 + lda]);
        b11 = fabs(b[0]);
        b22 = fabs(b[1 + ldb]);
        d21 = fabs(d[1]);
        d12 = fabs(d[ldd]);
        compg = false;

        if (a11 * b11 <= prec * a22 * b22) {
            if (a11 <= prec * a22) {
                q1[0] = ONE;
                q1[1] = ZERO;
                q1[ldq1] = ZERO;
                q1[1 + ldq1] = ONE;
                q2[0] = ONE;
                q2[1] = ZERO;
                q2[ldq2] = ZERO;
                q2[1 + ldq2] = ONE;
                q3[0] = ZERO;
                q3[1] = -ONE;
                q3[ldq3] = -ONE;
                q3[1 + ldq3] = ZERO;
            } else if (b11 <= prec * b22) {
                q1[0] = -ONE;
                q1[1] = ZERO;
                q1[ldq1] = ZERO;
                q1[1 + ldq1] = -ONE;
                q2[0] = ZERO;
                q2[1] = ONE;
                q2[ldq2] = ONE;
                q2[1 + ldq2] = ZERO;
                q3[0] = ZERO;
                q3[1] = ONE;
                q3[ldq3] = ONE;
                q3[1 + ldq3] = ZERO;
            } else {
                compg = true;
            }
        } else if (a22 * b22 <= prec * a11 * b11) {
            if (a22 <= prec * a11) {
                q1[0] = ZERO;
                q1[1] = ONE;
                q1[ldq1] = ONE;
                q1[1 + ldq1] = ZERO;
                q2[0] = ZERO;
                q2[1] = ONE;
                q2[ldq2] = ONE;
                q2[1 + ldq2] = ZERO;
                q3[0] = -ONE;
                q3[1] = ZERO;
                q3[ldq3] = ZERO;
                q3[1 + ldq3] = -ONE;
            } else if (b22 <= prec * b11) {
                q1[0] = ZERO;
                q1[1] = -ONE;
                q1[ldq1] = -ONE;
                q1[1 + ldq1] = ZERO;
                q2[0] = ONE;
                q2[1] = ZERO;
                q2[ldq2] = ZERO;
                q2[1 + ldq2] = ONE;
                q3[0] = ONE;
                q3[1] = ZERO;
                q3[ldq3] = ZERO;
                q3[1 + ldq3] = ONE;
            } else {
                compg = true;
            }
        } else if (d21 <= prec * d12) {
            q1[0] = ONE;
            q1[1] = ZERO;
            q1[ldq1] = ZERO;
            q1[1 + ldq1] = ONE;
            q2[0] = ONE;
            q2[1] = ZERO;
            q2[ldq2] = ZERO;
            q2[1 + ldq2] = ONE;
            q3[0] = ONE;
            q3[1] = ZERO;
            q3[ldq3] = ZERO;
            q3[1 + ldq3] = ONE;
        } else if (d12 <= prec * d21) {
            q1[0] = ZERO;
            q1[1] = ONE;
            q1[ldq1] = ONE;
            q1[1 + ldq1] = ZERO;
            q2[0] = ZERO;
            q2[1] = ONE;
            q2[ldq2] = ONE;
            q2[1 + ldq2] = ZERO;
            q3[0] = ZERO;
            q3[1] = ONE;
            q3[ldq3] = ONE;
            q3[1 + ldq3] = ZERO;
        } else {
            compg = true;
        }

        if (compg) {
            f64 sign_prod = (a[0] * b[0] * a[1 + lda] * b[1 + ldb] >= 0.0) ? ONE : -ONE;
            SLC_DLARTG(&(f64){sign_prod * sqrt(a22 * b22 * d12)},
                       &(f64){sqrt(a11 * b11 * d21)}, &co, &si, &tmp);
            q1[0] = co;
            q1[1] = -si;
            q1[ldq1] = si;
            q1[1 + ldq1] = co;

            f64 sign_aa = (a[0] * a[1 + lda] >= 0.0) ? ONE : -ONE;
            SLC_DLARTG(&(f64){sign_aa * sqrt(a22 * b11 * d12)},
                       &(f64){sqrt(a11 * b22 * d21)}, &co, &si, &tmp);
            q2[0] = co;
            q2[1] = -si;
            q2[ldq2] = si;
            q2[1 + ldq2] = co;

            SLC_DLARTG(&(f64){sqrt(a11 * b11 * d12)},
                       &(f64){sqrt(a22 * b22 * d21)}, &co, &si, &tmp);
            q3[0] = co;
            q3[1] = -si;
            q3[ldq3] = si;
            q3[1 + ldq3] = co;
        }
    }
}
