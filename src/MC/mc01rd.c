/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"

void mc01rd(i32 dp1, i32 dp2, i32 *dp3, f64 alpha, const f64 *p1,
            const f64 *p2, f64 *p3, i32 *info)
{
    *info = 0;
    if (dp1 < -1) {
        *info = -1;
        return;
    }
    if (dp2 < -1) {
        *info = -2;
        return;
    }
    if (*dp3 < -1) {
        *info = -3;
        return;
    }

    i32 d1 = dp1;
    while (d1 >= 0 && p1[d1] == 0.0) {
        d1--;
    }

    i32 d2 = dp2;
    while (d2 >= 0 && p2[d2] == 0.0) {
        d2--;
    }

    i32 d3 = (alpha == 0.0) ? -1 : *dp3;
    while (d3 >= 0 && p3[d3] == 0.0) {
        d3--;
    }

    i32 int1 = 1;
    i32 intm1 = -1;
    i32 d3_plus_1 = d3 + 1;
    SLC_DSCAL(&d3_plus_1, &alpha, p3, &int1);

    if (d1 == -1 || d2 == -1) {
        *dp3 = d3;
        return;
    }

    i32 dsum = d1 + d2;
    i32 dmax = (d1 > d2) ? d1 : d2;
    i32 dmin = dsum - dmax;

    if (d3 < dsum) {
        p3[d3 + 1] = 0.0;
        i32 count = dsum - d3 - 1;
        if (count > 0) {
            i32 zero_inc = 0;
            SLC_DCOPY(&count, &p3[d3 + 1], &zero_inc, &p3[d3 + 2], &int1);
        }
        d3 = dsum;
    }

    if (d1 == 0 || d2 == 0) {
        if (d1 != 0) {
            i32 len = d1 + 1;
            SLC_DAXPY(&len, &p2[0], p1, &int1, p3, &int1);
        } else {
            i32 len = d2 + 1;
            SLC_DAXPY(&len, &p1[0], p2, &int1, p3, &int1);
        }
    } else {
        for (i32 i = 1; i <= dmin + 1; i++) {
            p3[i - 1] += SLC_DDOT(&i, p1, &int1, p2, &intm1);
        }

        for (i32 i = dmin + 2; i <= dmax + 1; i++) {
            i32 len = dmin + 1;
            if (d1 > d2) {
                i32 k = i - d2 - 1;
                p3[i - 1] += SLC_DDOT(&len, &p1[k], &int1, p2, &intm1);
            } else {
                i32 k = i - d1 - 1;
                p3[i - 1] += SLC_DDOT(&len, &p2[k], &intm1, p1, &int1);
            }
        }

        i32 e3 = dsum + 2;
        for (i32 i = dmax + 2; i <= dsum + 1; i++) {
            i32 j = e3 - i;
            i32 k = i - dmin - 1;
            i32 l = i - dmax - 1;
            if (d1 > d2) {
                p3[i - 1] += SLC_DDOT(&j, &p1[k], &int1, &p2[l], &intm1);
            } else {
                p3[i - 1] += SLC_DDOT(&j, &p1[l], &intm1, &p2[k], &int1);
            }
        }
    }

    while (d3 >= 0 && p3[d3] == 0.0) {
        d3--;
    }
    *dp3 = d3;
}
