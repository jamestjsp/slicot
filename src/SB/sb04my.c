/**
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * SB04MY - Construct and solve linear algebraic system for 1x1 blocks
 *
 * Constructs and solves a linear algebraic system of order M whose
 * coefficient matrix is in upper Hessenberg form. Such systems
 * appear when solving Sylvester equations using the Hessenberg-Schur
 * method.
 */

#include "slicot.h"
#include "slicot_blas.h"

void sb04my(i32 n, i32 m, i32 ind, const f64* a, i32 lda,
            const f64* b, i32 ldb, f64* c, i32 ldc,
            f64* d, i32* ipr, i32* info)
{
    i32 i, i2, j, k, k1, k2, m1;

    for (i = ind + 1; i <= n; i++) {
        f64 neg_b_ind_i = -b[(ind - 1) + (i - 1) * ldb];
        i32 int1 = 1;
        SLC_DAXPY(&m, &neg_b_ind_i, &c[(i - 1) * ldc], &int1, &c[(ind - 1) * ldc], &int1);
    }

    m1 = m + 1;
    i2 = (m * m1) / 2 + m1;
    k2 = 1;
    k = m;

    for (i = 1; i <= m; i++) {
        j = m1 - k;
        i32 int1 = 1;
        SLC_DCOPY(&k, &a[(i - 1) + (j - 1) * lda], &lda, &d[k2 - 1], &int1);
        k1 = k2;
        k2 = k2 + k;
        if (i > 1) {
            k1 = k1 + 1;
            k = k - 1;
        }
        d[k1 - 1] = d[k1 - 1] + b[(ind - 1) + (ind - 1) * ldb];

        d[i2 - 1] = c[(i - 1) + (ind - 1) * ldc];
        i2 = i2 + 1;
    }

    sb04mw(m, d, ipr, info);

    if (*info != 0) {
        *info = ind;
    } else {
        for (i = 1; i <= m; i++) {
            c[(i - 1) + (ind - 1) * ldc] = d[ipr[i - 1] - 1];
        }
    }
}
