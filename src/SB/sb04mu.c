/**
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * SB04MU - Construct and solve linear algebraic system for 2x2 blocks
 *
 * Constructs and solves a linear algebraic system of order 2*M
 * whose coefficient matrix has zeros below the second subdiagonal.
 * Such systems appear when solving continuous-time Sylvester
 * equations using the Hessenberg-Schur method.
 */

#include "slicot.h"
#include "slicot_blas.h"

void sb04mu(i32 n, i32 m, i32 ind, const f64* a, i32 lda,
            const f64* b, i32 ldb, f64* c, i32 ldc,
            f64* d, i32* ipr, i32* info)
{
    const f64 zero = 0.0;

    i32 i, i2, ind1, j, k, k1, k2, m2;
    f64 temp;

    ind1 = ind - 1;

    for (i = ind + 1; i <= n; i++) {
        f64 neg_b_ind1_i = -b[(ind1 - 1) + (i - 1) * ldb];
        f64 neg_b_ind_i = -b[(ind - 1) + (i - 1) * ldb];
        i32 int1 = 1;
        SLC_DAXPY(&m, &neg_b_ind1_i, &c[(i - 1) * ldc], &int1, &c[(ind1 - 1) * ldc], &int1);
        SLC_DAXPY(&m, &neg_b_ind_i, &c[(i - 1) * ldc], &int1, &c[(ind - 1) * ldc], &int1);
    }

    k1 = -1;
    m2 = 2 * m;
    i2 = m * (m2 + 5);
    k = m2;

    for (i = 1; i <= m; i++) {
        i32 j_start = (1 > i - 1) ? 1 : i - 1;
        for (j = j_start; j <= m; j++) {
            k1 = k1 + 2;
            k2 = k1 + k;
            temp = a[(i - 1) + (j - 1) * lda];
            if (i != j) {
                d[k1 - 1] = temp;
                d[k1] = zero;
                if (j > i) d[k2 - 1] = zero;
                d[k2] = temp;
            } else {
                d[k1 - 1] = temp + b[(ind1 - 1) + (ind1 - 1) * ldb];
                d[k1] = b[(ind1 - 1) + (ind - 1) * ldb];
                d[k2 - 1] = b[(ind - 1) + (ind1 - 1) * ldb];
                d[k2] = temp + b[(ind - 1) + (ind - 1) * ldb];
            }
        }

        k1 = k2;
        i32 decr = (2 < i) ? 2 : i;
        k = k - decr;

        i2 = i2 + 2;
        d[i2 - 1] = c[(i - 1) + (ind - 1) * ldc];
        d[i2 - 2] = c[(i - 1) + (ind1 - 1) * ldc];
    }

    sb04mr(m2, d, ipr, info);

    if (*info != 0) {
        *info = ind;
    } else {
        i2 = 0;
        for (i = 1; i <= m; i++) {
            i2 = i2 + 2;
            c[(i - 1) + (ind1 - 1) * ldc] = d[ipr[i2 - 2] - 1];
            c[(i - 1) + (ind - 1) * ldc] = d[ipr[i2 - 1] - 1];
        }
    }
}
