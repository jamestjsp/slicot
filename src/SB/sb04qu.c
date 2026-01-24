/**
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * SB04QU - Construct and solve linear algebraic system for 2x2 blocks (discrete)
 *
 * Constructs and solves a linear algebraic system of order 2*M
 * whose coefficient matrix has zeros below the third subdiagonal,
 * and zero elements on the third subdiagonal with even column
 * indices. Such systems appear when solving discrete-time Sylvester
 * equations using the Hessenberg-Schur method.
 */

#include "slicot.h"
#include "slicot_blas.h"

void sb04qu(i32 n, i32 m, i32 ind, const f64* a, i32 lda,
            const f64* b, i32 ldb, f64* c, i32 ldc,
            f64* d, i32* ipr, i32* info)
{
    const f64 one = 1.0;
    const f64 zero = 0.0;

    i32 i, i2, ind1, j, k, k1, k2, m2;
    f64 temp;

    ind1 = ind - 1;

    if (ind < n) {
        i32 int1 = 1;
        for (i = 0; i < m; i++) {
            d[i] = zero;
        }

        for (i = ind + 1; i <= n; i++) {
            f64 b_val = b[(ind1 - 1) + (i - 1) * ldb];
            SLC_DAXPY(&m, &b_val, &c[(i - 1) * ldc], &int1, d, &int1);
        }

        for (i = 2; i <= m; i++) {
            c[(i - 1) + (ind1 - 1) * ldc] -= a[(i - 1) + (i - 2) * lda] * d[i - 2];
        }
        SLC_DTRMV("U", "N", "N", &m, a, &lda, d, &int1);
        for (i = 1; i <= m; i++) {
            c[(i - 1) + (ind1 - 1) * ldc] -= d[i - 1];
        }

        for (i = 0; i < m; i++) {
            d[i] = zero;
        }
        for (i = ind + 1; i <= n; i++) {
            f64 b_val = b[(ind - 1) + (i - 1) * ldb];
            SLC_DAXPY(&m, &b_val, &c[(i - 1) * ldc], &int1, d, &int1);
        }

        for (i = 2; i <= m; i++) {
            c[(i - 1) + (ind - 1) * ldc] -= a[(i - 1) + (i - 2) * lda] * d[i - 2];
        }
        SLC_DTRMV("U", "N", "N", &m, a, &lda, d, &int1);
        for (i = 1; i <= m; i++) {
            c[(i - 1) + (ind - 1) * ldc] -= d[i - 1];
        }
    }

    k1 = -1;
    m2 = 2 * m;
    i2 = m2 * (m + 3);
    k = m2;

    for (i = 1; i <= m; i++) {
        i32 j_start = (1 > i - 1) ? 1 : i - 1;
        for (j = j_start; j <= m; j++) {
            k1 = k1 + 2;
            k2 = k1 + k;
            temp = a[(i - 1) + (j - 1) * lda];
            d[k1 - 1] = temp * b[(ind1 - 1) + (ind1 - 1) * ldb];
            d[k1] = temp * b[(ind1 - 1) + (ind - 1) * ldb];
            d[k2 - 1] = temp * b[(ind - 1) + (ind1 - 1) * ldb];
            d[k2] = temp * b[(ind - 1) + (ind - 1) * ldb];
            if (i == j) {
                d[k1 - 1] += one;
                d[k2] += one;
            }
        }

        k1 = k2;
        if (i > 1) k = k - 2;

        i2 = i2 + 2;
        d[i2 - 1] = c[(i - 1) + (ind - 1) * ldc];
        d[i2 - 2] = c[(i - 1) + (ind1 - 1) * ldc];
    }

    sb04qr(m2, d, ipr, info);

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
