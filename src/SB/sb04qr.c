/**
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * SB04QR - Solve linear algebraic system with special compact storage
 *
 * Solves a linear algebraic system of order M whose coefficient matrix
 * has zeros below the third subdiagonal and zero elements on the third
 * subdiagonal with even column indices. The matrix is stored compactly,
 * row-wise.
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>

void sb04qr(i32 m, f64* d, i32* ipr, i32* info)
{
    const f64 zero = 0.0;

    i32 i, i1, i2, iprm, iprm1, j, k, l, m1, mpi, mpi1, mpi2;
    f64 d1, d2, d3, dmax;

    *info = 0;

    if (m == 0) return;

    i2 = m * m / 2 + 3 * m;
    mpi = m;
    iprm = i2;
    m1 = m;
    i1 = 1;

    for (i = 1; i <= m; i++) {
        mpi = mpi + 1;
        iprm = iprm + 1;
        ipr[mpi - 1] = i1;
        ipr[i - 1] = iprm;
        i1 = i1 + m1;
        if (i >= 4 && (i % 2) == 0) m1 = m1 - 2;
    }

    m1 = m - 1;
    mpi1 = m + 1;

    for (i = 1; i <= m1; i++) {
        mpi = mpi1;
        mpi1 = mpi1 + 1;
        iprm = ipr[mpi - 1];
        d1 = d[iprm - 1];
        i1 = 3;
        if ((i % 2) == 0) i1 = 2;
        if (i == m1) i1 = 1;
        mpi2 = mpi + i1;
        l = 0;
        dmax = fabs(d1);

        for (j = mpi1; j <= mpi2; j++) {
            d2 = d[ipr[j - 1] - 1];
            d3 = fabs(d2);
            if (d3 > dmax) {
                dmax = d3;
                d1 = d2;
                l = j - mpi;
            }
        }

        if (dmax == zero) {
            *info = 1;
            return;
        }

        if (l > 0) {
            k = iprm;
            j = mpi + l;
            iprm = ipr[j - 1];
            ipr[j - 1] = k;
            ipr[mpi - 1] = iprm;
            k = ipr[i - 1];
            i2 = i + l;
            ipr[i - 1] = ipr[i2 - 1];
            ipr[i2 - 1] = k;
        }
        iprm = iprm + 1;

        i2 = i;
        d3 = d[ipr[i - 1] - 1];

        for (j = mpi1; j <= mpi2; j++) {
            i2 = i2 + 1;
            iprm1 = ipr[j - 1];
            dmax = -d[iprm1 - 1] / d1;
            d[ipr[i2 - 1] - 1] = d[ipr[i2 - 1] - 1] + dmax * d3;
            i32 n_axpy = m - i;
            i32 inc1 = 1;
            SLC_DAXPY(&n_axpy, &dmax, &d[iprm - 1], &inc1, &d[iprm1], &inc1);
            ipr[j - 1] = ipr[j - 1] + 1;
        }
    }

    mpi = m + m;
    iprm = ipr[mpi - 1];

    if (d[iprm - 1] == zero) {
        *info = 1;
        return;
    }

    d[ipr[m - 1] - 1] = d[ipr[m - 1] - 1] / d[iprm - 1];

    for (i = m1; i >= 1; i--) {
        mpi = mpi - 1;
        iprm = ipr[mpi - 1];
        iprm1 = iprm;
        dmax = zero;

        for (k = i + 1; k <= m; k++) {
            iprm1 = iprm1 + 1;
            dmax = dmax + d[ipr[k - 1] - 1] * d[iprm1 - 1];
        }

        d[ipr[i - 1] - 1] = (d[ipr[i - 1] - 1] - dmax) / d[iprm - 1];
    }
}
