/**
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * SB04MW - Solve linear algebraic system with upper Hessenberg compact storage
 *
 * Solves a linear algebraic system of order M whose coefficient
 * matrix is in upper Hessenberg form, stored compactly, row-wise.
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>

void sb04mw(i32 m, f64* d, i32* ipr, i32* info)
{
    const f64 zero = 0.0;

    i32 i, i1, iprm, iprm1, k, m1, m2, mpi;
    f64 d1, d2, mult;

    *info = 0;

    if (m == 0) return;

    m1 = (m * (m + 3)) / 2;
    m2 = m + m;
    mpi = m;
    iprm = m1;
    m1 = m;
    i1 = 1;

    for (i = 1; i <= m; i++) {
        mpi = mpi + 1;
        iprm = iprm + 1;
        ipr[mpi - 1] = i1;
        ipr[i - 1] = iprm;
        i1 = i1 + m1;
        if (i > 1) m1 = m1 - 1;
    }

    m1 = m - 1;
    mpi = m;

    for (i = 1; i <= m1; i++) {
        i1 = i + 1;
        mpi = mpi + 1;
        iprm = ipr[mpi - 1];
        iprm1 = ipr[mpi];
        d1 = d[iprm - 1];
        d2 = d[iprm1 - 1];
        if (fabs(d1) <= fabs(d2)) {
            k = iprm;
            ipr[mpi - 1] = iprm1;
            iprm = iprm1;
            iprm1 = k;
            k = ipr[i - 1];
            ipr[i - 1] = ipr[i1 - 1];
            ipr[i1 - 1] = k;
            d1 = d2;
        }

        if (d1 == zero) {
            *info = 1;
            return;
        }

        mult = -d[iprm1 - 1] / d1;
        iprm1 = iprm1 + 1;
        ipr[mpi] = iprm1;

        d[ipr[i1 - 1] - 1] = d[ipr[i1 - 1] - 1] + mult * d[ipr[i - 1] - 1];
        i32 n_axpy = m - i;
        i32 inc1 = 1;
        SLC_DAXPY(&n_axpy, &mult, &d[iprm], &inc1, &d[iprm1 - 1], &inc1);
    }

    if (d[ipr[m2 - 1] - 1] == zero) {
        *info = 1;
        return;
    }

    d[ipr[m - 1] - 1] = d[ipr[m - 1] - 1] / d[ipr[m2 - 1] - 1];
    mpi = m2;

    for (i = m1; i >= 1; i--) {
        mpi = mpi - 1;
        iprm = ipr[mpi - 1];
        iprm1 = iprm;
        mult = zero;

        for (i1 = i + 1; i1 <= m; i1++) {
            iprm1 = iprm1 + 1;
            mult = mult + d[ipr[i1 - 1] - 1] * d[iprm1 - 1];
        }

        d[ipr[i - 1] - 1] = (d[ipr[i - 1] - 1] - mult) / d[iprm - 1];
    }
}
