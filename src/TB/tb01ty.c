/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>

void tb01ty(i32 mode, i32 ioff, i32 joff, i32 nrow, i32 ncol,
            f64 size, f64* x, i32 ldx, f64* bvect)
{
    const f64 one = 1.0;
    i32 int1 = 1;

    i32 base_int = (i32)SLC_DLAMCH("Base");
    f64 base = (f64)base_int;
    f64 eps = SLC_DLAMCH("Epsilon");
    f64 div = one / log(base);
    f64 abs_size = fabs(size);

    if (mode != 0) {
        /* Balance columns using column-sum norm */
        for (i32 j = joff; j < joff + ncol; j++) {
            /* DASUM(NROW, X(IOFF+1,J), 1) in Fortran
             * X(IOFF+1, J) = x[ioff + j*ldx] in C (0-based)
             */
            f64 abssum = SLC_DASUM(&nrow, &x[ioff + j * ldx], &int1) / abs_size;
            f64 test = abssum / (f64)nrow;

            if (test > eps) {
                /* Non-zero column: calculate and apply scale factor */
                f64 expt = -div * log(abssum);
                i32 iexpt = (i32)expt;

                /* Adjustment to get floor semantics instead of truncation:
                 * C (i32) truncates toward zero, but we need floor.
                 * Example: expt = -0.5, (i32)(-0.5) = 0, but floor(-0.5) = -1
                 * The Fortran code checks IEXPT<0, but that misses -1<EXPT<0.
                 * Use correct condition: if truncation > expt, decrement.
                 */
                if ((f64)iexpt > expt) {
                    iexpt = iexpt - 1;
                }

                f64 scale = pow(base, (f64)iexpt);
                bvect[j] = scale;
                SLC_DSCAL(&nrow, &scale, &x[ioff + j * ldx], &int1);
            } else {
                /* 'Numerically' zero column: do not rescale */
                bvect[j] = one;
            }
        }
    } else {
        /* Balance rows using row-sum norm */
        for (i32 i = ioff; i < ioff + nrow; i++) {
            /* DASUM(NCOL, X(I,JOFF+1), LDX) in Fortran
             * X(I, JOFF+1) = x[i + joff*ldx] in C (0-based)
             */
            f64 abssum = SLC_DASUM(&ncol, &x[i + joff * ldx], &ldx) / abs_size;
            f64 test = abssum / (f64)ncol;

            if (test > eps) {
                /* Non-zero row: calculate and apply scale factor */
                f64 expt = -div * log(abssum);
                i32 iexpt = (i32)expt;

                /* Adjustment to get floor semantics */
                if ((f64)iexpt > expt) {
                    iexpt = iexpt - 1;
                }

                f64 scale = pow(base, (f64)iexpt);
                bvect[i] = scale;
                SLC_DSCAL(&ncol, &scale, &x[i + joff * ldx], &ldx);
            } else {
                /* 'Numerically' zero row: do not rescale */
                bvect[i] = one;
            }
        }
    }
}
