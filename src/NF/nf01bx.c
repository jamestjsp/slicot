/*
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * NF01BX - Compute (J'*J + c*I)*x for single output Jacobian (CG method).
 *
 * Computes (J'*J + c*I)*x where J is an m-by-n real matrix, c is a scalar,
 * I is the identity matrix, and x is a vector. Used with MD03AD iterative
 * (conjugate gradients) solver for a single output.
 */

#include "slicot.h"
#include "slicot_blas.h"

void nf01bx(i32 n, i32 *ipar, i32 lipar, f64 *dpar, i32 ldpar,
            f64 *j, i32 ldj, f64 *x, i32 incx,
            f64 *dwork, i32 ldwork, i32 *info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    i32 int1 = 1;

    *info = 0;

    if (n < 0) {
        *info = -1;
    } else if (lipar < 1) {
        *info = -3;
    } else if (ldpar < 1) {
        *info = -5;
    } else if (incx == 0) {
        *info = -9;
    } else {
        i32 m = ipar[0];
        if (m < 0) {
            *info = -2;
        } else if (ldj < 1 || (m > 0 && ldj < m)) {
            *info = -7;
        } else if (ldwork < m) {
            *info = -11;
        }
    }

    if (*info != 0) {
        return;
    }

    if (n == 0) {
        return;
    }

    f64 c = dpar[0];
    i32 m = ipar[0];

    if (m == 0) {
        SLC_DSCAL(&n, &c, x, &incx);
        return;
    }

    /* x_out = J'*(J*x) + c*x */
    SLC_DGEMV("N", &m, &n, &ONE, j, &ldj, x, &incx, &ZERO, dwork, &int1);
    SLC_DGEMV("T", &m, &n, &ONE, j, &ldj, dwork, &int1, &c, x, &incx);
}
