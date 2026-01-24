/*
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * NF01BW - Compute (J'*J + c*I)*x for full Wiener system Jacobian (CG method).
 *
 * Computes (J'*J + c*I)*x for the Jacobian J as received from NF01BD which has
 * a block structure with diagonal blocks J_k and off-diagonal L_k. Used with
 * MD03AD iterative (conjugate gradients) solver for the full Wiener system.
 */

#include "slicot.h"
#include "slicot_blas.h"

void nf01bw(i32 n, i32 *ipar, i32 lipar, f64 *dpar, i32 ldpar,
            f64 *j, i32 ldj, f64 *x, i32 incx,
            f64 *dwork, i32 ldwork, i32 *info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    i32 int0 = 0;
    i32 int1 = 1;

    *info = 0;

    if (n < 0) {
        *info = -1;
    } else if (lipar < 4) {
        *info = -3;
    } else if (ldpar < 1) {
        *info = -5;
    } else if (incx < 1) {
        *info = -9;
    } else {
        i32 st = ipar[0];
        i32 bn = ipar[1];
        i32 bsm = ipar[2];
        i32 bsn = ipar[3];
        i32 nths = bn * bsn;
        i32 m = (bn > 1) ? bn * bsm : bsm;

        if (st < 0 || bn < 0 || bsm < 0 || bsn < 0) {
            *info = -2;
        } else if (n != nths + st) {
            *info = -1;
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
    i32 st = ipar[0];
    i32 bn = ipar[1];
    i32 bsm = ipar[2];
    i32 bsn = ipar[3];
    i32 m = (bn > 1) ? bn * bsm : bsm;

    if (m == 0) {
        SLC_DSCAL(&n, &c, x, &incx);
        return;
    }

    if (bn <= 1 || bsn == 0) {
        SLC_DGEMV("N", &m, &n, &ONE, j, &ldj, x, &incx, &ZERO, dwork, &int1);
        SLC_DGEMV("T", &m, &n, &ONE, j, &ldj, dwork, &int1, &c, x, &incx);
        return;
    }

    i32 jl = bsn;  /* column index of L block (0-based) */
    i32 ix = bsn * incx;
    i32 xl = bn * ix;

    if (st > 0) {
        SLC_DGEMV("N", &m, &st, &ONE, &j[jl * ldj], &ldj, &x[xl], &incx, &ZERO, dwork, &int1);
    } else {
        f64 zero = ZERO;
        SLC_DCOPY(&m, &zero, &int0, dwork, &int1);
    }

    i32 ibsn_idx = 0;  /* position in x array */
    for (i32 ibsm = 0; ibsm < m; ibsm += bsm) {
        SLC_DGEMV("N", &bsm, &bsn, &ONE, &j[ibsm], &ldj, &x[ibsn_idx], &incx,
                  &ONE, &dwork[ibsm], &int1);
        SLC_DGEMV("T", &bsm, &bsn, &ONE, &j[ibsm], &ldj, &dwork[ibsm], &int1,
                  &c, &x[ibsn_idx], &incx);
        ibsn_idx += ix;
    }

    if (st > 0) {
        SLC_DGEMV("T", &m, &st, &ONE, &j[jl * ldj], &ldj, dwork, &int1, &c, &x[xl], &incx);
    }
}
