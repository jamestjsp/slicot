/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"

/**
 * @brief Calculate the output of the Wiener system.
 *
 * Calculates the output y of the Wiener system:
 * x(t+1) = A*x(t) + B*u(t)
 * z(t)   = C*x(t) + D*u(t)
 * y(t)   = f(z(t), wb)
 *
 * @param[in] nsmp Number of training samples.
 * @param[in] m Length of each input sample.
 * @param[in] l Length of each output sample.
 * @param[in] ipar Integer parameters (n, nn).
 * @param[in] lipar Length of ipar.
 * @param[in] x Parameter vector (wb, theta).
 * @param[in] lx Length of x.
 * @param[in] u Input samples.
 * @param[in] ldu Leading dimension of u.
 * @param[out] y Simulated output.
 * @param[in] ldy Leading dimension of y.
 * @param[out] dwork Workspace.
 * @param[in] ldwork Length of dwork.
 * @param[out] info Exit code.
 */
void nf01ad(i32 nsmp, i32 m, i32 l, i32 *ipar, i32 lipar, f64 *x, i32 lx, 
            f64 *u, i32 ldu, f64 *y, i32 ldy, f64 *dwork, i32 ldwork, i32 *info)
{
    /* Local variables */
    i32 ac, bd, ix, jw, ldac, lths, n, nn, nths, z;
    i32 min_nsmp_l;

    *info = 0;
    if (nsmp < 0) *info = -1;
    else if (m < 0) *info = -2;
    else if (l < 0) *info = -3;
    else if (lipar < 2) *info = -5;
    else {
        n = ipar[0];
        nn = ipar[1];
        ldac = n + l;
        nths = (nn * (l + 2) + 1) * l;
        lths = n * (m + l + 1) + l * m;

        if (n < 0 || nn < 0) *info = -4;
        else if (lx < nths + lths) *info = -7;
        else if (ldu < (nsmp > 1 ? nsmp : 1)) *info = -9;
        else if (ldy < (nsmp > 1 ? nsmp : 1)) *info = -11;
        else {
            i32 jw_size;
            if (m > 0) {
                jw_size = (n * ldac > n + m + l) ? n * ldac : n + m + l;
            } else {
                jw_size = (n * ldac > l) ? n * ldac : l;
            }
            
            i32 min_ldwork = nsmp * l + ((2 * nn > ldac * (n + m) + 2 * n + jw_size) 
                                         ? 2 * nn 
                                         : ldac * (n + m) + 2 * n + jw_size);
            
            if (ldwork < min_ldwork) *info = -13;
        }
    }

    if (*info != 0) {
        i32 err_code = -(*info);
        SLC_XERBLA("NF01AD", &err_code);
        return;
    }

    min_nsmp_l = (nsmp < l) ? nsmp : l;
    if (min_nsmp_l == 0) return;

    /* Compute output of linear part */
    z = 0; /* 0-based index */
    ac = z + nsmp * l;
    bd = ac + ldac * n;
    ix = bd + ldac * m;
    jw = ix + n;

    /* Call TB01VY */
    /* Parameters: 'Apply', N, M, L, THETA, LTHETA, ... */
    /* THETA is at x[nths] */
    i32 lwork_tb = ldwork - jw;
    tb01vy("Apply", n, m, l, &x[nths], lths, &dwork[ac], ldac, 
           &dwork[bd], ldac, &dwork[ac + n], ldac, &dwork[bd + n], ldac, 
           &dwork[ix], &dwork[jw], lwork_tb, info);

    /* Call TF01MX */
    /* Params: N, M, L, NSMP, S, LDS, U, LDU, X, Y, LDY, DWORK, LDWORK, INFO */
    /* S is at dwork[ac], LDS is ldac.
       X (initial state) is at dwork[ix].
       Y (output of linear part) is at dwork[z]. LDY is NSMP.
    */
    i32 lwork_tf = ldwork - jw;
    tf01mx(n, m, l, nsmp, &dwork[ac], ldac, u, ldu, &dwork[ix], 
           &dwork[z], nsmp, &dwork[jw], lwork_tf, info);

    /* Simulate static nonlinearity */
    /* Call NF01AY */
    /* Params: NSMP, NZ, L, IPAR, LIPAR, WB, LWB, Z, LDZ, Y, LDY, DWORK, LDWORK, INFO */
    /* NZ is L (output of linear part is input to nonlinear part) */
    /* WB is x[0]. LWB is nths. */
    /* Z is dwork[z]. LDZ is NSMP. */
    /* Y is output y. */
    jw = ac;
    i32 lwork_nf = ldwork - jw;
    /* Pass &ipar[1] as NF01AY expects IPAR(1)=NN. Here ipar[1]=NN. */
    /* LIPAR-1 because we skip ipar[0]=N. */
    nf01ay(nsmp, l, l, &ipar[1], lipar - 1, x, nths, &dwork[z], nsmp, 
           y, ldy, &dwork[jw], lwork_nf, info);
}
