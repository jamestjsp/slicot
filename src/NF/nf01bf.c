/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <stdio.h>

/**
 * @brief Error function for Wiener system identification (Full parameter optimization).
 *
 * This is the FCN routine for optimizing all parameters of a Wiener system.
 *
 * @param[in,out] iflag Integer indicating the action to be performed.
 * @param[in] nfun Number of functions (M in MD03BD).
 * @param[in] lx Number of variables (N in MD03BD).
 * @param[in,out] ipar Integer parameters.
 * @param[in] lipar Length of ipar.
 * @param[in] u Input samples (DPAR1).
 * @param[in] ldu Leading dimension of U.
 * @param[in] y Output samples (DPAR2).
 * @param[in] ldy Leading dimension of Y.
 * @param[in] x Current estimate of parameters.
 * @param[out] nfevl Number of function evaluations.
 * @param[out] e Error vector.
 * @param[out] j Jacobian matrix.
 * @param[in,out] ldj Leading dimension of J.
 * @param[out] dwork Workspace.
 * @param[in] ldwork Length of dwork.
 * @param[out] info Exit code.
 */
void nf01bf(i32 *iflag, i32 nfun, i32 lx, i32 *ipar, i32 lipar, 
            f64 *u, i32 ldu, f64 *y, i32 ldy, f64 *x, 
            i32 *nfevl, f64 *e, f64 *j, i32 *ldj, 
            f64 *dwork, i32 ldwork, i32 *info)
{
    i32 l, m, n, nn, nsmp, st, bsn, jwork;
    bool full;
    f64 err;
    i32 inc = 1;
    f64 neg_one = -1.0;

    l = ipar[1];
    m = ipar[4];
    n = ipar[5];
    if (l == 0) nsmp = nfun;
    else nsmp = nfun / l;

    *info = 0;

    if (*iflag == 1) {
        /* Compute output and error */
        /* ipar[5] is N (passed as IPAR(6) to NF01AD? No, NF01AD uses IPAR(1)=N) */
        /* NF01BF passes &ipar[5] to NF01AD.
           NF01AD expects IPAR(1)=N, IPAR(2)=NN.
           In NF01BF, IPAR(6)=N, IPAR(7)=NN.
           So &ipar[5] is correct.
        */
        nf01ad(nsmp, m, l, &ipar[5], lipar - 5, x, lx, u, ldu, e, nsmp, 
               dwork, ldwork, info);
               
        for (i32 i = 0; i < l; i++) {
            SLC_DAXPY(&nsmp, &neg_one, &y[i * ldy], &inc, &e[i * nsmp], &inc);
        }
        
        /* Workspace reporting */
        nn = ipar[6];
        /* Formula from Fortran code */
        i32 ldac = n + l;
        i32 term1 = ldac * (n + m) + 2 * n;
        i32 jw;
        if (m > 0) jw = (n * ldac > n + m + l) ? n * ldac : n + m + l;
        else jw = (n * ldac > l) ? n * ldac : l;
        
        i32 wsize = nfun + ((2 * nn > term1 + jw) ? 2 * nn : term1 + jw);
        dwork[0] = (f64)wsize;
        
    } else if (*iflag == 2) {
        /* Compute Jacobian */
        const char *cjte = "N";
        /* Calls NF01BD. Passes &ipar[5]. */
        nf01bd(cjte, nsmp, m, l, &ipar[5], lipar - 5, x, lx, u, ldu, 
               e, j, ldj, dwork, dwork, ldwork, info);
               
        *nfevl = ipar[5] * (m + l + 1) + l * m;
        
        nn = ipar[6];
        i32 ldac = n + l;
        i32 term1 = ldac * (n + m) + 2 * n;
        i32 jw;
        if (m > 0) jw = (n * ldac > n + m + l) ? n * ldac : n + m + l;
        else jw = (n * ldac > l) ? n * ldac : l;
        
        i32 wsize = 2 * nfun + ((2 * nn > term1 + jw) ? 2 * nn : term1 + jw);
        dwork[0] = (f64)wsize;
        
    } else if (*iflag == 3) {
        /* Initialization */
        st = ipar[0];
        bsn = ipar[3];
        nn = ipar[6];
        full = (l <= 1 || bsn == 0);
        
        *ldj = nfun;
        ipar[0] = *ldj * (bsn + st);
        
        i32 ldac = n + l;
        if (m > 0) jwork = (n * ldac > n + m + l) ? n * ldac : n + m + l;
        else jwork = (n * ldac > l) ? n * ldac : l;
        
        i32 term1 = (n + l) * (n + m) + 2 * n + jwork;
        ipar[1] = *ldj + (term1 > 2 * nn ? term1 : 2 * nn);
        ipar[2] = *ldj + ipar[1];
        
        jwork = 1;
        if (full) {
            jwork = 4 * lx + 1;
        } else if (bsn > 0) {
            i32 t1 = 3 * bsn + 1;
            jwork = bsn + (t1 > st ? t1 : st);
            if (nsmp > bsn) {
                i32 t2 = 4 * st + 1;
                if (jwork < t2) jwork = t2;
                if (nsmp < 2 * bsn) {
                    i32 t3 = (nsmp - bsn) * (l - 1);
                    if (jwork < t3) jwork = t3;
                }
            }
        }
        ipar[3] = jwork;
        
        if (full) {
            jwork = 4 * lx;
        } else {
            i32 t1 = 2 * bsn;
            if (t1 < 2 * st) t1 = 2 * st;
            jwork = st * (lx - st) + 2 * lx + t1;
        }
        ipar[4] = jwork;
        
    } else if (*iflag == 0) {
        err = SLC_DNRM2(&nfun, e, &inc);
        printf(" Norm of current error = %15.6E\n", err);
    }
}
