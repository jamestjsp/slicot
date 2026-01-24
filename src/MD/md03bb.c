/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"

/**
 * @brief Compute Levenberg-Marquardt parameter for compressed Jacobian.
 *
 * This routine is an interface to SLICOT Library routine MD03BY.
 *
 * @param[in] cond Condition estimation mode ('E', 'N', 'U').
 * @param[in] n Order of matrix R.
 * @param[in] ipar Integer parameters (unused, for compatibility).
 * @param[in] lipar Length of ipar.
 * @param[in,out] r Upper triangular matrix R.
 * @param[in] ldr Leading dimension of R.
 * @param[in] ipvt Permutation matrix P.
 * @param[in] diag Diagonal scaling matrix D.
 * @param[in] qtb First n elements of Q'*b.
 * @param[in] delta Trust region radius.
 * @param[in,out] par Levenberg-Marquardt parameter.
 * @param[in,out] ranks Numerical rank.
 * @param[out] x Least squares solution.
 * @param[out] rx Residual -R*P'*x.
 * @param[in] tol Tolerance for rank estimation.
 * @param[out] dwork Workspace.
 * @param[in] ldwork Length of dwork.
 * @param[out] info Exit code.
 */
void md03bb(const char *cond, i32 n, const i32 *ipar, i32 lipar, f64 *r, 
            i32 ldr, const i32 *ipvt, const f64 *diag, const f64 *qtb, 
            f64 delta, f64 *par, i32 *ranks, f64 *x, f64 *rx, f64 tol, 
            f64 *dwork, i32 ldwork, i32 *info)
{
    /* Call MD03BY */
    /* ranks is array(1), pass address of first element to MD03BY which takes rank pointer?
       MD03BY signature: void md03by(..., i32* rank, ...);
       Here ranks is int*. So passing ranks is correct (as pointer to int).
       Wait, MD03BY takes i32* rank. 
       In Fortran MD03BB calls MD03BY with RANKS(1).
       So we pass &ranks[0] or just ranks.
    */
    
    md03by(cond, n, r, ldr, ipvt, diag, qtb, delta, par, ranks, x, rx, tol, 
           dwork, ldwork, info);
}
