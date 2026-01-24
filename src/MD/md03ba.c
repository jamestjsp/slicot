/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"

/**
 * @brief QR factorization with column pivoting for Levenberg-Marquardt.
 *
 * This routine is an interface to SLICOT Library routine MD03BX.
 *
 * @param[in] n Number of columns of Jacobian matrix J.
 * @param[in] ipar Integer parameters. ipar[0] must contain M (rows of J).
 * @param[in] lipar Length of ipar.
 * @param[in] fnorm Euclidean norm of error vector e.
 * @param[in,out] j Jacobian matrix (M x N). On exit, upper triangular R.
 * @param[in,out] ldj Leading dimension of J.
 * @param[in,out] e Error vector (M). On exit, Q'*e.
 * @param[out] jnorms Euclidean norms of columns of J.
 * @param[out] gnorm 1-norm of scaled gradient.
 * @param[out] ipvt Permutation matrix P.
 * @param[out] dwork Workspace.
 * @param[in] ldwork Length of dwork.
 * @param[out] info Exit code.
 */
void md03ba(i32 n, const i32 *ipar, i32 lipar, f64 fnorm, f64 *j, i32 *ldj, 
            f64 *e, f64 *jnorms, f64 *gnorm, i32 *ipvt, f64 *dwork, 
            i32 ldwork, i32 *info)
{
    i32 m;
    
    if (lipar < 1) {
        *info = -3;
        return;
    }
    
    m = ipar[0];
    
    /* Call MD03BX */
    md03bx(m, n, fnorm, j, ldj, e, jnorms, gnorm, ipvt, dwork, ldwork, info);
}
