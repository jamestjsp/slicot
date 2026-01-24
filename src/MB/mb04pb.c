/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 *
 * MB04PB - Reduce Hamiltonian matrix to PVL form (blocked version)
 *
 * Reduces a Hamiltonian matrix H = [[A, G], [Q, -A^T]] where G, Q symmetric
 * to Paige/Van Loan form using orthogonal symplectic U such that
 * U^T H U has upper Hessenberg A and diagonal Q.
 *
 * Note: This implementation delegates to mb04pu (unblocked version).
 * The blocked algorithm is complex and requires UE01MD which is not available.
 */

#include "slicot.h"
#include "slicot_blas.h"

void mb04pb(i32 n, i32 ilo,
            f64 *a, i32 lda,
            f64 *qg, i32 ldqg,
            f64 *cs, f64 *tau,
            f64 *dwork, i32 ldwork,
            i32 *info)
{
    mb04pu(n, ilo, a, lda, qg, ldqg, cs, tau, dwork, ldwork, info);
}
