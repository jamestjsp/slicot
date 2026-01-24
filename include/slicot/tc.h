/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#ifndef SLICOT_TC_H
#define SLICOT_TC_H

#include "../slicot_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Find dual polynomial matrix representation.
 *
 * Finds the dual right (left) polynomial matrix representation of a given
 * left (right) polynomial matrix representation, where the representations
 * are of the form Q(s)*inv(P(s)) and inv(P(s))*Q(s) respectively.
 *
 * The dual is found by transposing both the numerator matrix Q(s) and
 * denominator matrix P(s) for each polynomial coefficient.
 *
 * @param[in] leri Specifies input representation type:
 *                 'L' = left matrix fraction input
 *                 'R' = right matrix fraction input
 * @param[in] m Number of system inputs (m >= 0)
 * @param[in] p Number of system outputs (p >= 0)
 * @param[in] indlim Number of polynomial coefficient slices (indlim >= 1)
 *                   Equal to kpcoef + 1 where kpcoef is max polynomial degree
 * @param[in,out] pcoeff Denominator polynomial matrix coefficients
 *                       Dimension (ldpco1, ldpco2, indlim)
 *                       Leading porm-by-porm-by-indlim part used
 *                       where porm = P if LERI='L', porm = M if LERI='R'
 *                       PCOEFF(i,j,k) is coefficient in s^(indlim-k) of P(i,j)
 *                       On exit: transposed coefficients P'(s)
 * @param[in] ldpco1 Leading dimension of PCOEFF (>= max(1,P) if LERI='L',
 *                   >= max(1,M) if LERI='R')
 * @param[in] ldpco2 Second dimension of PCOEFF (same constraints as ldpco1)
 * @param[in,out] qcoeff Numerator polynomial matrix coefficients
 *                       Dimension (ldqco1, ldqco2, indlim)
 *                       Leading P-by-M-by-indlim part on input
 *                       QCOEFF(i,j,k) is coefficient in s^(indlim-k) of Q(i,j)
 *                       On exit: leading M-by-P-by-indlim contains Q'(s)
 * @param[in] ldqco1 Leading dimension of QCOEFF (>= max(1, m, p))
 * @param[in] ldqco2 Second dimension of QCOEFF (>= max(1, m, p))
 * @param[out] info Exit code: 0 = success, <0 = parameter -info invalid
 */
void tc01od(
    const char leri,
    const i32 m,
    const i32 p,
    const i32 indlim,
    f64* pcoeff,
    const i32 ldpco1,
    const i32 ldpco2,
    f64* qcoeff,
    const i32 ldqco1,
    const i32 ldqco2,
    i32* info
);

/**
 * @brief Convert polynomial matrix representation to state-space.
 *
 * Finds a state-space representation (A,B,C,D) with the same transfer
 * matrix T(s) as that of a given left or right polynomial matrix
 * representation, i.e.
 *
 *    C*inv(sI-A)*B + D = T(s) = inv(P(s))*Q(s) = Q(s)*inv(P(s))
 *
 * Uses Wolovich's Observable Structure Theorem to construct observable
 * companion form. For right matrix fractions, converts via duality.
 *
 * @param[in] leri 'L' for left PMR inv(P(s))*Q(s), 'R' for right PMR Q(s)*inv(P(s))
 * @param[in] m Number of system inputs (m >= 0)
 * @param[in] p Number of system outputs (p >= 0)
 * @param[in] index Array of row degrees (left) or column degrees (right)
 *                  Dimension: max(m,p). For left: INDEX(i) = max degree in row i of P(s)
 *                  For right: INDEX(i) = max degree in column i of P(s)
 * @param[in,out] pcoeff Denominator polynomial coefficients, dimension (ldpco1, ldpco2, kpcoef)
 *                       where kpcoef = max(INDEX) + 1
 *                       PCOEFF(i,j,k) = coeff in s^(INDEX(iorj)-k+1) of P(i,j)
 *                       iorj = i for left, j for right
 *                       For right PMR: modified and restored on exit
 * @param[in] ldpco1 Leading dimension of PCOEFF (>= max(1,p) if left, >= max(1,m) if right)
 * @param[in] ldpco2 Second dimension of PCOEFF (same as ldpco1)
 * @param[in,out] qcoeff Numerator polynomial coefficients, dimension (ldqco1, ldqco2, kpcoef)
 *                       For right PMR: modified and restored on exit
 * @param[in] ldqco1 Leading dimension of QCOEFF
 *                   >= max(1,p) if left, >= max(1,m,p) if right
 * @param[in] ldqco2 Second dimension of QCOEFF
 *                   >= max(1,m) if left, >= max(1,m,p) if right
 * @param[out] n Order of resulting state-space representation (sum of INDEX values)
 * @param[out] rcond Reciprocal condition number of leading coefficient matrix of P(s)
 *                   If nearly zero, P(s) is nearly row/column non-proper
 * @param[out] a State dynamics matrix, dimension (lda, n)
 * @param[in] lda Leading dimension of A (>= max(1,n))
 * @param[out] b Input matrix, dimension (ldb, max(m,p))
 *               Leading n-by-m part contains B; rest is workspace
 * @param[in] ldb Leading dimension of B (>= max(1,n))
 * @param[out] c Output matrix, dimension (ldc, n)
 *               Leading p-by-n part contains C; rest is workspace
 * @param[in] ldc Leading dimension of C (>= max(1,m,p))
 * @param[out] d Direct transmission matrix, dimension (ldd, max(m,p))
 *               Leading p-by-m part contains D; rest is workspace
 * @param[in] ldd Leading dimension of D (>= max(1,m,p))
 * @param[out] iwork Integer workspace, dimension (2*max(m,p))
 * @param[out] dwork Double workspace, dimension (ldwork)
 *                   On exit dwork[0] = optimal ldwork
 * @param[in] ldwork Workspace size (>= max(1, max(m,p)*(max(m,p)+4)))
 * @param[out] info Exit code: 0 = success, <0 = -info invalid param,
 *                  1 = P(s) not row/column proper
 */
void tc04ad(
    const char leri,
    const i32 m,
    const i32 p,
    const i32* index,
    f64* pcoeff,
    const i32 ldpco1,
    const i32 ldpco2,
    f64* qcoeff,
    const i32 ldqco1,
    const i32 ldqco2,
    i32* n,
    f64* rcond,
    f64* a,
    const i32 lda,
    f64* b,
    const i32 ldb,
    f64* c,
    const i32 ldc,
    f64* d,
    const i32 ldd,
    i32* iwork,
    f64* dwork,
    const i32 ldwork,
    i32* info
);

/**
 * @brief Evaluate transfer matrix at complex frequency.
 *
 * Evaluates the transfer matrix T(s) of a left polynomial matrix
 * representation [T(s) = inv(P(s))*Q(s)] or a right polynomial matrix
 * representation [T(s) = Q(s)*inv(P(s))] at a specified complex frequency
 * s = SVAL.
 *
 * For standard frequency response, supply SVAL as (0, omega) for frequency omega.
 *
 * @param[in] leri 'L' for left PMR inv(P(s))*Q(s), 'R' for right PMR Q(s)*inv(P(s))
 * @param[in] m Number of system inputs (m >= 0)
 * @param[in] p Number of system outputs (p >= 0)
 * @param[in] sval Complex frequency at which to evaluate
 * @param[in] index Array of polynomial degrees, dimension max(m,p)
 *                  For left: INDEX(i) = max degree in row i of P(s)
 *                  For right: INDEX(i) = max degree in column i of P(s)
 * @param[in,out] pcoeff Denominator polynomial coefficients, dimension (ldpco1, ldpco2, kpcoef)
 *                       where kpcoef = max(INDEX) + 1
 *                       For right PMR: modified and restored on exit
 * @param[in] ldpco1 Leading dimension of PCOEFF (>= max(1,p) if left, >= max(1,m) if right)
 * @param[in] ldpco2 Second dimension of PCOEFF (same as ldpco1)
 * @param[in,out] qcoeff Numerator polynomial coefficients, dimension (ldqco1, ldqco2, kpcoef)
 *                       For right PMR: modified and restored on exit
 * @param[in] ldqco1 Leading dimension of QCOEFF
 *                   >= max(1,p) if left, >= max(1,m,p) if right
 * @param[in] ldqco2 Second dimension of QCOEFF
 *                   >= max(1,m) if left, >= max(1,m,p) if right
 * @param[out] rcond Reciprocal condition number of P(SVAL)
 *                   If nearly zero, SVAL is approximately a system pole
 * @param[out] cfreqr Complex frequency response matrix T(SVAL)
 *                    Dimension (ldcfre, max(m,p)), leading p-by-m part used
 * @param[in] ldcfre Leading dimension of CFREQR
 *                   >= max(1,p) if left, >= max(1,m,p) if right
 * @param[out] iwork Integer workspace, dimension P if left, M if right
 * @param[out] dwork Double workspace, dimension 2*P if left, 2*M if right
 * @param[out] zwork Complex workspace, dimension P*(P+2) if left, M*(M+2) if right
 * @param[out] info Exit code: 0 = success, <0 = -info invalid param,
 *                  1 = P(SVAL) exactly or nearly singular
 */
void tc05ad(
    const char leri,
    const i32 m,
    const i32 p,
    const c128 sval,
    const i32* index,
    f64* pcoeff,
    const i32 ldpco1,
    const i32 ldpco2,
    f64* qcoeff,
    const i32 ldqco1,
    const i32 ldqco2,
    f64* rcond,
    c128* cfreqr,
    const i32 ldcfre,
    i32* iwork,
    f64* dwork,
    c128* zwork,
    i32* info
);

#ifdef __cplusplus
}
#endif

#endif /* SLICOT_TC_H */
