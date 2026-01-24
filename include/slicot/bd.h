/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#ifndef SLICOT_BD_H
#define SLICOT_BD_H

#include "../slicot_types.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Generate benchmark examples for continuous-time dynamical systems.
 *
 * Generates benchmark examples for time-invariant, continuous-time
 * dynamical systems:
 *
 *     E x'(t) = A x(t) + B u(t)
 *       y(t)  = C x(t) + D u(t)
 *
 * E, A are real N-by-N matrices, B is N-by-M, C is P-by-N, and D is P-by-M.
 * In many examples, E is the identity matrix and D is the zero matrix.
 *
 * This routine implements the CTDSX (Continuous-Time Dynamical Systems eXamples)
 * benchmark library described in SLICOT Working Note 1998-9.
 *
 * @param[in] def Parameter initialization mode:
 *               'D' = use default values for parameters
 *               'N' = use user-provided values in DPAR/IPAR
 *               This parameter is not referenced if NR(1) = 1.
 * @param[in] nr Example identifier array of size 2:
 *               nr[0] = group number (1-4):
 *                   1 = parameter-free problems of fixed size
 *                   2 = parameter-dependent problems of fixed size
 *                   3 = parameter-free problems of scalable size
 *                   4 = parameter-dependent problems of scalable size
 *               nr[1] = example number within group
 * @param[in,out] dpar Array of real parameters, size 7.
 *                     For Ex. 2.1, 2.2: dpar[0] = epsilon
 *                     For Ex. 2.4: dpar[0..6] = b, mu, r, r_c, k_l, sigma, a
 *                     For Ex. 2.7: dpar[0..1] = mu, nu
 *                     For Ex. 4.1: dpar[0..6] = a, b, c, beta_1, beta_2, gamma_1, gamma_2
 *                     For Ex. 4.2: dpar[0..2] = mu, delta, kappa
 *                     On exit if DEF='D': overwritten with default values
 * @param[in,out] ipar Array of integer parameters, size 1.
 *                     For Ex. 2.3, 2.5, 2.6: ipar[0] = s
 *                     For Ex. 3.1: ipar[0] = q
 *                     For Ex. 3.2, 3.3: ipar[0] = n
 *                     For Ex. 3.4: ipar[0] = l
 *                     For Ex. 4.1: ipar[0] = n
 *                     For Ex. 4.2: ipar[0] = l
 *                     On exit if DEF='D': overwritten with default values
 * @param[out] vec Boolean output flags, size 8:
 *                 vec[0..2] = N, M, P available (always true)
 *                 vec[3] = E is NOT identity matrix
 *                 vec[4..6] = A, B, C available (always true)
 *                 vec[7] = D is NOT zero matrix
 * @param[out] n State dimension (order of E and A)
 * @param[out] m Number of inputs (columns of B and D)
 * @param[out] p Number of outputs (rows of C and D)
 * @param[out] e Matrix E, dimension (lde,n). If vec[3]=false, contains I.
 * @param[in] lde Leading dimension of E (lde >= n)
 * @param[out] a State matrix A, dimension (lda,n)
 * @param[in] lda Leading dimension of A (lda >= n)
 * @param[out] b Input matrix B, dimension (ldb,m)
 * @param[in] ldb Leading dimension of B (ldb >= n)
 * @param[out] c Output matrix C, dimension (ldc,n)
 * @param[in] ldc Leading dimension of C (ldc >= p)
 * @param[out] d Feedthrough matrix D, dimension (ldd,m). If vec[7]=false, contains 0.
 * @param[in] ldd Leading dimension of D (ldd >= p)
 * @param[out] note String describing the chosen example (up to 70 chars)
 * @param[out] dwork Workspace array, dimension (ldwork)
 * @param[in] ldwork Size of workspace:
 *                   For Example 3.4: ldwork >= 4*ipar[0]
 *                   For other examples: ldwork >= 1
 * @param[out] info Error indicator:
 *                  0 = success
 *                  1 = data file required but not available
 *                  <0 = -i means i-th argument has illegal value
 *                  -3 = invalid DPAR value
 *                  -4 = invalid IPAR value
 */
void bd01ad(const char* def, const i32* nr, f64* dpar, i32* ipar,
            bool* vec, i32* n, i32* m, i32* p,
            f64* e, const i32 lde, f64* a, const i32 lda,
            f64* b, const i32 ldb, f64* c, const i32 ldc,
            f64* d, const i32 ldd, char* note,
            f64* dwork, const i32 ldwork, i32* info);

/**
 * @brief Generate benchmark examples for discrete-time dynamical systems.
 *
 * Generates benchmark examples for time-invariant, discrete-time
 * dynamical systems:
 *
 *     E x_{k+1} = A x_k + B u_k
 *           y_k = C x_k + D u_k
 *
 * E, A are real N-by-N matrices, B is N-by-M, C is P-by-N, and D is P-by-M.
 * In many examples, E is the identity matrix and D is the zero matrix.
 *
 * This routine implements the DTDSX (Discrete-Time Dynamical Systems eXamples)
 * benchmark library described in SLICOT Working Note 1998-10.
 *
 * @param[in] def Parameter initialization mode:
 *               'D' = use default values for parameters
 *               'N' = use user-provided values in DPAR/IPAR
 *               This parameter is not referenced if NR(1) = 1.
 * @param[in] nr Example identifier array of size 2:
 *               nr[0] = group number (1-4):
 *                   1 = parameter-free problems of fixed size
 *                   2 = parameter-dependent problems of fixed size
 *                   3 = parameter-free problems of scalable size
 *                   4 = parameter-dependent problems of scalable size
 *               nr[1] = example number within group
 * @param[in,out] dpar Array of real parameters, size 7.
 *                     For Ex. 2.1: dpar[0..2] = tau, delta, K
 *                     On exit if DEF='D': overwritten with default values
 * @param[in,out] ipar Array of integer parameters, size 1.
 *                     For Ex. 3.1: ipar[0] = n (>= 2)
 *                     On exit if DEF='D': overwritten with default values
 * @param[out] vec Boolean output flags, size 8:
 *                 vec[0..2] = N, M, P available (always true)
 *                 vec[3] = E is NOT identity matrix
 *                 vec[4..6] = A, B, C available (always true)
 *                 vec[7] = D is NOT zero matrix
 * @param[out] n State dimension (order of E and A)
 * @param[out] m Number of inputs (columns of B and D)
 * @param[out] p Number of outputs (rows of C and D)
 * @param[out] e Matrix E, dimension (lde,n). If vec[3]=false, contains I.
 * @param[in] lde Leading dimension of E (lde >= n)
 * @param[out] a State matrix A, dimension (lda,n)
 * @param[in] lda Leading dimension of A (lda >= n)
 * @param[out] b Input matrix B, dimension (ldb,m)
 * @param[in] ldb Leading dimension of B (ldb >= n)
 * @param[out] c Output matrix C, dimension (ldc,n)
 * @param[in] ldc Leading dimension of C (ldc >= p)
 * @param[out] d Feedthrough matrix D, dimension (ldd,m). If vec[7]=false, contains 0.
 * @param[in] ldd Leading dimension of D (ldd >= p)
 * @param[out] note String describing the chosen example (up to 70 chars)
 * @param[out] dwork Workspace array, dimension (ldwork)
 * @param[in] ldwork Size of workspace (ldwork >= 1)
 * @param[out] info Error indicator:
 *                  0 = success
 *                  1 = data file required but not available
 *                  <0 = -i means i-th argument has illegal value
 *                  -3 = invalid DPAR value
 *                  -4 = invalid IPAR value
 */
void bd02ad(const char* def, const i32* nr, f64* dpar, i32* ipar,
            bool* vec, i32* n, i32* m, i32* p,
            f64* e, const i32 lde, f64* a, const i32 lda,
            f64* b, const i32 ldb, f64* c, const i32 ldc,
            f64* d, const i32 ldd, char* note,
            f64* dwork, const i32 ldwork, i32* info);

#ifdef __cplusplus
}
#endif

#endif /* SLICOT_BD_H */
