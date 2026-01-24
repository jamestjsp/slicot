/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#ifndef SLICOT_BB_H
#define SLICOT_BB_H

#include "../slicot_types.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Generate benchmark examples for continuous-time algebraic Riccati equations.
 *
 * Generates benchmark examples for numerical solution of continuous-time
 * algebraic Riccati equations (CAREs) of the form:
 *     0 = Q + A'X + XA - XGX
 *
 * The symmetric matrices G and Q may be given in factored form:
 *     G = B * R^{-1} * B'
 *     Q = C' * W * C
 *
 * Examples from CAREX collection (Kenney/Laub/Wette 1989).
 *
 * @param[in] def Parameter initialization mode:
 *               'D' = use default values for parameters
 *               'N' = use user-provided values in DPAR/IPAR
 * @param[in] nr Example identifier array of size 2:
 *               nr[0] = group number (1-4)
 *               nr[1] = example number within group
 * @param[in,out] dpar Array of real parameters, size 7.
 *                     On entry: user values if DEF='N'. On exit: actual values used.
 * @param[in,out] ipar Array of integer parameters, size 4.
 *                     ipar[0] = N (problem size), ipar[1] = M, ipar[2] = P
 *                     On entry: user values if DEF='N'. On exit: actual values used.
 * @param[in] bpar Array of boolean parameters, size 6:
 *                 bpar[0] = true: compute G, false: return B and R factors
 *                 bpar[1] = true: G in full storage, false: packed storage
 *                 bpar[2] = true: G in upper packed, false: lower packed
 *                 bpar[3] = true: compute Q, false: return C and W factors
 *                 bpar[4] = true: Q in full storage, false: packed storage
 *                 bpar[5] = true: Q in upper packed, false: lower packed
 * @param[out] chpar Character string describing the example (max 255 chars)
 * @param[out] vec Boolean output flags, size 9:
 *                 vec[0..3] = A, (B or G), (C or Q), (R or W) available
 *                 vec[4] = true if R is identity (G = B*B')
 *                 vec[5] = true if W is identity (Q = C'*C)
 *                 vec[6..7] = reserved
 *                 vec[8] = true if solution X is available
 * @param[out] n Problem dimension (state size)
 * @param[out] m Number of inputs (columns of B)
 * @param[out] p Number of outputs (rows of C)
 * @param[out] a State matrix A, dimension (lda,n)
 * @param[in] lda Leading dimension of A (lda >= n)
 * @param[out] b Input matrix B, dimension (ldb,m) [or unused if bpar[0]=true]
 * @param[in] ldb Leading dimension of B (ldb >= n)
 * @param[out] c Output matrix C, dimension (ldc,n) [or unused if bpar[3]=true]
 * @param[in] ldc Leading dimension of C (ldc >= p)
 * @param[out] g Matrix G = B*R^{-1}*B', dimension depends on storage:
 *               - If bpar[0]=true: full (ldg,n) or packed n*(n+1)/2
 *               - If bpar[0]=false: contains R^{-1} in packed m*(m+1)/2
 * @param[in] ldg Leading dimension of G (ldg >= n if full storage)
 * @param[out] q Matrix Q = C'*W*C, dimension depends on storage:
 *               - If bpar[3]=true: full (ldq,n) or packed n*(n+1)/2
 *               - If bpar[3]=false: contains W in packed p*(p+1)/2
 * @param[in] ldq Leading dimension of Q (ldq >= n if full storage)
 * @param[out] x Solution matrix X, dimension (ldx,n) [if vec[8]=true]
 * @param[in] ldx Leading dimension of X (ldx >= n if solution available)
 * @param[out] dwork Workspace array, dimension (ldwork)
 * @param[in] ldwork Size of workspace (ldwork >= n*max(n,4))
 * @param[out] info Error indicator:
 *                  0 = success
 *                  1 = example not implemented (requires external data file)
 *                  2 = invalid parameter (e.g., division by zero)
 *                  3 = R matrix is singular
 *                  <0 = -i means i-th argument has illegal value
 */
void bb01ad(const char* def, const i32* nr, f64* dpar, i32* ipar,
            const bool* bpar, char* chpar, bool* vec, i32* n, i32* m, i32* p,
            f64* a, const i32 lda, f64* b, const i32 ldb,
            f64* c, const i32 ldc, f64* g, const i32 ldg,
            f64* q, const i32 ldq, f64* x, const i32 ldx,
            f64* dwork, const i32 ldwork, i32* info);

/**
 * @brief Generate benchmark examples for discrete-time algebraic Riccati equations.
 *
 * Generates benchmark examples for numerical solution of discrete-time
 * algebraic Riccati equations (DAREs) of the form:
 *     0 = A^T X A - X - (A^T X B + S)(R + B^T X B)^{-1}(B^T X A + S^T) + Q
 *
 * The symmetric matrices Q and R may be given in factored form:
 *     Q = C^T * Q0 * C
 * and if R is nonsingular and S = 0:
 *     G = B * R^{-1} * B^T
 *
 * Examples from DAREX collection (Benner/Laub/Mehrmann 1995).
 *
 * @param[in] def Parameter initialization mode:
 *               'D' = use default values for parameters
 *               'N' = use user-provided values in DPAR/IPAR
 * @param[in] nr Example identifier array of size 2:
 *               nr[0] = group number (1-4)
 *               nr[1] = example number within group
 * @param[in,out] dpar Array of real parameters, size 4.
 *                     On entry: user values if DEF='N'. On exit: actual values used.
 * @param[in,out] ipar Array of integer parameters, size 3.
 *                     ipar[0] = N (problem size), ipar[1] = M, ipar[2] = P
 *                     On entry: user values if DEF='N'. On exit: actual values used.
 * @param[in] bpar Array of boolean parameters, size 7:
 *                 bpar[0] = true: compute Q, false: return C and Q0 factors
 *                 bpar[1] = true: Q in full storage, false: packed storage
 *                 bpar[2] = true: Q in upper packed, false: lower packed
 *                 bpar[3] = true: compute G, false: return B and R factors
 *                 bpar[4] = true: R/G in full storage, false: packed storage
 *                 bpar[5] = true: R/G in upper packed, false: lower packed
 *                 bpar[6] = true: return S matrix, false: don't return S
 * @param[out] chpar Character string describing the example (max 255 chars)
 * @param[out] vec Boolean output flags, size 10:
 *                 vec[0..2] = N, M, P always available
 *                 vec[3] = A always available
 *                 vec[4] = B and R available if bpar[3]=false
 *                 vec[5] = C and Q0 available if bpar[0]=false
 *                 vec[6] = Q always available
 *                 vec[7] = R/G always available
 *                 vec[8] = S available if bpar[6]=true
 *                 vec[9] = X (solution) available for some examples
 * @param[out] n Problem dimension (state size)
 * @param[out] m Number of inputs (columns of B)
 * @param[out] p Number of outputs (rows of C)
 * @param[out] a State matrix A, dimension (lda,n)
 * @param[in] lda Leading dimension of A (lda >= n)
 * @param[out] b Input matrix B, dimension (ldb,m) [or used as workspace if bpar[3]=true]
 * @param[in] ldb Leading dimension of B (ldb >= n)
 * @param[out] c Output matrix C, dimension (ldc,n) [or used as workspace if bpar[0]=true]
 * @param[in] ldc Leading dimension of C (ldc >= p)
 * @param[out] q Matrix Q (or Q0 if factored), dimension depends on storage
 * @param[in] ldq Leading dimension of Q (ldq >= n if full storage)
 * @param[out] r Matrix R (or G if computed), dimension depends on storage
 * @param[in] ldr Leading dimension of R (ldr >= m or n depending on bpar[3])
 * @param[out] s Coefficient matrix S, dimension (lds,m) if bpar[6]=true
 * @param[in] lds Leading dimension of S (lds >= n if bpar[6]=true)
 * @param[out] x Solution matrix X, dimension (ldx,n) [if vec[9]=true]
 * @param[in] ldx Leading dimension of X (ldx >= n if solution available)
 * @param[out] dwork Workspace array, dimension (ldwork)
 * @param[in] ldwork Size of workspace (ldwork >= n*n)
 * @param[out] info Error indicator:
 *                  0 = success
 *                  1 = example requires external data file (not implemented)
 *                  2 = division by zero (invalid parameter)
 *                  3 = R matrix is singular
 *                  <0 = -i means i-th argument has illegal value
 */
void bb02ad(const char* def, const i32* nr, f64* dpar, i32* ipar,
            const bool* bpar, char* chpar, bool* vec, i32* n, i32* m, i32* p,
            f64* a, const i32 lda, f64* b, const i32 ldb,
            f64* c, const i32 ldc, f64* q, const i32 ldq,
            f64* r, const i32 ldr, f64* s, const i32 lds,
            f64* x, const i32 ldx, f64* dwork, const i32 ldwork, i32* info);

/**
 * @brief Generate benchmark examples for continuous-time Lyapunov equations.
 *
 * Generates benchmark examples of (generalized) continuous-time Lyapunov
 * equations:
 *     A^T X E + E^T X A = Y
 *
 * In some examples, the right hand side has the form Y = -B^T B
 * and the solution can be represented as X = U^T U.
 *
 * E, A, Y, X, and U are real N-by-N matrices, and B is M-by-N.
 * Note that E can be the identity matrix. For some examples, B, X, or U
 * are not provided.
 *
 * Implements the CTLEX benchmark collection (Kressner/Mehrmann/Penzl 1999).
 *
 * @param[in] def Parameter initialization mode:
 *               'D' = use default values for parameters
 *               'N' = use user-provided values in DPAR/IPAR
 * @param[in] nr Example identifier array of size 2:
 *               nr[0] = group number (only 4 is supported):
 *                   4 = parameter-dependent problems of scalable size
 *               nr[1] = example number within group (1-4 for group 4)
 * @param[in,out] dpar Array of real parameters, size 2.
 *                     For Example 4.1: dpar[0]=r, dpar[1]=s (both > 1)
 *                     For Example 4.2: dpar[0]=lambda (<0), dpar[1]=s (>1)
 *                     For Examples 4.3, 4.4: dpar[0]=t
 * @param[in,out] ipar Array of integer parameters, size 1.
 *                     For Examples 4.1-4.3: ipar[0]=n (>=2)
 *                     For Example 4.4: ipar[0]=q (n=3*q)
 * @param[out] vec Boolean output flags, size 8:
 *                 vec[0] = N available (always true)
 *                 vec[1] = M available (always true)
 *                 vec[2] = E is NOT identity (true if generalized)
 *                 vec[3] = A available (always true)
 *                 vec[4] = Y available (always true)
 *                 vec[5] = B provided
 *                 vec[6] = X (solution) provided
 *                 vec[7] = U (Cholesky factor) provided
 * @param[out] n Problem dimension (order of E, A)
 * @param[out] m Number of rows in B (0 if B not provided)
 * @param[out] e Matrix E, dimension (lde,n)
 * @param[in] lde Leading dimension of E (lde >= n)
 * @param[out] a Matrix A, dimension (lda,n)
 * @param[in] lda Leading dimension of A (lda >= n)
 * @param[out] y Right-hand side matrix Y, dimension (ldy,n)
 * @param[in] ldy Leading dimension of Y (ldy >= n)
 * @param[out] b Matrix B, dimension (ldb,n) (if vec[5]=true)
 * @param[in] ldb Leading dimension of B (ldb >= m)
 * @param[out] x Solution matrix X, dimension (ldx,n) (if vec[6]=true)
 * @param[in] ldx Leading dimension of X (ldx >= n)
 * @param[out] u Cholesky factor U, dimension (ldu,n) (if vec[7]=true)
 * @param[in] ldu Leading dimension of U (ldu >= n)
 * @param[out] note String describing the chosen example (up to 70 chars)
 * @param[out] dwork Workspace array, dimension (ldwork)
 * @param[in] ldwork Size of workspace:
 *                   For Examples 4.1, 4.2: ldwork >= 2*ipar[0]
 *                   For other examples: ldwork >= 1
 * @param[out] info Error indicator:
 *                  0 = success
 *                  <0 = -i means i-th argument has illegal value
 *                  -3 = invalid DPAR value
 *                  -4 = invalid IPAR value
 */
void bb03ad(const char* def, const i32* nr, f64* dpar, i32* ipar,
            bool* vec, i32* n, i32* m,
            f64* e, const i32 lde, f64* a, const i32 lda,
            f64* y, const i32 ldy, f64* b, const i32 ldb,
            f64* x, const i32 ldx, f64* u, const i32 ldu,
            char* note, f64* dwork, const i32 ldwork, i32* info);

/**
 * @brief Generate benchmark examples for discrete-time Lyapunov equations.
 *
 * Generates benchmark examples of (generalized) discrete-time Lyapunov
 * equations:
 *     A^T X A - E^T X E = Y
 *
 * In some examples, the right hand side has the form Y = -B^T B
 * and the solution can be represented as X = U^T U.
 *
 * E, A, Y, X, and U are real N-by-N matrices, and B is M-by-N.
 * Note that E can be the identity matrix. For some examples, B, X, or U
 * are not provided.
 *
 * Implements the DTLEX benchmark collection (Kressner/Mehrmann/Penzl 1999).
 *
 * @param[in] def Parameter initialization mode:
 *               'D' = use default values for parameters
 *               'N' = use user-provided values in DPAR/IPAR
 * @param[in] nr Example identifier array of size 2:
 *               nr[0] = group number (only 4 is supported):
 *                   4 = parameter-dependent problems of scalable size
 *               nr[1] = example number within group (1-4 for group 4)
 * @param[in,out] dpar Array of real parameters, size 2.
 *                     For Example 4.1: dpar[0]=r, dpar[1]=s (both > 1)
 *                     For Example 4.2: dpar[0]=lambda (in (-1,1)), dpar[1]=s (>1)
 *                     For Examples 4.3, 4.4: dpar[0]=t
 * @param[in,out] ipar Array of integer parameters, size 1.
 *                     For Examples 4.1-4.3: ipar[0]=n (>=2)
 *                     For Example 4.4: ipar[0]=q (n=3*q)
 * @param[out] vec Boolean output flags, size 8:
 *                 vec[0] = N available (always true)
 *                 vec[1] = M available (always true)
 *                 vec[2] = E is NOT identity (true if generalized)
 *                 vec[3] = A available (always true)
 *                 vec[4] = Y available (always true)
 *                 vec[5] = B provided
 *                 vec[6] = X (solution) provided
 *                 vec[7] = U (Cholesky factor) provided
 * @param[out] n Problem dimension (order of E, A)
 * @param[out] m Number of rows in B (0 if B not provided)
 * @param[out] e Matrix E, dimension (lde,n)
 * @param[in] lde Leading dimension of E (lde >= n)
 * @param[out] a Matrix A, dimension (lda,n)
 * @param[in] lda Leading dimension of A (lda >= n)
 * @param[out] y Right-hand side matrix Y, dimension (ldy,n)
 * @param[in] ldy Leading dimension of Y (ldy >= n)
 * @param[out] b Matrix B, dimension (ldb,n) (if vec[5]=true)
 * @param[in] ldb Leading dimension of B (ldb >= m)
 * @param[out] x Solution matrix X, dimension (ldx,n) (if vec[6]=true)
 * @param[in] ldx Leading dimension of X (ldx >= n)
 * @param[out] u Cholesky factor U, dimension (ldu,n) (if vec[7]=true)
 * @param[in] ldu Leading dimension of U (ldu >= n)
 * @param[out] note String describing the chosen example (up to 70 chars)
 * @param[out] dwork Workspace array, dimension (ldwork)
 * @param[in] ldwork Size of workspace:
 *                   For Examples 4.1, 4.2: ldwork >= 2*ipar[0]
 *                   For other examples: ldwork >= 1
 * @param[out] info Error indicator:
 *                  0 = success
 *                  <0 = -i means i-th argument has illegal value
 *                  -3 = invalid DPAR value
 *                  -4 = invalid IPAR value
 */
void bb04ad(const char* def, const i32* nr, f64* dpar, i32* ipar,
            bool* vec, i32* n, i32* m,
            f64* e, const i32 lde, f64* a, const i32 lda,
            f64* y, const i32 ldy, f64* b, const i32 ldb,
            f64* x, const i32 ldx, f64* u, const i32 ldu,
            char* note, f64* dwork, const i32 ldwork, i32* info);

#ifdef __cplusplus
}
#endif

#endif /* SLICOT_BB_H */
