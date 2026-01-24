/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#ifndef SLICOT_MB05_H
#define SLICOT_MB05_H

#include "../slicot_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Matrix exponential for a real non-defective matrix.
 *
 * Computes exp(A*delta) where A is a real N-by-N non-defective matrix
 * with real or complex eigenvalues and delta is a scalar value.
 *
 * Uses eigenvalue/eigenvector decomposition technique (Moler-Van Loan "Method 15").
 * The routine returns eigenvalues, eigenvectors of A, and intermediate matrix Y
 * such that exp(A*delta) = V*Y.
 *
 * Optionally computes a balancing transformation to improve the conditioning
 * of the eigenvalues and eigenvectors.
 *
 * @param[in] balanc 'N' = do not scale
 *                   'S' = diagonally scale (D*A*D^(-1))
 * @param[in] n Order of matrix A (n >= 0)
 * @param[in] delta Scalar value delta in exp(A*delta)
 * @param[in,out] a N-by-N matrix A, dimension (lda,n)
 *                  On exit: contains exp(A*delta)
 * @param[in] lda Leading dimension of A (lda >= max(1,n))
 * @param[out] v N-by-N eigenvector matrix, dimension (ldv,n)
 * @param[in] ldv Leading dimension of V (ldv >= max(1,n))
 * @param[out] y N-by-N intermediate result, dimension (ldy,n)
 *               exp(A*delta) = V*Y
 * @param[in] ldy Leading dimension of Y (ldy >= max(1,n))
 * @param[out] valr Real parts of eigenvalues, dimension (n)
 * @param[out] vali Imaginary parts of eigenvalues, dimension (n)
 * @param[out] iwork Integer workspace, dimension (n)
 * @param[out] dwork Double workspace, dimension (ldwork)
 *                   On exit: dwork[0] = optimal ldwork
 *                   dwork[1] = reciprocal condition number
 * @param[in] ldwork Workspace size (ldwork >= max(1,4*n))
 * @param[out] info Exit code:
 *                  0 = success
 *                  < 0 = invalid parameter -info
 *                  1..n = QR algorithm failed, eigenvalues info+1:n converged
 *                  n+1 = eigenvector matrix is singular
 *                  n+2 = matrix A is defective (near-singular eigenvectors)
 */
void mb05md(
    const char* balanc,
    const i32 n,
    const f64 delta,
    f64* a,
    const i32 lda,
    f64* v,
    const i32 ldv,
    f64* y,
    const i32 ldy,
    f64* valr,
    f64* vali,
    i32* iwork,
    f64* dwork,
    const i32 ldwork,
    i32* info
);

/**
 * @brief Schur form, eigenvalues, and right eigenvectors.
 *
 * Computes for an N-by-N real nonsymmetric matrix A:
 *   - Orthogonal matrix Q reducing A to real Schur form T
 *   - Eigenvalues (WR + i*WI)
 *   - Right eigenvectors R of T (upper triangular by construction)
 *
 * The right eigenvector r(j) of T satisfies: T * r(j) = lambda(j) * r(j)
 *
 * @param[in] balanc 'N' = no scaling, 'S' = diagonal scaling
 * @param[in] n Order of matrix A (n >= 0)
 * @param[in,out] a N-by-N matrix A, dimension (lda,n)
 *                  On exit: real Schur form T
 * @param[in] lda Leading dimension of A (lda >= max(1,n))
 * @param[out] wr Real parts of eigenvalues, dimension (n)
 * @param[out] wi Imaginary parts of eigenvalues, dimension (n)
 * @param[out] r N-by-N upper triangular matrix of right eigenvectors, dimension (ldr,n)
 * @param[in] ldr Leading dimension of R (ldr >= max(1,n))
 * @param[out] q N-by-N orthogonal matrix Q, dimension (ldq,n)
 * @param[in] ldq Leading dimension of Q (ldq >= max(1,n))
 * @param[out] dwork Workspace, dimension (ldwork)
 *                   On exit: dwork[0] = optimal ldwork
 *                   If balanc='S' and ldwork>0: dwork[1..n] = scaling factors
 * @param[in] ldwork Workspace size (ldwork >= max(1,4*n))
 *                   If ldwork=-1: workspace query
 * @param[out] info Exit code:
 *                  0 = success
 *                  < 0 = invalid parameter -info
 *                  > 0 = QR algorithm failed; eigenvalues info+1:n converged
 */
void mb05my(
    const char* balanc,
    const i32 n,
    f64* a,
    const i32 lda,
    f64* wr,
    f64* wi,
    f64* r,
    const i32 ldr,
    f64* q,
    const i32 ldq,
    f64* dwork,
    const i32 ldwork,
    i32* info
);

/**
 * @brief Matrix exponential and integral.
 *
 * Computes:
 *   (a) F(delta) = exp(A*delta)
 *   (b) H(delta) = integral from 0 to delta of exp(A*s) ds
 *
 * where A is a real N-by-N matrix and delta is a scalar value.
 *
 * Uses Pade approximation with scaling and squaring.
 *
 * @param[in] n Order of matrix A (n >= 0)
 * @param[in] delta Scalar time parameter
 * @param[in] a N-by-N matrix A, dimension (lda,n)
 * @param[in] lda Leading dimension of A (lda >= max(1,n))
 * @param[out] ex N-by-N matrix exp(A*delta), dimension (ldex,n)
 * @param[in] ldex Leading dimension of EX (ldex >= max(1,n))
 * @param[out] exint N-by-N integral matrix H(delta), dimension (ldexin,n)
 * @param[in] ldexin Leading dimension of EXINT (ldexin >= max(1,n))
 * @param[in] tol Tolerance for Pade approximation order. sqrt(eps) recommended.
 * @param[out] iwork Integer workspace, dimension (n)
 * @param[out] dwork Double workspace, dimension (ldwork)
 *                   On exit: dwork[0] = optimal ldwork
 * @param[in] ldwork Workspace size (ldwork >= max(1, n*(n+1)))
 *                   For optimal performance: ldwork >= 2*n*n
 * @param[out] info Exit code:
 *                  0 = success
 *                  < 0 = invalid parameter -info
 *                  1..n = Pade denominator singular at (info,info)
 *                  n+1 = delta*||A||_F too large for meaningful computation
 */
void mb05nd(
    const i32 n,
    const f64 delta,
    const f64* a,
    const i32 lda,
    f64* ex,
    const i32 ldex,
    f64* exint,
    const i32 ldexin,
    const f64 tol,
    i32* iwork,
    f64* dwork,
    const i32 ldwork,
    i32* info
);

/**
 * @brief Restore matrix after balancing transformations.
 *
 * Computes A <- P * D * A * D^{-1} * P' where P is a permutation matrix
 * and D is a diagonal scaling matrix, both determined by DGEBAL.
 *
 * @param[in] job 'N' = do nothing
 *                'P' = permutation only
 *                'S' = scaling only
 *                'B' = both permutation and scaling
 * @param[in] n Order of matrix A (n >= 0)
 * @param[in] low Low index from DGEBAL (1 <= low <= max(1,n))
 * @param[in] igh High index from DGEBAL (min(low,n) <= igh <= n)
 * @param[in,out] a N-by-N matrix, dimension (lda,n)
 *                  On exit: back-transformed matrix
 * @param[in] lda Leading dimension of A (lda >= max(1,n))
 * @param[in] scale Permutation/scaling factors from DGEBAL, dimension (n)
 * @param[out] info Exit code:
 *                  0 = success
 *                  < 0 = invalid parameter -info
 */
void mb05oy(
    const char* job,
    const i32 n,
    const i32 low,
    const i32 igh,
    f64* a,
    const i32 lda,
    const f64* scale,
    i32* info
);

/**
 * @brief Matrix exponential with accuracy estimate using Pade approximation.
 *
 * Computes exp(A*delta) where A is a real N-by-N matrix using diagonal
 * Pade approximation with scaling and squaring.
 *
 * @param[in] balanc Balancing option:
 *                'N' = no balancing
 *                'S' = balance using MB04MD
 * @param[in] n Order of matrix A (n >= 0)
 * @param[in] ndiag Order of diagonal Pade approximant (1 <= ndiag <= 15)
 * @param[in] delta Scalar multiplier for matrix A
 * @param[in,out] a N-by-N matrix A, overwritten with exp(A*delta)
 * @param[in] lda Leading dimension of A (lda >= max(1,n))
 * @param[out] mdig Minimal accurate mantissa digits in 1-norm
 * @param[out] idig Accurate mantissa digits at 95% confidence
 * @param[out] iwork Integer workspace, dimension at least N
 * @param[out] dwork Double workspace, dimension at least LDWORK
 * @param[in] ldwork Workspace size (>= 4*N*N + 2*N + 1)
 * @param[out] iwarn Warning indicator:
 *                0 = no warning
 *                1 = possible inaccuracy
 *                2 = severe inaccuracy warning
 *                3 = balancing not used (no improvement)
 * @param[out] info Exit code:
 *                0 = success
 *                1 = norm too large for accuracy
 *                2 = quasi-singular coefficient matrix
 *                3 = overflow detected
 */
void mb05od(
    const char* balanc,
    const i32 n,
    const i32 ndiag,
    const f64 delta,
    f64* a,
    const i32 lda,
    i32* mdig,
    i32* idig,
    i32* iwork,
    f64* dwork,
    const i32 ldwork,
    i32* iwarn,
    i32* info
);

#ifdef __cplusplus
}
#endif

#endif /* SLICOT_MB05_H */
