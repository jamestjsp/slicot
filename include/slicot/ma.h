/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#ifndef SLICOT_MA_H
#define SLICOT_MA_H

#include "../slicot_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Compute complex square root in real arithmetic.
 *
 * Computes the complex square root YR + i*YI of a complex number XR + i*XI.
 * The result satisfies: YR >= 0 and SIGN(YI) = SIGN(XI).
 *
 * Adapted from EISPACK subroutine CSROOT.
 *
 * @param[in] xr Real part of input complex number
 * @param[in] xi Imaginary part of input complex number
 * @param[out] yr Real part of complex square root
 * @param[out] yi Imaginary part of complex square root
 */
void ma01ad(f64 xr, f64 xi, f64 *yr, f64 *yi);

/**
 * @brief Compute general product of K real scalars without over/underflow.
 *
 * Computes a product of K scalars stored in array A, with each scalar either
 * multiplied or divided based on the signature array S. The result is
 * represented as ALPHA / BETA * BASE^SCAL to avoid overflow/underflow.
 *
 * @param[in] base Machine base
 * @param[in] lgbas Logarithm of BASE
 * @param[in] k Number of scalars (k >= 1)
 * @param[in] s Signature array, dimension (k). Each entry must be 1 or -1.
 *              S[i]=1 means multiply, S[i]=-1 means divide.
 * @param[in] a Array of real scalars, dimension ((k-1)*|inca|+1)
 * @param[in] inca Increment for array A (inca != 0)
 * @param[out] alpha Output scalar numerator
 * @param[out] beta Output scalar (0.0 or 1.0)
 * @param[out] scal Scaling factor exponent
 *
 * @note Result = ALPHA / BETA * BASE^SCAL = product of scalars
 */
void ma01bd(f64 base, f64 lgbas, i32 k, const i32 *s, const f64 *a, i32 inca,
            f64 *alpha, f64 *beta, i32 *scal);

/**
 * @brief Compute general product of K complex scalars without over/underflow.
 *
 * Computes a product of K complex scalars stored in array A, with each scalar
 * either multiplied or divided based on the signature array S. The result is
 * represented as ALPHA / BETA * BASE^SCAL to avoid overflow/underflow.
 *
 * @param[in] base Machine base
 * @param[in] k Number of scalars (k >= 1)
 * @param[in] s Signature array, dimension (k). Each entry must be 1 or -1.
 *              S[i]=1 means multiply, S[i]=-1 means divide.
 * @param[in] a Array of complex scalars, dimension ((k-1)*|inca|+1)
 * @param[in] inca Increment for array A (inca != 0)
 * @param[out] alpha Output complex scalar numerator with 1 <= |alpha| < BASE
 * @param[out] beta Output complex scalar (0 or 1)
 * @param[out] scal Scaling factor exponent
 *
 * @note Result = ALPHA / BETA * BASE^SCAL = product of scalars
 */
void ma01bz(f64 base, i32 k, const i32 *s, const c128 *a, i32 inca,
            c128 *alpha, c128 *beta, i32 *scal);

/**
 * @brief Compute sign of sum of two scaled numbers without overflow.
 *
 * Computes, without over- or underflow, the sign of the sum:
 *     A * BASE^IA + B * BASE^IB
 *
 * Any base can be used, but it should be the same for both numbers.
 *
 * @param[in] a First real scalar
 * @param[in] ia Exponent for first scalar (A * BASE^IA)
 * @param[in] b Second real scalar
 * @param[in] ib Exponent for second scalar (B * BASE^IB)
 * @return Sign of the sum: 1 (positive), 0 (zero), or -1 (negative)
 */
i32 ma01cd(f64 a, i32 ia, f64 b, i32 ib);

/**
 * @brief Compute approximate symmetric chordal metric for two complex numbers.
 *
 * Computes an approximate symmetric chordal metric for two complex numbers
 * A1 = AR1 + i*AI1 and A2 = AR2 + i*AI2 using the formula:
 *     D = MIN(|A1 - A2|, |1/A1 - 1/A2|)
 *
 * The chordal metric is finite even if both numbers are infinite, or if
 * one is infinite and the other is finite and nonzero.
 *
 * @param[in] ar1 Real part of first complex number A1
 * @param[in] ai1 Imaginary part of first complex number A1
 * @param[in] ar2 Real part of second complex number A2
 * @param[in] ai2 Imaginary part of second complex number A2
 * @param[in] eps Relative machine precision (DLAMCH('E'))
 * @param[in] safemn Safe minimum, such that 1/SAFEMN does not overflow (DLAMCH('S'))
 * @param[out] d The approximate symmetric chordal metric (D >= 0)
 */
void ma01dd(f64 ar1, f64 ai1, f64 ar2, f64 ai2, f64 eps, f64 safemn, f64 *d);

/**
 * @brief Compute approximate symmetric chordal metric for two complex rationals.
 *
 * Computes an approximate symmetric chordal metric for two complex numbers
 * A1 and A2 represented as rational numbers. Each Aj = (ARj + i*AIj) / Bj.
 *
 * The formula used is: D = MIN(|A1 - A2|, |1/A1 - 1/A2|)
 *
 * Special cases:
 *   - Bj = 0 with nonzero numerator means Aj is infinite
 *   - ARj = AIj = Bj = 0 means Aj is not a number (NaN)
 *
 * @param[in] ar1 Real part of numerator of A1
 * @param[in] ai1 Imaginary part of numerator of A1
 * @param[in] b1 Denominator of A1 (b1 >= 0)
 * @param[in] ar2 Real part of numerator of A2
 * @param[in] ai2 Imaginary part of numerator of A2
 * @param[in] b2 Denominator of A2 (b2 >= 0)
 * @param[in] eps Relative machine precision (DLAMCH('E'))
 * @param[in] safemn Safe minimum, such that 1/SAFEMN does not overflow (DLAMCH('S'))
 * @param[out] d1 Numerator of chordal metric D (d1 >= 0)
 * @param[out] d2 Denominator of chordal metric D (0 or 1)
 * @param[out] iwarn 0 = success, 1 = A1 or A2 is NaN (D1 = D2 = 0)
 */
void ma01dz(f64 ar1, f64 ai1, f64 b1, f64 ar2, f64 ai2, f64 b2,
            f64 eps, f64 safemn, f64 *d1, f64 *d2, i32 *iwarn);

/**
 * @brief Transpose all or part of a matrix.
 *
 * Transposes an M-by-N matrix A into N-by-M matrix B.
 * Supports full, upper triangular, or lower triangular transposition.
 *
 * @param[in] job Specifies part to transpose:
 *                'U' = upper triangular/trapezoidal part only
 *                'L' = lower triangular/trapezoidal part only
 *                Otherwise = full matrix
 * @param[in] m Number of rows of A (m >= 0)
 * @param[in] n Number of columns of A (n >= 0)
 * @param[in] a Input matrix, dimension (lda,n), column-major
 * @param[in] lda Leading dimension of A (lda >= max(1,m))
 * @param[out] b Output matrix (transpose), dimension (ldb,m), column-major
 * @param[in] ldb Leading dimension of B (ldb >= max(1,n))
 */
void ma02ad(const char* job, const i32 m, const i32 n,
            const f64* a, const i32 lda,
            f64* b, const i32 ldb);

/**
 * @brief Reverse order of rows and/or columns of a matrix.
 *
 * Pre-multiplies and/or post-multiplies a matrix A with a permutation
 * matrix P, where P is a square matrix with ones on the secondary diagonal.
 * This reverses the order of rows (P*A), columns (A*P), or both (P*A*P).
 *
 * @param[in] side Specifies the operation:
 *                 'L' = reverse rows (compute P*A)
 *                 'R' = reverse columns (compute A*P)
 *                 'B' = reverse both (compute P*A*P)
 * @param[in] m Number of rows of A (m >= 0)
 * @param[in] n Number of columns of A (n >= 0)
 * @param[in,out] a Matrix A, dimension (lda,n), column-major.
 *                  On entry: the M-by-N matrix to be permuted.
 *                  On exit: P*A, A*P, or P*A*P depending on SIDE.
 * @param[in] lda Leading dimension of A (lda >= max(1,m))
 */
void ma02bd(const char side, const i32 m, const i32 n, f64* a, const i32 lda);

/**
 * @brief Pertranspose the central band of a square matrix.
 *
 * Computes the pertranspose of the central band of a square matrix.
 * The pertranspose reverses elements along each antidiagonal within
 * the specified band (KL subdiagonals, main diagonal, KU superdiagonals).
 * This is equivalent to P*B'*P where B is the band matrix and P is
 * a permutation matrix with ones on the secondary diagonal.
 *
 * @param[in] n Order of the square matrix A (n >= 0)
 * @param[in] kl Number of subdiagonals to pertranspose (0 <= kl <= n-1)
 * @param[in] ku Number of superdiagonals to pertranspose (0 <= ku <= n-1)
 * @param[in,out] a Matrix A, dimension (lda,n), column-major.
 *                  On entry: square matrix whose central band will be pertransposed.
 *                  On exit: matrix with central band pertransposed.
 * @param[in] lda Leading dimension of A (lda >= max(1,n))
 */
void ma02cd(i32 n, i32 kl, i32 ku, f64 *a, i32 lda);

/**
 * @brief Pertranspose the central band of a complex square matrix.
 *
 * Computes the pertranspose of the central band of a square complex matrix.
 * The pertranspose reverses elements along each antidiagonal within
 * the specified band (KL subdiagonals, main diagonal, KU superdiagonals).
 * This is equivalent to P*B'*P where B is the band matrix and P is
 * a permutation matrix with ones on the secondary diagonal.
 *
 * @param[in] n Order of the square matrix A (n >= 0)
 * @param[in] kl Number of subdiagonals to pertranspose (0 <= kl <= n-1)
 * @param[in] ku Number of superdiagonals to pertranspose (0 <= ku <= n-1)
 * @param[in,out] a Complex matrix A, dimension (lda,n), column-major.
 *                  On entry: square matrix whose central band will be pertransposed.
 *                  On exit: matrix with central band pertransposed.
 * @param[in] lda Leading dimension of A (lda >= max(1,n))
 */
void ma02cz(i32 n, i32 kl, i32 ku, c128 *a, i32 lda);

/**
 * @brief Pack/unpack upper or lower triangle of symmetric matrix.
 *
 * Packs or unpacks the upper or lower triangle of a symmetric matrix.
 * The packed matrix is stored column-wise in a one-dimensional array.
 *
 * Packed storage order:
 *   Upper (UPLO='U'): 11, 12, 22, 13, 23, 33, ..., 1n, 2n, ..., nn
 *   Lower (UPLO='L'): 11, 21, 31, ..., n1, 22, 32, ..., n2, ..., nn
 *
 * @param[in] job 'P' = pack, 'U' = unpack
 * @param[in] uplo 'U' = upper triangular, 'L' = lower triangular
 * @param[in] n Order of matrix A (n >= 0)
 * @param[in,out] a Matrix A, dimension (lda,n), column-major.
 *                  Input if job='P', output if job='U'
 * @param[in] lda Leading dimension of A (lda >= max(1,n))
 * @param[in,out] ap Packed array, dimension n*(n+1)/2.
 *                   Output if job='P', input if job='U'
 */
void ma02dd(const char* job, const char* uplo, i32 n, f64* a, i32 lda, f64* ap);

/**
 * @brief Store by symmetry the upper or lower triangle of a symmetric matrix.
 *
 * Completes a symmetric matrix by copying one triangle to the other.
 * Given upper triangle, constructs lower triangle (or vice versa).
 *
 * @param[in] uplo Specifies which part is given:
 *                 'U' = upper triangular part given
 *                 'L' = lower triangular part given
 *                 Other values = no operation
 * @param[in] n Order of matrix A (n >= 0)
 * @param[in,out] a Matrix, dimension (lda,n), column-major
 *                  In: N-by-N upper/lower triangle contains data
 *                  Out: Full N-by-N symmetric matrix
 * @param[in] lda Leading dimension of A (lda >= max(1,n))
 */
void ma02ed(const char uplo, i32 n, f64 *a, i32 lda);

/**
 * @brief Store by skew-symmetry the upper or lower triangle of a skew-symmetric matrix.
 *
 * Completes a skew-symmetric matrix by negating one triangle to fill the other.
 * Given upper triangle, constructs lower triangle as -upper^T (or vice versa).
 * Diagonal entries are set to zero.
 *
 * @param[in] uplo Specifies which part is given:
 *                 'U' = upper triangular part given
 *                 'L' = lower triangular part given
 *                 Other values = no operation
 * @param[in] n Order of matrix A (n >= 0)
 * @param[in,out] a Matrix, dimension (lda,n), column-major
 *                  In: N-by-N upper/lower triangle contains data
 *                  Out: Full N-by-N skew-symmetric matrix (A = -A^T)
 * @param[in] lda Leading dimension of A (lda >= max(1,n))
 */
void ma02es(const char uplo, i32 n, f64 *a, i32 lda);

/**
 * @brief Compute coefficients for modified hyperbolic plane rotation.
 *
 * Computes c and s (c^2 + s^2 = 1) such that:
 *     y1 = (1/c) * x1 - (s/c) * x2 = sqrt(x1^2 - x2^2)
 *     y2 = -s * y1 + c * x2 = 0
 *
 * The input must satisfy either x1 = x2 = 0, or abs(x2) < abs(x1).
 *
 * @param[in,out] x1 On entry: x1. On exit: y1 = sqrt(x1^2 - x2^2)
 * @param[in] x2 The value x2
 * @param[out] c Cosine of the modified hyperbolic rotation
 * @param[out] s Sine of the modified hyperbolic rotation
 * @param[out] info 0 = success, 1 = abs(x2) >= abs(x1) with x1 != 0 or x2 != 0
 */
void ma02fd(f64 *x1, f64 x2, f64 *c, f64 *s, i32 *info);

/**
 * @brief Column interchanges on a matrix.
 *
 * Performs a series of column interchanges on matrix A. One column
 * interchange is initiated for each of columns K1 through K2 of A.
 * Column-oriented counterpart of LAPACK's DLASWP.
 *
 * @param[in] n Number of rows of matrix A. N >= 0.
 * @param[in,out] a Matrix of dimension (LDA, M). On exit, permuted matrix.
 * @param[in] lda Leading dimension of A. LDA >= max(1,N).
 * @param[in] k1 First element of IPIV for which interchange will be done. 1-based.
 * @param[in] k2 Last element of IPIV for which interchange will be done. 1-based.
 * @param[in] ipiv Pivot indices. IPIV(K)=L means swap columns K and L.
 * @param[in] incx Increment between successive values of IPIV.
 *                 Negative INCX applies interchanges in reverse order.
 */
void ma02gd(i32 n, f64* a, i32 lda, i32 k1, i32 k2, const i32* ipiv, i32 incx);

/**
 * @brief Compute norm of skew-Hamiltonian or Hamiltonian matrix.
 *
 * Computes the one norm, Frobenius norm, infinity norm, or max element
 * of a skew-Hamiltonian or Hamiltonian matrix:
 *     H = [A, G; Q, -A'] for Hamiltonian (G=G', Q=Q')
 *     X = [A, G; Q, A']  for skew-Hamiltonian (G=-G', Q=-Q')
 *
 * @param[in] typ 'S' = skew-Hamiltonian, 'H' = Hamiltonian
 * @param[in] norm '1'/'O' = one norm, 'F'/'E' = Frobenius, 'I' = infinity, 'M' = max element
 * @param[in] n Order of matrix A (n >= 0)
 * @param[in] a Matrix A, dimension (lda,n)
 * @param[in] lda Leading dimension of A (lda >= max(1,n))
 * @param[in] qg Matrix containing Q (lower) and G (upper), dimension (ldqg,n+1)
 * @param[in] ldqg Leading dimension of QG (ldqg >= max(1,n))
 * @param[out] dwork Workspace, dimension 2*n for 1/I/O norms, not used for F/E/M
 * @return The computed norm value
 */
f64 ma02id(const char *typ, const char *norm, i32 n, const f64 *a, i32 lda,
           const f64 *qg, i32 ldqg, f64 *dwork);

/**
 * @brief Compute the number of zero rows and zero columns of a real matrix.
 *
 * Scans the M-by-N matrix A to count rows and columns that contain only zeros.
 *
 * @param[in] m Number of rows of matrix A (m >= 0)
 * @param[in] n Number of columns of matrix A (n >= 0)
 * @param[in] a Matrix A, dimension (lda,n), column-major
 * @param[in] lda Leading dimension of A (lda >= max(1,m))
 * @param[out] nzr Number of zero rows in matrix A
 * @param[out] nzc Number of zero columns in matrix A
 */
void ma02pd(const i32 m, const i32 n, const f64* a, const i32 lda,
            i32* nzr, i32* nzc);

/**
 * @brief Transpose or conjugate-transpose a complex matrix.
 *
 * (Conjugate) transposes all or part of a two-dimensional complex
 * matrix A into another matrix B.
 *
 * @param[in] trans 'T' = transpose, 'C' = conjugate transpose
 * @param[in] job   'U' = upper triangular, 'L' = lower triangular, other = full
 * @param[in] m     Number of rows of A (m >= 0)
 * @param[in] n     Number of columns of A (n >= 0)
 * @param[in] a     Input matrix, dimension (lda,n), column-major
 * @param[in] lda   Leading dimension of A (lda >= max(1,m))
 * @param[out] b    Output matrix (transposed), dimension (ldb,m), column-major
 * @param[in] ldb   Leading dimension of B (ldb >= max(1,n))
 */
void ma02az(const char* trans, const char* job, const i32 m, const i32 n,
            const c128* a, const i32 lda, c128* b, const i32 ldb);

/**
 * @brief Reverse order of rows and/or columns of a complex matrix.
 *
 * Pre-multiplies and/or post-multiplies a complex matrix A with a permutation
 * matrix P, where P is a square matrix with ones on the secondary diagonal.
 * This reverses the order of rows (P*A), columns (A*P), or both (P*A*P).
 *
 * @param[in] side Specifies the operation:
 *                 'L' = reverse rows (compute P*A)
 *                 'R' = reverse columns (compute A*P)
 *                 'B' = reverse both (compute P*A*P)
 * @param[in] m Number of rows of A (m >= 0)
 * @param[in] n Number of columns of A (n >= 0)
 * @param[in,out] a Complex matrix A, dimension (lda,n), column-major.
 *                  On entry: the M-by-N matrix to be permuted.
 *                  On exit: P*A, A*P, or P*A*P depending on SIDE.
 * @param[in] lda Leading dimension of A (lda >= max(1,m))
 */
void ma02bz(const char side, const i32 m, const i32 n, c128* a, const i32 lda);

/**
 * @brief Store by (skew-)symmetry the upper or lower triangle of a complex matrix.
 *
 * Stores by (skew-)symmetry the upper or lower triangle of a
 * (skew-)symmetric/Hermitian complex matrix, given the other triangle.
 * The option SKEW = 'G' allows to suitably deal with the diagonal of a
 * general square triangular matrix.
 *
 * @param[in] uplo Specifies which part of the matrix is given:
 *                 'U' = upper triangular part given
 *                 'L' = lower triangular part given
 *                 Other values = no operation
 * @param[in] trans Specifies transposition type:
 *                  'T' = use transposition
 *                  'C' = use conjugate transposition
 * @param[in] skew Specifies symmetry type:
 *                 'G' = general (not symmetric/Hermitian)
 *                 'N' = symmetric/Hermitian
 *                 'S' = skew-symmetric/Hermitian
 * @param[in] n Order of matrix A (n >= 0)
 * @param[in,out] a Complex matrix, dimension (lda,n), column-major
 *                  In: N-by-N upper/lower triangle contains data
 *                  Out: Full N-by-N (skew-)symmetric/Hermitian matrix
 *                  For TRANS='C', SKEW='N': diagonal imag parts set to 0
 *                  For TRANS='C', SKEW='S': diagonal real parts set to 0
 * @param[in] lda Leading dimension of A (lda >= max(1,n))
 */
void ma02ez(const char uplo, const char trans, const char skew,
            i32 n, c128 *a, i32 lda);

/**
 * @brief Column interchanges on a complex matrix.
 *
 * Performs a series of column interchanges on complex matrix A. One column
 * interchange is initiated for each of columns K1 through K2 of A.
 * Column-oriented counterpart of LAPACK's ZLASWP.
 * Complex version of MA02GD.
 *
 * @param[in] n Number of rows of matrix A. N >= 0.
 * @param[in,out] a Complex matrix of dimension (LDA, M). On exit, permuted matrix.
 * @param[in] lda Leading dimension of A. LDA >= max(1,N).
 * @param[in] k1 First element of IPIV for which interchange will be done. 1-based.
 * @param[in] k2 Last element of IPIV for which interchange will be done. 1-based.
 * @param[in] ipiv Pivot indices. IPIV(K)=L means swap columns K and L.
 * @param[in] incx Increment between successive values of IPIV.
 *                 Negative INCX applies interchanges in reverse order.
 */
void ma02gz(i32 n, c128* a, i32 lda, i32 k1, i32 k2, const i32* ipiv, i32 incx);

/**
 * @brief Check if matrix equals scalar times identity-like matrix.
 *
 * Checks if A = DIAG*I, where I is an M-by-N matrix with ones on
 * the diagonal and zeros elsewhere.
 *
 * @param[in] job Specifies the part of matrix A to check:
 *                'U' = upper triangular/trapezoidal part only
 *                'L' = lower triangular/trapezoidal part only
 *                Otherwise = all of matrix A
 * @param[in] m Number of rows of matrix A (m >= 0)
 * @param[in] n Number of columns of matrix A (n >= 0)
 * @param[in] diag The scalar DIAG to compare diagonal elements against
 * @param[in] a Input matrix, dimension (lda,n), column-major
 * @param[in] lda Leading dimension of A (lda >= max(1,m))
 * @return true if A = DIAG*I in the specified region, false otherwise.
 *         Returns false if min(m,n) = 0.
 */
bool ma02hd(const char *job, i32 m, i32 n, f64 diag, const f64 *a, i32 lda);

/**
 * @brief Check if complex matrix equals scalar times identity-like matrix.
 *
 * Checks if A = DIAG*I, where I is an M-by-N matrix with ones on
 * the diagonal and zeros elsewhere. A is complex and DIAG is a complex scalar.
 *
 * @param[in] job Specifies the part of matrix A to check:
 *                'U' = upper triangular/trapezoidal part only
 *                'L' = lower triangular/trapezoidal part only
 *                Otherwise = all of matrix A
 * @param[in] m Number of rows of matrix A (m >= 0)
 * @param[in] n Number of columns of matrix A (n >= 0)
 * @param[in] diag The complex scalar DIAG to compare diagonal elements against
 * @param[in] a Input complex matrix, dimension (lda,n), column-major
 * @param[in] lda Leading dimension of A (lda >= max(1,m))
 * @return true if A = DIAG*I in the specified region, false otherwise.
 *         Returns false if min(m,n) = 0.
 */
bool ma02hz(const char *job, i32 m, i32 n, c128 diag, const c128 *a, i32 lda);

/**
 * @brief Compute norm of complex skew-Hamiltonian or Hamiltonian matrix.
 *
 * Computes the one norm, Frobenius norm, infinity norm, or max element
 * of a complex skew-Hamiltonian or Hamiltonian matrix:
 *     H = [A, G; Q, -A^H] for Hamiltonian (G=G^H, Q=Q^H)
 *     X = [A, G; Q, A^H]  for skew-Hamiltonian (G=-G^H, Q=-Q^H)
 *
 * Note that for these matrix types, the infinity norm equals the one norm.
 *
 * @param[in] typ 'S' = skew-Hamiltonian, 'H' = Hamiltonian
 * @param[in] norm '1'/'O' = one norm, 'F'/'E' = Frobenius, 'I' = infinity, 'M' = max element
 * @param[in] n Order of matrix A (n >= 0)
 * @param[in] a Complex matrix A, dimension (lda,n)
 * @param[in] lda Leading dimension of A (lda >= max(1,n))
 * @param[in] qg Complex matrix containing Q (lower) and G (upper), dimension (ldqg,n+1)
 * @param[in] ldqg Leading dimension of QG (ldqg >= max(1,n))
 * @param[out] dwork Workspace, dimension 2*n for 1/I/O norms, not used for F/E/M
 * @return The computed norm value
 */
f64 ma02iz(const char *typ, const char *norm, i32 n, const c128 *a, i32 lda,
           const c128 *qg, i32 ldqg, f64 *dwork);

/**
 * @brief Test if a matrix is an orthogonal symplectic matrix.
 *
 * Computes || Q^T Q - I ||_F for a matrix of the form:
 *     Q = [  op(Q1)   op(Q2) ]
 *         [ -op(Q2)   op(Q1) ]
 *
 * where Q1 and Q2 are N-by-N matrices. This residual can be used to test
 * whether Q is numerically an orthogonal symplectic matrix.
 *
 * @param[in] ltran1 If true, op(Q1) = Q1^T; otherwise op(Q1) = Q1
 * @param[in] ltran2 If true, op(Q2) = Q2^T; otherwise op(Q2) = Q2
 * @param[in] n Order of matrices Q1 and Q2 (n >= 0)
 * @param[in] q1 Matrix Q1, dimension (ldq1, n)
 * @param[in] ldq1 Leading dimension of Q1 (ldq1 >= max(1,n))
 * @param[in] q2 Matrix Q2, dimension (ldq2, n)
 * @param[in] ldq2 Leading dimension of Q2 (ldq2 >= max(1,n))
 * @param[out] res Workspace matrix, dimension (ldres, n)
 * @param[in] ldres Leading dimension of RES (ldres >= max(1,n))
 * @return The computed residual || Q^T Q - I ||_F
 */
f64 ma02jd(bool ltran1, bool ltran2, i32 n, const f64 *q1, i32 ldq1,
           const f64 *q2, i32 ldq2, f64 *res, i32 ldres);

/**
 * @brief Test if a complex matrix is a unitary symplectic matrix.
 *
 * Computes || Q^H Q - I ||_F for a complex matrix of the form:
 *     Q = [  op(Q1)   op(Q2) ]
 *         [ -op(Q2)   op(Q1) ]
 *
 * where Q1 and Q2 are N-by-N complex matrices. This residual can be used to
 * test whether Q is numerically a unitary symplectic matrix.
 *
 * This is the complex version of MA02JD.
 *
 * @param[in] ltran1 If true, op(Q1) = Q1'; otherwise op(Q1) = Q1
 * @param[in] ltran2 If true, op(Q2) = Q2'; otherwise op(Q2) = Q2
 * @param[in] n Order of matrices Q1 and Q2 (n >= 0)
 * @param[in] q1 Complex matrix Q1, dimension (ldq1, n)
 * @param[in] ldq1 Leading dimension of Q1 (ldq1 >= max(1,n))
 * @param[in] q2 Complex matrix Q2, dimension (ldq2, n)
 * @param[in] ldq2 Leading dimension of Q2 (ldq2 >= max(1,n))
 * @param[out] res Complex workspace matrix, dimension (ldres, n)
 * @param[in] ldres Leading dimension of RES (ldres >= max(1,n))
 * @return The computed residual || Q^H Q - I ||_F
 */
f64 ma02jz(bool ltran1, bool ltran2, i32 n, const c128 *q1, i32 ldq1,
           const c128 *q2, i32 ldq2, c128 *res, i32 ldres);

/**
 * @brief Compute norms of a real skew-symmetric matrix.
 *
 * Computes the value of the one norm, or the Frobenius norm, or
 * the infinity norm, or the element of largest absolute value
 * of a real skew-symmetric matrix.
 *
 * Note that for skew-symmetric matrices, the infinity norm equals the one norm.
 *
 * @param[in] norm Specifies the value to return:
 *                 '1' or 'O' = one norm
 *                 'F' or 'E' = Frobenius norm
 *                 'I' = infinity norm
 *                 'M' = max(abs(A(i,j)))
 * @param[in] uplo 'U' = upper triangular part stored, 'L' = lower triangular part stored
 * @param[in] n Order of matrix A (n >= 0). When n = 0, returns 0.
 * @param[in] a Skew-symmetric matrix, dimension (lda, n).
 *              If UPLO='U', strictly upper triangular part is referenced.
 *              If UPLO='L', strictly lower triangular part is referenced.
 *              Diagonal need not be set to zero.
 * @param[in] lda Leading dimension of A (lda >= max(1,n))
 * @param[out] dwork Workspace, dimension (n) when NORM='I','1','O'; not referenced otherwise
 * @return The computed norm value
 */
f64 ma02md(const char *norm, const char *uplo, i32 n, const f64 *a, i32 lda,
           f64 *dwork);

/**
 * @brief Compute norms of a complex skew-Hermitian matrix.
 *
 * Computes the value of the one norm, or the Frobenius norm, or
 * the infinity norm, or the element of largest absolute value
 * of a complex skew-Hermitian matrix.
 *
 * Note that for skew-Hermitian matrices, the infinity norm equals the one norm.
 *
 * @param[in] norm Specifies the value to return:
 *                 '1' or 'O' = one norm
 *                 'F' or 'E' = Frobenius norm
 *                 'I' = infinity norm
 *                 'M' = max(abs(A(i,j)))
 * @param[in] uplo 'U' = upper triangular part stored, 'L' = lower triangular part stored
 * @param[in] n Order of matrix A (n >= 0). When n = 0, returns 0.
 * @param[in] a Complex skew-Hermitian matrix, dimension (lda, n).
 *              If UPLO='U', upper triangular part is referenced.
 *              If UPLO='L', lower triangular part is referenced.
 *              Real parts of diagonal elements are assumed to be zero.
 * @param[in] lda Leading dimension of A (lda >= max(1,n))
 * @param[out] dwork Workspace, dimension (n) when NORM='I','1','O'; not referenced otherwise
 * @return The computed norm value
 */
f64 ma02mz(const char *norm, const char *uplo, i32 n, const c128 *a, i32 lda,
           f64 *dwork);

/**
 * @brief Permute two rows and corresponding columns of a (skew-)symmetric/Hermitian complex matrix.
 *
 * Permutes rows K and L and the corresponding columns K and L of a
 * (skew-)symmetric/Hermitian complex matrix, stored in triangular form.
 *
 * @param[in] uplo Specifies which part of matrix A is stored:
 *                 'U' = upper triangular part stored
 *                 'L' = lower triangular part stored
 * @param[in] trans Specifies transposition type:
 *                  'T' = use transposition
 *                  'C' = use conjugate transposition
 * @param[in] skew Specifies symmetry type:
 *                 'N' = symmetric/Hermitian
 *                 'S' = skew-symmetric/skew-Hermitian
 * @param[in] n Order of matrix A (n >= 0)
 * @param[in] k Smaller index of pair to permute (0 <= k <= l). If k = 0, routine returns.
 * @param[in] l Larger index of pair to permute (k <= l <= n). 1-based indices.
 * @param[in,out] a Complex matrix A, dimension (lda,n), column-major.
 *                  On entry: the (skew-)symmetric/Hermitian matrix in triangular form.
 *                  On exit: the permuted matrix.
 * @param[in] lda Leading dimension of A (lda >= max(1,n))
 */
void ma02nz(const char *uplo, const char *trans, const char *skew,
            i32 n, i32 k, i32 l, c128 *a, i32 lda);

/**
 * @brief Count zero rows of a real (skew-)Hamiltonian matrix.
 *
 * Computes the number of zero rows (and zero columns) of a real
 * (skew-)Hamiltonian matrix:
 *
 *       (  A    D   )
 *   H = (           )
 *       (  E  +/-A' )
 *
 * The matrix E is stored in the lower triangular part of DE (columns 1..M),
 * and D is stored in the upper triangular part of DE (columns 2..M+1).
 *
 * @param[in] skew Specifies matrix type:
 *                 'H' = Hamiltonian (D=D', E=E')
 *                 'S' = skew-Hamiltonian (D=-D', E=-E', diagonal assumed zero)
 * @param[in] m Order of matrices A, D, E (m >= 0)
 * @param[in] a Matrix A, dimension (lda, m)
 * @param[in] lda Leading dimension of A (lda >= max(1,m))
 * @param[in] de Matrix DE, dimension (ldde, m+1).
 *               Lower triangular part (columns 1..M) contains E.
 *               Upper triangular part (columns 2..M+1) contains D.
 *               For skew-Hamiltonian, diagonal and first superdiagonal not referenced.
 * @param[in] ldde Leading dimension of DE (ldde >= max(1,m))
 * @return The number of zero rows in H
 */
i32 ma02od(const char *skew, i32 m, const f64 *a, i32 lda,
           const f64 *de, i32 ldde);

/**
 * @brief Count zero rows of a complex (skew-)Hamiltonian matrix.
 *
 * Computes the number of zero rows (and zero columns) of a complex
 * (skew-)Hamiltonian matrix:
 *
 *       (  A    D   )
 *   H = (           )
 *       (  E  +/-A' )
 *
 * The matrix E is stored in the lower triangular part of DE (columns 1..M),
 * and D is stored in the upper triangular part of DE (columns 2..M+1).
 *
 * For Hamiltonian (SKEW='H'): D and E are Hermitian, real parts of diagonal used
 * For skew-Hamiltonian (SKEW='S'): D and E are skew-Hermitian, imaginary parts
 *                                  of diagonal used (real parts assumed zero)
 *
 * This is the complex version of MA02OD.
 *
 * @param[in] skew Specifies matrix type:
 *                 'H' = Hamiltonian (D=D^H, E=E^H)
 *                 'S' = skew-Hamiltonian (D=-D^H, E=-E^H)
 * @param[in] m Order of matrices A, D, E (m >= 0)
 * @param[in] a Complex matrix A, dimension (lda, m)
 * @param[in] lda Leading dimension of A (lda >= max(1,m))
 * @param[in] de Complex matrix DE, dimension (ldde, m+1).
 *               Lower triangular part (columns 1..M) contains E.
 *               Upper triangular part (columns 2..M+1) contains D.
 * @param[in] ldde Leading dimension of DE (ldde >= max(1,m))
 * @return The number of zero rows in H
 */
i32 ma02oz(const char *skew, i32 m, const c128 *a, i32 lda,
           const c128 *de, i32 ldde);

/**
 * @brief Compute the number of zero rows and zero columns of a complex matrix.
 *
 * Scans the M-by-N complex matrix A to count rows and columns that contain only zeros.
 * Complex version of MA02PD.
 *
 * @param[in] m Number of rows of matrix A (m >= 0)
 * @param[in] n Number of columns of matrix A (n >= 0)
 * @param[in] a Complex matrix A, dimension (lda,n), column-major
 * @param[in] lda Leading dimension of A (lda >= max(1,m))
 * @param[out] nzr Number of zero rows in matrix A
 * @param[out] nzc Number of zero columns in matrix A
 */
void ma02pz(const i32 m, const i32 n, const c128* a, const i32 lda,
            i32* nzr, i32* nzc);

/**
 * @brief Sort vector D and rearrange E with same permutation.
 *
 * Sorts the elements of an n-vector D in increasing (ID='I') or decreasing
 * (ID='D') order, and rearranges the elements of an n-vector E using the
 * same permutations.
 *
 * Uses Quick Sort, but reverts to Insertion sort on arrays of length <= 20.
 * Stack dimension limits N to about 2^32.
 *
 * Based on LAPACK DLASRT, but applies to E the same interchanges used for D.
 *
 * @param[in] id Specifies the desired order:
 *               'I' = sort D in increasing order
 *               'D' = sort D in decreasing order
 * @param[in] n Length of arrays D and E (n >= 0)
 * @param[in,out] d Array to sort, dimension (n).
 *                  On entry: vector to be sorted.
 *                  On exit: vector sorted in specified order.
 * @param[in,out] e Array to rearrange, dimension (n).
 *                  On entry: vector to be rearranged.
 *                  On exit: vector rearranged with same permutation as D.
 * @return info = 0 on success, -i if i-th argument had illegal value
 */
i32 ma02rd(const char id, i32 n, f64 *d, f64 *e);

/**
 * @brief Compute smallest nonzero absolute value of matrix elements.
 *
 * Computes the smallest nonzero absolute value of the elements of
 * a real M-by-N matrix A.
 *
 * @param[in] m Number of rows of matrix A (m >= 0)
 * @param[in] n Number of columns of matrix A (n >= 0)
 * @param[in] a Matrix A, dimension (lda, n), column-major
 * @param[in] lda Leading dimension of A (lda >= max(1,m))
 * @return The smallest nonzero |A(i,j)|. Returns 0 if M=0 or N=0.
 *         Returns overflow value if all elements are zero.
 */
f64 ma02sd(i32 m, i32 n, const f64 *a, i32 lda);

#ifdef __cplusplus
}
#endif

#endif /* SLICOT_MA_H */
