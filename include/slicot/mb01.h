/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#ifndef SLICOT_MB01_H
#define SLICOT_MB01_H

#include "../slicot_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Scale a matrix to safe numerical range.
 *
 * Scales a matrix so its norm is in the safe range [SMLNUM, BIGNUM],
 * or undoes such scaling. Scaling is performed if needed to ensure the
 * matrix norm is within representable floating-point numbers.
 *
 * The scaling factor is represented as a ratio: for scaling up (when
 * ANRM < SMLNUM), matrix is multiplied by SMLNUM/ANRM. For scaling
 * down (when ANRM > BIGNUM), matrix is multiplied by BIGNUM/ANRM.
 *
 * @param[in] scun Operation: 'S' = scale, 'U' = undo scaling
 * @param[in] type Matrix storage type:
 *                 'G' = full matrix
 *                 'L' = (block) lower triangular
 *                 'U' = (block) upper triangular
 *                 'H' = (block) upper Hessenberg
 *                 'B' = symmetric band (lower half)
 *                 'Q' = symmetric band (upper half)
 *                 'Z' = general band
 * @param[in] m Number of rows (M >= 0)
 * @param[in] n Number of columns (N >= 0)
 * @param[in] kl Lower bandwidth (for 'B', 'Q', 'Z' types)
 * @param[in] ku Upper bandwidth (for 'B', 'Q', 'Z' types)
 * @param[in] anrm Norm of the matrix (ANRM >= 0). Must be preserved
 *                 between scaling and undo operations.
 * @param[in] nbl Number of diagonal blocks (0 = no block structure)
 * @param[in] nrows Block sizes, dimension max(1,nbl), may be NULL if nbl=0
 * @param[in,out] a Matrix array, dimension (lda,n), column-major storage
 * @param[in] lda Leading dimension (lda >= max(1,m))
 * @param[out] info Exit code: 0 = success, < 0 = -i means i-th arg invalid
 */
void mb01pd(const char* scun, const char* type, i32 m, i32 n, i32 kl, i32 ku,
            f64 anrm, i32 nbl, const i32* nrows, f64* a, i32 lda, i32* info);

/**
 * @brief Multiply matrix by scalar CTO/CFROM without overflow/underflow.
 *
 * Multiplies M-by-N real matrix A by scalar CTO/CFROM, preventing overflow and
 * underflow through iterative scaling. The algorithm decomposes the scaling
 * factor into products of safe intermediate values (DBL_MIN or DBL_MAX), applying
 * them sequentially until the final result is achieved. This ensures that
 * intermediate values remain representable even when the final result is near
 * machine limits.
 *
 * Supports full, triangular, Hessenberg, and banded matrix storage with optional
 * block structure for efficient handling of structured matrices.
 *
 * Based on LAPACK routine DLASCL with extensions for block-structured matrices.
 * For efficiency, input parameters are not validated (caller responsibility).
 *
 * @param[in] type Matrix storage type:
 *                 'G' = full matrix
 *                 'L' = (block) lower triangular
 *                 'U' = (block) upper triangular
 *                 'H' = (block) upper Hessenberg
 *                 'B' = symmetric band (lower half)
 *                 'Q' = symmetric band (upper half)
 *                 'Z' = general band
 * @param[in] m Number of rows (M >= 0)
 * @param[in] n Number of columns (N >= 0)
 * @param[in] kl Lower bandwidth (for 'B', 'Q', 'Z' types)
 * @param[in] ku Upper bandwidth (for 'B', 'Q', 'Z' types)
 * @param[in] cfrom Denominator scalar (caller must ensure nonzero)
 * @param[in] cto Numerator scalar
 * @param[in] nbl Number of diagonal blocks (0 = no block structure)
 * @param[in] nrows Block sizes, dimension max(1,nbl), may be NULL if nbl=0
 * @param[in,out] a Matrix array, dimension (lda,n), column-major storage
 * @param[in] lda Leading dimension (lda >= max(1,m))
 * @param[out] info Exit code (always 0, reserved for future use)
 */
void mb01qd(char type, i32 m, i32 n, i32 kl, i32 ku,
            f64 cfrom, f64 cto, i32 nbl, const i32* nrows,
            f64* a, i32 lda, i32* info);

/**
 * @brief Block symmetric rank-k update (BLAS 3 version of MB01RX).
 *
 * Computes triangular part of matrix formula:
 *   R = alpha*R + beta*op(A)*B  (SIDE='L'), or
 *   R = alpha*R + beta*B*op(A)  (SIDE='R')
 *
 * where alpha, beta are scalars, R is m-by-m, and op(A) is A or A'.
 *
 * @param[in] side 'L' for R=alpha*R+beta*op(A)*B, 'R' for R=alpha*R+beta*B*op(A)
 * @param[in] uplo 'U' for upper triangle, 'L' for lower triangle
 * @param[in] trans 'N' for op(A)=A, 'T'/'C' for op(A)=A'
 * @param[in] m Order of R (m >= 0)
 * @param[in] n Inner dimension (n >= 0)
 * @param[in] alpha Scalar alpha
 * @param[in] beta Scalar beta
 * @param[in,out] r m-by-m matrix R, dimension (ldr,m)
 * @param[in] ldr Leading dimension of R (ldr >= max(1,m))
 * @param[in] a Matrix A
 * @param[in] lda Leading dimension of A
 * @param[in] b Matrix B
 * @param[in] ldb Leading dimension of B
 * @param[out] info Exit code (0=success, <0=invalid parameter)
 *
 * @note Main application: symmetric updates where B = X*op(A)' or B = op(A)'*X
 */
void mb01rb(const char* side, const char* uplo, const char* trans,
            const i32 m, const i32 n, const f64 alpha, const f64 beta,
            f64* r, const i32 ldr, const f64* a, const i32 lda,
            const f64* b, const i32 ldb, i32* info);

/**
 * @brief Symmetric rank-k matrix update with symmetric matrices.
 *
 * Computes: R = alpha*R + beta*op(A)*X*op(A)'
 * where R and X are symmetric matrices, and op(A) = A or A'.
 *
 * @param[in] uplo 'U' = upper triangle, 'L' = lower triangle stored
 * @param[in] trans 'N' = op(A)=A, 'T' or 'C' = op(A)=A'
 * @param[in] m Order of R and number of rows of op(A), m >= 0
 * @param[in] n Order of X and number of columns of op(A), n >= 0
 * @param[in] alpha Scalar multiplier for R
 * @param[in] beta Scalar multiplier for quadratic term
 * @param[in,out] r Symmetric matrix R, dimension (ldr,m). Upper/lower triangle
 *                  depending on uplo. On exit: result in same triangle.
 * @param[in] ldr Leading dimension of R, ldr >= max(1,m)
 * @param[in] a General matrix A, dimension (lda,k) where k=n if trans='N',
 *              k=m if trans='T' or 'C'
 * @param[in] lda Leading dimension of A, lda >= max(1,nrowa) where
 *                nrowa=m if trans='N', n otherwise
 * @param[in,out] x Symmetric matrix X, dimension (ldx,n). On exit:
 *                  diagonal elements halved.
 * @param[in] ldx Leading dimension of X, ldx >= max(1,n)
 * @param[out] dwork Workspace, dimension >= m*n if beta!=0, else 1
 * @param[in] ldwork Workspace size, ldwork >= max(1,m*n) if beta!=0
 * @param[out] info Exit code (0=success, <0=invalid parameter)
 */
void mb01rd(const char* uplo, const char* trans, i32 m, i32 n,
            f64 alpha, f64 beta, f64* r, i32 ldr,
            const f64* a, i32 lda, f64* x, i32 ldx,
            f64* dwork, i32 ldwork, i32* info);

/**
 * @brief Compute R = alpha*R + beta*op(A)*X*op(A)' with R, X symmetric.
 *
 * Computes the matrix formula:
 *     R := alpha*R + beta*op(A)*X*op(A)'
 * where alpha and beta are scalars, R and X are symmetric matrices,
 * A is a general matrix, and op(A) is A or A'.
 *
 * @param[in] uplo 'U': upper triangle stored, 'L': lower triangle stored
 * @param[in] trans 'N': op(A)=A, 'T'/'C': op(A)=A'
 * @param[in] m Order of R, rows of op(A). m >= 0.
 * @param[in] n Order of X, columns of op(A). n >= 0.
 * @param[in] alpha Scalar for R.
 * @param[in] beta Scalar for product.
 * @param[in,out] r On entry: symmetric matrix R (triangle per uplo).
 *                  On exit: updated R.
 * @param[in] ldr Leading dimension of R, ldr >= max(1,m).
 * @param[in] a Matrix A: m-by-n if trans='N', n-by-m if trans='T'/'C'.
 * @param[in] lda Leading dimension of A.
 * @param[in,out] x Symmetric matrix X. Diagonal modified internally, restored.
 * @param[in] ldx Leading dimension of X, ldx >= max(1,n).
 * @param[out] dwork Workspace, dimension m*n if beta!=0.
 * @param[in] ldwork Length of dwork.
 * @param[out] info 0 on success, -i if argument i invalid.
 */
void mb01ru(const char* uplo, const char* trans, i32 m, i32 n,
            f64 alpha, f64 beta, f64* r, i32 ldr, const f64* a, i32 lda,
            f64* x, i32 ldx, f64* dwork, i32 ldwork, i32* info);

/**
 * @brief Symmetric matrix transformation (BLAS 2 version).
 *
 * Computes: A := op(Z)*A*op(Z)'
 * where A is symmetric, Z is a general matrix, op(Z) = Z or Z'.
 *
 * @param[in] uplo 'U' = upper triangle, 'L' = lower triangle stored
 * @param[in] trans 'N' = op(Z)=Z, 'T' = op(Z)=Z'
 * @param[in] m Order of result and number of rows of Z if trans='N',
 *              columns if trans='T', m >= 0
 * @param[in] n Order of input A and number of columns of Z if trans='N',
 *              rows if trans='T', n >= 0
 * @param[in,out] a Symmetric matrix A, dimension (lda,max(m,n)).
 *                  On entry: N-by-N input in upper/lower triangle.
 *                  On exit: M-by-M result in same triangle.
 * @param[in] lda Leading dimension of A, lda >= max(1,m,n)
 * @param[in] z Transformation matrix, dimension (ldz,k) where
 *              k=n if trans='N', k=m if trans='T'
 * @param[in] ldz Leading dimension of Z, ldz >= max(1,m) if trans='N',
 *                ldz >= max(1,n) if trans='T'
 * @param[out] dwork Workspace, dimension (n)
 * @param[out] info Exit code (0=success, <0=invalid parameter)
 *
 * @note BLAS 2 version of MB01RD (simpler interface for transformations).
 */
void mb01rw(const char* uplo, const char* trans, i32 m, i32 n,
            f64* a, i32 lda, const f64* z, i32 ldz,
            f64* dwork, i32* info);

/**
 * @brief Compute triangle of R = alpha*R + beta*op(H)*B or beta*B*op(H).
 *
 * Computes the upper or lower triangular part of:
 *     R := alpha*R + beta*op(H)*B  (SIDE='L')
 *     R := alpha*R + beta*B*op(H)  (SIDE='R')
 * where H is upper Hessenberg and op(H) = H or H'.
 *
 * @param[in] side 'L': R = alpha*R + beta*op(H)*B, 'R': R = alpha*R + beta*B*op(H)
 * @param[in] uplo 'U': upper triangle, 'L': lower triangle
 * @param[in] trans 'N': op(H)=H, 'T'/'C': op(H)=H'
 * @param[in] m Order of matrices. m >= 0.
 * @param[in] alpha Scalar for R.
 * @param[in] beta Scalar for product.
 * @param[in,out] r On entry: m-by-m R (triangle per uplo). On exit: updated R.
 * @param[in] ldr Leading dimension of R, ldr >= max(1,m).
 * @param[in,out] h Upper Hessenberg m-by-m. First column may be modified but restored.
 * @param[in] ldh Leading dimension of H, ldh >= max(1,m).
 * @param[in] b m-by-m matrix B.
 * @param[in] ldb Leading dimension of B, ldb >= max(1,m).
 * @param[out] dwork Workspace, dimension m if beta!=0 and side='L'.
 * @param[out] info 0 on success, -i if argument i invalid.
 */
void mb01ry(const char* side, const char* uplo, const char* trans, i32 m,
            f64 alpha, f64 beta, f64* r, i32 ldr, f64* h, i32 ldh,
            const f64* b, i32 ldb, f64* dwork, i32* info);

/**
 * @brief Scale rows or columns of a matrix by a diagonal matrix.
 *
 * Computes one of:
 *   A := diag(R) * A        (jobs='R', row scaling)
 *   A := A * diag(C)        (jobs='C', column scaling)
 *   A := diag(R) * A * diag(C)  (jobs='B', both)
 *
 * @param[in] jobs Scaling operation: 'R'=row, 'C'=column, 'B'=both
 * @param[in] m Number of rows of A (m >= 0)
 * @param[in] n Number of columns of A (n >= 0)
 * @param[in,out] a M-by-N matrix, dimension (lda,n), scaled on exit
 * @param[in] lda Leading dimension of A (lda >= max(1,m))
 * @param[in] r Row scale factors, dimension (m), not used if jobs='C'
 * @param[in] c Column scale factors, dimension (n), not used if jobs='R'
 */
void mb01sd(const char jobs, const i32 m, const i32 n,
            f64* a, const i32 lda, const f64* r, const f64* c);

/**
 * @brief Scale a symmetric matrix using diagonal scaling factors.
 *
 * Scales a symmetric N-by-N matrix A using the row and column scaling
 * factors stored in the vector D.
 *
 * @param[in] jobs Scaling operation:
 *                 'D': A := diag(D)*A*diag(D) (row and column scaling with D)
 *                 'I': A := inv(diag(D))*A*inv(diag(D)) (scaling with inv(D))
 * @param[in] uplo Triangle storage:
 *                 'U': upper triangle of A is stored
 *                 'L': lower triangle of A is stored
 * @param[in] n Order of the symmetric matrix A. n >= 0.
 * @param[in,out] a On entry: the n-by-n upper or lower triangular part of the
 *                  symmetric matrix A (depending on uplo).
 *                  On exit: the corresponding triangular part of the scaled matrix.
 * @param[in] lda Leading dimension of a, lda >= max(1,n).
 * @param[in] d Diagonal scaling factors, dimension (n).
 */
void mb01ss(const char jobs, const char uplo, i32 n,
            f64* a, i32 lda, const f64* d);

/**
 * @brief Product of upper quasi-triangular matrices B := A * B.
 *
 * Computes the matrix product A * B, where A and B are upper quasi-triangular
 * matrices (block upper triangular with 1-by-1 or 2-by-2 diagonal blocks)
 * with the same structure. The result is returned in array B.
 *
 * @param[in] n Order of matrices A and B (n >= 0)
 * @param[in] a N-by-N upper quasi-triangular matrix, dimension (lda,n)
 * @param[in] lda Leading dimension of A (lda >= max(1,n))
 * @param[in,out] b N-by-N upper quasi-triangular matrix with same structure as A.
 *                  On exit: contains the product A * B.
 * @param[in] ldb Leading dimension of B (ldb >= max(1,n))
 * @param[out] dwork Workspace, dimension (n-1)
 * @param[out] info Exit code: 0=success, <0=invalid parameter,
 *                  1=A and B have different structures or are not quasi-triangular
 *
 * @note Useful for computing powers of real Schur form matrices.
 */
void mb01td(const i32 n, const f64* a, const i32 lda,
            f64* b, const i32 ldb, f64* dwork, i32* info);

/**
 * @brief Compute B = alpha*op(H)*A or B = alpha*A*op(H), H Hessenberg.
 *
 * @param[in] side 'L': B = alpha*op(H)*A, 'R': B = alpha*A*op(H)
 * @param[in] trans 'N': op(H)=H, 'T'/'C': op(H)=H'
 * @param[in] m Rows of A and B. m >= 0.
 * @param[in] n Columns of A and B. n >= 0.
 * @param[in] alpha Scalar multiplier.
 * @param[in,out] h Upper Hessenberg (m-by-m if side='L', n-by-n if side='R').
 *                  First column may be modified but restored.
 * @param[in] ldh Leading dimension of H.
 * @param[in] a m-by-n matrix A.
 * @param[in] lda Leading dimension of A, lda >= max(1,m).
 * @param[out] b m-by-n output matrix B.
 * @param[in] ldb Leading dimension of B, ldb >= max(1,m).
 * @param[out] info 0 on success, -i if argument i invalid.
 */
void mb01ud(const char* side, const char* trans, i32 m, i32 n,
            f64 alpha, f64* h, i32 ldh, const f64* a, i32 lda,
            f64* b, i32 ldb, i32* info);

/**
 * @brief Compute matrix product T := alpha*op(T)*A or T := alpha*A*op(T).
 *
 * Computes one of the matrix products:
 *   T := alpha*op(T)*A (SIDE='L'), or
 *   T := alpha*A*op(T) (SIDE='R'),
 * where alpha is a scalar, A is M-by-N, T is triangular, and op(T) is T or T^T.
 * Uses block algorithm with BLAS 3 when possible. Result overwrites T.
 *
 * @param[in] side 'L' for T:=alpha*op(T)*A, 'R' for T:=alpha*A*op(T)
 * @param[in] uplo 'U' for upper triangular T, 'L' for lower triangular
 * @param[in] trans 'N' for op(T)=T, 'T'/'C' for op(T)=T^T
 * @param[in] m Number of rows of A (m >= 0)
 * @param[in] n Number of columns of A (n >= 0)
 * @param[in] alpha Scalar multiplier (alpha=0: T set to zero)
 * @param[in,out] t DOUBLE PRECISION array, dimension (ldt,max(K,N)) if SIDE='L',
 *                  (ldt,K) if SIDE='R', where K=M if SIDE='L', K=N if SIDE='R'.
 *                  In: K-by-K triangular matrix T
 *                  Out: M-by-N result matrix
 * @param[in] ldt Leading dimension of T (ldt >= max(1,M) if SIDE='L',
 *                ldt >= max(1,M,N) if SIDE='R')
 * @param[in] a DOUBLE PRECISION array, dimension (lda,n), M-by-N matrix A
 * @param[in] lda Leading dimension of A (lda >= max(1,M))
 * @param[out] dwork DOUBLE PRECISION array, dimension (ldwork), workspace
 *                   On exit, dwork[0] returns optimal ldwork
 * @param[in] ldwork Workspace size (ldwork >= 1 if alpha=0 or min(M,N)=0,
 *                   ldwork >= M if SIDE='L', ldwork >= N if SIDE='R'.
 *                   If ldwork=-1, workspace query mode)
 * @param[out] info Exit code (0=success, <0=invalid parameter)
 */
void mb01uy(
    const char* side, const char* uplo, const char* trans,
    const i32 m, const i32 n,
    const f64 alpha,
    f64* t, const i32 ldt,
    const f64* a, const i32 lda,
    f64* dwork, const i32 ldwork,
    i32* info
);

/**
 * @brief Compute Kronecker product of two matrices.
 *
 * Performs C = alpha*kron(op(A), op(B)) + beta*C where kron denotes
 * the Kronecker product and op(M) is M or M'.
 *
 * @param[in] trana 'N' use A, 'T' use A'
 * @param[in] tranb 'N' use B, 'T' use B'
 * @param[in] ma Rows of op(A)
 * @param[in] na Columns of op(A)
 * @param[in] mb Rows of op(B)
 * @param[in] nb Columns of op(B)
 * @param[in] alpha Scalar multiplier for Kronecker product
 * @param[in] beta Scalar multiplier for C
 * @param[in] a Matrix A
 * @param[in] lda Leading dimension of A
 * @param[in] b Matrix B
 * @param[in] ldb Leading dimension of B
 * @param[in,out] c Result matrix, dimension (MC x NC)
 * @param[in] ldc Leading dimension of C
 * @param[out] mc Rows of C (= MA*MB)
 * @param[out] nc Columns of C (= NA*NB)
 * @param[out] info 0=success, <0=arg error
 */
void mb01vd(const char *trana, const char *tranb, i32 ma, i32 na, i32 mb, i32 nb,
            f64 alpha, f64 beta, const f64 *a, i32 lda, const f64 *b, i32 ldb,
            f64 *c, i32 ldc, i32 *mc, i32 *nc, i32 *info);

/**
 * @brief Triangular symmetric rank-k update.
 *
 * Computes either the upper or lower triangular part of:
 *   R = alpha*R + beta*op(A)*B  (SIDE='L')
 *   R = alpha*R + beta*B*op(A)  (SIDE='R')
 *
 * where op(A) = A or A', and only the specified triangle is computed.
 *
 * @param[in] side 'L' for left (R = alpha*R + beta*op(A)*B)
 *                 'R' for right (R = alpha*R + beta*B*op(A))
 * @param[in] uplo 'U' for upper triangle, 'L' for lower triangle
 * @param[in] trans 'N' for op(A)=A, 'T'/'C' for op(A)=A'
 * @param[in] m Order of matrix R
 * @param[in] n Dimension for product:
 *              SIDE='L': rows of B, columns of op(A)
 *              SIDE='R': rows of op(A), columns of B
 * @param[in] alpha Scalar multiplier for R
 * @param[in] beta Scalar multiplier for product
 * @param[in,out] r On entry: m-by-m matrix R (triangle only)
 *                  On exit: updated R (triangle only)
 * @param[in] ldr Leading dimension of R, >= max(1,m)
 * @param[in] a Matrix A with dimensions:
 *              SIDE='L', TRANS='N': m-by-n
 *              SIDE='L', TRANS='T': n-by-m
 *              SIDE='R', TRANS='N': n-by-m
 *              SIDE='R', TRANS='T': m-by-n
 * @param[in] lda Leading dimension of A
 * @param[in] b Matrix B with dimensions:
 *              SIDE='L': n-by-m
 *              SIDE='R': m-by-n
 * @param[in] ldb Leading dimension of B
 * @return 0 on success, -i if argument i had an illegal value
 *
 * @note Main application: computing symmetric updates where B = X*op(A)'
 *       or B = op(A)'*X with X symmetric.
 */
i32 slicot_mb01rx(
    char side,
    char uplo,
    char trans,
    i32 m,
    i32 n,
    f64 alpha,
    f64 beta,
    f64 *r,
    i32 ldr,
    const f64 *a,
    i32 lda,
    const f64 *b,
    i32 ldb
);

/**
 * @brief Skew-symmetric rank-2k update.
 *
 * Performs one of the skew-symmetric rank 2k operations:
 *   C := alpha*A*B' - alpha*B*A' + beta*C   (trans='N')
 *   C := alpha*A'*B - alpha*B'*A + beta*C   (trans='T' or 'C')
 *
 * where alpha and beta are scalars, C is a real N-by-N skew-symmetric
 * matrix and A, B are N-by-K matrices in the first case and K-by-N
 * matrices in the second case.
 *
 * This is the skew-symmetric variant of DSYR2K.
 *
 * @param[in] uplo 'U': strictly upper triangular part of C
 *                 'L': strictly lower triangular part of C
 * @param[in] trans 'N': C := alpha*A*B' - alpha*B*A' + beta*C
 *                  'T'/'C': C := alpha*A'*B - alpha*B'*A + beta*C
 * @param[in] n Order of matrix C. n >= 0.
 * @param[in] k If trans='N': columns of A and B. If trans='T'/'C': rows of A,B.
 * @param[in] alpha Scalar multiplier. If alpha=0 or n<=1 or k=0, A and B not used.
 * @param[in] a Matrix A: n-by-k if trans='N', k-by-n if trans='T'/'C'
 * @param[in] lda Leading dimension of A
 * @param[in] b Matrix B: n-by-k if trans='N', k-by-n if trans='T'/'C'
 * @param[in] ldb Leading dimension of B
 * @param[in] beta Scalar multiplier for C. If beta=0, C need not be set.
 * @param[in,out] c n-by-n skew-symmetric matrix (strictly upper/lower triangle)
 * @param[in] ldc Leading dimension of C, ldc >= max(1,n)
 * @param[out] info 0 on success, -i if argument i invalid
 */
void mb01kd(const char* uplo, const char* trans, i32 n, i32 k,
            f64 alpha, const f64* a, i32 lda, const f64* b, i32 ldb,
            f64 beta, f64* c, i32 ldc, i32* info);

/**
 * @brief Compute R = alpha*R + beta*op(A)*X*op(A)' with R, X skew-symmetric.
 *
 * Computes the matrix formula:
 *     R := alpha*R + beta*op(A)*X*op(A)'
 * where alpha and beta are scalars, R and X are skew-symmetric matrices,
 * A is a general matrix, and op(A) is A or A'.
 *
 * The result is overwritten on R.
 *
 * @param[in] uplo 'U': strictly upper triangular part of R and X given
 *                 'L': strictly lower triangular part of R and X given
 * @param[in] trans 'N': op(A) = A, 'T'/'C': op(A) = A'
 * @param[in] m Order of R and number of rows of op(A). m >= 0.
 * @param[in] n Order of X and number of columns of op(A). n >= 0.
 * @param[in] alpha Scalar for R. When alpha=0, R need not be set.
 * @param[in] beta Scalar for product. When beta=0 or n<=1 or m<=1, A,X not used.
 * @param[in,out] r On entry: strictly upper/lower triangle of skew-symmetric R.
 *                  On exit: updated R.
 * @param[in] ldr Leading dimension of R, ldr >= max(1,m).
 * @param[in] a Matrix A: m-by-n if trans='N', n-by-m if trans='T'/'C'.
 * @param[in] lda Leading dimension of A.
 * @param[in,out] x Skew-symmetric matrix X (strictly upper/lower triangle).
 *                  May be overwritten if workspace is insufficient.
 * @param[in] ldx Leading dimension of X.
 * @param[out] dwork Workspace, dimension ldwork.
 * @param[in] ldwork Length of dwork. ldwork >= n if beta!=0 and m>1 and n>1.
 *                   For optimum performance: ldwork >= m*(n-1).
 * @param[out] info 0 on success, -i if argument i invalid.
 */
void mb01ld(const char* uplo, const char* trans, i32 m, i32 n,
            f64 alpha, f64 beta, f64* r, i32 ldr, const f64* a, i32 lda,
            f64* x, i32 ldx, f64* dwork, i32 ldwork, i32* info);

/**
 * @brief Skew-symmetric matrix-vector multiply: y := alpha*A*x + beta*y.
 *
 * Performs the matrix-vector operation y := alpha*A*x + beta*y where
 * alpha and beta are scalars, x and y are vectors of length n, and
 * A is an n-by-n skew-symmetric matrix (A = -A').
 *
 * This is a modified version of the BLAS routine DSYMV for skew-symmetric
 * matrices. Since A is skew-symmetric, the diagonal is zero and only the
 * strictly upper or lower triangular part is stored.
 *
 * @param[in] uplo 'U': strictly upper triangular part of A is stored
 *                 'L': strictly lower triangular part of A is stored
 * @param[in] n Order of matrix A. n >= 0.
 * @param[in] alpha Scalar multiplier for A*x. If alpha=0, A is not referenced.
 * @param[in] a Skew-symmetric matrix A, dimension (lda,n). Contains strictly
 *              upper or lower triangular part depending on uplo.
 * @param[in] lda Leading dimension of A, lda >= max(1,n).
 * @param[in] x Input vector x, dimension 1+(n-1)*|incx|.
 * @param[in] incx Increment for x. incx != 0. If incx < 0, elements of x
 *                 are accessed in reverse order.
 * @param[in] beta Scalar multiplier for y. If beta=0, y need not be set.
 * @param[in,out] y Input/output vector y, dimension 1+(n-1)*|incy|.
 *                  On exit: y := alpha*A*x + beta*y.
 * @param[in] incy Increment for y. incy != 0. If incy < 0, elements of y
 *                 are accessed in reverse order.
 * @param[out] info Exit code: 0=success, i=parameter i had an illegal value.
 *
 * @note Performance may be lower than vendor BLAS DSYMV due to
 *       skew-symmetric structure handling.
 */
void mb01md(const char uplo, const i32 n, const f64 alpha,
            const f64* a, const i32 lda, const f64* x, const i32 incx,
            const f64 beta, f64* y, const i32 incy, i32* info);

/**
 * @brief Skew-symmetric rank 2 operation A := alpha*x*y' - alpha*y*x' + A.
 *
 * Performs the skew-symmetric rank 2 operation:
 *     A := alpha*x*y' - alpha*y*x' + A
 * where alpha is a scalar, x and y are vectors of length n, and A is
 * an n-by-n skew-symmetric matrix (A = -A').
 *
 * This is a modified version of the BLAS routine DSYR2 adapted for
 * skew-symmetric matrices.
 *
 * @param[in] uplo 'U': strictly upper triangular part of A is referenced
 *                 'L': strictly lower triangular part of A is referenced
 * @param[in] n Order of matrix A. n >= 0.
 * @param[in] alpha Scalar multiplier. If alpha=0, x and y are not referenced.
 * @param[in] x Input vector x, dimension 1+(n-1)*|incx|.
 * @param[in] incx Increment for x. incx != 0. If incx < 0, elements of x
 *                 are accessed in reverse order.
 * @param[in] y Input vector y, dimension 1+(n-1)*|incy|.
 * @param[in] incy Increment for y. incy != 0. If incy < 0, elements of y
 *                 are accessed in reverse order.
 * @param[in,out] a Skew-symmetric matrix A, dimension (lda,n). On entry,
 *                  contains strictly upper or lower triangle depending on uplo.
 *                  On exit: updated A := alpha*x*y' - alpha*y*x' + A.
 * @param[in] lda Leading dimension of A, lda >= max(1,n).
 * @param[out] info Exit code: 0=success, i=parameter i had an illegal value.
 *
 * @note Performance may be lower than vendor BLAS DSYR2 due to
 *       skew-symmetric structure handling.
 */
void mb01nd(const char uplo, const i32 n, const f64 alpha,
            const f64* x, const i32 incx, const f64* y, const i32 incy,
            f64* a, const i32 lda, i32* info);

/**
 * @brief Symmetric rank 2k update with Hessenberg matrix.
 *
 * Performs one of the special symmetric rank 2k operations:
 *     R := alpha*R + beta*H*X + beta*X*H'     (trans='N')
 *     R := alpha*R + beta*H'*X + beta*X*H     (trans='T' or 'C')
 *
 * where alpha and beta are scalars, R and X are N-by-N symmetric
 * matrices, and H is an N-by-N upper Hessenberg matrix.
 *
 * @param[in] uplo 'U': upper triangle of R and X stored
 *                 'L': lower triangle of R and X stored
 * @param[in] trans 'N': R := alpha*R + beta*H*X + beta*X*H'
 *                  'T'/'C': R := alpha*R + beta*H'*X + beta*X*H
 * @param[in] n Order of matrices R, H, X. n >= 0.
 * @param[in] alpha Scalar for R. When alpha=0, R need not be set.
 * @param[in] beta Scalar for products. When beta=0, H and X not used.
 * @param[in,out] r On entry: symmetric matrix R (triangle per uplo).
 *                  On exit: updated R (same triangle).
 * @param[in] ldr Leading dimension of R, ldr >= max(1,n).
 * @param[in] h Upper Hessenberg matrix H, dimension (ldh,n).
 * @param[in] ldh Leading dimension of H, ldh >= max(1,n).
 * @param[in] x Symmetric matrix X (triangle per uplo), dimension (ldx,n).
 * @param[in] ldx Leading dimension of X, ldx >= max(1,n).
 * @param[out] info 0 on success, -i if argument i invalid.
 */
void mb01oc(const char* uplo, const char* trans, i32 n,
            f64 alpha, f64 beta, f64* r, i32 ldr,
            const f64* h, i32 ldh, const f64* x, i32 ldx, i32* info);

/**
 * @brief Symmetric rank 2k update with Hessenberg and triangular matrices.
 *
 * Computes one of the symmetric rank 2k operations:
 *     R := alpha*R + beta*H*E' + beta*E*H'     (trans='N')
 *     R := alpha*R + beta*H'*E + beta*E'*H     (trans='T' or 'C')
 *
 * where alpha and beta are scalars, R is an N-by-N symmetric matrix,
 * H is an N-by-N upper Hessenberg matrix, and E is an N-by-N upper
 * triangular matrix.
 *
 * @param[in] uplo 'U': upper triangle of R stored
 *                 'L': lower triangle of R stored
 * @param[in] trans 'N': R := alpha*R + beta*H*E' + beta*E*H'
 *                  'T'/'C': R := alpha*R + beta*H'*E + beta*E'*H
 * @param[in] n Order of matrices R, H, E. n >= 0.
 * @param[in] alpha Scalar for R. When alpha=0, R need not be set.
 * @param[in] beta Scalar for products. When beta=0, H and E not used.
 * @param[in,out] r On entry: symmetric matrix R (triangle per uplo).
 *                  On exit: updated R (same triangle).
 * @param[in] ldr Leading dimension of R, ldr >= max(1,n).
 * @param[in] h Upper Hessenberg matrix H, dimension (ldh,n).
 * @param[in] ldh Leading dimension of H, ldh >= max(1,n).
 * @param[in] e Upper triangular matrix E, dimension (lde,n).
 * @param[in] lde Leading dimension of E, lde >= max(1,n).
 * @param[out] info 0 on success, -i if argument i invalid.
 */
void mb01oe(const char* uplo, const char* trans, i32 n,
            f64 alpha, f64 beta, f64* r, i32 ldr,
            const f64* h, i32 ldh, const f64* e, i32 lde, i32* info);

/**
 * @brief Compute R = alpha*R + beta*(op(H)*X*op(E)' + op(E)*X*op(H)').
 *
 * Computes the matrix formula:
 *     R := alpha*R + beta*(op(H)*X*op(E)' + op(E)*X*op(H)')
 * where alpha and beta are scalars, R and X are symmetric matrices,
 * H is an upper Hessenberg matrix, E is an upper triangular matrix,
 * and op(M) is M or M'.
 *
 * @param[in] uplo 'U': upper triangle of R and X stored
 *                 'L': lower triangle of R and X stored
 * @param[in] trans 'N': op(M) = M
 *                  'T'/'C': op(M) = M'
 * @param[in] n Order of matrices R, H, X, E. n >= 0.
 * @param[in] alpha Scalar for R. When alpha=0 and R!=X, R need not be set.
 * @param[in] beta Scalar for products. When beta=0, H and X not used.
 * @param[in,out] r On entry: symmetric matrix R (triangle per uplo).
 *                  On exit: updated R (same triangle).
 * @param[in] ldr Leading dimension of R, ldr >= max(1,n).
 * @param[in,out] h Upper Hessenberg matrix H, dimension (ldh,n).
 *                  If trans='N', elements 3..N of first column modified but restored.
 * @param[in] ldh Leading dimension of H, ldh >= max(1,n).
 * @param[in,out] x Symmetric matrix X (triangle per uplo), dimension (ldx,n).
 *                  Diagonal modified internally but restored.
 * @param[in] ldx Leading dimension of X, ldx >= max(1,n).
 * @param[in] e Upper triangular matrix E, dimension (lde,n).
 * @param[in] lde Leading dimension of E, lde >= max(1,n).
 * @param[out] dwork Workspace, dimension ldwork. Not used if beta=0 or n=0.
 * @param[in] ldwork Length of dwork. ldwork >= n*n if beta!=0, else 0.
 * @param[out] info 0 on success, -i if argument i invalid.
 */
void mb01od(const char* uplo, const char* trans, i32 n,
            f64 alpha, f64 beta, f64* r, i32 ldr,
            f64* h, i32 ldh, f64* x, i32 ldx,
            const f64* e, i32 lde, f64* dwork, i32 ldwork, i32* info);

/**
 * @brief Symmetric rank 2k update with two Hessenberg matrices.
 *
 * Computes one of the symmetric rank 2k operations:
 *     R := alpha*R + beta*H*A' + beta*A*H'     (trans='N')
 *     R := alpha*R + beta*H'*A + beta*A'*H     (trans='T' or 'C')
 *
 * where alpha and beta are scalars, R is an N-by-N symmetric matrix,
 * and H and A are N-by-N upper Hessenberg matrices.
 *
 * @param[in] uplo 'U': upper triangle of R stored
 *                 'L': lower triangle of R stored
 * @param[in] trans 'N': R := alpha*R + beta*H*A' + beta*A*H'
 *                  'T'/'C': R := alpha*R + beta*H'*A + beta*A'*H
 * @param[in] n Order of matrices R, H, A. n >= 0.
 * @param[in] alpha Scalar for R. When alpha=0, R need not be set.
 * @param[in] beta Scalar for products. When beta=0, H and A not used.
 * @param[in,out] r On entry: symmetric matrix R (triangle per uplo).
 *                  On exit: updated R (same triangle).
 * @param[in] ldr Leading dimension of R, ldr >= max(1,n).
 * @param[in] h Upper Hessenberg matrix H, dimension (ldh,n).
 * @param[in] ldh Leading dimension of H, ldh >= max(1,n).
 * @param[in] a Upper Hessenberg matrix A, dimension (lda,n).
 * @param[in] lda Leading dimension of A, lda >= max(1,n).
 * @param[out] info 0 on success, -i if argument i invalid.
 */
void mb01oh(const char* uplo, const char* trans, i32 n,
            f64 alpha, f64 beta, f64* r, i32 ldr,
            const f64* h, i32 ldh, const f64* a, i32 lda, i32* info);

/**
 * @brief Compute R = alpha*R + beta*op(H)*X*op(H)' with R, X symmetric and H Hessenberg.
 *
 * Computes the matrix formula:
 *     R := alpha*R + beta*op(H)*X*op(H)'
 * where alpha and beta are scalars, R and X are symmetric matrices,
 * H is an upper Hessenberg matrix, and op(H) is H or H'.
 *
 * @param[in] uplo 'U': upper triangle of R and X stored
 *                 'L': lower triangle of R and X stored
 * @param[in] trans 'N': op(H) = H
 *                  'T'/'C': op(H) = H'
 * @param[in] n Order of matrices R, H, X. n >= 0.
 * @param[in] alpha Scalar for R. When alpha=0 and R!=X, R need not be set.
 * @param[in] beta Scalar for quadratic form. When beta=0, H and X not used.
 * @param[in,out] r On entry: symmetric matrix R (triangle per uplo).
 *                  On exit: updated R (same triangle).
 * @param[in] ldr Leading dimension of R, ldr >= max(1,n).
 * @param[in,out] h Upper Hessenberg matrix H, dimension (ldh,n).
 *                  If trans='N', elements 3..N of first column modified but restored.
 * @param[in] ldh Leading dimension of H, ldh >= max(1,n).
 * @param[in,out] x Symmetric matrix X (triangle per uplo), dimension (ldx,n).
 *                  Diagonal modified internally but restored.
 * @param[in] ldx Leading dimension of X, ldx >= max(1,n).
 * @param[out] dwork Workspace, dimension ldwork. Not used if beta=0 or n=0.
 * @param[in] ldwork Length of dwork. ldwork >= n*n if beta!=0, else 0.
 * @param[out] info 0 on success, -i if argument i invalid.
 */
void mb01rh(const char* uplo, const char* trans, i32 n,
            f64 alpha, f64 beta, f64* r, i32 ldr,
            f64* h, i32 ldh, f64* x, i32 ldx,
            f64* dwork, i32 ldwork, i32* info);

/**
 * @brief Compute P = H*X or P = X*H with H Hessenberg and X symmetric.
 *
 * Computes the matrix product:
 *     P := H*X   (trans='N')
 *     P := X*H   (trans='T' or 'C')
 *
 * where H is an N-by-N upper Hessenberg matrix and X is an N-by-N symmetric
 * matrix. The symmetric matrix X is specified by its upper or lower triangular
 * part.
 *
 * @param[in] uplo 'U': upper triangular part of X given
 *                 'L': lower triangular part of X given
 * @param[in] trans 'N': compute P = H*X
 *                  'T'/'C': compute P = X*H
 * @param[in] n Order of matrices H, X, P. n >= 0.
 * @param[in] h Upper Hessenberg matrix H, dimension (ldh,n).
 * @param[in] ldh Leading dimension of H, ldh >= max(1,n).
 * @param[in] x Symmetric matrix X (triangle per uplo), dimension (ldx,n).
 * @param[in] ldx Leading dimension of X, ldx >= max(1,n).
 * @param[out] p Output matrix P, dimension (ldp,n).
 * @param[in] ldp Leading dimension of P, ldp >= max(1,n).
 * @param[out] info 0 on success, -i if argument i invalid.
 */
void mb01os(const char* uplo, const char* trans, i32 n,
            const f64* h, i32 ldh, const f64* x, i32 ldx,
            f64* p, i32 ldp, i32* info);

/**
 * @brief Compute P = op(H)*X*op(E)' or P' with H Hessenberg, X symmetric, E triangular.
 *
 * Computes either P or P', with P defined by the matrix formula:
 *     P = op(H)*X*op(E)'
 * where H is an upper Hessenberg matrix, X is a symmetric matrix,
 * E is an upper triangular matrix, and op(M) is M or M'.
 *
 * @param[in] uplo 'U': upper triangular part of X given
 *                 'L': lower triangular part of X given
 * @param[in] trans 'N': compute P = H*X*E'
 *                  'T'/'C': compute P' = E'*X*H
 * @param[in] n Order of matrices H, X, E, P. n >= 0.
 * @param[in] h Upper Hessenberg matrix H, dimension (ldh,n).
 * @param[in] ldh Leading dimension of H, ldh >= max(1,n).
 * @param[in] x Symmetric matrix X (triangle per uplo), dimension (ldx,n).
 * @param[in] ldx Leading dimension of X, ldx >= max(1,n).
 * @param[in] e Upper triangular matrix E, dimension (lde,n).
 * @param[in] lde Leading dimension of E, lde >= max(1,n).
 * @param[out] p Output matrix P, dimension (ldp,n).
 * @param[in] ldp Leading dimension of P, ldp >= max(1,n).
 * @param[out] info 0 on success, -i if argument i invalid.
 */
void mb01oo(const char* uplo, const char* trans, i32 n,
            const f64* h, i32 ldh, const f64* x, i32 ldx,
            const f64* e, i32 lde, f64* p, i32 ldp, i32* info);

/**
 * @brief Symmetric rank 2k update with two upper triangular matrices.
 *
 * Computes one of the symmetric rank 2k operations:
 *     R := alpha*R + beta*E*T' + beta*T*E'     (trans='N')
 *     R := alpha*R + beta*E'*T + beta*T'*E     (trans='T' or 'C')
 *
 * where alpha and beta are scalars, R is an N-by-N symmetric matrix,
 * and E and T are N-by-N upper triangular matrices.
 *
 * @param[in] uplo 'U': upper triangle of R stored
 *                 'L': lower triangle of R stored
 * @param[in] trans 'N': R := alpha*R + beta*E*T' + beta*T*E'
 *                  'T'/'C': R := alpha*R + beta*E'*T + beta*T'*E
 * @param[in] n Order of matrices R, E, T. n >= 0.
 * @param[in] alpha Scalar for R. When alpha=0, R need not be set.
 * @param[in] beta Scalar for products. When beta=0, E and T not used.
 * @param[in,out] r On entry: symmetric matrix R (triangle per uplo).
 *                  On exit: updated R (same triangle).
 * @param[in] ldr Leading dimension of R, ldr >= max(1,n).
 * @param[in] e Upper triangular matrix E, dimension (lde,n).
 * @param[in] lde Leading dimension of E, lde >= max(1,n).
 * @param[in] t Upper triangular matrix T, dimension (ldt,n).
 * @param[in] ldt Leading dimension of T, ldt >= max(1,n).
 * @param[out] info 0 on success, -i if argument i invalid.
 */
void mb01ot(const char* uplo, const char* trans, i32 n,
            f64 alpha, f64 beta, f64* r, i32 ldr,
            const f64* e, i32 lde, const f64* t, i32 ldt, i32* info);

/**
 * @brief Compute R = alpha*R + beta*op(E)*X*op(E)' with R, X symmetric and E upper triangular.
 *
 * Computes the matrix formula:
 *     R := alpha*R + beta*op(E)*X*op(E)'
 * where alpha and beta are scalars, R and X are symmetric matrices,
 * E is an upper triangular matrix, and op(E) is E or E'.
 *
 * @param[in] uplo 'U': upper triangle of R and X stored
 *                 'L': lower triangle of R and X stored
 * @param[in] trans 'N': op(E) = E
 *                  'T'/'C': op(E) = E'
 * @param[in] n Order of matrices R, E, X. n >= 0.
 * @param[in] alpha Scalar for R. When alpha=0 and R!=X, R need not be set.
 * @param[in] beta Scalar for quadratic form. When beta=0, E and X not used.
 * @param[in,out] r On entry: symmetric matrix R (triangle per uplo).
 *                  On exit: updated R (same triangle).
 * @param[in] ldr Leading dimension of R, ldr >= max(1,n).
 * @param[in] e Upper triangular matrix E, dimension (lde,n).
 * @param[in] lde Leading dimension of E, lde >= max(1,n).
 * @param[in,out] x Symmetric matrix X (triangle per uplo), dimension (ldx,n).
 *                  Diagonal modified internally but restored.
 * @param[in] ldx Leading dimension of X, ldx >= max(1,n).
 * @param[out] dwork Workspace, dimension ldwork. Not used if beta=0 or n=0.
 * @param[in] ldwork Length of dwork. ldwork >= n*n if beta!=0, else 0.
 * @param[out] info 0 on success, -i if argument i invalid.
 */
void mb01rt(const char* uplo, const char* trans, i32 n,
            f64 alpha, f64 beta, f64* r, i32 ldr,
            const f64* e, i32 lde, f64* x, i32 ldx,
            f64* dwork, i32 ldwork, i32* info);

/**
 * @brief Compute A := alpha*op(H)*A or A := alpha*A*op(H) with H Hessenberg.
 *
 * Computes one of the matrix products:
 *     A := alpha*op(H)*A   (SIDE='L')
 *     A := alpha*A*op(H)   (SIDE='R')
 * where alpha is a scalar, A is an m-by-n matrix, H is an upper Hessenberg
 * matrix, and op(H) is H or H'.
 *
 * @param[in] side 'L': A := alpha*op(H)*A, 'R': A := alpha*A*op(H)
 * @param[in] trans 'N': op(H)=H, 'T'/'C': op(H)=H'
 * @param[in] m Rows of A. m >= 0.
 * @param[in] n Columns of A. n >= 0.
 * @param[in] alpha Scalar multiplier. When alpha=0, H not referenced, A set to zero.
 * @param[in,out] h Upper Hessenberg (m-by-m if side='L', n-by-n if side='R').
 *                  First column elements below subdiagonal may be modified but restored.
 * @param[in] ldh Leading dimension of H.
 * @param[in,out] a m-by-n matrix. On exit: result.
 * @param[in] lda Leading dimension of A, lda >= max(1,m).
 * @param[out] dwork Workspace. For maximal efficiency ldwork >= m*n.
 * @param[in] ldwork Length of dwork. ldwork >= 0 if alpha=0 or min(m,n)=0,
 *                   ldwork >= m-1 if side='L', ldwork >= n-1 if side='R'.
 * @param[out] info 0 on success, -i if argument i invalid.
 */
void mb01uw(const char* side, const char* trans, i32 m, i32 n,
            f64 alpha, f64* h, i32 ldh, f64* a, i32 lda,
            f64* dwork, i32 ldwork, i32* info);

/**
 * @brief Compute A := alpha*op(T)*A or A := alpha*A*op(T) with T quasi-triangular.
 *
 * Computes one of the matrix products:
 *     A := alpha*op(T)*A   (SIDE='L')
 *     A := alpha*A*op(T)   (SIDE='R')
 * where alpha is a scalar, A is an m-by-n matrix, T is a quasi-triangular
 * matrix (upper or lower), and op(T) is T or T'.
 *
 * @param[in] side 'L': A := alpha*op(T)*A, 'R': A := alpha*A*op(T)
 * @param[in] uplo 'U': T is upper quasi-triangular, 'L': T is lower quasi-triangular
 * @param[in] trans 'N': op(T)=T, 'T'/'C': op(T)=T'
 * @param[in] m Rows of A. m >= 0.
 * @param[in] n Columns of A. n >= 0.
 * @param[in] alpha Scalar multiplier. When alpha=0, T not referenced, A set to zero.
 * @param[in] t Quasi-triangular matrix T (m-by-m if side='L', n-by-n if side='R').
 *              If UPLO='U', upper Hessenberg (upper triangular + subdiagonal).
 *              If UPLO='L', lower Hessenberg (lower triangular + superdiagonal).
 * @param[in] ldt Leading dimension of T, ldt >= max(1,k) where k=m if side='L', k=n if side='R'.
 * @param[in,out] a m-by-n matrix. On exit: result overwrites A.
 * @param[in] lda Leading dimension of A, lda >= max(1,m).
 * @param[out] dwork Workspace array, dimension (ldwork).
 *                   On exit, dwork[0] returns optimal ldwork.
 * @param[in] ldwork Length of dwork. ldwork >= 1 if alpha=0 or min(m,n)=0.
 *                   ldwork >= 2*(m-1) if side='L'. ldwork >= 2*(n-1) if side='R'.
 *                   If ldwork=-1, workspace query mode.
 * @param[out] info 0 on success, -i if argument i invalid.
 */
void mb01ux(const char* side, const char* uplo, const char* trans,
            i32 m, i32 n, f64 alpha, f64* t, i32 ldt, f64* a, i32 lda,
            f64* dwork, i32 ldwork, i32* info);

/**
 * @brief Compute T := alpha*op(T)*A or T := alpha*A*op(T) with T triangular (complex).
 *
 * Computes one of the matrix products:
 *     T := alpha*op(T)*A   (SIDE='L')
 *     T := alpha*A*op(T)   (SIDE='R')
 * where alpha is a complex scalar, A is an m-by-n complex matrix, T is a
 * triangular (upper or lower) complex matrix, and op(T) is T, T', or conj(T').
 *
 * A block-row/column algorithm is used when possible. Result overwrites T.
 *
 * @param[in] side 'L': T := alpha*op(T)*A, 'R': T := alpha*A*op(T)
 * @param[in] uplo 'U': T is upper triangular, 'L': T is lower triangular
 * @param[in] trans 'N': op(T)=T, 'T': op(T)=T', 'C': op(T)=conj(T')
 * @param[in] m Rows of A. m >= 0.
 * @param[in] n Columns of A. n >= 0.
 * @param[in] alpha Complex scalar multiplier. When alpha=0, T set to zero.
 * @param[in,out] t COMPLEX*16 array, dimension (ldt,max(K,N)) if SIDE='L',
 *                  (ldt,K) if SIDE='R', where K=M if SIDE='L', K=N if SIDE='R'.
 *                  In: K-by-K triangular matrix T
 *                  Out: M-by-N result matrix
 * @param[in] ldt Leading dimension of T (ldt >= max(1,M) if SIDE='L',
 *                ldt >= max(1,M,N) if SIDE='R')
 * @param[in] a COMPLEX*16 array, dimension (lda,n), M-by-N matrix A
 * @param[in] lda Leading dimension of A (lda >= max(1,M))
 * @param[out] zwork COMPLEX*16 workspace array, dimension (lzwork).
 *                   On exit, zwork[0] returns optimal lzwork.
 * @param[in] lzwork Workspace size (lzwork >= 1 if alpha=0 or min(M,N)=0,
 *                   lzwork >= M if SIDE='L', lzwork >= N if SIDE='R'.
 *                   If lzwork=-1, workspace query mode)
 * @param[out] info Exit code (0=success, <0=invalid parameter)
 */
void mb01uz(const char* side, const char* uplo, const char* trans,
            i32 m, i32 n, c128 alpha, c128* t, i32 ldt, const c128* a, i32 lda,
            c128* zwork, i32 lzwork, i32* info);

/**
 * @brief Compute U'*U or L*L' for triangular matrix (block algorithm).
 *
 * Computes the matrix product U'*U or L*L', where U and L are upper and
 * lower triangular matrices, respectively, stored in the corresponding
 * triangular part of the array A. The result overwrites the input triangle.
 *
 * If UPLO = 'U', computes U'*U and stores upper triangle of result.
 * If UPLO = 'L', computes L*L' and stores lower triangle of result.
 *
 * This routine uses BLAS 3 operations (block algorithm) when the matrix
 * size exceeds the block size, otherwise it uses the unblocked algorithm
 * MB01XY. This is a counterpart of LAPACK DLAUUM which computes U*U' or L'*L.
 *
 * @param[in] uplo 'U': upper triangular U given, compute U'*U
 *                 'L': lower triangular L given, compute L*L'
 * @param[in] n Order of triangular matrix. n >= 0.
 * @param[in,out] a On entry: N-by-N triangular matrix (upper or lower per uplo).
 *                  On exit: corresponding triangle contains result.
 *                  Strictly opposite triangle is not referenced.
 * @param[in] lda Leading dimension of A, lda >= max(1,n).
 * @param[out] info 0 on success, -i if argument i invalid.
 */
void mb01xd(const char* uplo, i32 n, f64* a, i32 lda, i32* info);

/**
 * @brief Compute U'*U or L*L' for triangular matrix (unblocked algorithm).
 *
 * Computes the matrix product U'*U or L*L', where U and L are upper and
 * lower triangular matrices, respectively, stored in the corresponding
 * triangular part of the array A. The result overwrites the input triangle.
 *
 * If UPLO = 'U', computes U'*U and stores upper triangle of result.
 * If UPLO = 'L', computes L*L' and stores lower triangle of result.
 *
 * This is a counterpart of LAPACK DLAUU2 which computes U*U' or L'*L.
 *
 * @param[in] uplo 'U': upper triangular U given, compute U'*U
 *                 'L': lower triangular L given, compute L*L'
 * @param[in] n Order of triangular matrix. n >= 0.
 * @param[in,out] a On entry: N-by-N triangular matrix (upper or lower per uplo).
 *                  On exit: corresponding triangle contains result.
 *                  Strictly opposite triangle is not referenced.
 * @param[in] lda Leading dimension of A, lda >= max(1,n).
 * @param[out] info 0 on success, -i if argument i invalid.
 */
void mb01xy(const char* uplo, i32 n, f64* a, i32 lda, i32* info);

/**
 * @brief Symmetric rank k operation with banded matrix.
 *
 * Performs the symmetric rank k operations:
 *     C := alpha*op(A)*op(A)' + beta*C
 * where alpha and beta are scalars, C is an n-by-n symmetric matrix,
 * op(A) is an n-by-k matrix, and op(A) is A or A'.
 *
 * The matrix A has l nonzero codiagonals, either upper or lower:
 * - If UPLO = 'U': A has L nonzero subdiagonals
 * - If UPLO = 'L': A has L nonzero superdiagonals
 *
 * This is a specialization of DSYRK for banded matrices.
 *
 * @param[in] uplo 'U': upper triangle of C, A has upper triang + L subdiag
 *                 'L': lower triangle of C, A has lower triang + L superdiag
 * @param[in] trans 'N': op(A) = A, 'T'/'C': op(A) = A'
 * @param[in] n Order of matrix C. n >= 0.
 * @param[in] k Columns of op(A). k >= 0.
 * @param[in] l Number of nonzero codiagonals. See constraints below.
 *              For UPLO='U': 0 <= L <= max(0, NROW-1)
 *              For UPLO='L': 0 <= L <= max(0, NCOL-1)
 *              where NROW, NCOL are dimensions of A.
 * @param[in] alpha Scalar multiplier. If alpha=0, A not referenced.
 * @param[in] beta Scalar multiplier for C. If beta=0, C need not be set.
 * @param[in] a Matrix A, dimension (lda, NC) where NC=K if trans='N', NC=N otherwise.
 *              Only upper triangular + L subdiagonals (UPLO='U') or
 *              lower triangular + L superdiagonals (UPLO='L') referenced.
 * @param[in] lda Leading dimension of A, lda >= max(1, NR) where
 *                NR=N if trans='N', NR=K otherwise.
 * @param[in,out] c On entry: symmetric matrix C (triangle per uplo).
 *                  On exit: updated C (same triangle).
 * @param[in] ldc Leading dimension of C, ldc >= max(1,n).
 * @param[out] info 0 on success, -i if argument i invalid.
 */
void mb01yd(const char* uplo, const char* trans, i32 n, i32 k, i32 l,
            f64 alpha, f64 beta, const f64* a, i32 lda,
            f64* c, i32 ldc, i32* info);

/**
 * @brief Compute H := alpha*op(T)*H or H := alpha*H*op(T), H Hessenberg-like.
 *
 * Computes the matrix product:
 *     H := alpha*op(T)*H   (SIDE='L'), or
 *     H := alpha*H*op(T)   (SIDE='R'),
 * where alpha is a scalar, H is an m-by-n upper or lower Hessenberg-like
 * matrix (with L nonzero subdiagonals or superdiagonals), T is a triangular
 * matrix, and op(T) is T or T'.
 *
 * @param[in] side 'L': H := alpha*op(T)*H, 'R': H := alpha*H*op(T)
 * @param[in] uplo 'U': T upper triangular, H upper Hessenberg-like
 *                 'L': T lower triangular, H lower Hessenberg-like
 * @param[in] transt 'N': op(T)=T, 'T'/'C': op(T)=T'
 * @param[in] diag 'U': T is unit triangular, 'N': T is non-unit triangular
 * @param[in] m Rows of H. m >= 0.
 * @param[in] n Columns of H. n >= 0.
 * @param[in] l If UPLO='U': H has L nonzero subdiagonals (0 <= L <= max(0,M-1)).
 *              If UPLO='L': H has L nonzero superdiagonals (0 <= L <= max(0,N-1)).
 * @param[in] alpha Scalar. If alpha=0, T not referenced, H set to zero.
 * @param[in] t Triangular matrix T, dimension (ldt,k) where k=m if SIDE='L', k=n if SIDE='R'.
 * @param[in] ldt Leading dimension of T. ldt >= max(1,m) if SIDE='L', ldt >= max(1,n) if SIDE='R'.
 * @param[in,out] h On entry: m-by-n Hessenberg-like matrix.
 *                  On exit: result alpha*op(T)*H or alpha*H*op(T).
 * @param[in] ldh Leading dimension of H, ldh >= max(1,m).
 * @param[out] info 0 on success, -i if argument i invalid.
 */
void mb01zd(const char* side, const char* uplo, const char* transt,
            const char* diag, i32 m, i32 n, i32 l, f64 alpha,
            const f64* t, i32 ldt, f64* h, i32 ldh, i32* info);

/**
 * @brief Compute residuals of Lyapunov or Stein equations for Cholesky factored solutions.
 *
 * Computes the matrix formula:
 *   R = alpha*(op(A)'*op(T)'*op(T) + op(T)'*op(T)*op(A)) + beta*R  (DICO='C')
 * or
 *   R = alpha*(op(A)'*op(T)'*op(T)*op(A) - op(T)'*op(T)) + beta*R  (DICO='D')
 *
 * where R is symmetric, T is triangular, A is general or Hessenberg,
 * and op(M) = M or M'.
 *
 * @param[in] dico 'C': continuous-time formula (1), 'D': discrete-time formula (2)
 * @param[in] uplo 'U': upper triangular parts of R and T given,
 *                 'L': lower triangular parts of R and T given
 * @param[in] trans 'N': op(M) = M, 'T'/'C': op(M) = M'
 * @param[in] hess 'F': A is full, 'H': A is Hessenberg
 * @param[in] n Order of R, A, T. n >= 0.
 * @param[in] alpha Scalar. If alpha=0, A and T not referenced.
 * @param[in] beta Scalar. If beta=0, R need not be set on entry.
 * @param[in,out] r Symmetric matrix R. On exit, the computed result.
 * @param[in] ldr Leading dimension of R. ldr >= max(1,n).
 * @param[in,out] a Matrix A. On exit: alpha*T'*T*A (DICO='C', TRANS='N') or
 *                  alpha*A*T*T' (DICO='C', TRANS='T') or T*A (DICO='D', TRANS='N')
 *                  or A*T (DICO='D', TRANS='T').
 * @param[in] lda Leading dimension of A. lda >= max(1,n).
 * @param[in] t Triangular matrix T.
 * @param[in] ldt Leading dimension of T. ldt >= max(1,n).
 * @param[out] info 0 on success, -i if argument i is invalid.
 */
void mb01wd(const char* dico, const char* uplo, const char* trans,
            const char* hess, i32 n, f64 alpha, f64 beta,
            f64* r, i32 ldr, f64* a, i32 lda,
            const f64* t, i32 ldt, i32* info);

/**
 * @brief Computes triangular part of alpha*R + beta*op(A)*B or alpha*R + beta*B*op(A).
 *
 * Computes either upper or lower triangular part of:
 *   R = alpha*R + beta*op(A)*B (SIDE='L'), or
 *   R = alpha*R + beta*B*op(A) (SIDE='R')
 * where op(A) = A or A'.
 *
 * @param[in] side 'L': R := alpha*R + beta*op(A)*B,
 *                 'R': R := alpha*R + beta*B*op(A)
 * @param[in] uplo 'U': upper triangular part computed, 'L': lower triangular part
 * @param[in] trans 'N': op(A) = A, 'T'/'C': op(A) = A'
 * @param[in] m Order of result matrix R. m >= 0.
 * @param[in] n Number of columns in B (SIDE='L') or rows in B (SIDE='R'). n >= 0.
 * @param[in] alpha Scalar multiplier for R.
 * @param[in] beta Scalar multiplier for matrix product.
 * @param[in,out] r M-by-M result matrix. Only triangle specified by UPLO modified.
 * @param[in] ldr Leading dimension of R. ldr >= max(1,m).
 * @param[in] a Matrix A. Dimension depends on SIDE/TRANS.
 * @param[in] lda Leading dimension of A.
 * @param[in] b Matrix B. Dimension depends on SIDE.
 * @param[in] ldb Leading dimension of B.
 * @param[out] info 0 on success, -i if argument i invalid.
 */
void mb01rx(const char* side, const char* uplo, const char* trans, i32 m,
            i32 n, f64 alpha, f64 beta, f64* r, i32 ldr, const f64* a,
            i32 lda, const f64* b, i32 ldb, i32* info);

#ifdef __cplusplus
}
#endif

#endif /* SLICOT_MB01_H */
