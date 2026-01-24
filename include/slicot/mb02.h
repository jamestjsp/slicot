/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#ifndef SLICOT_MB02_H
#define SLICOT_MB02_H

#include "../slicot_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Cholesky factorization of positive definite block Toeplitz matrix.
 *
 * Computes the Cholesky factor and the generator and/or the Cholesky factor
 * of the inverse of a symmetric positive definite (s.p.d.) block Toeplitz
 * matrix T, defined by either its first block row, or its first block column,
 * depending on the routine parameter TYPET.
 *
 * @param[in] job Specifies the output:
 *                'G' = only compute generator G of inverse
 *                'R' = compute G and Cholesky factor R of T
 *                'L' = compute G and Cholesky factor L of inv(T)
 *                'A' = compute G, L, and R
 *                'O' = only compute Cholesky factor R of T
 * @param[in] typet Type of T:
 *                  'R' = T contains first block row, R/L upper/lower triangular
 *                  'C' = T contains first block column, R/L lower/upper triangular
 * @param[in] k Block size (number of rows/columns per block). k >= 0
 * @param[in] n Number of blocks. n >= 0
 * @param[in,out] t First block row/column of T, dimension (ldt, n*k)/(ldt, k).
 *                  On exit: Cholesky factor of T(1:k,1:k) and Householder info.
 * @param[in] ldt Leading dimension of t.
 *                ldt >= max(1,k) if typet='R', ldt >= max(1,n*k) if typet='C'.
 * @param[out] g Generator of inverse, dimension (ldg, n*k)/(ldg, 2*k).
 *               To get actual generator, set G(k+1:2*k,1:k)=0 if typet='R',
 *               or G(1:k,k+1:2*k)=0 if typet='C'.
 * @param[in] ldg Leading dimension of g.
 *                ldg >= max(1,2*k) if typet='R' and job!='O',
 *                ldg >= max(1,n*k) if typet='C' and job!='O',
 *                ldg >= 1 if job='O'.
 * @param[out] r Cholesky factor of T, dimension (ldr, n*k).
 *               Upper triangular if typet='R', lower triangular if typet='C'.
 * @param[in] ldr Leading dimension of r.
 *                ldr >= max(1,n*k) if job='R','A','O', ldr >= 1 otherwise.
 * @param[out] l Cholesky factor of inv(T), dimension (ldl, n*k).
 *               Lower triangular if typet='R', upper triangular if typet='C'.
 * @param[in] ldl Leading dimension of l.
 *                ldl >= max(1,n*k) if job='L','A', ldl >= 1 otherwise.
 * @param[out] cs Transformation info for MB02DD, dimension (lcs).
 * @param[in] lcs Length of cs. lcs >= 3*(n-1)*k.
 * @param[out] dwork Workspace, dimension (ldwork).
 *                   On exit: dwork[0] = optimal ldwork.
 * @param[in] ldwork Workspace size. ldwork >= max(1,(n-1)*k).
 * @param[out] info Exit code:
 *                  = 0: success
 *                  < 0: -i means parameter i had illegal value
 *                  = 1: matrix is not numerically positive definite
 */
void mb02cd(const char* job, const char* typet, i32 k, i32 n,
            f64* t, i32 ldt, f64* g, i32 ldg, f64* r, i32 ldr,
            f64* l, i32 ldl, f64* cs, i32 lcs, f64* dwork, i32 ldwork,
            i32* info);

/**
 * @brief Update Cholesky factorization of positive definite block Toeplitz matrix.
 *
 * Updates the Cholesky factor and the generator and/or the Cholesky factor of
 * the inverse of a symmetric positive definite (s.p.d.) block Toeplitz matrix T,
 * given the information from a previous factorization (MB02CD) and additional
 * blocks in TA of its first block row or column.
 *
 * @param[in] job Specifies the output:
 *                'R' = update generator G and compute new columns/rows for R
 *                'A' = update G, compute new columns/rows for R and L
 *                'O' = only compute new columns/rows for R
 * @param[in] typet Type of T:
 *                  'R' = T contains first block row, R/L upper/lower triangular
 *                  'C' = T contains first block column, R/L lower/upper triangular
 * @param[in] k Block size (number of rows/columns per block). k >= 0
 * @param[in] m Number of blocks in TA (additional blocks). m >= 0
 * @param[in] n Number of blocks in T (initial blocks). n >= 0
 * @param[in,out] ta Additional blocks from MB02CD, dimension (ldta, m*k)/(ldta, k).
 *                   On exit: Householder transformation info for subsequent calls.
 * @param[in] ldta Leading dimension of ta.
 *                 ldta >= max(1,k) if typet='R', ldta >= max(1,m*k) if typet='C'.
 * @param[in] t Transformation info from MB02CD, dimension (ldt, n*k)/(ldt, k).
 * @param[in] ldt Leading dimension of t.
 *                ldt >= max(1,k) if typet='R', ldt >= max(1,n*k) if typet='C'.
 * @param[in,out] g Generator of inverse, dimension (ldg, (n+m)*k)/(ldg, 2*k).
 *                  On entry: generator from MB02CD for T.
 *                  On exit: updated generator for extended matrix.
 * @param[in] ldg Leading dimension of g.
 *                ldg >= max(1,2*k) if typet='R' and job!='O',
 *                ldg >= max(1,(n+m)*k) if typet='C' and job!='O',
 *                ldg >= 1 if job='O'.
 * @param[in,out] r Cholesky factor, dimension (ldr, m*k)/(ldr, (n+m)*k).
 *                  On input: last block column/row of previous R.
 *                  On exit: last m*k columns/rows of updated R.
 * @param[in] ldr Leading dimension of r.
 *                ldr >= max(1,(n+m)*k) if typet='R', ldr >= max(1,m*k) if typet='C'.
 * @param[out] l Cholesky factor of inverse, dimension (ldl, (n+m)*k)/(ldl, m*k).
 *               Last m*k rows/columns of L if job='A'.
 * @param[in] ldl Leading dimension of l.
 *                ldl >= max(1,m*k) if typet='R' and job='A',
 *                ldl >= max(1,(n+m)*k) if typet='C' and job='A',
 *                ldl >= 1 otherwise.
 * @param[in,out] cs Transformation info, dimension (lcs).
 *                   On input: 3*(n-1)*k values from MB02CD.
 *                   On exit: 3*(n+m-1)*k values for all transformations.
 * @param[in] lcs Length of cs. lcs >= 3*(n+m-1)*k.
 * @param[out] dwork Workspace, dimension (ldwork).
 *                   On exit: dwork[0] = optimal ldwork.
 * @param[in] ldwork Workspace size. ldwork >= max(1,(n+m-1)*k).
 * @param[out] info Exit code:
 *                  = 0: success
 *                  < 0: -i means parameter i had illegal value
 *                  = 1: matrix is not numerically positive definite
 */
void mb02dd(const char* job, const char* typet, i32 k, i32 m, i32 n,
            f64* ta, i32 ldta, f64* t, i32 ldt, f64* g, i32 ldg,
            f64* r, i32 ldr, f64* l, i32 ldl, f64* cs, i32 lcs,
            f64* dwork, i32 ldwork, i32* info);

/**
 * @brief Bring first part of generator to proper form (block or rank-deficient).
 *
 * Brings the first part of a generator in proper form using orthogonal
 * transformations and modified hyperbolic rotations. Used for computing
 * Cholesky factor of symmetric positive semi-definite block Toeplitz
 * matrices and for solving associated linear systems.
 *
 * @param[in] typeg Generator type:
 *                  'D' = column oriented, rank deficient
 *                  'C' = column oriented, not rank deficient
 *                  'R' = row oriented, not rank deficient
 * @param[in] k Number of rows in A1 to process. k >= 0
 * @param[in] p Columns of positive generator. p >= k
 * @param[in] q Columns in B. If typeg='D': q >= k; else q >= 0
 * @param[in] nb Block size for typeg='C' or 'R'. nb <= 0 means unblocked.
 * @param[in,out] a1 K-by-K matrix, dimension (lda1,k).
 *                   On entry: triangular part of first block.
 *                   On exit: transformed first block.
 * @param[in] lda1 Leading dimension of a1. lda1 >= max(1,k)
 * @param[in,out] a2 Additional part of positive generator.
 *                   Dimension (lda2, p-k) for 'C'/'D', (lda2, k) for 'R'.
 *                   On exit: transformed additional part and Householder info.
 * @param[in] lda2 Leading dimension of a2.
 * @param[in,out] b Negative generator, dimension (ldb, q) for 'C'/'D', (ldb, k) for 'R'.
 *                  On exit: Householder transformation information.
 * @param[in] ldb Leading dimension of b.
 * @param[out] rnk If typeg='D': numerical rank determined (0 <= rnk <= k).
 *                 Otherwise not referenced.
 * @param[out] ipvt If typeg='D': pivot indices, dimension (k).
 *                  Otherwise not referenced.
 * @param[out] cs Rotation/Householder info.
 *                Length depends on typeg and p,q,k.
 * @param[in] tol If typeg='D': tolerance for rank determination.
 *                Otherwise not referenced.
 * @param[out] dwork Workspace, dimension (ldwork).
 * @param[in] ldwork Workspace size.
 *                   If typeg='D': ldwork >= 4*k
 *                   Otherwise: ldwork >= max(1, nb*k, k)
 * @param[out] info Exit code:
 *                  = 0: success
 *                  < 0: -i means parameter i had illegal value
 *                  = 1: matrix is not numerically positive definite
 */
void mb02cu(const char* typeg, i32 k, i32 p, i32 q, i32 nb,
            f64* a1, i32 lda1, f64* a2, i32 lda2,
            f64* b, i32 ldb, i32* rnk, i32* ipvt, f64* cs,
            f64 tol, f64* dwork, i32 ldwork, i32* info);

/**
 * @brief Bring first blocks of generator to proper form.
 *
 * Brings the first blocks of a generator into proper form using
 * QR/LQ decomposition, Householder transformations, and modified
 * hyperbolic rotations. Transformation information is stored for
 * later use by MB02CY.
 *
 * @param[in] typet Generator type:
 *                  'R' = A, B are first blocks of rows of positive/negative generators
 *                  'C' = A, B are first blocks of columns of positive/negative generators
 * @param[in] p Number of rows/columns in A (positive generator). p >= 0
 * @param[in] q Number of rows/columns in B (negative generator). q >= 0
 * @param[in] k Number of columns/rows in A and B to process. p >= k >= 0
 * @param[in,out] a Positive generator, dimension (lda,k)/(lda,p).
 *                  On entry: upper/lower triangular part contains positive generator.
 *                  On exit: transformed positive generator in proper form.
 * @param[in] lda Leading dimension of a.
 *                lda >= max(1,p) if typet='R', lda >= max(1,k) if typet='C'
 * @param[in,out] b Negative generator, dimension (ldb,k)/(ldb,q).
 *                  On entry: negative generator.
 *                  On exit: Householder transformation information.
 * @param[in] ldb Leading dimension of b.
 *                ldb >= max(1,q) if typet='R', ldb >= max(1,k) if typet='C'
 * @param[out] cs Rotation/Householder info, dimension (lcs).
 *                Contains 2*k hyperbolic rotation params + min(k,q) Householder factors.
 * @param[in] lcs Length of cs. lcs >= 2*k + min(k,q)
 * @param[out] dwork Workspace, dimension (ldwork).
 *                   On exit: dwork[0] = optimal ldwork
 * @param[in] ldwork Workspace size. ldwork >= max(1,k)
 * @param[out] info Exit code:
 *                  = 0: success
 *                  < 0: -i means parameter i had illegal value
 *                  = 1: matrix is not numerically positive definite
 */
void mb02cx(const char* typet, i32 p, i32 q, i32 k,
            f64* a, i32 lda, f64* b, i32 ldb,
            f64* cs, i32 lcs, f64* dwork, i32 ldwork, i32* info);

/**
 * @brief Apply hyperbolic transformations to generator columns/rows.
 *
 * Applies transformations computed by MB02CX on other columns/rows of
 * the generator contained in arrays A (positive generator) and B
 * (negative generator).
 *
 * @param[in] typet Generator type:
 *                  'R' = A, B are additional columns of the generator
 *                  'C' = A, B are additional rows of the generator
 * @param[in] strucg Structure info:
 *                   'T' = trailing block of positive generator is triangular,
 *                         trailing block of negative generator is zero
 *                   'N' = no special structure
 * @param[in] p Number of rows/columns in A (positive generators). p >= 0
 * @param[in] q Number of rows/columns in B (negative generators). q >= 0
 * @param[in] n Number of columns/rows in A and B to process. n >= 0
 * @param[in] k Number of columns/rows in H. p >= k >= 0
 * @param[in,out] a Positive generator, dimension (lda,n)/(lda,p).
 *                  On exit: transformed positive generator
 * @param[in] lda Leading dimension of a.
 *                lda >= max(1,p) if typet='R', lda >= max(1,n) if typet='C'
 * @param[in,out] b Negative generator, dimension (ldb,n)/(ldb,q).
 *                  On exit: transformed negative generator
 * @param[in] ldb Leading dimension of b.
 *                ldb >= max(1,q) if typet='R', ldb >= max(1,n) if typet='C'
 * @param[in] h Householder transformation info from MB02CX, dimension (ldh,k)/(ldh,q)
 * @param[in] ldh Leading dimension of h.
 *                ldh >= max(1,q) if typet='R', ldh >= max(1,k) if typet='C'
 * @param[in] cs Rotation/Householder info from MB02CX, dimension (lcs).
 *               Contains 2*k + min(k,q) elements
 * @param[in] lcs Length of cs. lcs >= 2*k + min(k,q)
 * @param[out] dwork Workspace, dimension (ldwork).
 *                   On exit: dwork[0] = optimal ldwork
 * @param[in] ldwork Workspace size. ldwork >= max(1,n)
 * @param[out] info Exit code: 0=success, <0=invalid parameter
 */
void mb02cy(const char* typet, const char* strucg, i32 p, i32 q, i32 n, i32 k,
            f64* a, i32 lda, f64* b, i32 ldb, f64* h, i32 ldh,
            f64* cs, i32 lcs, f64* dwork, i32 ldwork, i32* info);

/**
 * @brief Apply MB02CU transformations to other generator columns/rows.
 *
 * Applies the Householder transformations and modified hyperbolic rotations
 * computed by MB02CU to other columns/rows of the generator contained in
 * arrays F1, F2 and G.
 *
 * @param[in] typeg Generator type:
 *                  'D' = column oriented, rank deficient
 *                  'C' = column oriented, not rank deficient
 *                  'R' = row oriented, not rank deficient
 * @param[in] strucg Structure info:
 *                   'T' = trailing block of positive generator is triangular,
 *                         trailing block of negative generator is zero
 *                   'N' = no special structure
 * @param[in] k Number of rows in A1 to process. k >= 0
 * @param[in] n If TYPEG='D' or 'C': rows in F1; if TYPEG='R': columns in F1. n >= 0
 * @param[in] p Columns of positive generator. p >= k
 * @param[in] q Columns in B. If TYPEG='D': q >= k; else q >= 0
 * @param[in] nb Block size for TYPEG='C' or 'R' (must match MB02CU)
 * @param[in] rnk If TYPEG='D': rank from MB02CU (0 <= rnk <= k); else unused
 * @param[in] a1 If TYPEG='D': K-by-K matrix A1 from MB02CU. Else not referenced.
 * @param[in] lda1 Leading dimension of A1
 * @param[in] a2 Matrix A2 from MB02CU
 * @param[in] lda2 Leading dimension of A2
 * @param[in] b Matrix B from MB02CU
 * @param[in] ldb Leading dimension of B
 * @param[in,out] f1 First part of positive generator to transform
 * @param[in] ldf1 Leading dimension of F1
 * @param[in,out] f2 Second part of positive generator to transform
 * @param[in] ldf2 Leading dimension of F2
 * @param[in,out] g Negative part of generator to transform
 * @param[in] ldg Leading dimension of G
 * @param[in] cs Rotation/Householder info from MB02CU
 * @param[out] dwork Workspace. On error -23, dwork[0] = minimum required size.
 * @param[in] ldwork Workspace size
 * @param[out] info Exit code: 0=success, <0=invalid parameter
 */
void mb02cv(const char* typeg, const char* strucg, i32 k, i32 n, i32 p, i32 q,
            i32 nb, i32 rnk, f64* a1, i32 lda1, f64* a2, i32 lda2,
            f64* b, i32 ldb, f64* f1, i32 ldf1, f64* f2, i32 ldf2,
            f64* g, i32 ldg, const f64* cs, f64* dwork, i32 ldwork, i32* info);

/**
 * @brief Solve symmetric positive definite block Toeplitz system.
 *
 * Solves T*X = B or X*T = B where T is a symmetric positive definite
 * block Toeplitz matrix defined by its first block row or column.
 *
 * @param[in] typet Type of T:
 *                  'R' = T contains first block row, solve X*T = B
 *                  'C' = T contains first block column, solve T*X = B
 * @param[in] k Block size (number of rows/columns in each block). k >= 0
 * @param[in] n Number of blocks. n >= 0
 * @param[in] nrhs Number of right-hand sides. nrhs >= 0
 * @param[in,out] t Block Toeplitz data, dimension (ldt, n*k) or (ldt, k).
 *                  On entry: first block row/column of T.
 *                  On exit: last row/column of Cholesky factor of inv(T).
 * @param[in] ldt Leading dimension of t.
 *                ldt >= max(1,k) if typet='R', ldt >= max(1,n*k) if typet='C'
 * @param[in,out] b Right-hand side matrix, dimension (ldb, n*k) or (ldb, nrhs).
 *                  On entry: matrix B.
 *                  On exit: solution matrix X.
 * @param[in] ldb Leading dimension of b.
 *                ldb >= max(1,nrhs) if typet='R', ldb >= max(1,n*k) if typet='C'
 * @param[out] dwork Workspace, dimension (ldwork).
 *                   On exit: dwork[0] = optimal ldwork
 * @param[in] ldwork Workspace size. ldwork >= max(1, n*k*k + (n+2)*k)
 * @param[out] info Exit code:
 *                  = 0: success
 *                  < 0: -i means parameter i had illegal value
 *                  = 1: matrix is not numerically positive definite
 */
void mb02ed(const char* typet, i32 k, i32 n, i32 nrhs,
            f64* t, i32 ldt, f64* b, i32 ldb,
            f64* dwork, i32 ldwork, i32* info);

/**
 * @brief Solve linear equations with LU factorization and iterative refinement.
 *
 * Solves op(A)*X = B using LU factorization with optional equilibration
 * and iterative refinement.
 *
 * @param[in] fact Factorization: 'F'=factored, 'N'=factor A, 'E'=equilibrate+factor
 * @param[in] trans Form: 'N'=A*X=B, 'T'/'C'=A'*X=B
 * @param[in] n Order of A (n >= 0)
 * @param[in] nrhs Number of right-hand sides (nrhs >= 0)
 * @param[in,out] a N-by-N matrix A, dimension (lda,n)
 * @param[in] lda Leading dimension of A
 * @param[in,out] af N-by-N LU factors, dimension (ldaf,n)
 * @param[in] ldaf Leading dimension of AF
 * @param[in,out] ipiv Pivot indices, dimension (n)
 * @param[in,out] equed Equilibration type: 'N','R','C','B'
 * @param[in,out] r Row scale factors, dimension (n)
 * @param[in,out] c Column scale factors, dimension (n)
 * @param[in,out] b N-by-NRHS right-hand side B, dimension (ldb,nrhs)
 * @param[in] ldb Leading dimension of B
 * @param[out] x N-by-NRHS solution X, dimension (ldx,nrhs)
 * @param[in] ldx Leading dimension of X
 * @param[out] rcond Reciprocal condition number estimate
 * @param[out] ferr Forward error bounds, dimension (nrhs)
 * @param[out] berr Backward error bounds, dimension (nrhs)
 * @param[out] iwork Integer workspace, dimension (n)
 * @param[out] dwork Double workspace, dimension (4*n)
 * @param[out] info 0=success, <0=invalid param, >0=singular
 */
void mb02pd(
    const char* fact,
    const char* trans,
    const i32 n,
    const i32 nrhs,
    f64* a,
    const i32 lda,
    f64* af,
    const i32 ldaf,
    i32* ipiv,
    char* equed,
    f64* r,
    f64* c,
    f64* b,
    const i32 ldb,
    f64* x,
    const i32 ldx,
    f64* rcond,
    f64* ferr,
    f64* berr,
    i32* iwork,
    f64* dwork,
    i32* info);

/**
 * @brief Minimum-norm least squares solution using rank-revealing QR.
 *
 * Determines the minimum-norm solution to a real linear least squares
 * problem: minimize || A * X - B ||, using the rank-revealing QR
 * factorization of a real general M-by-N matrix A, computed by MB03OD.
 *
 * The input data for MB02QY are the transformed matrices Q'*A and Q'*B.
 * Matrix Q is not needed.
 *
 * @param[in] m Number of rows of matrices A and B (m >= 0)
 * @param[in] n Number of columns of matrix A (n >= 0)
 * @param[in] nrhs Number of columns of matrix B (nrhs >= 0)
 * @param[in] rank Effective rank of A, as returned by MB03OD (0 <= rank <= min(m,n))
 * @param[in,out] a DOUBLE PRECISION array, dimension (lda,n).
 *                  On entry: leading min(m,n)-by-n upper trapezoidal part
 *                  contains triangular factor R from QR factorization.
 *                  On exit if rank < n: leading rank-by-rank upper triangular
 *                  part contains factor R from complete orthogonal factorization;
 *                  submatrix (1:rank, rank+1:n) with tau represents orthogonal
 *                  matrix Z as product of rank elementary reflectors.
 *                  Unchanged if rank = n.
 * @param[in] lda Leading dimension of a (lda >= max(1,m))
 * @param[in] jpvt INTEGER array, dimension (n).
 *                 Column permutation from MB03OD: column i of A*P was column
 *                 jpvt[i] of original matrix A.
 * @param[in,out] b DOUBLE PRECISION array, dimension (ldb,nrhs).
 *                  On entry if nrhs > 0: leading m-by-nrhs part contains
 *                  matrix B (corresponding to transformed matrix A from MB03OD).
 *                  On exit if nrhs > 0: leading n-by-nrhs part contains
 *                  solution matrix X. If m >= n and rank = n, residual
 *                  sum-of-squares for column i is sum of squares of
 *                  elements n+1:m in that column.
 * @param[in] ldb Leading dimension of b.
 *                ldb >= max(1,m,n) if nrhs > 0, ldb >= 1 if nrhs = 0.
 * @param[out] tau DOUBLE PRECISION array, dimension (min(m,n)).
 *                 Scalar factors of elementary reflectors.
 *                 Not referenced if rank = n.
 * @param[out] dwork DOUBLE PRECISION array, dimension (ldwork).
 *                   On exit: dwork[0] = optimal ldwork.
 * @param[in] ldwork Workspace size. ldwork >= max(1,n,nrhs).
 *                   If ldwork = -1, workspace query is assumed.
 * @param[out] info Exit code: 0 = success, < 0 = -i means param i invalid.
 */
void mb02qy(i32 m, i32 n, i32 nrhs, i32 rank, f64 *a, i32 lda,
            const i32 *jpvt, f64 *b, i32 ldb, f64 *tau,
            f64 *dwork, i32 ldwork, i32 *info);

/**
 * @brief Solve linear system with Hessenberg LU factorization.
 *
 * Solves a system of linear equations H*X = B or H'*X = B with an upper
 * Hessenberg N-by-N matrix H using the LU factorization computed by MB02SD.
 *
 * @param[in] trans 'N' = H*X = B, 'T' or 'C' = H'*X = B
 * @param[in] n Order of matrix H (n >= 0)
 * @param[in] nrhs Number of right-hand sides (nrhs >= 0)
 * @param[in] h LU factors from MB02SD, dimension (ldh,n)
 * @param[in] ldh Leading dimension of H (ldh >= max(1,n))
 * @param[in] ipiv Pivot indices from MB02SD, dimension (n)
 * @param[in,out] b On entry: RHS matrix B. On exit: solution X.
 * @param[in] ldb Leading dimension of B (ldb >= max(1,n))
 * @param[out] info Exit code: 0 = success, < 0 = invalid arg
 */
void mb02rd(const char* trans, i32 n, i32 nrhs, const f64* h, i32 ldh,
            const i32* ipiv, f64* b, i32 ldb, i32* info);

/**
 * @brief LU factorization of upper Hessenberg matrix.
 *
 * Computes an LU factorization of an n-by-n upper Hessenberg matrix H
 * using partial pivoting with row interchanges.
 *
 * The factorization has the form H = P * L * U where P is a permutation
 * matrix, L is lower triangular with unit diagonal elements (and one
 * nonzero subdiagonal), and U is upper triangular.
 *
 * @param[in] n Order of matrix H (n >= 0)
 * @param[in,out] h On entry: n-by-n upper Hessenberg matrix.
 *                  On exit: L and U factors; unit diagonal of L not stored.
 * @param[in] ldh Leading dimension of H (ldh >= max(1,n))
 * @param[out] ipiv Pivot indices, dimension (n). Row i was interchanged
 *                  with row ipiv[i].
 * @param[out] info Exit code: 0 = success, < 0 = invalid arg,
 *                  > 0 = U(i,i) is exactly zero (singular)
 */
void mb02sd(i32 n, f64* h, i32 ldh, i32* ipiv, i32* info);

/**
 * @brief Estimate reciprocal condition number of upper Hessenberg matrix.
 *
 * Estimates the reciprocal of the condition number of an upper Hessenberg
 * matrix H, in either the 1-norm or the infinity-norm, using the LU
 * factorization computed by mb02sd.
 *
 * @param[in] norm '1' or 'O' for 1-norm, 'I' for infinity-norm
 * @param[in] n Order of matrix H. n >= 0
 * @param[in] hnorm The norm of the original matrix H (1-norm or inf-norm)
 * @param[in] h LU factors from mb02sd, dimension (ldh, n)
 * @param[in] ldh Leading dimension of h. ldh >= max(1, n)
 * @param[in] ipiv Pivot indices from mb02sd, dimension (n)
 * @param[out] rcond Reciprocal condition number = 1/(norm(H)*norm(inv(H)))
 * @param[out] iwork Integer workspace, dimension (n)
 * @param[out] dwork Double workspace, dimension (3*n)
 * @param[out] info Exit code: 0 = success, < 0 = invalid arg -i
 */
void mb02td(const char* norm, i32 n, f64 hnorm, const f64* h, i32 ldh,
            const i32* ipiv, f64* rcond, i32* iwork, f64* dwork, i32* info);

/**
 * @brief Minimum norm least squares solution using SVD.
 *
 * Computes the minimum norm least squares solution of one of:
 *   op(R)*X = alpha*B     (SIDE='L')
 *   X*op(R) = alpha*B     (SIDE='R')
 *
 * where alpha is a real scalar, op(R) is R or R', R is an L-by-L real
 * upper triangular matrix, B is an M-by-N real matrix, and L = M for
 * SIDE='L' or L = N for SIDE='R'. Uses singular value decomposition,
 * R = Q*S*P', assuming R may be rank deficient.
 *
 * @param[in] fact 'N' to compute SVD, 'F' if SVD already factored
 * @param[in] side 'L' for op(R)*X = alpha*B, 'R' for X*op(R) = alpha*B
 * @param[in] trans 'N' for op(R)=R, 'T'/'C' for op(R)=R'
 * @param[in] jobp 'P' to compute/use pinv(R), 'N' otherwise
 * @param[in] m Number of rows of B (m >= 0)
 * @param[in] n Number of columns of B (n >= 0)
 * @param[in] alpha Scalar multiplier (if 0, X set to zero)
 * @param[in] rcond Rank threshold: SV(i) <= RCOND*SV(1) treated as zero
 *                  If RCOND <= 0, uses machine epsilon. Not used if FACT='F'.
 * @param[in,out] rank Rank of R. Input if FACT='F', output if FACT='N'.
 * @param[in,out] r DOUBLE PRECISION array, dimension (ldr,L)
 *                  If FACT='N': In: upper triangular R
 *                              Out: P' (scaled by inv(S) if JOBP='P')
 *                  If FACT='F': In: P' from prior SVD
 * @param[in] ldr Leading dimension of r (ldr >= max(1,L))
 * @param[in,out] q DOUBLE PRECISION array, dimension (ldq,L)
 *                  If FACT='N': Out: orthogonal matrix Q
 *                  If FACT='F': In: orthogonal matrix Q from prior SVD
 * @param[in] ldq Leading dimension of q (ldq >= max(1,L))
 * @param[in,out] sv DOUBLE PRECISION array, dimension (L)
 *                   If FACT='N': Out: first RANK entries are 1/SV(1:RANK),
 *                                rest are SV(RANK+1:L) in descending order
 *                   If FACT='F': In: reciprocal SVs and remaining SVs
 * @param[in,out] b DOUBLE PRECISION array, dimension (ldb,n)
 *                  In: matrix B (if alpha != 0)
 *                  Out: solution X
 * @param[in] ldb Leading dimension of b (ldb >= max(1,m))
 * @param[out] rp DOUBLE PRECISION array, dimension (ldrp,L)
 *                If JOBP='P' and RANK > 0: pinv(R)
 *                Not referenced if JOBP='N'
 * @param[in] ldrp Leading dimension of rp (ldrp >= L if JOBP='P', else >= 1)
 * @param[out] dwork DOUBLE PRECISION array, dimension (ldwork)
 *                   On exit: dwork[0] = optimal ldwork
 * @param[in] ldwork Workspace size (>= max(1,L) if FACT='F', >= max(1,5*L) if FACT='N')
 *                   Optimal: max(1,L,M*N) if FACT='F', max(1,5*L,M*N) if FACT='N'
 *                   If ldwork=-1, workspace query
 * @param[out] info Exit code (0=success, <0=invalid parameter,
 *                  >0=SVD did not converge)
 */
void mb02ud(
    const char* fact, const char* side, const char* trans, const char* jobp,
    const i32 m, const i32 n, const f64 alpha, const f64 rcond,
    i32* rank, f64* r, const i32 ldr, f64* q, const i32 ldq,
    f64* sv, f64* b, const i32 ldb, f64* rp, const i32 ldrp,
    f64* dwork, const i32 ldwork, i32* info
);

/**
 * @brief Solve linear system using LU factorization with complete pivoting.
 *
 * Solves A*x = scale*rhs using the LU factorization computed by MB02UV:
 *     A = P * L * U * Q
 * where P and Q are permutation matrices, L is unit lower triangular,
 * and U is upper triangular.
 *
 * @param[in] n Order of matrix A
 * @param[in] a LU factors from MB02UV, dimension (lda,n)
 * @param[in] lda Leading dimension of A, lda >= max(1,n)
 * @param[in,out] rhs On entry: right-hand side vector, dimension (n).
 *                    On exit: solution vector x.
 * @param[in] ipiv Row pivot indices from MB02UV, dimension (n)
 * @param[in] jpiv Column pivot indices from MB02UV, dimension (n)
 * @param[out] scale Scale factor (0 < scale <= 1) applied to prevent overflow
 *
 * @note For speed, no input validation. Use only for small matrices.
 */
void mb02uu(i32 n, const f64* a, i32 lda, f64* rhs, const i32* ipiv,
            const i32* jpiv, f64* scale);

/**
 * @brief LU factorization with complete pivoting.
 *
 * Computes an LU factorization with complete pivoting:
 *     A = P * L * U * Q
 * where P and Q are permutation matrices, L is unit lower triangular
 * with diagonal elements 1, and U is upper triangular.
 *
 * If U(k,k) appears to be less than SMIN (a threshold), it is given
 * the value SMIN, yielding a nonsingular perturbed system (warning).
 *
 * @param[in] n Order of matrix A
 * @param[in,out] a On entry: N-by-N matrix to factor, dimension (lda,n).
 *                  On exit: L (unit diagonal not stored) and U factors.
 * @param[in] lda Leading dimension of A, lda >= max(1,n)
 * @param[out] ipiv Row pivot indices, dimension (n). Row i was interchanged
 *                  with row ipiv[i].
 * @param[out] jpiv Column pivot indices, dimension (n). Column j was
 *                  interchanged with column jpiv[j].
 * @param[out] info Exit code: 0=success, k>0=U(k,k) was perturbed (warning)
 *
 * @note For speed, no input validation. Use only for small matrices.
 */
void mb02uv(i32 n, f64* a, i32 lda, i32* ipiv, i32* jpiv, i32* info);

/**
 * @brief Solve X * op(A) = B using LU factorization.
 *
 * Solves the system X * op(A) = B for X, where op(A) is either A or A'.
 * Uses LU factorization with partial pivoting.
 *
 * @param[in] trans Specifies op(A): 'N' = A, 'T' or 'C' = A'
 * @param[in] m Number of rows of B (m >= 0)
 * @param[in] n Order of A (n >= 0)
 * @param[in,out] a On entry: n-by-n matrix A. On exit: LU factors of A.
 * @param[in] lda Leading dimension of A (lda >= max(1,n))
 * @param[out] ipiv Pivot indices from DGETRF, dimension (n)
 * @param[in,out] b On entry: m-by-n RHS matrix B. On exit: m-by-n solution X.
 * @param[in] ldb Leading dimension of B (ldb >= max(1,m))
 * @param[out] info Exit code: 0 = success, < 0 = -i means i-th arg invalid,
 *                  > 0 = U(i,i) is exactly zero (singular matrix)
 */
void mb02vd(const char* trans, i32 m, i32 n, f64* a, i32 lda,
            i32* ipiv, f64* b, i32 ldb, i32* info);

/**
 * @brief Function type for implicit matrix-vector product in MB02WD.
 *
 * For FORM='F', MB02WD requires a user-supplied function that computes
 * the matrix-vector product y = f(A, x), where A is implicitly defined.
 *
 * @param n Dimension of vector x (n >= 0)
 * @param ipar Integer parameters describing matrix structure
 * @param lipar Length of ipar array
 * @param dpar Real parameters for the problem
 * @param ldpar Length of dpar array
 * @param a Compressed representation of matrix A
 * @param lda Leading dimension of A
 * @param x On entry: input vector. On exit: output y = f(A, x)
 * @param incx Increment for elements of x
 * @param dwork Workspace array
 * @param ldwork Workspace size
 * @param info Error indicator (0 = success)
 */
typedef void (*mb02wd_func)(
    i32 n, i32* ipar, i32 lipar, f64* dpar, i32 ldpar,
    f64* a, i32 lda, f64* x, i32 incx,
    f64* dwork, i32 ldwork, i32* info
);

/**
 * @brief Conjugate gradient solver for SPD linear systems.
 *
 * Solves the system of linear equations Ax = b, with A symmetric,
 * positive definite, or in the implicit form f(A, x) = b, where
 * y = f(A, x) is a symmetric positive definite linear mapping,
 * using the conjugate gradient (CG) algorithm without preconditioning.
 *
 * The CG iteration used is:
 *   Start: q(0) = r(0) = Ax - b
 *   alpha(k) = -<q(k), r(k)> / <q(k), A*q(k)>
 *   x(k+1) = x(k) - alpha(k) * q(k)
 *   r(k+1) = r(k) - alpha(k) * A*q(k)
 *   beta(k) = <r(k+1), r(k+1)> / <r(k), r(k)>
 *   q(k+1) = r(k+1) + beta(k) * q(k)
 *
 * @param[in] form Equation form:
 *                 'U' = Ax=b using upper triangle of A
 *                 'L' = Ax=b using lower triangle of A
 *                 'F' = implicit form f(A,x)=b
 * @param[in] f Function computing y=f(A,x) when form='F', NULL otherwise
 * @param[in] n Dimension of x (n >= 0). Also size of A for form='U'/'L'.
 * @param[in] ipar Integer parameters for function f (ignored if form!='F')
 * @param[in] lipar Length of ipar (lipar >= 0)
 * @param[in] dpar Real parameters for function f (ignored if form!='F')
 * @param[in] ldpar Length of dpar (ldpar >= 0)
 * @param[in] itmax Maximum iterations (itmax >= 0)
 * @param[in] a SPD matrix A, dimension (lda,n) for form='U'/'L',
 *              or compressed representation for form='F'
 * @param[in] lda Leading dimension of A (lda >= max(1,n) for form='U'/'L')
 * @param[in] b Right-hand side vector, dimension (1+(n-1)*incb)
 * @param[in] incb Increment for b (incb > 0)
 * @param[in,out] x On entry: initial approximation (use zeros if unknown).
 *                  On exit: computed solution. Dimension (1+(n-1)*incx).
 * @param[in] incx Increment for x (incx > 0)
 * @param[in] tol Absolute tolerance. If tol > 0, algorithm stops when
 *                ||Ax-b||_2 <= tol. If tol <= 0, uses default n*eps*||b||_2.
 * @param[out] dwork Workspace, dimension (ldwork).
 *                   On exit: dwork[0] = iterations performed,
 *                            dwork[1] = final residual ||Ax-b||_2.
 * @param[in] ldwork Workspace size. Must be >= max(2, 3*n) for form='U'/'L',
 *                   >= max(2, 3*n + workspace_for_f) for form='F'.
 * @param[out] iwarn Warning indicator:
 *                   0 = no warning
 *                   1 = max iterations reached without achieving tol
 *                   2 = itmax is zero (dwork[1] not set)
 * @param[out] info Error indicator:
 *                  0 = success
 *                  < 0 = -i means i-th argument is invalid
 *                  > 0 = function f returned with error
 */
void mb02wd(
    const char* form,
    mb02wd_func f,
    i32 n,
    i32* ipar,
    i32 lipar,
    f64* dpar,
    i32 ldpar,
    i32 itmax,
    f64* a,
    i32 lda,
    const f64* b,
    i32 incb,
    f64* x,
    i32 incx,
    f64 tol,
    f64* dwork,
    i32 ldwork,
    i32* iwarn,
    i32* info
);

/**
 * @brief Callback function type for MB02XD implicit form.
 *
 * Computes A'*A for the implicit form f(A) = A'*A.
 *
 * @param[in] stor Storage scheme ('F' full, 'P' packed)
 * @param[in] uplo Which triangle ('U' upper, 'L' lower)
 * @param[in] n Order of matrix A'*A
 * @param[in] ipar Integer parameters for matrix structure
 * @param[in] lipar Length of ipar
 * @param[in] dpar Real parameters
 * @param[in] ldpar Length of dpar
 * @param[in] a Matrix A
 * @param[in] lda Leading dimension of a
 * @param[out] ata Output matrix A'*A
 * @param[in] ldata Leading dimension of ata
 * @param[in,out] dwork Workspace
 * @param[in] ldwork Length of dwork
 * @param[out] info Error indicator
 */
typedef void (*mb02xd_callback)(
    const char* stor,
    const char* uplo,
    const i32* n,
    const i32* ipar,
    const i32* lipar,
    const f64* dpar,
    const i32* ldpar,
    const f64* a,
    const i32* lda,
    f64* ata,
    const i32* ldata,
    f64* dwork,
    const i32* ldwork,
    i32* info
);

/**
 * @brief Solve A'*A*X = B using Cholesky factorization.
 *
 * Solves a set of systems of linear equations A'*A*X = B, or in implicit
 * form f(A)*X = B, with A'*A or f(A) positive definite, using symmetric
 * Gaussian elimination (Cholesky factorization).
 *
 * @param[in] form Form of matrix ('S' standard, 'F' function/implicit)
 * @param[in] stor Storage scheme ('F' full, 'P' packed)
 * @param[in] uplo Which triangle to store ('U' upper, 'L' lower)
 * @param[in] f Callback for implicit form (NULL if form='S')
 * @param[in] m Number of rows of A (m >= 0)
 * @param[in] n Number of columns of A, order of A'*A (n >= 0)
 * @param[in] nrhs Number of right-hand sides (nrhs >= 0)
 * @param[in] ipar Integer parameters for implicit form (ignored if form='S')
 * @param[in] lipar Length of ipar (lipar >= 0)
 * @param[in] dpar Real parameters for implicit form (ignored if form='S')
 * @param[in] ldpar Length of dpar (ldpar >= 0)
 * @param[in] a Matrix A, dimension (lda, n) if form='S'
 * @param[in] lda Leading dimension of a (lda >= max(1,m) if form='S')
 * @param[in,out] b Right-hand side B, dimension (ldb, nrhs)
 *                  On exit: solution X if info=0 and m>0
 * @param[in] ldb Leading dimension of b (ldb >= max(1,n))
 * @param[out] ata Cholesky factor of A'*A
 *                 If stor='F': dimension (ldata, n)
 *                 If stor='P': dimension (n*(n+1)/2)
 * @param[in] ldata Leading dimension of ata
 *                  If stor='F': ldata >= max(1,n)
 *                  If stor='P': ldata >= 1
 * @param[in,out] dwork Workspace
 * @param[in] ldwork Length of dwork
 * @param[out] info Error indicator:
 *                  = 0: success
 *                  < 0: parameter -info had illegal value
 *                  > 0: if info <= n, (info,info) element of Cholesky factor
 *                       is zero (A'*A singular)
 *                       if info > n, callback f returned info-n
 */
void mb02xd(
    const char* form,
    const char* stor,
    const char* uplo,
    mb02xd_callback f,
    const i32* m,
    const i32* n,
    const i32* nrhs,
    const i32* ipar,
    const i32* lipar,
    const f64* dpar,
    const i32* ldpar,
    const f64* a,
    const i32* lda,
    f64* b,
    const i32* ldb,
    f64* ata,
    const i32* ldata,
    f64* dwork,
    const i32* ldwork,
    i32* info
);

/**
 * @brief Solve triangular matrix equation with condition estimation.
 *
 * Solves one of the matrix equations:
 *   op(A)*X = alpha*B,   or   X*op(A) = alpha*B,
 * where alpha is a scalar, X and B are m-by-n matrices, A is a unit
 * or non-unit, upper or lower triangular matrix, and op(A) is A or A'.
 *
 * Computes reciprocal condition number RCOND = 1/(norm(A)*norm(inv(A)))
 * and only solves if RCOND > TOL.
 *
 * @param[in] side 'L' for op(A)*X = alpha*B, 'R' for X*op(A) = alpha*B
 * @param[in] uplo 'U' for upper triangular, 'L' for lower triangular
 * @param[in] trans 'N' for op(A)=A, 'T' or 'C' for op(A)=A'
 * @param[in] diag 'U' for unit triangular, 'N' for non-unit triangular
 * @param[in] norm '1' or 'O' for 1-norm, 'I' for infinity-norm
 * @param[in] m Rows of B (m >= 0)
 * @param[in] n Columns of B (n >= 0)
 * @param[in] alpha Scalar multiplier. If 0, A is not referenced.
 * @param[in] a Triangular matrix, dimension (lda, k) where k=m if side='L', k=n if side='R'
 * @param[in] lda Leading dimension of a. lda >= max(1,m) if side='L', >= max(1,n) if side='R'
 * @param[in,out] b On entry: m-by-n RHS matrix B. On exit: solution X if info=0.
 * @param[in] ldb Leading dimension of b (ldb >= max(1,m))
 * @param[out] rcond Reciprocal condition number estimate
 * @param[in] tol Tolerance. If tol > 0, used as lower bound for rcond.
 *                If tol <= 0, uses k*k*epsilon as default.
 * @param[out] iwork Integer workspace, dimension (k)
 * @param[out] dwork Real workspace, dimension (3*k)
 * @param[out] info 0=success, <0=invalid parameter, 1=matrix singular (rcond <= tol)
 */
void mb02od(
    const char* side,
    const char* uplo,
    const char* trans,
    const char* diag,
    const char* norm,
    i32 m,
    i32 n,
    f64 alpha,
    const f64* a,
    i32 lda,
    f64* b,
    i32 ldb,
    f64* rcond,
    f64 tol,
    i32* iwork,
    f64* dwork,
    i32* info);

/**
 * @brief Solve augmented system A*x = b, D*x = 0 in least squares sense.
 *
 * Determines vector x which solves the system of linear equations
 * A*x = b, D*x = 0 in the least squares sense, where A is m-by-n,
 * D is n-by-n diagonal, and b is m-vector. Assumes QR factorization
 * with column pivoting of A is available: A*P = Q*R.
 *
 * Uses Givens rotations to annihilate diagonal matrix D, updating
 * upper triangular matrix R and first n elements of Q'*b.
 *
 * @param[in] cond Condition estimation mode:
 *                 'E': use incremental condition estimation
 *                 'N': check diagonal entries for zero
 *                 'U': use rank already stored in RANK
 * @param[in] n Order of matrix R (n >= 0)
 * @param[in,out] r DOUBLE PRECISION array, dimension (ldr,n)
 *                  In: upper triangular matrix R
 *                  Out: full upper triangle unaltered,
 *                       strict lower triangle contains strict upper triangle
 *                       (transposed) of upper triangular matrix S
 * @param[in] ldr Leading dimension of r (ldr >= max(1,n))
 * @param[in] ipvt INTEGER array, dimension (n)
 *                 Permutation matrix P: column j of P is column ipvt[j]
 *                 of identity matrix (1-based indices)
 * @param[in] diag DOUBLE PRECISION array, dimension (n)
 *                 Diagonal elements of matrix D
 * @param[in] qtb DOUBLE PRECISION array, dimension (n)
 *                First n elements of Q'*b
 * @param[in,out] rank INTEGER
 *                     In (COND='U'): numerical rank of S
 *                     Out (COND='E' or 'N'): estimated numerical rank of S
 * @param[out] x DOUBLE PRECISION array, dimension (n)
 *               Least squares solution of A*x = b, D*x = 0
 * @param[in] tol DOUBLE PRECISION
 *                Tolerance for rank determination (COND='E' only)
 *                tol > 0: lower bound for reciprocal condition number
 *                tol <= 0: use default n*eps
 * @param[out] dwork DOUBLE PRECISION array, dimension (ldwork)
 *                   On exit: first n elements contain diagonal of S,
 *                           next n elements contain solution z
 * @param[in] ldwork Length of dwork
 *                   COND='E': ldwork >= 4*n
 *                   COND!='E': ldwork >= 2*n
 * @param[out] info Exit code (0=success, <0=invalid parameter -info)
 */
void mb02yd(
    const char* cond,
    const i32 n,
    f64* r,
    const i32 ldr,
    const i32* ipvt,
    const f64* diag,
    const f64* qtb,
    i32* rank,
    f64* x,
    const f64 tol,
    f64* dwork,
    const i32 ldwork,
    i32* info
);

/**
 * @brief Solve complex Hessenberg system using LU factorization.
 *
 * Solves H*X=B, H'*X=B, or H^H*X=B using LU factorization from slicot_mb02sz.
 *
 * @param[in] trans 'N' = no transpose, 'T' = transpose, 'C' = conjugate transpose
 * @param[in] n Order of matrix H (n >= 0)
 * @param[in] nrhs Number of right-hand sides (nrhs >= 0)
 * @param[in] h LU factors from slicot_mb02sz, dimension (ldh,n)
 * @param[in] ldh Leading dimension of h (ldh >= max(1,n))
 * @param[in] ipiv Pivot indices from slicot_mb02sz, dimension (n)
 * @param[in,out] b Right-hand sides / solutions, dimension (ldb,nrhs)
 * @param[in] ldb Leading dimension of b (ldb >= max(1,n))
 * @return Exit code: 0 = success, < 0 = -i means i-th argument invalid
 */
i32 slicot_mb02rz(char trans, i32 n, i32 nrhs, const c128* h, i32 ldh,
                  const i32* ipiv, c128* b, i32 ldb);

/**
 * @brief Solve complex Hessenberg system using LU factors.
 *
 * Solves op(H)*X = B where H is factored by slicot_mb02sz.
 * op(H) can be H, H^T (transpose), or H^H (conjugate transpose).
 *
 * @param[in] trans 'N' = H*X=B, 'T' = H^T*X=B, 'C' = H^H*X=B
 * @param[in] n Order of matrix H (n >= 0)
 * @param[in] nrhs Number of RHS columns (nrhs >= 0)
 * @param[in] h LU factors from slicot_mb02sz, dimension (ldh,n)
 * @param[in] ldh Leading dimension of H (ldh >= max(1,n))
 * @param[in] ipiv Pivot indices from slicot_mb02sz, dimension (n)
 * @param[in,out] b On entry: RHS. On exit: solution X. Dim (ldb,nrhs)
 * @param[in] ldb Leading dimension of B (ldb >= max(1,n))
 * @return Info code: 0 = success, < 0 = invalid arg
 */
i32 slicot_mb02rz(char trans, i32 n, i32 nrhs, const c128* h, i32 ldh,
                  const i32* ipiv, c128* b, i32 ldb);

/**
 * @brief LU factorization of complex upper Hessenberg matrix.
 *
 * Computes LU factorization H = P*L*U of a complex n-by-n upper Hessenberg
 * matrix using partial pivoting with row interchanges.
 *
 * @param[in] n Order of matrix H (n >= 0)
 * @param[in,out] h Complex matrix, dimension (ldh,n)
 *                  On entry: upper Hessenberg matrix
 *                  On exit: L and U factors (unit diagonal of L not stored)
 * @param[in] ldh Leading dimension of h (ldh >= max(1,n))
 * @param[out] ipiv Pivot indices, dimension (n). 1-based indexing.
 * @return Exit code:
 *         0 = success
 *         < 0 = -i means i-th argument invalid
 *         > 0 = i means U(i,i) is exactly zero (factorization completed but U is singular)
 */
i32 slicot_mb02sz(i32 n, c128* h, i32 ldh, i32* ipiv);

/**
 * @brief LU factorization of complex upper Hessenberg matrix.
 *
 * Computes LU factorization with partial pivoting of a complex
 * upper Hessenberg matrix H (single sub-diagonal only).
 *
 * @param[in] n Order of matrix H (n >= 0)
 * @param[in,out] h Complex Hessenberg matrix, dimension (ldh,n)
 *                  On exit: L,U factors in compact form
 * @param[in] ldh Leading dimension of H (ldh >= max(1,n))
 * @param[out] ipiv Pivot indices (1-based), dimension (n)
 * @return Info code: 0 = success, k > 0 = H(k,k) is exactly zero
 */
i32 slicot_mb02sz(i32 n, c128* h, i32 ldh, i32* ipiv);

/**
 * @brief Estimate condition number of complex Hessenberg matrix.
 *
 * Estimates reciprocal of condition number of factored complex Hessenberg
 * matrix using LU factorization from slicot_mb02sz.
 *
 * @param[in] norm '1' or 'O' for 1-norm, 'I' for infinity-norm
 * @param[in] n Order of matrix (n >= 0)
 * @param[in] hnorm Norm of original matrix H (before factorization)
 * @param[in] h LU factors from slicot_mb02sz, dimension (ldh,n)
 * @param[in] ldh Leading dimension of h (ldh >= max(1,n))
 * @param[in] ipiv Pivot indices from slicot_mb02sz, dimension (n)
 * @param[out] rcond Reciprocal condition number estimate
 * @param[out] dwork Real workspace, dimension (n)
 * @param[out] zwork Complex workspace, dimension (2*n)
 * @return Exit code: 0 = success, < 0 = -i means i-th argument invalid
 */
i32 slicot_mb02tz(char norm, i32 n, f64 hnorm, const c128* h, i32 ldh,
                  const i32* ipiv, f64* rcond, f64* dwork, c128* zwork);

/**
 * @brief Incomplete Cholesky factor of positive definite block Toeplitz matrix.
 *
 * Computes the incomplete Cholesky (ICC) factor of a symmetric
 * positive definite (s.p.d.) block Toeplitz matrix T, defined by
 * either its first block row, or its first block column.
 *
 * By subsequent calls, further rows/columns of the Cholesky factor can be added.
 * The generator of the Schur complement is also available.
 *
 * @param[in] typet Type of T:
 *                  'R' = T contains first block row, R is upper trapezoidal
 *                  'C' = T contains first block column, R is lower trapezoidal
 * @param[in] k Block size (rows/columns per block). k >= 0
 * @param[in] n Number of blocks. n >= 0
 * @param[in] p Number of previously computed block rows/columns of R. 0 <= p <= n
 * @param[in] s Number of block rows/columns of R to compute. 0 <= s <= n-p
 * @param[in,out] t Block Toeplitz data, dimension (ldt, (n-p)*k) / (ldt, k).
 *                  On entry if P=0: first block row/column of T.
 *                  On entry if P>0: negative generator of Schur complement.
 *                  On exit: Cholesky factor of T(1:k,1:k) and transformation info.
 * @param[in] ldt Leading dimension of t.
 *                ldt >= max(1,k) if typet='R', ldt >= max(1,(n-p)*k) if typet='C'.
 * @param[in,out] r ICC factor, dimension (ldr, n*k)/(ldr, s*k) for P=0,
 *                  (ldr, (n-p+1)*k)/(ldr, (s+1)*k) for P>0.
 *                  On entry if P>0: last block row/column from previous call.
 *                  On exit: upper/lower trapezoidal ICC factor.
 * @param[in] ldr Leading dimension of r.
 *                ldr >= max(1, s*k) if typet='R' and p=0,
 *                ldr >= max(1, (s+1)*k) if typet='R' and p>0,
 *                ldr >= max(1, n*k) if typet='C' and p=0,
 *                ldr >= max(1, (n-p+1)*k) if typet='C' and p>0.
 * @param[out] dwork Workspace, dimension (ldwork).
 *                   On exit: dwork[0] = optimal ldwork.
 * @param[in] ldwork Workspace size.
 *                   ldwork >= max(1,(n+1)*k,4*k) if p=0,
 *                   ldwork >= max(1,(n-p+2)*k,4*k) if p>0.
 * @param[out] info Exit code:
 *                  = 0: success
 *                  < 0: -i means parameter i had illegal value
 *                  = 1: matrix is not numerically positive definite
 */
void mb02fd(const char* typet, i32 k, i32 n, i32 p, i32 s,
            f64* t, i32 ldt, f64* r, i32 ldr,
            f64* dwork, i32 ldwork, i32* info);

/**
 * @brief Cholesky factorization of banded symmetric positive definite block Toeplitz matrix.
 *
 * Computes the Cholesky factor of a banded symmetric positive definite (s.p.d.)
 * block Toeplitz matrix, defined by either its first block row, or its first
 * block column, depending on the routine parameter TYPET.
 *
 * By subsequent calls of this routine the Cholesky factor can be computed
 * block column by block column.
 *
 * @param[in] typet Type of T:
 *                  'R' = T contains first block row, Cholesky factor is upper triangular
 *                  'C' = T contains first block column, Cholesky factor is lower triangular
 * @param[in] triu Structure of last block in T:
 *                 'N' = no special structure
 *                 'T' = last block is lower/upper triangular for typet='R'/'C'
 * @param[in] k Block size (number of rows/columns in T). k >= 0
 * @param[in] n Number of blocks. If triu='N': n >= 1; if triu='T': n >= 2
 * @param[in] nl Lower block bandwidth.
 *               If triu='N': 0 <= nl < n; if triu='T': 1 <= nl < n
 * @param[in] p Number of previously computed block rows/columns. 0 <= p <= n
 * @param[in] s Number of block rows/columns to compute. 0 <= s <= n-p
 * @param[in,out] t Block Toeplitz data, dimension (ldt, (nl+1)*k) for typet='R',
 *                  (ldt, k) for typet='C'.
 *                  On entry if p=0: first block row/column of s.p.d. block Toeplitz matrix.
 *                  On entry if p>0: p-th block row/column of Cholesky factor.
 *                  On exit: (p+s)-th block row/column of Cholesky factor.
 * @param[in] ldt Leading dimension of t.
 *                ldt >= max(1,k) if typet='R', ldt >= max(1,(nl+1)*k) if typet='C'
 * @param[in,out] rb Cholesky factor in banded storage format.
 *                   On exit: columns (p*k+1) to (min(p+nl+s,n)*k) of Cholesky factor.
 * @param[in] ldrb Leading dimension of rb.
 *                 If triu='N': ldrb >= max((nl+1)*k, 1)
 *                 If triu='T': ldrb >= nl*k+1
 * @param[out] dwork Workspace, dimension (ldwork).
 *                   On exit: dwork[0] = optimal ldwork.
 *                   First 1+(nl+1)*k*k elements should be preserved between calls.
 * @param[in] ldwork Workspace size. ldwork >= 1 + (nl+1)*k*k + nl*k.
 *                   If ldwork=-1, workspace query is assumed.
 * @param[out] info Exit code:
 *                  = 0: success
 *                  < 0: -i means parameter i had illegal value
 *                  = 1: matrix is not numerically positive definite
 */
void mb02gd(const char* typet, const char* triu, i32 k, i32 n, i32 nl,
            i32 p, i32 s, f64* t, i32 ldt, f64* rb, i32 ldrb,
            f64* dwork, i32 ldwork, i32* info);

/**
 * @brief Cholesky factorization of T'T for banded block Toeplitz matrix.
 *
 * Computes, for a banded K*M-by-L*N block Toeplitz matrix T with block size
 * (K,L), specified by the nonzero blocks of its first block column TC and
 * row TR, a LOWER triangular matrix R (in band storage scheme) such that
 *                      T
 *                     T  T  =  R R'.
 *
 * It is assumed that the first MIN(M*K, N*L) columns of T are linearly
 * independent. By subsequent calls of this routine, the matrix R can be
 * computed block column by block column.
 *
 * @param[in] triu Structure of last blocks:
 *                 'N' = TC and TR have no special structure
 *                 'T' = TC and TR are upper/lower triangular
 * @param[in] k Number of rows in blocks of T (k >= 0)
 * @param[in] l Number of columns in blocks of T (l >= 0)
 * @param[in] m Number of blocks in first block column of T (m >= 1)
 * @param[in] ml Lower block bandwidth (0 <= ml < m, (ml+1)*k >= l)
 * @param[in] n Number of blocks in first block row of T (n >= 1)
 * @param[in] nu Upper block bandwidth (constraints depend on triu)
 * @param[in] p Number of previously computed block columns of R (p >= 0)
 * @param[in] s Number of block columns of R to compute (s >= 0)
 * @param[in] tc First block column, dimension (ldtc, l).
 *               If p=0: (ml+1)*k-by-l nonzero blocks of first block column.
 * @param[in] ldtc Leading dimension of tc (ldtc >= max(1,(ml+1)*k) if p=0)
 * @param[in] tr Blocks 2 to nu+1 of first block row, dimension (ldtr, nu*l).
 *               If p=0: k-by-nu*l part.
 * @param[in] ldtr Leading dimension of tr (ldtr >= max(1,k) if p=0)
 * @param[out] rb Lower R factor in band storage, dimension (ldrb, ncols)
 *                where ncols = min(s*l, min(m*k,n*l) - p*l).
 *                For TRIU='N': leading min(ml+nu+1,n)*l rows used.
 *                For TRIU='T': leading min((ml+nu)*l+1,n*l) rows used.
 * @param[in] ldrb Leading dimension of rb.
 *                 TRIU='N': ldrb >= max(min(ml+nu+1,n)*l, 1)
 *                 TRIU='T': ldrb >= max(min((ml+nu)*l+1,n*l), 1)
 * @param[out] dwork Workspace, dimension (ldwork).
 *                   On exit: dwork[0] = optimal ldwork.
 *                   First 1+2*min(ml+nu+1,n)*l*(k+l) elements preserved between calls.
 * @param[in] ldwork Workspace size.
 *                   Let x = min(ml+nu+1,n), then:
 *                   If p=0: ldwork >= 1 + max(x*l*l + (2*nu+1)*l*k, 2*x*l*(k+l) + (6+x)*l)
 *                   If p>0: ldwork >= 1 + 2*x*l*(k+l) + (6+x)*l
 *                   If ldwork=-1: workspace query.
 * @param[out] info Exit code:
 *                  = 0: success
 *                  < 0: -i means parameter i had illegal value
 *                  = 1: full rank condition violated
 */
void mb02hd(const char* triu, i32 k, i32 l, i32 m, i32 ml, i32 n, i32 nu,
            i32 p, i32 s, f64* tc, i32 ldtc, f64* tr, i32 ldtr,
            f64* rb, i32 ldrb, f64* dwork, i32 ldwork, i32* info);

/**
 * @brief Block Toeplitz matrix-matrix product.
 *
 * Computes C = alpha*op(T)*B + beta*C where T is a block Toeplitz matrix
 * specified by its first block column TC and first block row TR.
 * op(T) is T or T' depending on TRANS parameter.
 *
 * @param[in] ldblk Where T(1,1)-block is stored:
 *                  'C' = in first block of TC
 *                  'R' = in first block of TR
 * @param[in] trans Form of op(T):
 *                  'N' = op(T) = T
 *                  'T' or 'C' = op(T) = T'
 * @param[in] k Number of rows in blocks of T (k >= 0)
 * @param[in] l Number of columns in blocks of T (l >= 0)
 * @param[in] m Number of blocks in first block column of T (m >= 0)
 * @param[in] n Number of blocks in first block row of T (n >= 0)
 * @param[in] r Number of columns in B and C (r >= 0)
 * @param[in] alpha Scalar multiplier. If zero, TC/TR/B not referenced.
 * @param[in] beta Scalar. If zero, C need not be set on entry.
 * @param[in] tc First block column of T.
 *               If ldblk='C': dimension (ldtc, l), leading M*K-by-L used
 *               If ldblk='R': dimension (ldtc, l), leading (M-1)*K-by-L used
 * @param[in] ldtc Leading dimension of tc.
 *                 ldtc >= max(1,M*K) if ldblk='C'
 *                 ldtc >= max(1,(M-1)*K) if ldblk='R'
 * @param[in] tr First block row of T (excluding/including T(1,1) block).
 *               If ldblk='C': dimension (ldtr, (N-1)*L), leading K-by-(N-1)*L used
 *               If ldblk='R': dimension (ldtr, N*L), leading K-by-N*L used
 * @param[in] ldtr Leading dimension of tr. ldtr >= max(1,K)
 * @param[in] b Input matrix B.
 *              If trans='N': dimension (ldb, r), leading N*L-by-R used
 *              If trans='T'/'C': dimension (ldb, r), leading M*K-by-R used
 * @param[in] ldb Leading dimension of b.
 *                ldb >= max(1,N*L) if trans='N'
 *                ldb >= max(1,M*K) if trans='T'/'C'
 * @param[in,out] c Input/output matrix C.
 *                  If trans='N': dimension (ldc, r), leading M*K-by-R used
 *                  If trans='T'/'C': dimension (ldc, r), leading N*L-by-R used
 *                  On exit: updated C = alpha*op(T)*B + beta*C
 * @param[in] ldc Leading dimension of c.
 *                ldc >= max(1,M*K) if trans='N'
 *                ldc >= max(1,N*L) if trans='T'/'C'
 * @param[out] dwork Workspace, dimension (ldwork).
 *                   On exit: dwork[0] = optimal ldwork.
 *                   On error -19: dwork[0] = minimum ldwork.
 * @param[in] ldwork Workspace size (ldwork >= 1).
 *                   If ldwork=-1, workspace query is performed.
 * @param[out] info Exit code:
 *                  = 0: success
 *                  < 0: -i means parameter i had illegal value
 */
void mb02kd(const char* ldblk, const char* trans, i32 k, i32 l, i32 m, i32 n,
            i32 r, f64 alpha, f64 beta, f64* tc, i32 ldtc, f64* tr, i32 ldtr,
            f64* b, i32 ldb, f64* c, i32 ldc, f64* dwork, i32 ldwork, i32* info);

/**
 * @brief Total Least Squares solution using SVD approach.
 *
 * Solves the Total Least Squares (TLS) problem using a Singular Value
 * Decomposition (SVD) approach. The TLS problem assumes an overdetermined
 * set of linear equations AX = B, where both the data matrix A as well as
 * the observation matrix B are inaccurate. Also solves determined and
 * underdetermined sets of equations by computing the minimum norm solution.
 *
 * @param[in] job Determines whether RANK and/or TOL are computed:
 *                'R' = compute RANK only (TOL must be specified)
 *                'T' = compute TOL only (RANK must be specified)
 *                'B' = compute both RANK and TOL
 *                'N' = compute neither (both must be specified)
 * @param[in] m Number of rows in A and B (m >= 0)
 * @param[in] n Number of columns in A (n >= 0)
 * @param[in] l Number of columns in B (l >= 0)
 * @param[in,out] rank TLS approximation rank.
 *                     Input: if JOB='T'/'N', must specify rank <= min(m,n)
 *                     Output: if JOB='R'/'B', contains computed rank
 * @param[in,out] c Matrix [A|B], dimension (ldc, n+l).
 *                  On entry: first N columns = A, last L columns = B.
 *                  On exit: transformed right singular vectors of C.
 * @param[in] ldc Leading dimension of c (ldc >= max(1, max(m, n+l)))
 * @param[out] s Singular values of C, dimension min(m, n+l).
 *               In descending order: S(1) >= S(2) >= ... >= 0.
 * @param[out] x TLS solution, dimension (ldx, l).
 *               Leading N-by-L part contains solution X.
 * @param[in] ldx Leading dimension of x (ldx >= max(1, n))
 * @param[in,out] tol Tolerance for rank determination.
 *                    Input: if JOB='R'/'N', specifies tolerance (<=0 uses eps)
 *                           if JOB='T'/'B', specifies sdev >= 0
 *                    Output: may be updated internally
 * @param[out] iwork Integer workspace, dimension (l)
 * @param[out] dwork Real workspace, dimension (ldwork).
 *                   On exit: dwork[0] = optimal ldwork,
 *                            dwork[1] = reciprocal condition number of F.
 * @param[in] ldwork Workspace size.
 *                   If m >= n+l: ldwork >= max(2, 3*(n+l)+m, 5*(n+l))
 *                   If m < n+l: ldwork >= max(2, m*(n+l)+max(3*m+n+l, 5*m), 3*l)
 *                   If ldwork=-1, workspace query.
 * @param[out] iwarn Warning indicator:
 *                   0 = no warnings
 *                   1 = rank lowered due to singular value multiplicity > 1
 *                   2 = rank lowered because F is numerically singular
 * @param[out] info Exit code:
 *                  = 0: success
 *                  < 0: -i means parameter i had illegal value
 *                  > 0: SVD did not converge (S may be incorrect)
 */
void mb02md(const char* job, i32 m, i32 n, i32 l, i32* rank, f64* c,
            i32 ldc, f64* s, f64* x, i32 ldx, f64* tol,
            i32* iwork, f64* dwork, i32 ldwork, i32* iwarn, i32* info);

/**
 * @brief Separate zero singular value of bidiagonal submatrix.
 *
 * Separates a zero singular value of a bidiagonal submatrix of order k,
 * k <= p = min(M,N), of a bidiagonal matrix J by annihilating one or two
 * superdiagonal elements E(i-1) (if i > 1) and/or E(i) (if i < k).
 *
 * The bidiagonal matrix J has diagonal Q and superdiagonal E:
 *        |Q(1) E(1)  0    ...   0   |
 *        | 0   Q(2) E(2)        .   |
 *    J = | .                    .   |
 *        | .                  E(p-1)|
 *        | 0   ...  ...   ...  Q(p) |
 *
 * When Q(i) is negligible, Givens rotations are used to annihilate
 * E(i) (rotations from left, stored in U) and E(i-1) (rotations from right,
 * stored in V).
 *
 * @param[in] updatu Whether to update U with left Givens rotations:
 *                   false = do not form U
 *                   true  = update U (postmultiply by rotations S)
 * @param[in] updatv Whether to update V with right Givens rotations:
 *                   false = do not form V
 *                   true  = update V (postmultiply by rotations T)
 * @param[in] m Number of rows of matrix U. m >= 0.
 * @param[in] n Number of rows of matrix V. n >= 0.
 * @param[in] i Index of negligible diagonal entry Q(i), i <= p = min(m,n).
 *              (1-based index)
 * @param[in] k Index of last diagonal entry of considered bidiagonal
 *              submatrix, i.e., E(k-1) is considered negligible. k <= p.
 *              (1-based index)
 * @param[in,out] q Diagonal entries of J, dimension (p).
 *                  On exit: transformed diagonal S' J T.
 * @param[in,out] e Superdiagonal entries of J, dimension (p-1).
 *                  On exit: transformed superdiagonal S' J T.
 * @param[in,out] u Left transformation matrix, dimension (ldu, p).
 *                  Updated with left Givens rotations if updatu=true.
 *                  Not referenced if updatu=false.
 * @param[in] ldu Leading dimension of u.
 *                ldu >= max(1,m) if updatu=true, ldu >= 1 otherwise.
 * @param[in,out] v Right transformation matrix, dimension (ldv, p).
 *                  Updated with right Givens rotations if updatv=true.
 *                  Not referenced if updatv=false.
 * @param[in] ldv Leading dimension of v.
 *                ldv >= max(1,n) if updatv=true, ldv >= 1 otherwise.
 * @param[out] dwork Workspace, dimension depends on updatu and updatv:
 *                   >= 2*max(k-i, i-1) if both true
 *                   >= 2*(k-i) if updatu only
 *                   >= 2*(i-1) if updatv only
 *                   >= 1 otherwise.
 */
void mb02ny(bool updatu, bool updatv, i32 m, i32 n, i32 i, i32 k,
            f64* q, f64* e, f64* u, i32 ldu, f64* v, i32 ldv, f64* dwork);

/**
 * @brief Linear least squares solution using complete orthogonal factorization.
 *
 * Computes a solution, optionally corresponding to specified free elements,
 * to a real linear least squares problem: minimize || A * X - B || using a
 * complete orthogonal factorization of the M-by-N matrix A, which may be
 * rank-deficient.
 *
 * @param[in] job Specifies whether to compute standard least squares or with
 *                free elements:
 *                'L' = Compute standard least squares solution (Y = 0)
 *                'F' = Compute solution with specified free elements (given in Y)
 * @param[in] iniper Specifies whether initial column permutation is performed:
 *                   'P' = Perform initial column permutation defined by JPVT
 *                   'N' = Do not perform initial column permutation
 * @param[in] m Number of rows of matrix A (m >= 0)
 * @param[in] n Number of columns of matrix A (n >= 0)
 * @param[in] nrhs Number of right hand sides (nrhs >= 0)
 * @param[in] rcond Used to determine effective rank. Largest leading submatrix
 *                  R11 with condition number < 1/RCOND. 0 <= rcond <= 1.
 * @param[in] svlmax If A is submatrix of C, should be estimate of largest
 *                   singular value of C. Otherwise use 0.0. svlmax >= 0.
 * @param[in,out] a DOUBLE PRECISION array, dimension (lda,n).
 *                  On entry: M-by-N matrix A.
 *                  On exit: Complete orthogonal factorization details.
 * @param[in] lda Leading dimension of a (lda >= max(1,m))
 * @param[in,out] b DOUBLE PRECISION array, dimension (ldb,nrhs).
 *                  On entry: M-by-NRHS right hand side matrix B.
 *                  On exit: N-by-NRHS solution matrix X.
 * @param[in] ldb Leading dimension of b (ldb >= max(1,m,n))
 * @param[in] y DOUBLE PRECISION array, dimension (n*nrhs).
 *              If job='F', elements Y(1:(n-rank)*nrhs) used as free elements.
 *              If job='L' or nrhs=0, not referenced. May be NULL if job='L'.
 * @param[in,out] jpvt INTEGER array, dimension (n).
 *                     On entry with iniper='P': if jpvt[i] != 0, column i is
 *                     initial column, otherwise free column.
 *                     On exit: permutation info.
 * @param[out] rank Effective rank of A (0 <= rank <= min(m,n))
 * @param[out] sval DOUBLE PRECISION array, dimension (3).
 *                  sval[0]: largest singular value of R(1:rank,1:rank)
 *                  sval[1]: smallest singular value of R(1:rank,1:rank)
 *                  sval[2]: smallest singular value of R(1:rank+1,1:rank+1)
 * @param[out] dwork Workspace, dimension (ldwork).
 *                   On exit: dwork[0] = optimal ldwork.
 * @param[in] ldwork Workspace size.
 *                   ldwork >= max(min(m,n)+3*n+1, 2*min(m,n)+nrhs)
 * @param[out] info Exit code: 0 = success, < 0 = -i means param i invalid
 */
void mb02qd(const char* job, const char* iniper, i32 m, i32 n, i32 nrhs,
            f64 rcond, f64 svlmax, f64* a, i32 lda, f64* b, i32 ldb,
            const f64* y, i32* jpvt, i32* rank, f64* sval, f64* dwork,
            i32 ldwork, i32* info);

/**
 * @brief Full QR factorization of a block Toeplitz matrix of full rank.
 *
 * Computes a lower triangular matrix R and a matrix Q with Q^T Q = I such that
 * T = Q R^T, where T is a K*M-by-L*N block Toeplitz matrix with blocks of size
 * (K,L). The first column of T is denoted by TC and the first row by TR.
 *
 * It is assumed that the first MIN(M*K, N*L) columns of T have full rank.
 * By subsequent calls the factors Q and R can be computed block column by
 * block column.
 *
 * @param[in] job Specifies the output:
 *                'Q' = compute Q and R
 *                'R' = only compute R
 * @param[in] k Number of rows in one block of T. k >= 0
 * @param[in] l Number of columns in one block of T. l >= 0
 * @param[in] m Number of blocks in one block column of T. m >= 0
 * @param[in] n Number of blocks in one block row of T. n >= 0
 * @param[in] p Number of previously computed block columns of R.
 *              p*l < min(m*k, n*l) + l and p >= 0
 * @param[in] s Number of block columns of R to compute.
 *              (p+s)*l < min(m*k, n*l) + l and s >= 0
 * @param[in] tc First block column of T, dimension (ldtc, l).
 *               On entry, if p=0, must contain M*K-by-L leading part.
 * @param[in] ldtc Leading dimension of tc. ldtc >= max(1, m*k)
 * @param[in] tr First block row of T without leading block, dimension (ldtr, (n-1)*l).
 *               On entry, if p=0, must contain K-by-(N-1)*L leading part.
 * @param[in] ldtr Leading dimension of tr. ldtr >= max(1, k)
 * @param[in,out] q Matrix Q, dimension (ldq, min(s*l, min(m*k,n*l)-p*l)).
 *                  On entry if job='Q' and p>0: last block column of Q from previous call.
 *                  On exit if job='Q': P-th to (P+S)-th block columns of Q.
 * @param[in] ldq Leading dimension of q.
 *                ldq >= max(1, m*k) if job='Q', ldq >= 1 if job='R'
 * @param[in,out] r Lower triangular factor R, dimension (ldr, min(s*l, min(m*k,n*l)-p*l)).
 *                  On entry if p>0: nonzero part of last block column from previous call.
 *                  On exit: nonzero parts of P-th to (P+S)-th block columns of R.
 * @param[in] ldr Leading dimension of r. ldr >= max(1, min(n, n-p+1)*l)
 * @param[out] dwork Workspace, dimension (ldwork).
 *                   On exit: dwork[0] = optimal ldwork.
 *                   If ldwork=-1, workspace query is performed.
 * @param[in] ldwork Workspace size. See documentation for detailed requirements.
 * @param[out] info Exit code:
 *                  0 = success
 *                  < 0 = -i means argument i invalid
 *                  1 = full rank condition numerically violated
 */
void mb02jd(const char* job, i32 k, i32 l, i32 m, i32 n, i32 p, i32 s,
            const f64* tc, i32 ldtc, const f64* tr, i32 ldtr,
            f64* q, i32 ldq, f64* r, i32 ldr, f64* dwork, i32 ldwork, i32* info);

/**
 * @brief Low rank QR factorization with column pivoting of a block Toeplitz matrix.
 *
 * Computes a low rank QR factorization with column pivoting of a K*M-by-L*N
 * block Toeplitz matrix T with blocks of size (K,L):
 *
 *     T P = Q R^T
 *
 * where R is lower trapezoidal, P is a block permutation matrix, and Q^T Q = I.
 * The number of columns in R (RNK) is the numerical rank of T with respect to
 * the given tolerance TOL1. Note that the pivoting scheme is local, i.e., only
 * columns belonging to the same block in T are permuted.
 *
 * @param[in] job Specifies the output:
 *                'Q' = compute Q and R
 *                'R' = only compute R
 * @param[in] k Number of rows in one block of T. k >= 0
 * @param[in] l Number of columns in one block of T. l >= 0
 * @param[in] m Number of blocks in one block column of T. m >= 0
 * @param[in] n Number of blocks in one block row of T. n >= 0
 * @param[in] tc First block column of T, dimension (ldtc, l).
 *               Leading M*K-by-L part contains first block column.
 * @param[in] ldtc Leading dimension of tc. ldtc >= max(1, m*k)
 * @param[in] tr First block row of T without leading K-by-L block,
 *               dimension (ldtr, (n-1)*l). Leading K-by-(N-1)*L part used.
 * @param[in] ldtr Leading dimension of tr. ldtr >= max(1, k)
 * @param[out] rnk Number of columns in R, equals numerical rank of T
 * @param[out] q If JOB='Q', the M*K-by-RNK factor Q. Not referenced if JOB='R'.
 * @param[in] ldq Leading dimension of q.
 *                ldq >= max(1, m*k) if JOB='Q', ldq >= 1 if JOB='R'
 * @param[out] r The N*L-by-RNK lower trapezoidal factor R
 * @param[in] ldr Leading dimension of r. ldr >= max(1, n*l)
 * @param[out] jpvt Integer array recording column pivoting.
 *                  jpvt[j]=k means j-th column of T*P was k-th column of T.
 *                  Dimension min(m*k, n*l)
 * @param[in] tol1 Diagonal tolerance. If tol1 < 0, default used.
 * @param[in] tol2 Offdiagonal tolerance. If tol2 < 0, default used.
 * @param[out] dwork Workspace. On exit: dwork[0]=optimal ldwork, dwork[1]=used tol1,
 *                   dwork[2]=used tol2. If INFO=-19, dwork[0]=minimum ldwork.
 * @param[in] ldwork Workspace size. See documentation for detailed requirements.
 * @param[out] info Exit code:
 *                  0 = success
 *                  < 0 = -i means argument i invalid
 *                  1 = generator indefinite (diagonal in negative generator > positive)
 *                  2 = columns not linearly dependent but diagonals equal
 */
void mb02jx(const char* job, i32 k, i32 l, i32 m, i32 n,
            const f64* tc, i32 ldtc, const f64* tr, i32 ldtr,
            i32* rnk, f64* q, i32 ldq, f64* r, i32 ldr, i32* jpvt,
            f64 tol1, f64 tol2, f64* dwork, i32 ldwork, i32* info);

/**
 * @brief Solve over/underdetermined linear systems with block Toeplitz matrix.
 *
 * Solves overdetermined or underdetermined real linear systems involving an
 * M*K-by-N*L block Toeplitz matrix T that is specified by its first block
 * column and row. It is assumed that T has full rank.
 *
 * Options:
 * 1. If JOB='O' or 'A': find least squares solution minimize ||B - T*X||
 * 2. If JOB='U' or 'A': find minimum norm solution of T' * X = C
 *
 * @param[in] job Specifies problem to solve:
 *                'O' = solve overdetermined system (1)
 *                'U' = solve underdetermined system (2)
 *                'A' = solve both (1) and (2)
 * @param[in] k Number of rows in the blocks of T. k >= 0
 * @param[in] l Number of columns in the blocks of T. l >= 0
 * @param[in] m Number of blocks in first block column of T. m >= 0
 * @param[in] n Number of blocks in first block row of T. 0 <= n <= m*k/l
 * @param[in] rb If JOB='O' or 'A', number of columns in B. rb >= 0
 * @param[in] rc If JOB='U' or 'A', number of columns in C. rc >= 0
 * @param[in] tc First block column of T, dimension (ldtc, l)
 * @param[in] ldtc Leading dimension of tc. ldtc >= max(1, m*k)
 * @param[in] tr Blocks 2 to N of first block row of T, dimension (ldtr, (n-1)*l)
 * @param[in] ldtr Leading dimension of tr. ldtr >= max(1, k)
 * @param[in,out] b Right-hand side matrix B of overdetermined system, dimension (ldb, rb).
 *                  On exit: leading N*L-by-RB part contains solution.
 * @param[in] ldb Leading dimension of b.
 *                ldb >= max(1, m*k) if JOB='O' or 'A', ldb >= 1 if JOB='U'
 * @param[in,out] c Right-hand side matrix C of underdetermined system, dimension (ldc, rc).
 *                  On exit: leading M*K-by-RC part contains solution.
 * @param[in] ldc Leading dimension of c.
 *                ldc >= 1 if JOB='O', ldc >= max(1, m*k) if JOB='U' or 'A'
 * @param[out] dwork Workspace array, dimension (ldwork).
 *                   On exit: dwork[0] = optimal ldwork.
 * @param[in] ldwork Workspace size. If ldwork=-1, workspace query performed.
 * @param[out] info Exit code:
 *                  0 = success
 *                  < 0 = -i means argument i invalid
 *                  1 = Toeplitz matrix is numerically not of full rank
 */
void mb02id(const char* job, i32 k, i32 l, i32 m, i32 n,
            i32 rb, i32 rc, f64* tc, i32 ldtc, f64* tr, i32 ldtr,
            f64* b, i32 ldb, f64* c, i32 ldc,
            f64* dwork, i32 ldwork, i32* info);

/**
 * @brief Solve 1x1 or 2x2 linear system with scaling and perturbation.
 *
 * Solves a system of the form A*X = s*B or A'*X = s*B with possible scaling
 * ("s") and perturbation of A. A is an N-by-N real matrix, and X and B are
 * N-by-M matrices. N may be 1 or 2. The scalar "s" is a scaling factor
 * (<= 1), computed by this subroutine, which is chosen so that X can be
 * computed without overflow.
 *
 * @param[in] ltrans Specifies if A or A-transpose is to be used:
 *                   true  = A-transpose will be used
 *                   false = A will be used (not transposed)
 * @param[in] n Order of the matrix A. May only be 1 or 2.
 * @param[in] m Number of right-hand side vectors.
 * @param[in] par Machine related parameters, dimension (3):
 *                par[0] = PREC: (machine precision)*base, DLAMCH('P')
 *                par[1] = SFMIN: safe minimum, DLAMCH('S')
 *                par[2] = SMIN: desired lower bound on singular values of A
 * @param[in] a Matrix A, dimension (lda, n). Leading N-by-N part contains A.
 * @param[in] lda Leading dimension of a. lda >= n.
 * @param[in,out] b On entry, dimension (ldb, m), leading N-by-M part contains
 *                  right-hand side matrix B.
 *                  On exit, leading N-by-M part contains solution matrix X.
 * @param[in] ldb Leading dimension of b. ldb >= n.
 * @param[out] scale Scale factor so A*X = scale*B (or A'*X = scale*B).
 *                   scale will be at most 1.
 * @param[out] iwarn Warning indicator:
 *                   0 = no warning (A did not have to be perturbed)
 *                   1 = A was perturbed to make its smallest singular value
 *                       greater than SMIN
 */
void mb02uw(bool ltrans, i32 n, i32 m, const f64* par,
            const f64* a, i32 lda, f64* b, i32 ldb,
            f64* scale, i32* iwarn);

/**
 * @brief Total Least Squares solution using Partial SVD.
 *
 * Solves the Total Least Squares (TLS) problem using a Partial
 * Singular Value Decomposition (PSVD) approach. The TLS problem
 * assumes an overdetermined set of linear equations AX = B, where
 * both the data matrix A and observation matrix B are inaccurate.
 * Also solves determined and underdetermined systems by computing
 * the minimum norm solution.
 *
 * @param[in] m Number of rows in data matrix A and observation matrix B. m >= 0.
 * @param[in] n Number of columns in data matrix A. n >= 0.
 * @param[in] l Number of columns in observation matrix B. l >= 0.
 * @param[in,out] rank On entry, if rank < 0, the rank is computed using theta.
 *                     Otherwise, specifies the rank. rank <= min(m,n).
 *                     On exit, contains computed or adjusted rank.
 * @param[in,out] theta On entry, if rank < 0, used to compute rank as
 *                      min(m,n+l) - d where d is # singular values <= theta.
 *                      If rank >= 0 and theta < 0, theta is computed.
 *                      On exit, contains computed bound if rank >= 0 on entry.
 * @param[in,out] c Matrix C = [A|B], dimension (ldc, n+l).
 *                  On entry, first n columns contain A, last l columns contain B.
 *                  On exit, contains transformed singular vectors.
 * @param[in] ldc Leading dimension of c. ldc >= max(1, m, n+l).
 * @param[out] x Solution matrix X, dimension (ldx, l). Contains N-by-L solution.
 * @param[in] ldx Leading dimension of x. ldx >= max(1, n).
 * @param[out] q Bidiagonal matrix elements, dimension (max(1, 2*min(m,n+l)-1)).
 *               First p entries are diagonal, next p-1 are superdiagonal.
 * @param[out] inul Boolean array, dimension (n+l). TRUE indicates columns
 *                  used for TLS solution.
 * @param[in] tol Tolerance for singular value multiplicity and convergence.
 *                If tol <= 0, default from MB04YD is used.
 * @param[in] reltol Relative tolerance for bisection convergence.
 *                   If reltol < BASE*EPS, defaults to BASE*EPS.
 * @param[out] iwork Integer workspace, dimension (n + 2*l).
 * @param[out] dwork Double workspace, dimension (ldwork).
 *                   On exit, dwork[0] = optimal ldwork, dwork[1] = rcond of F.
 * @param[in] ldwork Workspace size. If -1, workspace query is performed.
 *                   Required: max(2, max(m,n+l) + 2*min(m,n+l),
 *                                 min(m,n+l) + LW + max(6*(n+l)-5, l*l+max(n+l,3*l)))
 *                   where LW = (n+l)*(n+l-1)/2 if m >= n+l, else m*(n+l-(m-1)/2).
 * @param[out] bwork Boolean workspace, dimension (n+l).
 * @param[out] iwarn Warning indicator:
 *                   0 = no warnings
 *                   1 = rank lowered due to singular value multiplicity
 *                   2 = rank lowered because F is singular
 * @param[out] info Exit code:
 *                  0 = success
 *                  < 0 = -i means argument i invalid
 *                  1 = max QR/QL iterations exceeded
 *                  2 = computed rank exceeds min(m,n)
 */
void mb02nd(i32 m, i32 n, i32 l, i32* rank, f64* theta,
            f64* c, i32 ldc, f64* x, i32 ldx, f64* q, bool* inul,
            f64 tol, f64 reltol, i32* iwork, f64* dwork, i32 ldwork,
            bool* bwork, i32* iwarn, i32* info);

#ifdef __cplusplus
}
#endif

#endif /* SLICOT_MB02_H */
