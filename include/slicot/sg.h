/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#ifndef SLICOT_SG_H
#define SLICOT_SG_H

#include "../slicot_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Solve generalized algebraic Riccati equation for descriptor systems.
 *
 * Solves the continuous-time Riccati equation:
 *   Q + A'XE + E'XA - (L+E'XB)R^{-1}(L+E'XB)' = 0
 *
 * Or the discrete-time Riccati equation:
 *   E'XE = A'XA - (L+A'XB)(R+B'XB)^{-1}(L+A'XB)' + Q
 *
 * Uses the method of deflating subspaces based on reordering eigenvalues
 * in a generalized Schur matrix pair.
 *
 * @param[in] dico 'C' for continuous-time, 'D' for discrete-time
 * @param[in] jobb 'B' if B and R given, 'G' if G=BR^{-1}B' given
 * @param[in] fact 'N' for Q,R not factored, 'C' for Q=C'C, 'D' for R=D'D, 'B' for both
 * @param[in] uplo 'U' for upper triangle, 'L' for lower triangle stored
 * @param[in] jobl 'Z' if L is zero, 'N' if L is nonzero
 * @param[in] scal 'G' to use scaling, 'N' for no scaling (only if JOBB='B')
 * @param[in] sort 'S' for stable eigenvalues first, 'U' for unstable first
 * @param[in] acc 'R' for iterative refinement, 'N' for no refinement
 * @param[in] n State dimension (n >= 0)
 * @param[in] m Number of inputs (m >= 0, if JOBB='B')
 * @param[in] p Number of outputs (p >= 0, if FACT='C','D','B')
 * @param[in] a DOUBLE PRECISION array, dimension (lda,n), state matrix A
 * @param[in] lda Leading dimension of A (lda >= max(1,n))
 * @param[in] e DOUBLE PRECISION array, dimension (lde,n), descriptor matrix E
 * @param[in] lde Leading dimension of E (lde >= max(1,n))
 * @param[in] b DOUBLE PRECISION array, dimension (ldb,*), input matrix B or G
 * @param[in] ldb Leading dimension of B (ldb >= max(1,n))
 * @param[in,out] q DOUBLE PRECISION array, dimension (ldq,n), weighting matrix Q or C
 * @param[in] ldq Leading dimension of Q
 * @param[in,out] r DOUBLE PRECISION array, dimension (ldr,*), weighting matrix R or D
 * @param[in] ldr Leading dimension of R
 * @param[in,out] l DOUBLE PRECISION array, dimension (ldl,*), cross-weighting L
 * @param[in] ldl Leading dimension of L
 * @param[out] rcondu Reciprocal condition number of solution system
 * @param[out] x DOUBLE PRECISION array, dimension (ldx,n), solution matrix X
 * @param[in] ldx Leading dimension of X (ldx >= max(1,n))
 * @param[out] alfar DOUBLE PRECISION array, dimension (2*n), real parts of eigenvalues
 * @param[out] alfai DOUBLE PRECISION array, dimension (2*n), imaginary parts
 * @param[out] beta DOUBLE PRECISION array, dimension (2*n), eigenvalue denominators
 * @param[out] s DOUBLE PRECISION array, dimension (lds,*), Schur form S
 * @param[in] lds Leading dimension of S (lds >= max(1,2*n+m) if JOBB='B', else 2*n)
 * @param[out] t DOUBLE PRECISION array, dimension (ldt,2*n), Schur form T
 * @param[in] ldt Leading dimension of T
 * @param[out] u DOUBLE PRECISION array, dimension (ldu,2*n), transformation matrix U
 * @param[in] ldu Leading dimension of U (ldu >= max(1,2*n))
 * @param[in] tol Tolerance for singularity test (tol <= 0 uses default EPS)
 * @param[out] iwork INTEGER array, dimension (max(1,m,2*n))
 * @param[out] dwork DOUBLE PRECISION array, dimension (ldwork)
 * @param[in] ldwork Workspace size (see docs for minimum)
 * @param[out] iwarn 0=ok, 1=solution may be inaccurate
 * @param[out] info 0=success, <0=invalid param, 1=singular, 2=QZ failed,
 *                  3=reordering failed, 4=eigenvalues changed, 5=dim mismatch,
 *                  6=spectrum too close to boundary, 7=singular in solution
 */
void sg02ad(
    const char* dico,
    const char* jobb,
    const char* fact,
    const char* uplo,
    const char* jobl,
    const char* scal,
    const char* sort,
    const char* acc,
    const i32 n,
    const i32 m,
    const i32 p,
    f64* a,
    const i32 lda,
    f64* e,
    const i32 lde,
    f64* b,
    const i32 ldb,
    f64* q,
    const i32 ldq,
    f64* r,
    const i32 ldr,
    f64* l,
    const i32 ldl,
    f64* rcondu,
    f64* x,
    const i32 ldx,
    f64* alfar,
    f64* alfai,
    f64* beta,
    f64* s,
    const i32 lds,
    f64* t,
    const i32 ldt,
    f64* u,
    const i32 ldu,
    const f64 tol,
    i32* iwork,
    f64* dwork,
    const i32 ldwork,
    i32* iwarn,
    i32* info
);

/**
 * @brief Solve generalized Lyapunov equation for descriptor systems.
 *
 * Solves the continuous-time generalized Lyapunov equation:
 *   op(A)' * X * op(E) + op(E)' * X * op(A) = SCALE * Y
 *
 * Or the discrete-time generalized Lyapunov equation:
 *   op(A)' * X * op(A) - op(E)' * X * op(E) = SCALE * Y
 *
 * where op(M) is M or M^T and Y is symmetric. Provides estimates of
 * separation and forward error.
 *
 * @param[in] dico 'C' for continuous-time, 'D' for discrete-time
 * @param[in] job 'X' for solution only, 'S' for separation only, 'B' for both
 * @param[in] fact 'N' if factorization needed, 'F' if factorization supplied
 * @param[in] trans 'N' for op(M)=M, 'T' for op(M)=M^T
 * @param[in] uplo 'U' if upper triangle of X needed, 'L' for lower
 * @param[in] n Order of matrices (n >= 0)
 * @param[in,out] a DOUBLE PRECISION array, dimension (lda,n), matrix A / Schur factor
 * @param[in] lda Leading dimension of A (lda >= max(1,n))
 * @param[in,out] e DOUBLE PRECISION array, dimension (lde,n), matrix E / Schur factor
 * @param[in] lde Leading dimension of E (lde >= max(1,n))
 * @param[in,out] q DOUBLE PRECISION array, dimension (ldq,n), orthogonal Q
 * @param[in] ldq Leading dimension of Q (ldq >= max(1,n))
 * @param[in,out] z DOUBLE PRECISION array, dimension (ldz,n), orthogonal Z
 * @param[in] ldz Leading dimension of Z (ldz >= max(1,n))
 * @param[in,out] x DOUBLE PRECISION array, dimension (ldx,n), RHS Y / solution X
 * @param[in] ldx Leading dimension of X (ldx >= max(1,n))
 * @param[out] scale Scaling factor (0 < scale <= 1)
 * @param[out] sep Separation estimate (if JOB='S' or 'B')
 * @param[out] ferr Forward error bound (if JOB='B')
 * @param[out] alphar DOUBLE PRECISION array, dimension (n), real parts of eigenvalues
 * @param[out] alphai DOUBLE PRECISION array, dimension (n), imaginary parts
 * @param[out] beta DOUBLE PRECISION array, dimension (n), eigenvalue denominators
 * @param[out] iwork INTEGER array, dimension (n^2) (not used if JOB='X')
 * @param[out] dwork DOUBLE PRECISION array, dimension (ldwork)
 * @param[in] ldwork Workspace size (see docs for minimum, -1 for query)
 * @param[out] info 0=success, <0=invalid param, 1=not quasitriangular,
 *                  2=factorization failed, 3=reciprocal eigenvalues (discrete),
 *                  4=degenerate eigenvalues (continuous)
 */
void sg03ad(
    const char* dico,
    const char* job,
    const char* fact,
    const char* trans,
    const char* uplo,
    const i32 n,
    f64* a,
    const i32 lda,
    f64* e,
    const i32 lde,
    f64* q,
    const i32 ldq,
    f64* z,
    const i32 ldz,
    f64* x,
    const i32 ldx,
    f64* scale,
    f64* sep,
    f64* ferr,
    f64* alphar,
    f64* alphai,
    f64* beta,
    i32* iwork,
    f64* dwork,
    const i32 ldwork,
    i32* info
);

/**
 * @brief Solve reduced generalized discrete-time Lyapunov equation.
 *
 * Solves for X either the reduced generalized discrete-time Lyapunov equation:
 *     A' * X * A - E' * X * E = scale * Y    (TRANS='N')
 * or
 *     A * X * A' - E * X * E' = scale * Y    (TRANS='T')
 *
 * where Y is symmetric, A is upper quasitriangular, E is upper triangular.
 *
 * @param[in] trans Specifies equation: 'N' or 'T'
 * @param[in] n Order of matrices (n >= 0)
 * @param[in] a N-by-N upper quasi-triangular matrix A, dimension (lda,n)
 * @param[in] lda Leading dimension of A (lda >= max(1,n))
 * @param[in] e N-by-N upper triangular matrix E, dimension (lde,n)
 * @param[in] lde Leading dimension of E (lde >= max(1,n))
 * @param[in,out] x On entry: symmetric RHS Y. On exit: solution X.
 * @param[in] ldx Leading dimension of X (ldx >= max(1,n))
 * @param[out] scale Scale factor (0 < scale <= 1)
 * @param[out] info Exit code: 0 = success, < 0 = invalid arg,
 *                  1 = nearly singular (perturbed values used)
 */
void sg03ax(const char* trans, i32 n, const f64* a, i32 lda,
            const f64* e, i32 lde, f64* x, i32 ldx, f64* scale, i32* info);

/**
 * @brief Solve reduced generalized continuous-time Lyapunov equation.
 *
 * Solves for X either the reduced generalized continuous-time
 * Lyapunov equation:
 *     A' * X * E + E' * X * A = SCALE * Y    (TRANS='N')
 * or
 *     A * X * E' + E * X * A' = SCALE * Y    (TRANS='T')
 *
 * where the right hand side Y is symmetric. A, E, Y, and the solution X
 * are N-by-N matrices. The pencil A - lambda * E must be in generalized
 * Schur form (A upper quasitriangular, E upper triangular). SCALE is an
 * output scale factor, set to avoid overflow in X.
 *
 * @param[in] trans Specifies the equation:
 *                  'N' = solve equation (1)
 *                  'T' = solve equation (2)
 * @param[in] n Order of matrix A (n >= 0)
 * @param[in] a N-by-N upper quasitriangular matrix A, dimension (lda,n)
 * @param[in] lda Leading dimension of A (lda >= max(1,n))
 * @param[in] e N-by-N upper triangular matrix E, dimension (lde,n)
 * @param[in] lde Leading dimension of E (lde >= max(1,n))
 * @param[in,out] x On entry: N-by-N symmetric RHS matrix Y (upper tri used)
 *                  On exit: N-by-N solution matrix X, dimension (ldx,n)
 * @param[in] ldx Leading dimension of X (ldx >= max(1,n))
 * @param[out] scale Scale factor (0 < scale <= 1)
 * @param[out] info Exit code:
 *                  0 = success
 *                  < 0 = -i means i-th argument invalid
 *                  1 = equation is nearly singular (perturbed values used)
 */
void sg03ay(const char* trans, i32 n, const f64* a, i32 lda,
            const f64* e, i32 lde, f64* x, i32 ldx, f64* scale, i32* info);

/**
 * @brief Solve generalized Lyapunov equation for Cholesky factor.
 *
 * Computes Cholesky factor U (X = op(U)^T * op(U)) solving the generalized
 * c-stable continuous-time Lyapunov equation:
 *
 *   op(A)^T * X * op(E) + op(E)^T * X * op(A) = -SCALE^2 * op(B)^T * op(B),
 *
 * or the generalized d-stable discrete-time Lyapunov equation:
 *
 *   op(A)^T * X * op(A) - op(E)^T * X * op(E) = -SCALE^2 * op(B)^T * op(B),
 *
 * where op(K) is either K or K^T. A, E are N-by-N, op(B) is M-by-N.
 * Result U is N-by-N upper triangular with non-negative diagonal entries.
 * SCALE is set to avoid overflow in U.
 *
 * If FACT='N', pencil A-lambda*E is reduced to generalized Schur form.
 * If FACT='F', generalized Schur factors must be supplied on entry.
 *
 * @param[in] dico 'C' for continuous-time, 'D' for discrete-time
 * @param[in] fact 'N' to compute factorization, 'F' if factorization supplied
 * @param[in] trans 'N' for op(K)=K, 'T' for op(K)=K^T
 * @param[in] n Order of matrices A and E (n >= 0)
 * @param[in] m Number of rows in op(B) (m >= 0)
 * @param[in,out] a DOUBLE PRECISION array, dimension (lda,n)
 *                  In: Matrix A (if FACT='F': generalized Schur factor)
 *                  Out: Generalized Schur factor (if FACT='N')
 * @param[in] lda Leading dimension of A (lda >= max(1,n))
 * @param[in,out] e DOUBLE PRECISION array, dimension (lde,n)
 *                  In: Matrix E (if FACT='F': generalized Schur factor)
 *                  Out: Generalized Schur factor (if FACT='N')
 * @param[in] lde Leading dimension of E (lde >= max(1,n))
 * @param[in,out] q DOUBLE PRECISION array, dimension (ldq,n)
 *                  In: Orthogonal matrix Q (if FACT='F')
 *                  Out: Orthogonal matrix Q from factorization (if FACT='N')
 * @param[in] ldq Leading dimension of Q (ldq >= max(1,n))
 * @param[in,out] z DOUBLE PRECISION array, dimension (ldz,n)
 *                  In: Orthogonal matrix Z (if FACT='F')
 *                  Out: Orthogonal matrix Z from factorization (if FACT='N')
 * @param[in] ldz Leading dimension of Z (ldz >= max(1,n))
 * @param[in,out] b DOUBLE PRECISION array, dimension (ldb,n1)
 *                  In: Matrix B (size depends on TRANS)
 *                  Out: Cholesky factor U
 * @param[in] ldb Leading dimension of B
 * @param[out] scale Scaling factor (0 < scale <= 1)
 * @param[out] alphar DOUBLE PRECISION array, dimension (n)
 *                    Real parts of eigenvalues of pencil A-lambda*E
 * @param[out] alphai DOUBLE PRECISION array, dimension (n)
 *                    Imaginary parts of eigenvalues of pencil A-lambda*E
 * @param[out] beta DOUBLE PRECISION array, dimension (n)
 *                  Scaling factors for eigenvalues
 * @param[out] dwork DOUBLE PRECISION array, dimension (ldwork)
 *                   Workspace, dwork[0] returns optimal ldwork
 * @param[in] ldwork Workspace size (ldwork >= max(1,4*n,6*n-6) if FACT='N',
 *                   ldwork >= max(1,2*n,6*n-6) if FACT='F')
 *                   If ldwork=-1, workspace query
 * @param[out] info Exit code (0=success, <0=invalid parameter, 1=singular,
 *                  2=not quasitriangular, 3=eigenvalues not conjugate,
 *                  4=factorization failed, 5=not c-stable, 6=not d-stable,
 *                  7=DSYEVX failed)
 */
void sg03bd(
    const char* dico, const char* fact, const char* trans,
    const i32 n, const i32 m,
    f64* a, const i32 lda,
    f64* e, const i32 lde,
    f64* q, const i32 ldq,
    f64* z, const i32 ldz,
    f64* b, const i32 ldb,
    f64* scale,
    f64* alphar, f64* alphai, f64* beta,
    f64* dwork, const i32 ldwork,
    i32* info
);

/**
 * @brief Compute complex Givens rotation in real arithmetic.
 *
 * Computes parameters for complex Givens rotation such that:
 *
 *     (    C      SR+SI*I )   ( XR+XI*I )   ( ZR+ZI*I )
 *     (                   ) * (         ) = (         )
 *     ( -SR+SI*I     C    )   ( YR+YI*I )   (    0    )
 *
 * where C**2 + |SR+SI*I|**2 = 1.
 *
 * Adapted from LAPACK ZLARTG for real data representation.
 * Avoids unnecessary overflow/underflow.
 *
 * @param[in] xr Real part of X
 * @param[in] xi Imaginary part of X
 * @param[in] yr Real part of Y
 * @param[in] yi Imaginary part of Y
 * @param[out] c Cosine parameter (real)
 * @param[out] sr Real part of sine parameter
 * @param[out] si Imaginary part of sine parameter
 * @param[out] zr Real part of result Z
 * @param[out] zi Imaginary part of result Z
 */
void sg03br(
    const f64 xr, const f64 xi, const f64 yr, const f64 yi,
    f64* c, f64* sr, f64* si, f64* zr, f64* zi
);

/**
 * @brief Solve generalized discrete-time Lyapunov equation for Cholesky factor.
 *
 * Computes Cholesky factor U (X = U^T * U or X = U * U^T) solving
 * generalized d-stable discrete-time Lyapunov equation:
 *
 *   TRANS='N': A^T * X * A - E^T * X * E = -SCALE^2 * B^T * B
 *   TRANS='T': A * X * A^T - E * X * E^T = -SCALE^2 * B * B^T
 *
 * A quasitriangular, E and B upper triangular. Pencil A-lambda*E must be
 * d-stable (eigenvalue moduli < 1).
 *
 * @param[in] trans 'N' for equation (1), 'T' for equation (2)
 * @param[in] n Order of matrices (N >= 0)
 * @param[in] a DOUBLE PRECISION array, dimension (lda,N), quasitriangular A
 * @param[in] lda Leading dimension of A (lda >= max(1,N))
 * @param[in] e DOUBLE PRECISION array, dimension (lde,N), upper triangular E
 * @param[in] lde Leading dimension of E (lde >= max(1,N))
 * @param[in,out] b DOUBLE PRECISION array, dimension (ldb,N)
 *                  On entry: upper triangular B
 *                  On exit: Cholesky factor U
 * @param[in] ldb Leading dimension of B (ldb >= max(1,N))
 * @param[out] scale Scaling factor (0 < scale <= 1)
 * @param[out] dwork DOUBLE PRECISION workspace, dimension (6*N-6)
 * @param[out] info Exit code (0=success, <0=parameter error, 1=near singular,
 *                  2=not complex conjugate, 3=not d-stable, 4=DSYEVX failed)
 */
void sg03bu(
    const char* trans, const i32 n,
    const f64* a, const i32 lda,
    const f64* e, const i32 lde,
    f64* b, const i32 ldb,
    f64* scale, f64* dwork,
    i32* info
);

/**
 * @brief Solve generalized continuous-time Lyapunov equation for Cholesky factor.
 *
 * Computes Cholesky factor U (X = U^T * U or X = U * U^T) solving
 * generalized c-stable continuous-time Lyapunov equation:
 *
 *   TRANS='N': A^T * X * E + E^T * X * A = -SCALE^2 * B^T * B
 *   TRANS='T': A * X * E^T + E * X * A^T = -SCALE^2 * B * B^T
 *
 * A quasitriangular, E and B upper triangular. Pencil A-lambda*E must be
 * c-stable (eigenvalues with negative real parts).
 *
 * @param[in] trans 'N' for equation (1), 'T' for equation (2)
 * @param[in] n Order of matrices (N >= 0)
 * @param[in] a DOUBLE PRECISION array, dimension (lda,N), quasitriangular A
 * @param[in] lda Leading dimension of A (lda >= max(1,N))
 * @param[in] e DOUBLE PRECISION array, dimension (lde,N), upper triangular E
 * @param[in] lde Leading dimension of E (lde >= max(1,N))
 * @param[in,out] b DOUBLE PRECISION array, dimension (ldb,N)
 *                  On entry: upper triangular B
 *                  On exit: Cholesky factor U
 * @param[in] ldb Leading dimension of B (ldb >= max(1,N))
 * @param[out] scale Scaling factor (0 < scale <= 1)
 * @param[out] dwork DOUBLE PRECISION workspace, dimension (6*N-6)
 * @param[out] info Exit code (0=success, <0=parameter error, 1=near singular,
 *                  2=not complex conjugate, 3=not c-stable)
 */
void sg03bv(
    const char* trans, const i32 n,
    const f64* a, const i32 lda,
    const f64* e, const i32 lde,
    f64* b, const i32 ldb,
    f64* scale, f64* dwork,
    i32* info
);

/**
 * @brief Solve generalized Sylvester equation for small systems.
 *
 * Solves for X the generalized Sylvester equation:
 *
 *     A^T * X * C + E^T * X * D = SCALE * Y,    (TRANS='N')
 *
 * or the transposed equation:
 *
 *     A * X * C^T + E * X * D^T = SCALE * Y,    (TRANS='T')
 *
 * where A and E are M-by-M matrices (A upper quasitriangular, E upper triangular),
 * C and D are N-by-N matrices, X and Y are M-by-N matrices. N must be 1 or 2.
 * The pencil A - lambda*E must be in generalized real Schur form.
 * SCALE is set to avoid overflow in X.
 *
 * @param[in] trans 'N' for equation (1), 'T' for transposed equation
 * @param[in] m Order of matrices A and E (m >= 0)
 * @param[in] n Order of matrices C and D (n = 1 or 2)
 * @param[in] a DOUBLE PRECISION array, dimension (lda,m)
 *              Upper quasitriangular matrix A
 * @param[in] lda Leading dimension of A (lda >= max(1,m))
 * @param[in] c DOUBLE PRECISION array, dimension (ldc,n)
 *              Matrix C
 * @param[in] ldc Leading dimension of C (ldc >= max(1,n))
 * @param[in] e DOUBLE PRECISION array, dimension (lde,m)
 *              Upper triangular matrix E
 * @param[in] lde Leading dimension of E (lde >= max(1,m))
 * @param[in] d DOUBLE PRECISION array, dimension (ldd,n)
 *              Matrix D
 * @param[in] ldd Leading dimension of D (ldd >= max(1,n))
 * @param[in,out] x DOUBLE PRECISION array, dimension (ldx,n)
 *                  In: Right-hand side Y
 *                  Out: Solution X
 * @param[in] ldx Leading dimension of X (ldx >= max(1,m))
 * @param[out] scale Scaling factor (0 < scale <= 1)
 * @param[out] info Exit code (0 = success, <0 = invalid parameter,
 *                  1 = nearly singular, perturbed values used)
 */
void sg03bw(
    const char* trans,
    const i32 m, const i32 n,
    const f64* a, const i32 lda,
    const f64* c, const i32 ldc,
    const f64* e, const i32 lde,
    const f64* d, const i32 ldd,
    f64* x, const i32 ldx,
    f64* scale,
    i32* info
);

/**
 * @brief Solve 2-by-2 generalized Lyapunov equation.
 *
 * Solves for Cholesky factor U (X = op(U)^T * op(U)) the generalized
 * continuous-time or discrete-time Lyapunov equation:
 *
 *   Continuous (DICO='C'):
 *     op(A)^T * X * op(E) + op(E)^T * X * op(A) = -SCALE^2 * op(B)^T * op(B)
 *
 *   Discrete (DICO='D'):
 *     op(A)^T * X * op(A) - op(E)^T * X * op(E) = -SCALE^2 * op(B)^T * op(B)
 *
 * where op(K) = K or K^T, A,B,E,U are 2x2 real matrices, E and B upper triangular.
 * Pencil A-lambda*E must have complex conjugate eigenvalues in the stability region.
 * Also computes auxiliary matrices M1 and M2.
 *
 * @param[in] dico 'C' for continuous-time, 'D' for discrete-time
 * @param[in] trans 'N' for op(K)=K, 'T' for op(K)=K^T
 * @param[in] a DOUBLE PRECISION array, dimension (lda,2), matrix A
 * @param[in] lda Leading dimension of A (lda >= 2)
 * @param[in] e DOUBLE PRECISION array, dimension (lde,2), upper triangular E
 * @param[in] lde Leading dimension of E (lde >= 2)
 * @param[in] b DOUBLE PRECISION array, dimension (ldb,2), upper triangular B
 * @param[in] ldb Leading dimension of B (ldb >= 2)
 * @param[out] u DOUBLE PRECISION array, dimension (ldu,2), Cholesky factor
 * @param[in] ldu Leading dimension of U (ldu >= 2)
 * @param[out] scale Scaling factor (0 < scale <= 1)
 * @param[out] m1 DOUBLE PRECISION array, dimension (ldm1,2), auxiliary matrix
 * @param[in] ldm1 Leading dimension of M1 (ldm1 >= 2)
 * @param[out] m2 DOUBLE PRECISION array, dimension (ldm2,2), auxiliary matrix
 * @param[in] ldm2 Leading dimension of M2 (ldm2 >= 2)
 * @param[out] info Exit code (0=success, 2=not complex conjugate,
 *                  3=eigenvalues not stable, 4=ZSTEIN failed)
 */
void sg03bx(
    const char* dico, const char* trans,
    const f64* a, const i32 lda,
    const f64* e, const i32 lde,
    const f64* b, const i32 ldb,
    f64* u, const i32 ldu,
    f64* scale,
    f64* m1, const i32 ldm1,
    f64* m2, const i32 ldm2,
    i32* info
);

/**
 * @brief Solve generalized discrete-time Lyapunov equation (complex Hammarling).
 *
 * Computes the Cholesky factor U of X = U^H*U or X = U*U^H for the
 * generalized d-stable discrete-time Lyapunov equation:
 *
 *   TRANS='N': A^H * X * A - E^H * X * E = -SCALE^2 * B^H * B
 *   TRANS='C': A * X * A^H - E * X * E^H = -SCALE^2 * B * B^H
 *
 * where A, E, B are complex N-by-N upper triangular matrices.
 * The pencil A-lambda*E must be in complex generalized Schur form with
 * eigenvalues inside the unit circle (d-stable).
 *
 * Uses the complex Hammarling algorithm with recursive 3-step approach.
 *
 * @param[in] trans 'N' for equation (1), 'C' for equation (2)
 * @param[in] n Order of matrices (n >= 0)
 * @param[in,out] a COMPLEX*16 array (lda,n), upper triangular A;
 *                  lower part used as workspace, diagonal restored
 * @param[in] lda Leading dimension of A (lda >= max(1,n))
 * @param[in,out] e COMPLEX*16 array (lde,n), upper triangular E;
 *                  strictly lower part used as workspace if TRANS='N'
 * @param[in] lde Leading dimension of E (lde >= max(1,n))
 * @param[in,out] b COMPLEX*16 array (ldb,n), input: upper triangular B;
 *                  output: upper triangular Cholesky factor U
 * @param[in] ldb Leading dimension of B (ldb >= max(1,n))
 * @param[out] scale Scaling factor (0 < scale <= 1)
 * @param[out] dwork DOUBLE PRECISION workspace, dimension max(n-1,10) if n>1
 * @param[out] zwork COMPLEX*16 workspace, dimension max(3*n-3,0)
 * @param[out] info Exit code: 0=success, -i=invalid param i,
 *                  3=not d-stable, 4=ZSTEIN failed
 */
void sg03bs(
    const char* trans, const i32 n,
    c128* a, const i32 lda,
    c128* e, const i32 lde,
    c128* b, const i32 ldb,
    f64* scale,
    f64* dwork, c128* zwork,
    i32* info
);

/**
 * @brief Solve c-stable generalized continuous-time Lyapunov equation (complex).
 *
 * Computes the Cholesky factor U of X = U^H*U or X = U*U^H for the
 * generalized c-stable continuous-time Lyapunov equation:
 *
 *   TRANS='N': A^H * X * E + E^H * X * A = -SCALE^2 * B^H * B
 *   TRANS='C': A * X * E^H + E * X * A^H = -SCALE^2 * B * B^H
 *
 * where A, E, B are complex N-by-N upper triangular matrices.
 * The pencil A-lambda*E must be in complex generalized Schur form with
 * eigenvalues having negative real parts (c-stable).
 *
 * Uses the complex Hammarling algorithm with recursive 3-step approach.
 *
 * @param[in] trans 'N' for equation (1), 'C' for equation (2)
 * @param[in] n Order of matrices (n >= 0)
 * @param[in,out] a COMPLEX*16 array (lda,n), upper triangular A;
 *                  lower part used as workspace, diagonal restored
 * @param[in] lda Leading dimension of A (lda >= max(1,n))
 * @param[in,out] e COMPLEX*16 array (lde,n), upper triangular E;
 *                  strictly lower part used as workspace if TRANS='N'
 * @param[in] lde Leading dimension of E (lde >= max(1,n))
 * @param[in,out] b COMPLEX*16 array (ldb,n), input: upper triangular B;
 *                  output: upper triangular Cholesky factor U
 * @param[in] ldb Leading dimension of B (ldb >= max(1,n))
 * @param[out] scale Scaling factor (0 < scale <= 1)
 * @param[out] dwork DOUBLE PRECISION workspace, dimension max(n-1,0)
 * @param[out] zwork COMPLEX*16 workspace, dimension max(3*n-3,0)
 * @param[out] info Exit code: 0=success, -i=invalid param i, 3=not c-stable
 */
void sg03bt(
    const char* trans, const i32 n,
    c128* a, const i32 lda,
    c128* e, const i32 lde,
    c128* b, const i32 ldb,
    f64* scale,
    f64* dwork, c128* zwork,
    i32* info
);

/**
 * @brief Compute residual for continuous/discrete-time Lyapunov equation.
 *
 * Computes the residual matrix R for a continuous-time or discrete-time
 * "reduced" Lyapunov equation, using the formulas:
 *
 *   Continuous (DICO='C', JOBE='I'): R = op(A)'*X + X*op(A) + Q
 *   Continuous (DICO='C', JOBE='G'): R = op(A)'*X*op(E) + op(E)'*X*op(A) + Q
 *   Discrete (DICO='D', JOBE='I'):   R = op(A)'*X*op(A) - X + Q
 *   Discrete (DICO='D', JOBE='G'):   R = op(A)'*X*op(A) - op(E)'*X*op(E) + Q
 *
 * where X and Q are symmetric, A is upper real Schur, E is upper triangular,
 * and op(W) = W or W'.
 *
 * @param[in] dico 'C' for continuous-time, 'D' for discrete-time
 * @param[in] job 'R' residual only, 'N' or 'B' residual and norms
 * @param[in] jobe 'G' general E, 'I' identity E
 * @param[in] uplo 'U' upper triangular, 'L' lower triangular
 * @param[in] trans 'N' for op(W)=W, 'T' or 'C' for op(W)=W'
 * @param[in] n Order of matrices (n >= 0)
 * @param[in,out] a N-by-N upper real Schur matrix, dimension (lda,n)
 * @param[in] lda Leading dimension of A (lda >= max(1,n))
 * @param[in] e N-by-N upper triangular E (if jobe='G'), dimension (lde,n)
 * @param[in] lde Leading dimension of E
 * @param[in,out] x Symmetric matrix X, dimension (ldx,n)
 * @param[in] ldx Leading dimension of X (ldx >= max(1,n))
 * @param[in,out] r On entry: Q; on exit: residual R, dimension (ldr,n)
 * @param[in] ldr Leading dimension of R (ldr >= max(1,n))
 * @param[out] norms Frobenius norms of product terms (if job != 'R')
 * @param[out] dwork Workspace array, dimension (ldwork)
 * @param[in] ldwork Workspace size; -1 or -2 for query
 * @param[out] info Exit code: 0 = success, < 0 = invalid arg
 */
void sg02cv(
    const char* dico, const char* job, const char* jobe,
    const char* uplo, const char* trans,
    const i32 n,
    f64* a, const i32 lda,
    const f64* e, const i32 lde,
    f64* x, const i32 ldx,
    f64* r, const i32 ldr,
    f64* norms,
    f64* dwork, const i32 ldwork,
    i32* info
);

/**
 * @brief Compute residual for continuous/discrete-time Riccati equation.
 *
 * Computes the residual matrix R for a continuous-time or discrete-time
 * algebraic Riccati equation and/or the "closed-loop system" matrix C.
 *
 * Continuous-time formulas:
 *   R = op(A)'*X + X*op(A) +/- X*G*X + Q  (JOBE='I')
 *   R = op(A)'*X*op(E) + op(E)'*X*op(A) +/- op(E)'*X*G*X*op(E) + Q  (JOBE='G')
 *   C = op(A) +/- G*X  or  C = op(A) +/- G*X*op(E)
 *
 * Discrete-time formulas:
 *   R = op(A)'*X*op(A) - X +/- op(A)'*X*G*X*op(A) + Q  (JOBE='I')
 *   R = op(A)'*X*op(A) - op(E)'*X*op(E) +/- op(A)'*X*G*X*op(A) + Q  (JOBE='G')
 *   C = op(A) +/- G*X*op(A)
 *
 * @param[in] dico 'C' for continuous-time, 'D' for discrete-time
 * @param[in] job 'A' R and C, 'R' R only, 'C' C only, 'N' R,C,norms, 'B' R,norms
 * @param[in] jobe 'G' general E, 'I' identity E
 * @param[in] flag 'P' plus sign, 'M' minus sign
 * @param[in] jobg 'G' G given, 'D' D given (G=D*D'), 'F' F given, 'H' H,K given
 * @param[in] uplo 'U' upper triangular, 'L' lower triangular
 * @param[in] trans 'N' for op(W)=W, 'T' or 'C' for op(W)=W'
 * @param[in] n Order of matrices A, E, Q, X, C, R (n >= 0)
 * @param[in] m Number of columns in D, F, H, K' (m >= 0 if jobg != 'G')
 * @param[in] a N-by-N matrix A, dimension (lda,n)
 * @param[in] lda Leading dimension of A (lda >= max(1,n))
 * @param[in] e N-by-N matrix E (if jobe='G'), dimension (lde,n)
 * @param[in] lde Leading dimension of E
 * @param[in,out] g Matrix G/D/B depending on jobg, dimension (ldg,*)
 * @param[in] ldg Leading dimension of G
 * @param[in,out] x Symmetric matrix X, dimension (ldx,n)
 * @param[in] ldx Leading dimension of X (ldx >= max(1,n))
 * @param[in] f Matrix F or H (if jobg='F' or 'H'), dimension (ldf,*)
 * @param[in] ldf Leading dimension of F
 * @param[in] k Matrix K (if jobg='H'), dimension (ldk,*)
 * @param[in] ldk Leading dimension of K
 * @param[in] xe Matrix product X*E or X*A (if needed), dimension (ldxe,*)
 * @param[in] ldxe Leading dimension of XE
 * @param[in,out] r On entry: Q; on exit: residual R (if job != 'C')
 * @param[in] ldr Leading dimension of R
 * @param[out] c Closed-loop matrix (if job != 'R' and job != 'B')
 * @param[in] ldc Leading dimension of C
 * @param[out] norms Frobenius norms of product terms (if job='N' or 'B')
 * @param[out] dwork Workspace array, dimension (ldwork)
 * @param[in] ldwork Workspace size; -1 for optimal query, -2 for minimum query
 * @param[out] info Exit code: 0 = success, < 0 = invalid arg
 */
void sg02cw(
    const char* dico, const char* job, const char* jobe, const char* flag,
    const char* jobg, const char* uplo, const char* trans,
    const i32 n, const i32 m,
    f64* a, const i32 lda,
    const f64* e, const i32 lde,
    f64* g, const i32 ldg,
    f64* x, const i32 ldx,
    const f64* f, const i32 ldf,
    const f64* k, const i32 ldk,
    const f64* xe, const i32 ldxe,
    f64* r, const i32 ldr,
    f64* c, const i32 ldc,
    f64* norms,
    f64* dwork, const i32 ldwork,
    i32* info
);

/**
 * @brief Line search parameter minimizing Riccati residual norm.
 *
 * Finds the line search parameter alpha minimizing the Frobenius norm:
 *   P(alpha) := ||R(X+alpha*S)||_F = ||(1-alpha)*R(X) +/- alpha^2*V||_F
 *
 * where R(X) is the residual of a (generalized) continuous-time algebraic
 * Riccati equation:
 *   0 = op(A)'*X + X*op(A) +/- X*G*X + Q  =:  R(X)
 * or
 *   0 = op(A)'*X*op(E) + op(E)'*X*op(A) +/- op(E)'*X*G*X*op(E) + Q  =:  R(X)
 *
 * V = op(E)'*S*G*S*op(E), where S is the Newton step.
 *
 * The algorithm sets up a cubic polynomial whose roots in [0,2] are candidates
 * for the solution, then solves a 3x3 generalized eigenproblem.
 *
 * @param[in] jobe 'G' general E matrix, 'I' identity E
 * @param[in] flag 'P' plus sign, 'M' minus sign
 * @param[in] jobg 'G' G given, 'D' D given (G=D*D'), 'F' F given, 'H' H,K given
 * @param[in] uplo 'U' upper triangle, 'L' lower triangle of symmetric matrices
 * @param[in] trans 'N' op(W)=W, 'T'/'C' op(W)=W'
 * @param[in] n Order of matrices E, R, S (n >= 0)
 * @param[in] m Number of columns of D/F/K' (m >= 0 if jobg != 'G')
 * @param[in] e N-by-N matrix E if jobe='G' and jobg='G' or 'D'
 * @param[in] lde Leading dimension of E
 * @param[in] r N-by-N symmetric residual R(X)
 * @param[in] ldr Leading dimension of R (ldr >= max(1,n))
 * @param[in] s Newton step S; N-by-N symmetric if jobg='G'/'D', M-by-N if jobg='H'
 * @param[in] lds Leading dimension of S
 * @param[in,out] g Matrix G/D/F/H depending on jobg
 * @param[in] ldg Leading dimension of G (ldg >= max(1,n))
 * @param[out] alpha Optimal line search parameter in [0,2]
 * @param[out] rnorm Frobenius norm of residual R(X+alpha*S)
 * @param[out] dwork Workspace, returns V in leading N-by-N triangle
 * @param[in] ldwork Workspace size; -1 for optimal, -2 for minimum query
 * @param[out] iwarn 0=ok, 2=no optimal alpha in [0,2] found, set to 1
 * @param[out] info 0=success, <0=invalid param, 1=MC01XD eigenproblem failed
 */
void sg02cx(
    const char* jobe, const char* flag, const char* jobg, const char* uplo,
    const char* trans,
    const i32 n, const i32 m,
    const f64* e, const i32 lde,
    const f64* r, const i32 ldr,
    const f64* s, const i32 lds,
    f64* g, const i32 ldg,
    f64* alpha, f64* rnorm,
    f64* dwork, const i32 ldwork,
    i32* iwarn, i32* info
);

/**
 * @brief Compute optimal gain matrix for discrete/continuous Riccati problems.
 *
 * Computes:
 * - Discrete:   K = (R + B'XB)^{-1} (B'X*op(A) + L')
 * - Continuous: K = R^{-1} (B'X*op(E) + L')
 *
 * R may be specified in factored form. If R or R + B'XB is positive definite,
 * let C be its Cholesky factor. Optionally returns H = op(E)'XB + L (continuous)
 * or H = op(A)'XB + L (discrete), or the matrix F = H/C.
 *
 * @param[in] dico 'C' continuous-time (eq. 2), 'D' discrete-time (eq. 1)
 * @param[in] jobe 'G' general E matrix, 'I' identity E (not used for DICO='D')
 * @param[in] job 'K' compute K only, 'H' compute H and K, 'F' compute F if possible,
 *                'D' H and K with pre-transformed B,L, 'C' F with pre-transformed B,L
 * @param[in] jobx 'C' compute op(X*op(E)) or op(X*op(A)), 'N' do not compute
 * @param[in] fact 'N' R unfactored, 'D' R=D'D, 'C' Cholesky factor, 'U' UdU'/LdL'
 * @param[in] uplo 'U' upper triangle, 'L' lower triangle stored
 * @param[in] jobl 'Z' L is zero, 'N' L is nonzero
 * @param[in] trans 'N' op(W)=W, 'T'/'C' op(W)=W'
 * @param[in] n Order of matrices A and X (n >= 0)
 * @param[in] m Order of matrix R and columns of B,L (m >= 0)
 * @param[in] p Number of rows of D (p >= m for DICO='C', p >= 0 for DICO='D')
 * @param[in] a N-by-N state matrix A (used only if DICO='D')
 * @param[in] lda Leading dimension of A
 * @param[in] e N-by-N matrix E (used only if DICO='C' and JOBE='G')
 * @param[in] lde Leading dimension of E
 * @param[in,out] b N-by-M input matrix B; may be overwritten
 * @param[in] ldb Leading dimension of B (ldb >= max(1,n))
 * @param[in,out] r M-by-M matrix R (factored or not); contains factor on exit
 * @param[in] ldr Leading dimension of R
 * @param[in,out] ipiv Pivot indices for UdU'/LdL' factorization
 * @param[in] l N-by-M cross-weighting matrix L (if JOBL='N')
 * @param[in] ldl Leading dimension of L
 * @param[in,out] x N-by-N solution matrix X; may contain Cholesky factor on exit
 * @param[in] ldx Leading dimension of X (ldx >= max(1,n))
 * @param[in] rnorm 1-norm of original R (used only if FACT='U')
 * @param[out] k M-by-N gain matrix K
 * @param[in] ldk Leading dimension of K (ldk >= max(1,m))
 * @param[out] h N-by-M matrix H or F (if JOB != 'K')
 * @param[in] ldh Leading dimension of H
 * @param[out] xe N-by-N matrix X*E or X*A (if JOBX='C')
 * @param[in] ldxe Leading dimension of XE
 * @param[out] oufact[2] Factorization info: oufact[0]=1 Cholesky/2 UdU';
 *                       oufact[1]=1 Cholesky(X)/2 spectral(X)
 * @param[out] dwork Workspace; dwork[1] contains rcond on exit
 * @param[in] ldwork Workspace size (-1 for optimal query, -2 for minimum query)
 * @param[out] info 0=success, <0=invalid param, i=d[i] zero, m+1=singular,
 *                  m+2=eigenvalue failed, m+3=X indefinite
 */
void sg02nd(
    const char* dico, const char* jobe, const char* job, const char* jobx,
    const char* fact, const char* uplo, const char* jobl, const char* trans,
    const i32 n, const i32 m, const i32 p,
    const f64* a, const i32 lda,
    const f64* e, const i32 lde,
    f64* b, const i32 ldb,
    f64* r, const i32 ldr,
    i32* ipiv,
    const f64* l, const i32 ldl,
    f64* x, const i32 ldx,
    const f64 rnorm,
    f64* k, const i32 ldk,
    f64* h, const i32 ldh,
    f64* xe, const i32 ldxe,
    i32* oufact,
    f64* dwork, const i32 ldwork,
    i32* info
);

/**
 * @brief Compute a complex Givens rotation in real arithmetic.
 *
 * Computes parameters for the complex Givens rotation:
 *
 *     (  CR-CI*I   SR-SI*I )   ( XR+XI*I )   ( Z )
 *     (                    ) * (         ) = (   )
 *     ( -SR-SI*I   CR+CI*I )   ( YR+YI*I )   ( 0 )
 *
 * where CR, CI, SR, SI, XR, XI, YR, YI are real numbers and I is the
 * imaginary unit. Z is a non-negative real number.
 *
 * The routine avoids overflow using max-norm scaling.
 *
 * @param[in] xr Real part of first complex input
 * @param[in] xi Imaginary part of first complex input
 * @param[in] yr Real part of second complex input
 * @param[in] yi Imaginary part of second complex input
 * @param[out] cr Real part of cosine factor
 * @param[out] ci Imaginary part of cosine factor
 * @param[out] sr Real part of sine factor
 * @param[out] si Imaginary part of sine factor
 * @param[out] z Non-negative real result (norm of input vector)
 */
void sg03by(
    const f64 xr, const f64 xi,
    const f64 yr, const f64 yi,
    f64* cr, f64* ci,
    f64* sr, f64* si,
    f64* z
);

/**
 * @brief Solve stable generalized continuous/discrete-time Lyapunov equation.
 *
 * Computes the Cholesky factor U of the matrix X,
 *     X = op(U)^H * op(U),
 * which is the solution of either:
 *
 * Continuous-time (DICO='C'):
 *     op(A)^H * X * op(E) + op(E)^H * X * op(A) = -SCALE^2 * op(B)^H * op(B)
 *
 * Discrete-time (DICO='D'):
 *     op(A)^H * X * op(A) - op(E)^H * X * op(E) = -SCALE^2 * op(B)^H * op(B)
 *
 * where op(K) = K (TRANS='N') or op(K) = K^H (TRANS='C').
 *
 * The pencil A - lambda*E must be c-stable (continuous) or d-stable (discrete).
 *
 * @param[in] dico 'C' for continuous-time, 'D' for discrete-time
 * @param[in] fact 'N' to compute Schur factorization, 'F' if supplied
 * @param[in] trans 'N' for op(K)=K, 'C' for op(K)=K^H
 * @param[in] n Order of matrices A, E (n >= 0)
 * @param[in] m Rows in op(B) (m >= 0)
 * @param[in,out] a N-by-N matrix; Schur factor on exit if FACT='N'
 * @param[in] lda Leading dimension of A (lda >= max(1,n))
 * @param[in,out] e N-by-N matrix; Schur factor on exit if FACT='N'
 * @param[in] lde Leading dimension of E (lde >= max(1,n))
 * @param[in,out] q N-by-N unitary transformation matrix
 * @param[in] ldq Leading dimension of Q (ldq >= max(1,n))
 * @param[in,out] z N-by-N unitary transformation matrix
 * @param[in] ldz Leading dimension of Z (ldz >= max(1,n))
 * @param[in,out] b Input matrix; Cholesky factor U on exit
 * @param[in] ldb Leading dimension of B
 * @param[out] scale Scale factor (0 < scale <= 1)
 * @param[out] alpha Eigenvalue numerators (alpha[j]/beta[j])
 * @param[out] beta Eigenvalue denominators (non-negative real)
 * @param[out] dwork Real workspace
 * @param[out] zwork Complex workspace; zwork[0] returns optimal size
 * @param[in] lzwork Workspace size (-1 for query)
 * @param[out] info 0=success, 4=ZGGES failed, 5=not c-stable, 6=not d-stable, 7=ZSTEIN failed
 */
void sg03bz(
    const char* dico, const char* fact, const char* trans,
    const i32 n, const i32 m,
    c128* a, const i32 lda,
    c128* e, const i32 lde,
    c128* q, const i32 ldq,
    c128* z, const i32 ldz,
    c128* b, const i32 ldb,
    f64* scale,
    c128* alpha, c128* beta,
    f64* dwork, c128* zwork, const i32 lzwork,
    i32* info
);

#ifdef __cplusplus
}
#endif

#endif /* SLICOT_SG_H */
