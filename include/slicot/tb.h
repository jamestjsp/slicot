/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#ifndef SLICOT_TB_H
#define SLICOT_TB_H

#include "../slicot_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Balance complex system matrix for state-space representation.
 *
 * Reduces the 1-norm of the system matrix S = [[A, B], [C, 0]] by applying
 * diagonal similarity transformations iteratively. The transformation is:
 *     diag(D,I)^(-1) * S * diag(D,I)
 *
 * The balancing can be performed on:
 * - All matrices (JOB='A')
 * - B and A only (JOB='B')
 * - C and A only (JOB='C')
 * - A only (JOB='N')
 *
 * @param[in] job Matrix selection:
 *                'A' = All matrices involved
 *                'B' = B and A matrices only
 *                'C' = C and A matrices only
 *                'N' = Only A matrix (B, C not involved)
 * @param[in] n Order of A, number of rows of B, columns of C (n >= 0)
 * @param[in] m Number of columns of B (m >= 0)
 * @param[in] p Number of rows of C (p >= 0)
 * @param[in,out] maxred Maximum allowed norm reduction.
 *                       On entry: if > 0, must be > 1. If <= 0, default 10.0 used.
 *                       On exit: ratio of original to balanced 1-norm.
 * @param[in,out] a Complex state matrix, dimension (lda,n)
 *                  On exit: balanced matrix inv(D)*A*D
 * @param[in] lda Leading dimension of a (lda >= max(1,n))
 * @param[in,out] b Complex input matrix, dimension (ldb,m)
 *                  On exit: balanced matrix inv(D)*B
 * @param[in] ldb Leading dimension of b (ldb >= max(1,n) if m > 0, else >= 1)
 * @param[in,out] c Complex output matrix, dimension (ldc,n)
 *                  On exit: balanced matrix C*D
 * @param[in] ldc Leading dimension of c (ldc >= max(1,p))
 * @param[out] scale Scaling factors D(j), dimension (n)
 * @return Exit code:
 *         0 = success
 *         < 0 = -i means i-th argument invalid
 */
i32 slicot_tb01iz(char job, i32 n, i32 m, i32 p, f64* maxred,
                  c128* a, i32 lda, c128* b, i32 ldb, c128* c, i32 ldc,
                  f64* scale);

/**
 * @brief Frequency response matrix of state-space system.
 *
 * Computes the complex frequency response matrix (transfer matrix) G(freq) of
 * state-space representation (A,B,C):
 *
 *     G(freq) = C * (freq*I - A)^(-1) * B
 *
 * where A is N-by-N, B is N-by-M, C is P-by-N, and freq is complex.
 *
 * The matrix A is optionally balanced and reduced to upper Hessenberg form.
 * The same transformations are applied to B and C.
 *
 * @param[in] baleig Balance/eigenvalue option:
 *                   'N' = no balancing, no eigenvalues, no condition estimate
 *                   'C' = no balancing, compute condition estimate
 *                   'B' or 'E' = balance and compute eigenvalues (requires inita='G')
 *                   'A' = balance, eigenvalues, and condition estimate (requires inita='G')
 * @param[in] inita 'G' = general matrix (will reduce to Hessenberg)
 *                  'H' = already in upper Hessenberg form
 * @param[in] n Order of state matrix A (n >= 0)
 * @param[in] m Number of inputs (m >= 0)
 * @param[in] p Number of outputs (p >= 0)
 * @param[in] freq Complex frequency at which to evaluate G(freq)
 * @param[in,out] a State matrix, dimension (lda,n). If inita='G', on exit
 *                  contains upper Hessenberg form.
 * @param[in] lda Leading dimension of a (lda >= max(1,n))
 * @param[in,out] b Input matrix, dimension (ldb,m). If inita='G', on exit
 *                  contains Q^T * B where Q is the Hessenberg transformation.
 * @param[in] ldb Leading dimension of b (ldb >= max(1,n))
 * @param[in,out] c Output matrix, dimension (ldc,n). If inita='G', on exit
 *                  contains C * Q where Q is the Hessenberg transformation.
 * @param[in] ldc Leading dimension of c (ldc >= max(1,p))
 * @param[out] rcond Reciprocal condition number (if baleig='C' or 'A')
 * @param[out] g Frequency response matrix, dimension (ldg,m), complex
 * @param[in] ldg Leading dimension of g (ldg >= max(1,p))
 * @param[out] evre Real parts of eigenvalues, dimension (n) (if baleig='B','E','A' and inita='G')
 * @param[out] evim Imaginary parts of eigenvalues, dimension (n)
 * @param[out] hinvb (freq*I-A)^(-1)*B, dimension (ldhinv,m), complex
 * @param[in] ldhinv Leading dimension of hinvb (ldhinv >= max(1,n))
 * @param[out] dwork Real workspace (see documentation for size requirements)
 * @param[in] ldwork Size of dwork (varies by options, see routine)
 * @param[out] zwork Complex workspace, dimension (lzwork)
 * @param[in] lzwork Size of zwork (>= n*n+2*n if baleig='C'/'A', else >= n*n)
 * @return Exit code:
 *         0 = success
 *         1 = eigenvalue computation failed (results may still be valid)
 *         2 = freq too close to eigenvalue or matrix nearly singular
 */
i32 slicot_tb05ad(char baleig, char inita, i32 n, i32 m, i32 p, c128 freq,
                  f64* a, i32 lda, f64* b, i32 ldb, f64* c, i32 ldc,
                  f64* rcond, c128* g, i32 ldg, f64* evre, f64* evim,
                  c128* hinvb, i32 ldhinv, f64* dwork, i32 ldwork,
                  c128* zwork, i32 lzwork);

/**
 * @brief Balance system matrices (A,B,C) using diagonal similarity transformation.
 *
 * Reduces the 1-norm of the system matrix S = [A B; C 0] by balancing.
 * Applies diagonal similarity transformation inv(D)*A*D iteratively
 * to make rows and columns of diag(D,I)^{-1} * S * diag(D,I) as close
 * in norm as possible.
 *
 * The balancing can be performed on:
 *   - S = [A B; C 0] (JOB='A')
 *   - S = [A B]      (JOB='B')
 *   - S = [A; C]     (JOB='C')
 *   - S = A          (JOB='N')
 *
 * @param[in] job Specifies which matrices are involved:
 *                'A' = All matrices (A, B, C)
 *                'B' = B and A matrices only
 *                'C' = C and A matrices only
 *                'N' = A matrix only (B and C not involved)
 * @param[in] n Order of A, rows of B, columns of C (n >= 0)
 * @param[in] m Number of columns of B (m >= 0)
 * @param[in] p Number of rows of C (p >= 0)
 * @param[in,out] maxred On entry: maximum allowed reduction in 1-norm if zero
 *                       rows/columns encountered. If maxred > 0, must be > 1.
 *                       If maxred <= 0, default value 10.0 is used.
 *                       On exit: ratio of original to balanced matrix 1-norm.
 * @param[in,out] a State matrix, dimension (lda,n)
 *                  In: N-by-N matrix A
 *                  Out: Balanced matrix inv(D)*A*D
 * @param[in] lda Leading dimension of A (lda >= max(1,n))
 * @param[in,out] b Input matrix, dimension (ldb,m)
 *                  In: N-by-M matrix B (if m > 0)
 *                  Out: Balanced matrix inv(D)*B
 * @param[in] ldb Leading dimension of B (ldb >= max(1,n) if m > 0, else >= 1)
 * @param[in,out] c Output matrix, dimension (ldc,n)
 *                  In: P-by-N matrix C (if p > 0)
 *                  Out: Balanced matrix C*D
 * @param[in] ldc Leading dimension of C (ldc >= max(1,p))
 * @param[out] scale Scaling factors, dimension (n)
 *                   scale[j] = D(j,j) for j = 0,...,n-1
 * @param[out] info Exit code:
 *                  0 = success
 *                  -i = i-th parameter had illegal value
 */
void tb01id(const char* job, i32 n, i32 m, i32 p, f64* maxred,
            f64* a, i32 lda, f64* b, i32 ldb, f64* c, i32 ldc,
            f64* scale, i32* info);

/**
 * @brief Stable/unstable decomposition of a state-space system.
 *
 * Computes an additive spectral decomposition of the transfer function matrix
 * of (A,B,C) by reducing A to block-diagonal form. The leading diagonal block
 * has eigenvalues in a specified domain of interest.
 *
 * Transformation: A <- inv(U)*A*U, B <- inv(U)*B, C <- C*U
 *
 * The domain of interest is defined by DICO, STDOM, and ALPHA:
 * - Continuous (DICO='C'): Re(lambda) < ALPHA (STDOM='S') or > ALPHA (STDOM='U')
 * - Discrete (DICO='D'): |lambda| < ALPHA (STDOM='S') or > ALPHA (STDOM='U')
 *
 * @param[in] dico System type: 'C' continuous, 'D' discrete
 * @param[in] stdom Domain type: 'S' stability, 'U' instability
 * @param[in] joba Matrix shape: 'S' Schur form, 'G' general
 * @param[in] n System order (n >= 0)
 * @param[in] m Number of inputs (m >= 0)
 * @param[in] p Number of outputs (p >= 0)
 * @param[in] alpha Domain boundary (alpha >= 0 for discrete)
 * @param[in,out] a State matrix (n x n). On exit: block-diagonal in Schur form
 * @param[in] lda Leading dimension of A
 * @param[in,out] b Input matrix (n x m). On exit: inv(U)*B
 * @param[in] ldb Leading dimension of B
 * @param[in,out] c Output matrix (p x n). On exit: C*U
 * @param[in] ldc Leading dimension of C
 * @param[out] ndim Number of eigenvalues in domain of interest
 * @param[out] u Transformation matrix (n x n)
 * @param[in] ldu Leading dimension of U
 * @param[out] wr Real parts of eigenvalues
 * @param[out] wi Imaginary parts of eigenvalues
 * @param[out] dwork Workspace
 * @param[in] ldwork Workspace size (>= n if JOBA='S', >= 3*n if JOBA='G')
 * @param[out] info 0=success, 1=QR failed, 2=reordering failed, 3=Sylvester failed
 */
void tb01kd(const char* dico, const char* stdom, const char* joba, i32 n, i32 m, i32 p,
            f64 alpha, f64* a, i32 lda, f64* b, i32 ldb, f64* c, i32 ldc,
            i32* ndim, f64* u, i32 ldu, f64* wr, f64* wi, f64* dwork, i32 ldwork, i32* info);

/**
 * @brief Block-diagonalize state-space system with Schur form input.
 *
 * Computes an additive spectral decomposition of the transfer-function matrix
 * of system (A,B,C) by reducing A to block-diagonal form. A must be in real
 * Schur form with leading NDIM eigenvalues distinct from trailing eigenvalues.
 *
 * Transformation: A <- V*A*U, B <- V*B, C <- C*U, where V = inv(U)
 *
 * The algorithm solves the Sylvester equation A11*X - X*A22 = A12 to construct
 * the similarity transformation T = [[I, -X], [0, I]] that block-diagonalizes A.
 *
 * @param[in] n Order of state matrix A (n >= 0)
 * @param[in] m Number of inputs (columns of B, m >= 0)
 * @param[in] p Number of outputs (rows of C, p >= 0)
 * @param[in] ndim Dimension of leading diagonal block (0 <= ndim <= n)
 * @param[in,out] a State matrix in real Schur form (lda, n)
 *                  On exit: block-diagonal with A12=0, lower elements zeroed
 * @param[in] lda Leading dimension of A (lda >= max(1,n))
 * @param[in,out] b Input matrix (ldb, m). On exit: V*B
 * @param[in] ldb Leading dimension of B (ldb >= max(1,n))
 * @param[in,out] c Output matrix (ldc, n). On exit: C*U
 * @param[in] ldc Leading dimension of C (ldc >= max(1,p))
 * @param[in,out] u Transformation matrix (ldu, n)
 *                  On entry: initial transformation (typically identity)
 *                  On exit: accumulated transformation matrix
 * @param[in] ldu Leading dimension of U (ldu >= max(1,n))
 * @param[out] v Inverse transformation matrix (ldv, n). V = inv(U)
 * @param[in] ldv Leading dimension of V (ldv >= max(1,n))
 * @param[out] info 0=success, <0=invalid parameter, 1=Sylvester solve failed
 */
void tb01kx(i32 n, i32 m, i32 p, i32 ndim, f64* a, i32 lda,
            f64* b, i32 ldb, f64* c, i32 ldc, f64* u, i32 ldu,
            f64* v, i32 ldv, i32* info);

/**
 * @brief Reduce state matrix to ordered Schur form.
 *
 * Reduces system (A,B,C) to ordered upper real Schur form using orthogonal
 * similarity transformation. The leading block has eigenvalues in the
 * specified domain of interest.
 *
 * @param[in] dico 'C' continuous-time, 'D' discrete-time
 * @param[in] stdom 'S' stability domain, 'U' instability domain
 * @param[in] joba 'S' A is Schur, 'G' A is general
 * @param[in] n Order of A
 * @param[in] m Number of inputs
 * @param[in] p Number of outputs
 * @param[in] alpha Eigenvalue domain boundary
 * @param[in,out] a State matrix (n x n), output is ordered Schur form
 * @param[in] lda Leading dimension of A
 * @param[in,out] b Input matrix (n x m), output is U'*B
 * @param[in] ldb Leading dimension of B
 * @param[in,out] c Output matrix (p x n), output is C*U
 * @param[in] ldc Leading dimension of C
 * @param[out] ndim Number of eigenvalues in domain of interest
 * @param[out] u Transformation matrix (n x n)
 * @param[in] ldu Leading dimension of U
 * @param[out] wr Real parts of eigenvalues
 * @param[out] wi Imaginary parts of eigenvalues
 * @param[out] dwork Workspace
 * @param[in] ldwork Workspace size
 * @param[out] info 0=success, 1=QR failed, 2=reordering failed
 */
void tb01ld(const char* dico, const char* stdom, const char* joba, i32 n, i32 m, i32 p,
            f64 alpha, f64* a, i32 lda, f64* b, i32 ldb, f64* c, i32 ldc,
            i32* ndim, f64* u, i32 ldu, f64* wr, f64* wi, f64* dwork, i32 ldwork, i32* info);

/**
 * @brief Reduce (B,A) to upper or lower controller Hessenberg form.
 *
 * Computes a unitary state-space transformation U which reduces the pair (B,A)
 * to controller Hessenberg form using Householder transformations.
 *
 * For UPLO='U' (upper):
 *     [U'B | U'AU] has upper triangular B and upper Hessenberg A
 *
 * For UPLO='L' (lower):
 *     [U'AU | U'B] has lower Hessenberg A and lower triangular B
 *
 * @param[in] jobu Transformation accumulation mode:
 *                 'N' = do not form U
 *                 'I' = initialize U to identity, accumulate transformations
 *                 'U' = update given U with transformations
 * @param[in] uplo Hessenberg form type:
 *                 'U' = upper controller Hessenberg form
 *                 'L' = lower controller Hessenberg form
 * @param[in] n State dimension (n >= 0)
 * @param[in] m Input dimension (m >= 0)
 * @param[in,out] a State matrix (lda, n). On exit, U' * A * U.
 * @param[in] lda Leading dimension of a (lda >= max(1, n))
 * @param[in,out] b Input matrix (ldb, m). On exit, U' * B.
 * @param[in] ldb Leading dimension of b (ldb >= max(1, n))
 * @param[in,out] u Transformation matrix (ldu, n):
 *                  JOBU='N': not referenced
 *                  JOBU='I': on exit, orthogonal transformation matrix
 *                  JOBU='U': on entry, given matrix; on exit, updated
 * @param[in] ldu Leading dimension of u (ldu >= 1 if JOBU='N', else >= max(1,n))
 * @param[out] dwork Workspace (max(n, m-1))
 * @param[out] info 0 = success, -i = parameter i invalid
 */
void tb01md(const char* jobu, const char* uplo, i32 n, i32 m,
            f64* a, i32 lda, f64* b, i32 ldb,
            f64* u, i32 ldu, f64* dwork, i32* info);

/**
 * @brief Reduce (A,C) pair to observer Hessenberg form.
 *
 * Reduces the pair (A,C) to lower or upper observer Hessenberg form
 * using (and optionally accumulating) unitary state-space transformations.
 *
 * The transformation is: A_out = U' * A * U, C_out = C * U
 *
 * @param[in] jobu 'N': don't form U; 'I': U = identity then accumulate;
 *                 'U': update given U
 * @param[in] uplo 'U': upper observer Hessenberg; 'L': lower
 * @param[in] n State dimension (order of A). N >= 0.
 * @param[in] p Output dimension (rows of C). 0 <= P <= N.
 * @param[in,out] a State matrix (lda, n). On exit: U' * A * U.
 * @param[in] lda Leading dimension of a. LDA >= max(1,N).
 * @param[in,out] c Output matrix (ldc, n). On exit: C * U.
 * @param[in] ldc Leading dimension of c. LDC >= max(1,P).
 * @param[in,out] u Transformation matrix (ldu, n).
 * @param[in] ldu Leading dimension of u.
 * @param[out] dwork Workspace of size max(N, P-1).
 * @param[out] info 0 = success, < 0 = -i means i-th argument invalid.
 */
void tb01nd(const char* jobu, const char* uplo, i32 n, i32 p,
            f64* a, i32 lda, f64* c, i32 ldc,
            f64* u, i32 ldu, f64* dwork, i32* info);

/**
 * @brief Compute a minimal realization of a state-space system.
 *
 * Computes a minimal or controllable/observable realization for the
 * linear time-invariant multi-input/multi-output system:
 *     dX/dt = A * X + B * U
 *        Y  = C * X
 *
 * Uses the staircase algorithm of Varga to reduce (A, B, C).
 *
 * @param[in] job Reduction type:
 *                'M' = minimal realization (controllable & observable)
 *                'C' = controllable realization only
 *                'O' = observable realization only
 * @param[in] equil Balancing option:
 *                  'S' = scale state-space matrices first
 *                  'N' = no balancing
 * @param[in] n Order of state-space representation (N >= 0)
 * @param[in] m Number of system inputs (M >= 0)
 * @param[in] p Number of system outputs (P >= 0)
 * @param[in,out] a State dynamics matrix, dimension (LDA,N).
 *                  On exit: reduced matrix A in upper block Hessenberg form.
 * @param[in] lda Leading dimension of A (LDA >= max(1,N))
 * @param[in,out] b Input matrix, dimension (LDB,max(M,P)).
 *                  On exit: reduced matrix B.
 * @param[in] ldb Leading dimension of B (LDB >= max(1,N))
 * @param[in,out] c Output matrix, dimension (LDC,N).
 *                  On exit: reduced matrix C.
 * @param[in] ldc Leading dimension of C (LDC >= max(1,max(M,P)) if N > 0, else >= 1)
 * @param[out] nr Order of reduced system (0 <= NR <= N)
 * @param[in] tol Tolerance for rank determination. If TOL <= 0, default N*N*EPS is used.
 * @param[out] iwork Integer workspace, dimension (N + max(M,P))
 * @param[out] dwork Double workspace, dimension (LDWORK)
 * @param[in] ldwork Workspace size (LDWORK >= max(1, N + max(N, 3*max(M,P))))
 * @param[out] info Exit code:
 *                  = 0: success
 *                  < 0: if INFO = -i, the i-th argument had an illegal value
 */
void tb01pd(
    const char* job, const char* equil, i32 n, i32 m, i32 p,
    f64* a, i32 lda,
    f64* b, i32 ldb,
    f64* c, i32 ldc,
    i32* nr, f64 tol,
    i32* iwork, f64* dwork, i32 ldwork,
    i32* info
);

/**
 * @brief Compute minimal realization of state-space system.
 *
 * Finds a reduced (controllable, observable, or minimal) state-space
 * representation (Ar,Br,Cr) for any original representation (A,B,C).
 * The matrix Ar is in upper block Hessenberg form.
 *
 * @param[in] job 'M' minimal (controllable+observable), 'C' controllable only, 'O' observable only
 * @param[in] equil 'S' to scale (A,B,C), 'N' no scaling
 * @param[in] n State dimension (n >= 0)
 * @param[in] m Input dimension (m >= 0)
 * @param[in] p Output dimension (p >= 0)
 * @param[in,out] a State matrix (lda, n). On exit, Ar in upper block Hessenberg (nr x nr).
 * @param[in] lda Leading dimension of a (lda >= max(1, n))
 * @param[in,out] b Input matrix (ldb, m or max(m,p)). On exit, Br (nr x m).
 * @param[in] ldb Leading dimension of b (ldb >= max(1, n))
 * @param[in,out] c Output matrix (ldc, n). On exit, Cr (p x nr).
 * @param[in] ldc Leading dimension of c (ldc >= max(1,m,p) if n>0, else >= 1)
 * @param[out] nr Order of reduced system (nr <= n)
 * @param[in] tol Tolerance for rank decisions (0 for default)
 * @param[out] iwork Integer workspace (n + max(m,p)). First elements are block sizes.
 * @param[out] dwork Workspace. On exit, dwork[0] = optimal ldwork.
 * @param[in] ldwork Workspace size (>= max(1, n + max(n, 3*max(m,p))))
 * @param[out] info 0 = success, -i = parameter i invalid
 */
void tb01pd(const char* job, const char* equil, i32 n, i32 m, i32 p,
            f64* a, i32 lda, f64* b, i32 ldb, f64* c, i32 ldc,
            i32* nr, f64 tol, i32* iwork, f64* dwork, i32 ldwork, i32* info);

/**
 * @brief Reduce state-space system to controllability staircase form.
 *
 * Finds a controllable realization for the linear time-invariant multi-input
 * system:
 *     dX/dt = A * X + B * U
 *        Y  = C * X
 *
 * The system (A, B, C) is reduced to (Ac, Bc, Cc) where:
 *     Ac = Z' * A * Z,  Bc = Z' * B,  Cc = C * Z
 *
 * with Ac in upper block Hessenberg form:
 *     Ac = [ Acont     *    ]    Bc = [ Bcont ]
 *          [   0    Auncont ]         [   0   ]
 *
 * The blocks B1, A21, ..., Ap,p-1 have full row ranks.
 *
 * @param[in] jobz Transformation matrix option:
 *                 'N' = do not form Z
 *                 'F' = store transformations in factored form
 *                 'I' = initialize Z to identity and accumulate transformations
 * @param[in] n Order of state-space representation (N >= 0)
 * @param[in] m Number of system inputs (M >= 0)
 * @param[in] p Number of system outputs (P >= 0)
 * @param[in,out] a State dynamics matrix, dimension (LDA,N)
 *                  On entry: original matrix A
 *                  On exit: transformed matrix Ac
 * @param[in] lda Leading dimension of A (LDA >= max(1,N))
 * @param[in,out] b Input matrix, dimension (LDB,M)
 *                  On entry: original matrix B
 *                  On exit: transformed matrix Bc
 * @param[in] ldb Leading dimension of B (LDB >= max(1,N))
 * @param[in,out] c Output matrix, dimension (LDC,N)
 *                  On entry: original matrix C
 *                  On exit: transformed matrix Cc
 * @param[in] ldc Leading dimension of C (LDC >= max(1,P))
 * @param[out] ncont Order of controllable state-space representation
 * @param[out] indcon Controllability index
 * @param[out] nblk Block dimensions array, dimension (N)
 *                  Leading INDCON elements contain diagonal block sizes
 * @param[out] z Transformation matrix, dimension (LDZ,N)
 *               If JOBZ='I': orthogonal transformation matrix
 *               If JOBZ='F': factored form with TAU
 *               If JOBZ='N': not referenced
 * @param[in] ldz Leading dimension of Z (LDZ >= max(1,N) if JOBZ='I'/'F', else >= 1)
 * @param[out] tau Scalar factors of elementary reflectors, dimension (N)
 * @param[in] tol Tolerance for rank determination. If TOL <= 0, default N*N*EPS is used.
 * @param[out] iwork Integer workspace, dimension (M)
 * @param[out] dwork Double workspace, dimension (LDWORK)
 * @param[in] ldwork Workspace size (LDWORK >= max(1, N, 3*M, P))
 * @param[out] info Exit code:
 *                  = 0: success
 *                  < 0: if INFO = -i, the i-th argument had an illegal value
 */
void tb01ud(
    const char* jobz, i32 n, i32 m, i32 p,
    f64* a, i32 lda,
    f64* b, i32 ldb,
    f64* c, i32 ldc,
    i32* ncont, i32* indcon, i32* nblk,
    f64* z, i32 ldz, f64* tau, f64 tol,
    i32* iwork, f64* dwork, i32 ldwork,
    i32* info
);

/**
 * @brief Observable-unobservable decomposition of a standard system.
 *
 * Computes orthogonal transformation Z which reduces N-th order system (A,B,C) to:
 *
 *    Z'*A*Z = [ Ano  *  ]     Z'*B = [ Bno ]     C*Z = [ 0  Co ]
 *            [  0  Ao  ]            [ Bo  ]
 *
 * where (Ao,Bo,Co) is observable with order NOBSV. Ano contains unobservable
 * eigenvalues. The pencil (Ao-lambda*I; Co) has full column rank for all lambda
 * and is in observability staircase form.
 *
 * @param[in] compz  'N': do not compute Z
 *                   'I': Z initialized to I, orthogonal Z returned
 * @param[in] n      System order (N >= 0)
 * @param[in] m      Number of inputs (M >= 0)
 * @param[in] p      Number of outputs (P >= 0)
 * @param[in,out] a  On entry: N-by-N state matrix.
 *                   On exit: transformed Z'*A*Z
 * @param[in] lda    Leading dimension of A (>= max(1,N))
 * @param[in,out] b  On entry: N-by-M input matrix.
 *                   On exit: transformed Z'*B. LDB >= max(1,N) if M>0 or P>0
 * @param[in] ldb    Leading dimension of B
 * @param[in,out] c  On entry: P-by-N output matrix.
 *                   On exit: transformed C*Z = [0 Co]
 * @param[in] ldc    Leading dimension of C (>= max(1,M,P) if N>0)
 * @param[in,out] z  If COMPZ='I': on exit, N-by-N orthogonal transformation.
 *                   If COMPZ='N': not referenced.
 * @param[in] ldz    Leading dimension of Z (>= 1 if 'N', >= max(1,N) if 'I')
 * @param[out] nobsv Order of observable subsystem (Ao)
 * @param[out] nlblck Number of staircase blocks in (Ao; Co)
 * @param[out] ctau  Block dimensions, dimension (N). CTAU(i) for i=1..NLBLCK
 * @param[in] tol    Tolerance for rank determination. If <= 0, default N*N*EPS used
 * @param[out] iwork Integer workspace, dimension (P)
 * @param[out] dwork Workspace, dimension (N + max(1, N, 3*P, M))
 * @param[out] info  0: success, <0: -i means param i invalid
 */
void tb01ux(
    const char* compz, i32 n, i32 m, i32 p,
    f64* a, i32 lda,
    f64* b, i32 ldb,
    f64* c, i32 ldc,
    f64* z, i32 ldz,
    i32* nobsv, i32* nlblck, i32* ctau,
    f64 tol,
    i32* iwork, f64* dwork,
    i32* info
);

/**
 * @brief Convert discrete-time system to output normal form.
 *
 * Converts a stable discrete-time system (A, B, C, D) with initial state x0
 * into the output normal form, producing parameter vector THETA.
 *
 * The parameter vector THETA contains:
 * - THETA[0:N*L-1]: parameters for A and C matrices
 * - THETA[N*L:N*(L+M)-1]: parameters for B matrix
 * - THETA[N*(L+M):N*(L+M)+L*M-1]: parameters for D matrix
 * - THETA[N*(L+M)+L*M:N*(L+M+1)+L*M-1]: initial state x0
 *
 * Algorithm:
 * 1. Solve Lyapunov equation A'*Q*A - Q = -scale^2*C'*C in Cholesky factor T
 * 2. Transform system using T
 * 3. QR factorization of transposed observability matrix
 * 4. Extract parameters via N orthogonal transformations
 *
 * @param[in] apply Bijective mapping mode:
 *                  'A' = apply bijective mapping to remove norm(THETA_i) < 1 constraint
 *                  'N' = no bijective mapping
 * @param[in] n System order (N >= 0)
 * @param[in] m Number of inputs (M >= 0)
 * @param[in] l Number of outputs (L >= 0)
 * @param[in,out] a State matrix, dimension (LDA,N), column-major.
 *                  On entry: original system matrix (must be stable).
 *                  On exit: transformed system matrix.
 * @param[in] lda Leading dimension of A (>= max(1,N))
 * @param[in,out] b Input matrix, dimension (LDB,M), column-major.
 *                  On entry: original input matrix.
 *                  On exit: transformed input matrix.
 * @param[in] ldb Leading dimension of B (>= max(1,N))
 * @param[in,out] c Output matrix, dimension (LDC,N), column-major.
 *                  On entry: original output matrix.
 *                  On exit: transformed output matrix.
 * @param[in] ldc Leading dimension of C (>= max(1,L))
 * @param[in] d Feedthrough matrix, dimension (LDD,M), column-major (read-only)
 * @param[in] ldd Leading dimension of D (>= max(1,L))
 * @param[in,out] x0 Initial state vector, dimension (N).
 *                   On entry: original initial state.
 *                   On exit: transformed initial state.
 * @param[out] theta Parameter vector, dimension (LTHETA)
 * @param[in] ltheta Length of THETA array (>= N*(L+M+1)+L*M)
 * @param[out] scale Scale factor from Lyapunov equation solver
 * @param[out] dwork Workspace array, dimension (LDWORK)
 * @param[in] ldwork Length of DWORK
 * @param[out] info Exit code:
 *                  = 0: success
 *                  < 0: if INFO = -i, the i-th argument had an illegal value
 *                  = 1: Lyapunov equation could only be solved with scale = 0
 *                  = 2: matrix A is not discrete-time stable
 *                  = 3: QR algorithm failed to converge for matrix A
 */
void tb01vd(const char* apply, i32 n, i32 m, i32 l, f64* a, i32 lda,
            f64* b, i32 ldb, f64* c, i32 ldc, const f64* d, i32 ldd,
            f64* x0, f64* theta, i32 ltheta, f64* scale,
            f64* dwork, i32 ldwork, i32* info);

/**
 * @brief Convert output normal form to state-space representation.
 *
 * Converts a discrete-time system from output normal form (parameter vector THETA)
 * to standard state-space representation (A, B, C, D) with initial state x0.
 *
 * The parameter vector THETA contains:
 * - THETA[0:N*L-1]: parameters for A and C matrices
 * - THETA[N*L:N*(L+M)-1]: parameters for B matrix
 * - THETA[N*(L+M):N*(L+M)+L*M-1]: parameters for D matrix
 * - THETA[N*(L+M)+L*M:N*(L+M+1)+L*M-1]: initial state x0
 *
 * @param[in] apply Bijective mapping mode:
 *                  'A' = apply bijective mapping to remove norm(THETA_i) < 1 constraint
 *                  'N' = no bijective mapping
 * @param[in] n System order (N >= 0)
 * @param[in] m Number of inputs (M >= 0)
 * @param[in] l Number of outputs (L >= 0)
 * @param[in] theta Parameter vector, dimension (LTHETA)
 * @param[in] ltheta Length of THETA array (>= N*(L+M+1)+L*M)
 * @param[out] a State matrix, dimension (LDA,N), column-major
 * @param[in] lda Leading dimension of A (>= max(1,N))
 * @param[out] b Input matrix, dimension (LDB,M), column-major
 * @param[in] ldb Leading dimension of B (>= max(1,N))
 * @param[out] c Output matrix, dimension (LDC,N), column-major
 * @param[in] ldc Leading dimension of C (>= max(1,L))
 * @param[out] d Feedthrough matrix, dimension (LDD,M), column-major
 * @param[in] ldd Leading dimension of D (>= max(1,L))
 * @param[out] x0 Initial state vector, dimension (N)
 * @param[out] dwork Workspace array, dimension (LDWORK)
 * @param[in] ldwork Length of DWORK (>= N*(N+L+1))
 * @param[out] info Exit code: 0 = success, <0 = invalid parameter
 */
void tb01vy(const char* apply, i32 n, i32 m, i32 l, const f64* theta,
            i32 ltheta, f64* a, i32 lda, f64* b, i32 ldb, f64* c, i32 ldc,
            f64* d, i32 ldd, f64* x0, f64* dwork, i32 ldwork, i32* info);

/**
 * @brief Reduce state matrix A to real Schur form via orthogonal transformation
 *
 * Reduces system state matrix A to upper real Schur form by orthogonal similarity
 * transformation A <- U'*A*U, and applies transformation to B <- U'*B and C <- C*U.
 *
 * @param[in] n Order of state matrix A (n >= 0)
 * @param[in] m Number of inputs (columns of B, m >= 0)
 * @param[in] p Number of outputs (rows of C, p >= 0)
 * @param[in,out] a State matrix, dimension (lda,n)
 *                  On entry: original state dynamics matrix
 *                  On exit: real Schur form U'*A*U
 * @param[in] lda Leading dimension of A (lda >= max(1,n))
 * @param[in,out] b Input matrix, dimension (ldb,m)
 *                  On entry: original input matrix
 *                  On exit: transformed matrix U'*B
 * @param[in] ldb Leading dimension of B (ldb >= max(1,n))
 * @param[in,out] c Output matrix, dimension (ldc,n)
 *                  On entry: original output matrix
 *                  On exit: transformed matrix C*U
 * @param[in] ldc Leading dimension of C (ldc >= max(1,p))
 * @param[out] u Orthogonal transformation matrix, dimension (ldu,n)
 *               Schur vectors of A
 * @param[in] ldu Leading dimension of U (ldu >= max(1,n))
 * @param[out] wr Real parts of eigenvalues, dimension (n)
 * @param[out] wi Imaginary parts of eigenvalues, dimension (n)
 * @param[out] dwork Workspace array, dimension (ldwork)
 * @param[in] ldwork Workspace size (ldwork >= 3*n, larger for optimal performance)
 * @param[out] info Exit code: 0=success, <0=invalid parameter, >0=QR failed
 */
void tb01wd(
    const i32 n, const i32 m, const i32 p,
    f64* a, const i32 lda,
    f64* b, const i32 ldb,
    f64* c, const i32 ldc,
    f64* u, const i32 ldu,
    f64* wr, f64* wi,
    f64* dwork, const i32 ldwork,
    i32* info
);

/**
 * @brief Special similarity transformation of dual state-space system.
 *
 * Applies the transformation:
 *   A <-- P * A' * P,  B <-- P * C',  C <-- B' * P
 *
 * where P is a matrix with 1 on the secondary diagonal (anti-identity).
 * Matrix A can be specified as a band matrix. Optionally, matrix D
 * is transposed.
 *
 * This is a special similarity transformation of the dual system.
 *
 * @param[in] jobd Specifies D handling:
 *                 'D' = D is present and will be transposed
 *                 'Z' = D is zero (not referenced)
 * @param[in] n Order of matrix A, rows of B, columns of C (N >= 0)
 * @param[in] m Number of columns of B (M >= 0)
 * @param[in] p Number of rows of C (P >= 0)
 * @param[in] kl Number of subdiagonals of A to transform (0 <= KL <= max(0,N-1))
 * @param[in] ku Number of superdiagonals of A to transform (0 <= KU <= max(0,N-1))
 * @param[in,out] a State matrix, dimension (LDA,N).
 *                  On entry: original matrix A.
 *                  On exit: transformed matrix P*A'*P.
 * @param[in] lda Leading dimension of A (LDA >= max(1,N))
 * @param[in,out] b Input matrix, dimension (LDB,max(M,P)).
 *                  On entry: N-by-M original input matrix.
 *                  On exit: N-by-P dual input matrix P*C'.
 * @param[in] ldb Leading dimension of B (LDB >= max(1,N) if M>0 or P>0, else >= 1)
 * @param[in,out] c Output matrix, dimension (LDC,N).
 *                  On entry: P-by-N original output matrix.
 *                  On exit: M-by-N dual output matrix B'*P.
 * @param[in] ldc Leading dimension of C (LDC >= max(1,M,P) if N>0, else >= 1)
 * @param[in,out] d Direct transmission matrix, dimension (LDD,max(M,P)).
 *                  On entry if JOBD='D': P-by-M original D matrix.
 *                  On exit if JOBD='D': M-by-P transposed D matrix.
 *                  Not referenced if JOBD='Z'.
 * @param[in] ldd Leading dimension of D (LDD >= max(1,M,P) if JOBD='D', else >= 1)
 * @param[out] info Exit code:
 *                  = 0: success
 *                  < 0: if INFO = -i, the i-th argument had an illegal value
 */
void tb01xd(
    const char* jobd,
    const i32 n, const i32 m, const i32 p,
    const i32 kl, const i32 ku,
    f64* a, const i32 lda,
    f64* b, const i32 ldb,
    f64* c, const i32 ldc,
    f64* d, const i32 ldd,
    i32* info
);

/**
 * @brief Pertranspose state-space dual system (complex version).
 *
 * Applies special transformation:
 *   A <- P * A' * P,  B <- P * C',  C <- B' * P
 * where P has 1s on the secondary diagonal (anti-identity).
 * Matrix A can be specified as a band matrix. Optionally, matrix D
 * is transposed.
 *
 * This is a special similarity transformation of the dual system.
 * Complex version of tb01xd.
 *
 * @param[in] jobd Specifies D handling:
 *                 'D' = D is present and will be transposed
 *                 'Z' = D is zero (not referenced)
 * @param[in] n Order of matrix A, rows of B, columns of C (N >= 0)
 * @param[in] m Number of columns of B (M >= 0)
 * @param[in] p Number of rows of C (P >= 0)
 * @param[in] kl Number of subdiagonals of A to transform (0 <= KL <= max(0,N-1))
 * @param[in] ku Number of superdiagonals of A to transform (0 <= KU <= max(0,N-1))
 * @param[in,out] a State matrix, dimension (LDA,N).
 *                  On entry: original matrix A.
 *                  On exit: transformed matrix P*A'*P.
 * @param[in] lda Leading dimension of A (LDA >= max(1,N))
 * @param[in,out] b Input matrix, dimension (LDB,max(M,P)).
 *                  On entry: N-by-M original input matrix.
 *                  On exit: N-by-P dual input matrix P*C'.
 * @param[in] ldb Leading dimension of B (LDB >= max(1,N) if M>0 or P>0, else >= 1)
 * @param[in,out] c Output matrix, dimension (LDC,N).
 *                  On entry: P-by-N original output matrix.
 *                  On exit: M-by-N dual output matrix B'*P.
 * @param[in] ldc Leading dimension of C (LDC >= max(1,M,P) if N>0, else >= 1)
 * @param[in,out] d Direct transmission matrix, dimension (LDD,max(M,P)).
 *                  On entry if JOBD='D': P-by-M original D matrix.
 *                  On exit if JOBD='D': M-by-P transposed D matrix.
 *                  Not referenced if JOBD='Z'.
 * @param[in] ldd Leading dimension of D (LDD >= max(1,M,P) if JOBD='D', else >= 1)
 * @param[out] info Exit code:
 *                  = 0: success
 *                  < 0: if INFO = -i, the i-th argument had an illegal value
 */
void tb01xz(
    const char* jobd,
    const i32 n, const i32 m, const i32 p,
    const i32 kl, const i32 ku,
    c128* a, const i32 lda,
    c128* b, const i32 ldb,
    c128* c, const i32 ldc,
    c128* d, const i32 ldd,
    i32* info
);

/**
 * @brief Apply secondary diagonal permutation to state-space system.
 *
 * Applies a special similarity transformation:
 *   A <- P * A * P,  B <- P * B,  C <- C * P
 * where P has 1 on secondary diagonal and 0 elsewhere.
 *
 * @param[in] n Order of matrix A, rows of B, columns of C (n >= 0)
 * @param[in] m Number of columns of B (m >= 0)
 * @param[in] p Number of rows of C (p >= 0)
 * @param[in,out] a State matrix (lda, n). Transformed in-place.
 * @param[in] lda Leading dimension of a (lda >= max(1,n))
 * @param[in,out] b Input matrix (ldb, m). Transformed in-place.
 * @param[in] ldb Leading dimension of b (ldb >= n if m>0, else >= 1)
 * @param[in,out] c Output matrix (ldc, n). Transformed in-place.
 * @param[in] ldc Leading dimension of c (ldc >= max(1,p))
 * @param[out] info 0 = success, -i = parameter i invalid
 */
void tb01yd(i32 n, i32 m, i32 p, f64* a, i32 lda,
            f64* b, i32 ldb, f64* c, i32 ldc, i32* info);

/**
 * @brief Controllable realization for single-input systems.
 *
 * Finds a controllable realization for the linear time-invariant
 * single-input system dX/dt = A*X + B*U, Y = C*X, where A is an N-by-N
 * matrix, B is an N element vector, and C is a P-by-N matrix.
 * Reduces (A, B) to orthogonal canonical form using orthogonal
 * similarity transformations which are also applied to C.
 *
 * The algorithm:
 * 1. Finds Householder matrix Z1 reducing B to [*, 0, ..., 0]^T
 * 2. Applies transformation: A <- Z1' * A * Z1, C <- C * Z1
 * 3. Reduces A to upper Hessenberg form via DGEHRD
 * 4. Determines controllable order NCONT from sub-diagonal elements
 *
 * @param[in] jobz Mode parameter:
 *                 'N' = do not form Z, do not store transformations
 *                 'F' = do not form Z, store transformations in factored form
 *                 'I' = return Z as the orthogonal transformation matrix
 * @param[in] n Order of the system (n >= 0)
 * @param[in] p Number of system outputs (p >= 0)
 * @param[in,out] a State matrix, dimension (lda, n). On exit, the leading
 *                  NCONT-by-NCONT upper Hessenberg part contains the
 *                  canonical form of the controllable realization.
 * @param[in] lda Leading dimension of a (lda >= max(1, n))
 * @param[in,out] b Input vector, dimension (n). On exit, leading NCONT
 *                  elements contain canonical form with all but B(0) zero.
 * @param[in,out] c Output matrix, dimension (ldc, n). On exit, contains
 *                  transformed output matrix C*Z.
 * @param[in] ldc Leading dimension of c (ldc >= max(1, p))
 * @param[out] ncont Order of controllable realization
 * @param[out] z Orthogonal transformation matrix, dimension (ldz, n).
 *               If JOBZ='I', contains accumulated transformations Z1*Z2.
 *               If JOBZ='F', contains factored form with TAU.
 *               If JOBZ='N', not referenced.
 * @param[in] ldz Leading dimension of z (ldz >= max(1,n) if JOBZ='I'/'F', else >= 1)
 * @param[out] tau Scalar factors of elementary reflectors, dimension (n)
 * @param[in] tol Tolerance for controllability. If TOL <= 0, uses default
 *                TOLDEF = N*EPS*max(norm(A), norm(B))
 * @param[out] dwork Workspace, dimension (ldwork). On exit, dwork[0] = optimal ldwork.
 * @param[in] ldwork Workspace size (ldwork >= max(1, n, p))
 * @param[out] info 0 = success, -i = parameter i invalid
 */
void tb01zd(
    const char* jobz, i32 n, i32 p,
    f64* a, i32 lda,
    f64* b,
    f64* c, i32 ldc,
    i32* ncont,
    f64* z, i32 ldz, f64* tau, f64 tol,
    f64* dwork, i32 ldwork,
    i32* info
);

/**
 * @brief Convert state-space to polynomial matrix fraction.
 *
 * Computes a relatively prime left polynomial matrix representation
 * inv(P(s))*Q(s) or right polynomial matrix representation Q(s)*inv(P(s))
 * with the same transfer matrix T(s) as a given state-space representation:
 *
 *   inv(P(s))*Q(s) = Q(s)*inv(P(s)) = T(s) = C*inv(s*I-A)*B + D
 *
 * @param[in] leri 'L' for left, 'R' for right polynomial matrix fraction
 * @param[in] equil 'S' to balance (A,B,C), 'N' for no balancing
 * @param[in] n Order of state-space representation (n >= 0)
 * @param[in] m Number of inputs (m >= 0)
 * @param[in] p Number of outputs (p >= 0)
 * @param[in,out] a State matrix (lda, n). On exit: upper block Hessenberg Amin.
 * @param[in] lda Leading dimension of a (lda >= max(1,n))
 * @param[in,out] b Input matrix (ldb, max(m,p)). On exit: transformed Bmin.
 * @param[in] ldb Leading dimension of b (ldb >= max(1,n))
 * @param[in,out] c Output matrix (ldc, n). On exit: transformed Cmin.
 * @param[in] ldc Leading dimension of c (ldc >= max(1,m,p))
 * @param[in] d Feedthrough matrix (ldd, max(m,p)). Used as workspace.
 * @param[in] ldd Leading dimension of d (ldd >= max(1,m,p))
 * @param[out] nr Order of minimal state-space representation
 * @param[out] index Row/column degrees of P(s). Dimension p (left) or m (right).
 * @param[out] pcoeff Denominator P(s) coefficients (ldpco1, ldpco2, n+1)
 * @param[in] ldpco1 First dimension of pcoeff
 * @param[in] ldpco2 Second dimension of pcoeff
 * @param[out] qcoeff Numerator Q(s) coefficients (ldqco1, ldqco2, n+1)
 * @param[in] ldqco1 First dimension of qcoeff
 * @param[in] ldqco2 Second dimension of qcoeff
 * @param[out] vcoeff Intermediate V(s) coefficients (ldvco1, ldvco2, n+1)
 * @param[in] ldvco1 First dimension of vcoeff
 * @param[in] ldvco2 Second dimension of vcoeff (ldvco2 >= max(1,n))
 * @param[in] tol Tolerance for rank determination (0 for default)
 * @param[out] iwork Integer workspace (n + max(m,p))
 * @param[out] dwork Workspace. On exit, dwork[0] = optimal ldwork.
 * @param[in] ldwork Workspace size
 * @param[out] info 0=success, 1=singular V(s) computation, 2=singular P(s)
 */
void tb03ad(const char* leri, const char* equil, i32 n, i32 m, i32 p,
            f64* a, i32 lda, f64* b, i32 ldb, f64* c, i32 ldc,
            f64* d, i32 ldd, i32* nr, i32* index,
            f64* pcoeff, i32 ldpco1, i32 ldpco2,
            f64* qcoeff, i32 ldqco1, i32 ldqco2,
            f64* vcoeff, i32 ldvco1, i32 ldvco2,
            f64 tol, i32* iwork, f64* dwork, i32 ldwork, i32* info);

/**
 * @brief Compute polynomial matrix V(s) block by block.
 *
 * Internal helper for TB03AD. Calculates a PWORK-by-NR polynomial matrix V(s)
 * one block V:L-1(s) at a time in reverse order (L = INDBLK,...,1).
 *
 * At each stage:
 * - W(s) = V2(s) * A2 (V2 = already computed part, A2 = subdiagonal part)
 * - Wbar(s) = s * V:L(s) - W(s)
 * - V:L-1(s) = Wbar(s) * inv(R) where R is upper triangular from A
 *
 * NOTE: This routine does not check inputs for errors (speed optimization).
 *
 * @param[in] nr Total state dimension
 * @param[in] a State matrix in upper block Hessenberg form (lda, nr)
 * @param[in] lda Leading dimension of a
 * @param[in] indblk Number of blocks
 * @param[in] nblk Block sizes array of length indblk
 * @param[in,out] vcoeff V(s) coefficient array (ldvco1, ldvco2, indblk+1)
 * @param[in] ldvco1 First leading dimension of vcoeff
 * @param[in] ldvco2 Second leading dimension of vcoeff
 * @param[in,out] pcoeff P(s) coefficient array (ldpco1, ldpco2, indblk+1), used as workspace
 * @param[in] ldpco1 First leading dimension of pcoeff
 * @param[in] ldpco2 Second leading dimension of pcoeff
 * @param[out] info 0 = success, >0 = singular R block (i-th diagonal is zero)
 */
void tb03ay(i32 nr, const f64* a, i32 lda, i32 indblk, const i32* nblk,
            f64* vcoeff, i32 ldvco1, i32 ldvco2,
            f64* pcoeff, i32 ldpco1, i32 ldpco2, i32* info);

/**
 * @brief State-space to transfer function conversion.
 *
 * Computes the transfer matrix T(s) of a state-space representation (A,B,C,D).
 * T(s) is expressed as either row or column polynomial vectors over monic
 * least common denominator polynomials.
 *
 * For ROWCOL='R': T(s) = diag(s^INDEX(i)) * (U(s,0) + U(s,1)/s + ...)
 *                        / (D(s,0) + D(s,1)/s + ...)  (per row)
 *
 * Algorithm uses the Orthogonal Structure Theorem. For column factorization,
 * operates on the dual system.
 *
 * @param[in] rowcol 'R' = rows over common denominators
 *                   'C' = columns over common denominators
 * @param[in] n Order of state matrix A (n >= 0)
 * @param[in] m Number of inputs (m >= 0)
 * @param[in] p Number of outputs (p >= 0)
 * @param[in,out] a State matrix (lda, n). On exit, transformed A.
 * @param[in] lda Leading dimension of a
 * @param[in,out] b Input matrix. On exit, transformed B.
 * @param[in] ldb Leading dimension of b
 * @param[in,out] c Output matrix. On exit, transformed C.
 * @param[in] ldc Leading dimension of c
 * @param[in,out] d Direct transmission matrix. Modified internally if ROWCOL='C'.
 * @param[in] ldd Leading dimension of d
 * @param[out] nr Order of transformed state-space representation
 * @param[out] index Degrees of denominator polynomials (P or M elements)
 * @param[out] dcoeff Denominator polynomial coefficients (lddcoe, n+1)
 * @param[in] lddcoe Leading dimension of dcoeff
 * @param[out] ucoeff Numerator polynomial coefficients (lduco1, lduco2, n+1)
 * @param[in] lduco1 First leading dimension of ucoeff
 * @param[in] lduco2 Second leading dimension of ucoeff
 * @param[in] tol1 Tolerance for row determination (0 for default)
 * @param[in] tol2 Tolerance for controllability separation (0 for default)
 * @param[out] iwork Integer workspace (n + max(m,p))
 * @param[out] dwork Workspace. On exit, dwork[0] = optimal ldwork.
 * @param[in] ldwork Workspace size
 * @param[out] info 0 = success, < 0 = -i means parameter i invalid
 */
void tb04ad(const char* rowcol, i32 n, i32 m, i32 p,
            f64* a, i32 lda, f64* b, i32 ldb,
            f64* c, i32 ldc, f64* d, i32 ldd,
            i32* nr, i32* index, f64* dcoeff, i32 lddcoe,
            f64* ucoeff, i32 lduco1, i32 lduco2,
            f64 tol1, f64 tol2,
            i32* iwork, f64* dwork, i32 ldwork, i32* info);

/**
 * @brief Compute transfer matrix rows using Orthogonal Structure Theorem.
 *
 * Internal helper for TB04AD. Calculates the (PWORK x MWORK) transfer matrix
 * T(s) in the form of polynomial row vectors over monic least common
 * denominator polynomials. Each row is computed by:
 * 1. Separating controllable subsystem using TB01UD
 * 2. For each output row, forming dual SIMO system and applying TB01ZD
 * 3. Computing monic denominator and numerator polynomials
 *
 * @param[in] n Order of original system
 * @param[in] mwork Number of system inputs
 * @param[in] pwork Number of system outputs
 * @param[in,out] a State matrix (lda, n), modified on exit
 * @param[in] lda Leading dimension of a
 * @param[in,out] b Input matrix (ldb, mwork), modified on exit
 * @param[in] ldb Leading dimension of b
 * @param[in,out] c Output matrix (ldc, n), modified on exit
 * @param[in] ldc Leading dimension of c
 * @param[in] d Direct transmission matrix (ldd, mwork)
 * @param[in] ldd Leading dimension of d
 * @param[out] ncont Order of controllable subsystem
 * @param[out] indexd Degrees of denominator polynomials (pwork elements)
 * @param[out] dcoeff Denominator coefficients (lddcoe, n+1)
 * @param[in] lddcoe Leading dimension of dcoeff
 * @param[out] ucoeff Numerator coefficients (lduco1, lduco2, n+1)
 * @param[in] lduco1 First leading dimension of ucoeff
 * @param[in] lduco2 Second leading dimension of ucoeff
 * @param[out] at Workspace for transformed A (n1, n)
 * @param[in] n1 Leading dimension of at (at least max(1,n))
 * @param[out] tau Workspace for reflectors (n elements)
 * @param[in] tol1 Tolerance for TB01ZD (controllability per row)
 * @param[in] tol2 Tolerance for TB01UD (overall controllability)
 * @param[out] iwork Integer workspace (n + max(mwork, pwork))
 * @param[out] dwork Workspace. On exit, dwork[0] = optimal ldwork.
 * @param[in] ldwork Workspace size
 * @param[out] info 0 = success
 */
void tb04ay(i32 n, i32 mwork, i32 pwork,
            f64* a, i32 lda, f64* b, i32 ldb,
            f64* c, i32 ldc, f64* d, i32 ldd,
            i32* ncont, i32* indexd, f64* dcoeff, i32 lddcoe,
            f64* ucoeff, i32 lduco1, i32 lduco2,
            f64* at, i32 n1, f64* tau, f64 tol1, f64 tol2,
            i32* iwork, f64* dwork, i32 ldwork, i32* info);

/**
 * @brief Computes minimal/controllable/observable realization
 *
 * TB01PX finds a reduced (controllable, observable, or minimal) state-space
 * representation (Ar,Br,Cr) for any original state-space representation
 * (A,B,C). The matrix Ar is in an upper block Hessenberg staircase form.
 *
 * Two-phase reduction:
 * - Phase 1 (job='M' or 'C'): Remove uncontrollable part
 * - Phase 2 (job='M' or 'O'): Remove unobservable part
 *
 * @param[in] job   'M': minimal, 'C': controllable, 'O': observable
 * @param[in] equil 'S': balance (scale), 'N': no balancing
 * @param[in] n     Order of original system (n >= 0)
 * @param[in] m     Number of inputs (m >= 0)
 * @param[in] p     Number of outputs (p >= 0)
 * @param[in,out] a State matrix (n x n), on exit contains Ar in (nr x nr)
 * @param[in] lda   Leading dimension of a (lda >= max(1,n))
 * @param[in,out] b Input matrix (n x m or n x max(m,p) if job != 'C')
 * @param[in] ldb   Leading dimension of b (ldb >= max(1,n))
 * @param[in,out] c Output matrix (p x n or max(m,p) x n if job != 'C')
 * @param[in] ldc   Leading dimension of c (ldc >= max(1,m,p) if n > 0)
 * @param[out] nr   Order of reduced system
 * @param[out] infred Information on reduction (4 elements)
 * @param[in] tol   Tolerance for rank determination (if <= 0, default used)
 * @param[out] iwork Integer workspace (c*n+max(m,p) where c=2 if job='M', else 1)
 * @param[out] dwork Real workspace
 * @param[in] ldwork Length of dwork (>= n + max(n, 3*m, 3*p) if n > 0)
 * @param[out] info  0: success, <0: -i means param i invalid
 */
void tb01px(const char* job, const char* equil, i32 n, i32 m, i32 p,
            f64* a, i32 lda, f64* b, i32 ldb, f64* c, i32 ldc,
            i32* nr, i32* infred, f64 tol,
            i32* iwork, f64* dwork, i32 ldwork, i32* info);

/**
 * @brief Balance state-space (A,B,C,D) by permutations and scalings
 *
 * TB01TD reduces (A,B,C,D) to balanced form using state permutations
 * and state/input/output scalings. Uses DGEBAL for A, then scales B,C,D.
 *
 * @param[in] n     Order of A (n >= 0)
 * @param[in] m     Number of inputs (m >= 0)
 * @param[in] p     Number of outputs (p >= 0)
 * @param[in,out] a N-by-N state matrix, returns balanced A
 * @param[in] lda   Leading dimension of a (>= max(1,n))
 * @param[in,out] b N-by-M input matrix, returns balanced B
 * @param[in] ldb   Leading dimension of b (>= max(1,n))
 * @param[in,out] c P-by-N output matrix, returns balanced C
 * @param[in] ldc   Leading dimension of c (>= max(1,p))
 * @param[in,out] d P-by-M feedthrough matrix, returns scaled D
 * @param[in] ldd   Leading dimension of d (>= max(1,p))
 * @param[out] low  Lower index of balanced submatrix (1-based)
 * @param[out] igh  Upper index of balanced submatrix (1-based)
 * @param[out] scstat State transformation info from DGEBAL (n elements)
 * @param[out] scin Input scalings (m elements), j-th input *= scin[j-1]
 * @param[out] scout Output scalings (p elements), i-th output *= scout[i-1]
 * @param[out] dwork Workspace (n elements)
 * @param[out] info  0: success, <0: -i means param i invalid
 */
void tb01td(i32 n, i32 m, i32 p,
            f64* a, i32 lda, f64* b, i32 ldb, f64* c, i32 ldc, f64* d, i32 ldd,
            i32* low, i32* igh, f64* scstat, f64* scin, f64* scout,
            f64* dwork, i32* info);

/**
 * @brief Balances rows/columns of a matrix block using integer powers of BASE
 *
 * TB01TY balances the rows (MODE == 0) or columns (MODE != 0) of the
 * (NROW x NCOL) block of matrix X with offset (IOFF, JOFF). Each non-zero
 * row/column is scaled by BASE^IEXPT where IEXPT is the largest integer
 * <= EXPT, such that the scaled 1-norm satisfies:
 *
 *     (SIZE / BASE) < ABSSUM <= SIZE
 *
 * This form of scaling uses integer powers of the floating-point base,
 * introducing no rounding errors.
 *
 * @param[in] mode   0: balance rows, != 0: balance columns
 * @param[in] ioff   Row offset (0-based in C)
 * @param[in] joff   Column offset (0-based in C)
 * @param[in] nrow   Number of rows in block
 * @param[in] ncol   Number of columns in block
 * @param[in] size   Target 1-norm (absolute value used)
 * @param[in,out] x  Matrix to balance (ldx x *)
 * @param[in] ldx    Leading dimension of x
 * @param[out] bvect Scale factors: bvect[ioff..ioff+nrow-1] for rows,
 *                   bvect[joff..joff+ncol-1] for columns
 */
void tb01ty(i32 mode, i32 ioff, i32 joff, i32 nrow, i32 ncol,
            f64 size, f64* x, i32 ldx, f64* bvect);

/**
 * @brief Controllable realization for M1+M2 input system
 *
 * TB01UY finds a controllable realization for the linear time-invariant
 * multi-input system dX/dt = A*X + B1*U1 + B2*U2, Y = C*X, where the
 * compound input matrix [B1,B2] is reduced to orthogonal canonical form.
 *
 * The transformation alternates between B1 (M1 columns) and B2 (M2 columns),
 * producing a staircase form with blocks having full row rank.
 *
 * @param[in] jobz   'N': no Z; 'F': factored form; 'I': accumulate Z
 * @param[in] n      Order of system (>= 0)
 * @param[in] m1     Number of B1 columns (>= 0)
 * @param[in] m2     Number of B2 columns (>= 0)
 * @param[in] p      Number of outputs (>= 0)
 * @param[in,out] a  N-by-N state matrix, returns Ac = Z'*A*Z
 * @param[in] lda    Leading dimension of a (>= max(1,n))
 * @param[in,out] b  N-by-(M1+M2) input matrix, returns [Bc1,Bc2] = Z'*[B1,B2]
 * @param[in] ldb    Leading dimension of b (>= max(1,n))
 * @param[in,out] c  P-by-N output matrix, returns Cc = C*Z
 * @param[in] ldc    Leading dimension of c (>= max(1,p))
 * @param[out] ncont Order of controllable realization
 * @param[out] indcon Controllability index (always even)
 * @param[out] nblk  Block dimensions (2*N elements), INDCON/2 odd+even pairs
 * @param[out] z     N-by-N transformation matrix (if jobz='I' or 'F')
 * @param[in] ldz    Leading dimension of z (>= max(1,n) if jobz='I'/'F', else 1)
 * @param[out] tau   Elementary reflector scalars (min(n,m1+m2) elements)
 * @param[in] tol    Tolerance for rank detection (0 = default N*N*EPS)
 * @param[out] iwork Integer workspace (max(m1,m2) elements)
 * @param[out] dwork Double workspace (ldwork elements)
 * @param[in] ldwork Workspace size (>= max(n,3*max(m1,m2),p) if min(n,m)>0, else 1)
 *                   Use -1 for workspace query
 * @param[out] info  0: success, <0: -i means param i invalid
 */
void tb01uy(const char* jobz, i32 n, i32 m1, i32 m2, i32 p,
            f64* a, i32 lda, f64* b, i32 ldb, f64* c, i32 ldc,
            i32* ncont, i32* indcon, i32* nblk,
            f64* z, i32 ldz, f64* tau, f64 tol,
            i32* iwork, f64* dwork, i32 ldwork, i32* info);

/**
 * @brief Reduce state matrix A to upper Hessenberg form via orthogonal similarity.
 *
 * Reduces the system state matrix A to upper Hessenberg form using
 * orthogonal similarity transformation A <- U'*A*U and applies the
 * transformation to matrices B and C: B <- U'*B and C <- C*U.
 *
 * @param[in] compu Transformation matrix option:
 *                  'N' = do not compute U
 *                  'I' = initialize U to identity, return orthogonal U
 *                  'U' = update given U1, return U1*U
 * @param[in] n Order of state matrix A (n >= 0)
 * @param[in] m Number of inputs (columns of B, m >= 0)
 * @param[in] p Number of outputs (rows of C, p >= 0)
 * @param[in,out] a State matrix, dimension (lda,n)
 *                  On entry: original state dynamics matrix
 *                  On exit: upper Hessenberg form U'*A*U
 * @param[in] lda Leading dimension of A (lda >= max(1,n))
 * @param[in,out] b Input matrix, dimension (ldb,m)
 *                  On entry: original input matrix
 *                  On exit: transformed matrix U'*B
 * @param[in] ldb Leading dimension of B (ldb >= max(1,n))
 * @param[in,out] c Output matrix, dimension (ldc,n)
 *                  On entry: original output matrix
 *                  On exit: transformed matrix C*U
 * @param[in] ldc Leading dimension of C (ldc >= max(1,p))
 * @param[in,out] u Transformation matrix, dimension (ldu,n)
 *                  COMPU='N': not referenced
 *                  COMPU='I': on exit, orthogonal transformation matrix
 *                  COMPU='U': on entry, given matrix U1; on exit, U1*U
 * @param[in] ldu Leading dimension of U (ldu >= 1 if COMPU='N', else >= max(1,n))
 * @param[out] dwork Workspace array, dimension (ldwork)
 *                   On exit, dwork[0] = optimal ldwork
 * @param[in] ldwork Workspace size (ldwork >= 1, if n > 0: >= n-1+max(n,m,p))
 *                   If ldwork = -1, workspace query mode
 * @param[out] info Exit code: 0=success, <0=invalid parameter
 */
void tb01wx(const char* compu, i32 n, i32 m, i32 p,
            f64* a, i32 lda, f64* b, i32 ldb, f64* c, i32 ldc,
            f64* u, i32 ldu, f64* dwork, i32 ldwork, i32* info);

/**
 * @brief Separate strictly proper part from constant part of transfer function matrix.
 *
 * Separates a proper P-by-M transfer function matrix G into:
 *   G = G0 + D
 * where G0 is strictly proper (degree(num) < degree(den)) and D is constant.
 *
 * @param[in] order Polynomial coefficient ordering:
 *                  'I' = Increasing order of powers
 *                  'D' = Decreasing order of powers
 * @param[in] p Number of system outputs (rows of G), p >= 0
 * @param[in] m Number of system inputs (columns of G), m >= 0
 * @param[in] md Maximum polynomial degree + 1 (md = max(igd) + 1)
 * @param[in,out] ign Numerator degrees array, dimension (ldign, m)
 *                    On entry: degrees of numerator polynomials
 *                    On exit: degrees of strictly proper numerators G0
 * @param[in] ldign Leading dimension of ign (ldign >= max(1, p))
 * @param[in] igd Denominator degrees array, dimension (ldigd, m)
 * @param[in] ldigd Leading dimension of igd (ldigd >= max(1, p))
 * @param[in,out] gn Numerator coefficients array, dimension (p*m*md)
 *                   Polynomials stored column-wise, md locations each
 *                   On exit: coefficients of strictly proper numerators
 * @param[in] gd Denominator coefficients array, dimension (p*m*md)
 * @param[out] d Feedthrough matrix D, dimension (ldd, m)
 * @param[in] ldd Leading dimension of d (ldd >= max(1, p))
 * @param[in] tol Tolerance for negligible leading coefficients
 *                tol > 0: absolute tolerance
 *                tol <= 0: default tolerance ign*EPS*||num||_inf
 * @param[out] info Exit code:
 *                  0: success
 *                  -i: parameter i had illegal value
 *                  1: transfer function is not proper
 *                  2: denominator polynomial is null
 */
void tb04bv(const char* order, i32 p, i32 m, i32 md,
            i32* ign, i32 ldign, const i32* igd, i32 ldigd,
            f64* gn, const f64* gd, f64* d, i32 ldd, f64 tol, i32* info);

/**
 * @brief Add real matrix D to rational matrix G.
 *
 * Computes the sum G + D where G is a P-by-M rational matrix (polynomial
 * ratios) and D is a P-by-M real matrix. The (i,j) entry of D is added
 * to the corresponding rational entry g(i,j) = num(i,j)/den(i,j).
 *
 * Result: (num + D*den) / den
 *
 * If g(i,j) = 0 (both degrees zero and numerator zero), it is assumed
 * that its denominator is 1.
 *
 * @param[in] order Polynomial coefficient ordering:
 *                  'I' = Increasing order of powers
 *                  'D' = Decreasing order of powers
 * @param[in] p Number of system outputs (rows of G and D), p >= 0
 * @param[in] m Number of system inputs (columns of G and D), m >= 0
 * @param[in] md Maximum polynomial degree + 1 (md = max(ign,igd) + 1)
 * @param[in,out] ign Numerator degrees array, dimension (ldign, m)
 *                    On entry: degrees of numerator polynomials
 *                    On exit: degrees of numerators in G + D
 * @param[in] ldign Leading dimension of ign (ldign >= max(1, p))
 * @param[in] igd Denominator degrees array, dimension (ldigd, m)
 * @param[in] ldigd Leading dimension of igd (ldigd >= max(1, p))
 * @param[in,out] gn Numerator coefficients array, dimension (p*m*md)
 *                   Polynomials stored column-wise, md locations each.
 *                   The (i,j) polynomial starts at ((j-1)*p+i-1)*md+1.
 *                   On exit: coefficients of numerators in G + D.
 * @param[in] gd Denominator coefficients array, dimension (p*m*md)
 * @param[in] d Real matrix D, dimension (ldd, m)
 * @param[in] ldd Leading dimension of d (ldd >= max(1, p))
 * @param[out] info Exit code:
 *                  0: success
 *                  -i: parameter i had illegal value
 */
void tb04bw(const char* order, i32 p, i32 m, i32 md,
            i32* ign, i32 ldign, const i32* igd, i32 ldigd,
            f64* gn, const f64* gd, const f64* d, i32 ldd, i32* info);

/**
 * @brief Compute gain of SISO system from state-space, poles, and zeros.
 *
 * Computes the gain of a single-input single-output linear system,
 * given its state-space representation (A,b,c,d), and its poles and
 * zeros. The matrix A is assumed to be in upper Hessenberg form.
 *
 * The gain is computed using the formula:
 *
 *     g = (c*(S0*I - A)^(-1)*b + d) * prod(S0 - Pi) / prod(S0 - Zi)
 *
 * where Pi (i=1:IP) and Zj (j=1:IZ) are poles and zeros respectively,
 * and S0 is a real scalar different from all poles and zeros.
 *
 * @param[in] ip Number of system poles. ip >= 0
 * @param[in] iz Number of system zeros. iz >= 0
 * @param[in,out] a State dynamics matrix in upper Hessenberg form,
 *                  dimension (lda, ip). On exit: LU factorization of A - S0*I.
 * @param[in] lda Leading dimension of a. lda >= max(1, ip)
 * @param[in,out] b System input vector, dimension (ip).
 *                  On exit: solution of (A - S0*I)*x = b.
 * @param[in] c System output vector, dimension (ip)
 * @param[in] d System feedthrough scalar
 * @param[in] pr Real parts of system poles, dimension (ip).
 *               Complex conjugate pairs must be consecutive.
 * @param[in] pi Imaginary parts of system poles, dimension (ip)
 * @param[in] zr Real parts of system zeros, dimension (iz).
 *               Complex conjugate pairs must be consecutive.
 * @param[in] zi Imaginary parts of system zeros, dimension (iz)
 * @param[out] gain Computed gain of the system
 * @param[out] iwork Integer workspace, dimension (ip).
 *                   Contains pivot indices on exit.
 */
void tb04bx(i32 ip, i32 iz, f64* a, i32 lda, f64* b, const f64* c, f64 d,
            const f64* pr, const f64* pi, const f64* zr, const f64* zi,
            f64* gain, i32* iwork);

/**
 * @brief Transfer function matrix via pole-zero method.
 *
 * Computes the transfer function matrix G of a state-space
 * representation (A,B,C,D) of a linear time-invariant multivariable
 * system, using the pole-zeros method. Each element of the transfer
 * function matrix is returned in a cancelled, minimal form, with
 * numerator and denominator polynomials stored either in increasing
 * or decreasing order of the powers of the indeterminate.
 *
 * @param[in] jobd  'D': D matrix present; 'Z': D is zero
 * @param[in] order 'I': increasing powers; 'D': decreasing powers
 * @param[in] equil 'S': equilibrate (A,B,C); 'N': no equilibration
 * @param[in] n     System order, n >= 0
 * @param[in] m     Number of inputs, m >= 0
 * @param[in] p     Number of outputs, p >= 0
 * @param[in] md    Max polynomial degree + 1, md >= 1 (upper bound: n+1)
 * @param[in,out] a State matrix, dimension (lda,n). Modified if equil='S'
 * @param[in] lda   Leading dimension of a, lda >= max(1,n)
 * @param[in,out] b Input matrix, dimension (ldb,m). Destroyed on exit
 * @param[in] ldb   Leading dimension of b, ldb >= max(1,n)
 * @param[in,out] c Output matrix, dimension (ldc,n). Modified if equil='S'
 * @param[in] ldc   Leading dimension of c, ldc >= max(1,p)
 * @param[in] d     Feedthrough matrix, dimension (ldd,m). Not referenced if jobd='Z'
 * @param[in] ldd   Leading dimension of d
 * @param[out] ign  Numerator degrees, dimension (ldign,m)
 * @param[in] ldign Leading dimension of ign, ldign >= max(1,p)
 * @param[out] igd  Denominator degrees, dimension (ldigd,m)
 * @param[in] ldigd Leading dimension of igd, ldigd >= max(1,p)
 * @param[out] gn   Numerator coefficients, dimension (p*m*md)
 * @param[out] gd   Denominator coefficients, dimension (p*m*md)
 * @param[in] tol   Controllability tolerance. If <= 0, default used
 * @param[out] iwork Integer workspace, dimension (n)
 * @param[out] dwork Workspace, dimension (ldwork)
 * @param[in] ldwork Workspace size, >= max(1, n*(n+p) + max(n+max(n,p), n*(2*n+5)))
 * @param[out] info Exit code: 0=success, <0: -i-th arg invalid,
 *                  1: QR failed computing zeros, 2: QR failed computing poles
 */
void tb04bd(const char* jobd, const char* order, const char* equil,
            i32 n, i32 m, i32 p, i32 md,
            f64* a, i32 lda, f64* b, i32 ldb, f64* c, i32 ldc,
            const f64* d, i32 ldd,
            i32* ign, i32 ldign, i32* igd, i32 ldigd,
            f64* gn, f64* gd, f64 tol,
            i32* iwork, f64* dwork, i32 ldwork, i32* info);

/**
 * @brief State-space to minimal pole-zero-gain form
 *
 * Computes the transfer function matrix G of a state-space representation
 * (A,B,C,D) of a linear time-invariant multivariable system, using the
 * pole-zeros method. The transfer function matrix is returned in a minimal
 * pole-zero-gain form.
 *
 * @param[in] jobd 'D': D matrix present; 'Z': D is zero
 * @param[in] equil 'S': equilibrate (A,B,C); 'N': no equilibration
 * @param[in] n System order, n >= 0
 * @param[in] m Number of inputs, m >= 0
 * @param[in] p Number of outputs, p >= 0
 * @param[in] npz Max number of poles/zeros per SISO channel (upper bound: n)
 * @param[in,out] a State matrix, dimension (lda,n). Modified if equil='S'
 * @param[in] lda Leading dimension of a, lda >= max(1,n)
 * @param[in,out] b Input matrix, dimension (ldb,m). Destroyed on exit
 * @param[in] ldb Leading dimension of b, ldb >= max(1,n)
 * @param[in,out] c Output matrix, dimension (ldc,n). Modified if equil='S'
 * @param[in] ldc Leading dimension of c, ldc >= max(1,p)
 * @param[in] d Feedthrough matrix, dimension (ldd,m). Not referenced if jobd='Z'
 * @param[in] ldd Leading dimension of d
 * @param[out] nz Number of zeros per element, dimension (ldnz,m)
 * @param[in] ldnz Leading dimension of nz, ldnz >= max(1,p)
 * @param[out] np Number of poles per element, dimension (ldnp,m)
 * @param[in] ldnp Leading dimension of np, ldnp >= max(1,p)
 * @param[out] zerosr Real parts of zeros, dimension (p*m*npz)
 * @param[out] zerosi Imaginary parts of zeros, dimension (p*m*npz)
 * @param[out] polesr Real parts of poles, dimension (p*m*npz)
 * @param[out] polesi Imaginary parts of poles, dimension (p*m*npz)
 * @param[out] gains Transfer function gains, dimension (ldgain,m)
 * @param[in] ldgain Leading dimension of gains, ldgain >= max(1,p)
 * @param[in] tol Controllability tolerance. If <= 0, default used
 * @param[out] iwork Integer workspace, dimension (n)
 * @param[out] dwork Workspace, dimension (ldwork)
 * @param[in] ldwork Workspace size, >= max(1, n*(n+p) + max(n+max(n,p), n*(2*n+3)))
 * @param[out] info Exit code: 0=success, <0: -i-th arg invalid,
 *                  1: QR failed computing zeros, 2: QR failed computing poles
 */
void tb04cd(const char* jobd, const char* equil, i32 n, i32 m, i32 p, i32 npz,
            f64* a, i32 lda, f64* b, i32 ldb, f64* c, i32 ldc,
            const f64* d, i32 ldd,
            i32* nz, i32 ldnz, i32* np, i32 ldnp,
            f64* zerosr, f64* zerosi, f64* polesr, f64* polesi,
            f64* gains, i32 ldgain, f64 tol,
            i32* iwork, f64* dwork, i32 ldwork, i32* info);

#ifdef __cplusplus
}
#endif

#endif /* SLICOT_TB_H */
