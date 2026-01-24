/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#ifndef SLICOT_AB_H
#define SLICOT_AB_H

#include "../slicot_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Controllable realization for single-input systems.
 *
 * Finds a controllable realization for the linear time-invariant
 * single-input system dX/dt = A * X + B * U, where A is an N-by-N matrix
 * and B is an N element vector. The routine reduces the system to
 * orthogonal canonical form using orthogonal similarity transformations.
 *
 * The algorithm:
 * 1. Finds a Householder matrix that reduces B to [*, 0, ..., 0]^T
 * 2. Applies this transformation to A as Z1' * A * Z1
 * 3. Reduces the resulting A to upper Hessenberg form
 * 4. Determines controllable order by checking sub-diagonal elements
 *
 * @param[in] jobz Mode parameter:
 *                 'N' = do not form Z, do not store transformations
 *                 'F' = do not form Z, store transformations in factored form
 *                 'I' = return Z as the orthogonal transformation matrix
 * @param[in] n Order of the system (n >= 0)
 * @param[in,out] a State matrix, dimension (lda, n). On exit, contains
 *                  canonical form in leading NCONT-by-NCONT upper Hessenberg part.
 * @param[in] lda Leading dimension of a (lda >= max(1, n))
 * @param[in,out] b Input vector, dimension (n). On exit, leading NCONT elements
 *                  contain canonical form with all but B(1) set to zero.
 * @param[out] ncont Order of controllable realization
 * @param[out] z Orthogonal transformation matrix, dimension (ldz, n).
 *               If JOBZ='I', contains accumulated transformations.
 *               If JOBZ='F', contains factored form (use DORGQR to reconstruct).
 *               If JOBZ='N', not referenced.
 * @param[in] ldz Leading dimension of z (ldz >= max(1,n) if JOBZ='I'/'F', else ldz >= 1)
 * @param[out] tau Scalar factors of elementary reflectors, dimension (n)
 * @param[in] tol Tolerance for controllability. If TOL <= 0, uses default
 *                TOLDEF = N*EPS*max(norm(A), norm(B))
 * @param[out] dwork Workspace, dimension (ldwork). On exit, dwork[0] = optimal ldwork.
 * @param[in] ldwork Workspace size (ldwork >= max(1, n))
 * @param[out] info 0 = success, -i = parameter i invalid
 */
void ab01md(const char* jobz, i32 n, f64* a, i32 lda, f64* b, i32* ncont,
            f64* z, i32 ldz, f64* tau, f64 tol, f64* dwork, i32 ldwork,
            i32* info);

/**
 * @brief Controllable realization for multi-input systems.
 *
 * Finds a controllable realization for the linear time-invariant
 * multi-input system dX/dt = A * X + B * U, where A is an N-by-N matrix
 * and B is an N-by-M matrix. Reduces (A, B) to orthogonal canonical form
 * using orthogonal similarity transformations.
 *
 * The pair (A, B) is reduced to (Ac, Bc) where:
 *   Ac = Z' * A * Z = [Acont  *   ]    Bc = Z' * B = [Bcont]
 *                     [  0   Aunc]                   [  0  ]
 *
 * Acont is upper block Hessenberg with each subdiagonal block having
 * full row rank. The size of Auncont equals the dimension of the
 * uncontrollable subspace.
 *
 * @param[in] jobz Mode parameter:
 *                 'N' = do not form Z, do not store transformations
 *                 'F' = do not form Z, store transformations in factored form
 *                 'I' = return Z as the orthogonal transformation matrix
 * @param[in] n Order of the system (n >= 0)
 * @param[in] m Number of inputs (m >= 0)
 * @param[in,out] a State matrix, dimension (lda, n). On exit, contains
 *                  canonical form in leading NCONT-by-NCONT upper block
 *                  Hessenberg part.
 * @param[in] lda Leading dimension of a (lda >= max(1, n))
 * @param[in,out] b Input matrix, dimension (ldb, m). On exit, leading
 *                  NCONT rows contain Bcont with all but first block zero.
 * @param[in] ldb Leading dimension of b (ldb >= max(1, n))
 * @param[out] ncont Order of controllable realization
 * @param[out] indcon Controllability index of the controllable part
 * @param[out] nblk Array of dimension (n). Leading INDCON elements contain
 *                  the orders of the diagonal blocks of Acont.
 * @param[out] z Orthogonal transformation matrix, dimension (ldz, n).
 *               If JOBZ='I', contains accumulated transformations.
 *               If JOBZ='F', contains factored form (use DORGQR to reconstruct).
 *               If JOBZ='N', not referenced.
 * @param[in] ldz Leading dimension of z (ldz >= max(1,n) if JOBZ='I'/'F', else ldz >= 1)
 * @param[out] tau Scalar factors of elementary reflectors, dimension (n)
 * @param[in] tol Tolerance for rank determination. If TOL <= 0, uses default
 *                TOLDEF = N*N*EPS
 * @param[in] iwork Integer workspace, dimension (m)
 * @param[out] dwork Workspace, dimension (ldwork). On exit, dwork[0] = optimal ldwork.
 * @param[in] ldwork Workspace size (ldwork >= max(1, n, 3*m))
 * @param[out] info 0 = success, -i = parameter i invalid
 */
void ab01nd(const char* jobz, i32 n, i32 m, f64* a, i32 lda, f64* b, i32 ldb,
            i32* ncont, i32* indcon, i32* nblk, f64* z, i32 ldz, f64* tau,
            f64 tol, i32* iwork, f64* dwork, i32 ldwork, i32* info);

/**
 * @brief Reduce (A,B) pair to upper staircase form using orthogonal transformations.
 *
 * Reduces the matrices A and B using orthogonal state-space and input-space
 * transformations U and V such that Ac = U'*A*U and Bc = U'*B*V are in upper
 * staircase form. The controllable part Acont is upper block Hessenberg with
 * full row rank subdiagonal blocks.
 *
 * @param[in] stages 'F' forward stage only, 'B' backward stage only, 'A' all stages
 * @param[in] jobu 'N' don't form U, 'I' initialize and return U
 * @param[in] jobv 'N' don't form V, 'I' initialize and return V (STAGES != 'F')
 * @param[in] n State dimension (n >= 0)
 * @param[in] m Input dimension (m >= 0)
 * @param[in,out] a State matrix, dimension (lda, n). On exit, U'*A*U.
 * @param[in] lda Leading dimension of a (lda >= max(1, n))
 * @param[in,out] b Input matrix, dimension (ldb, m). On exit, U'*B*V or U'*B.
 * @param[in] ldb Leading dimension of b (ldb >= max(1, n))
 * @param[in,out] u Transformation matrix, dimension (ldu, n).
 * @param[in] ldu Leading dimension of u (ldu >= max(1, n) if jobu='I', else ldu >= 1)
 * @param[out] v Input transformation matrix, dimension (ldv, m).
 * @param[in] ldv Leading dimension of v
 * @param[in,out] ncont Order of controllable part. Input if stages='B'.
 * @param[in,out] indcon Controllability index. Input if stages='B'.
 * @param[in,out] kstair Array (n) of staircase dimensions. Input if stages='B'.
 * @param[in] tol Tolerance for rank determination
 * @param[out] iwork Integer workspace, dimension (m)
 * @param[out] dwork Workspace. On exit, dwork[0] = optimal ldwork.
 * @param[in] ldwork Workspace size
 * @param[out] info 0 = success, -i = parameter i invalid
 */
void ab01od(const char* stages, const char* jobu, const char* jobv,
            i32 n, i32 m, f64* a, i32 lda, f64* b, i32 ldb,
            f64* u, i32 ldu, f64* v, i32 ldv,
            i32* ncont, i32* indcon, i32* kstair, f64 tol,
            i32* iwork, f64* dwork, i32 ldwork, i32* info);

/**
 * @brief Bilinear transformation of state-space system.
 *
 * Performs discrete-time <-> continuous-time conversion via bilinear
 * transformation of the state-space matrices (A,B,C,D).
 *
 * For TYPE='D' (discrete -> continuous):
 *   A_out = beta * (alpha*I + A)^{-1} * (A - alpha*I)
 *   B_out = sqrt(2*alpha*beta) * (alpha*I + A)^{-1} * B
 *   C_out = sqrt(2*alpha*beta) * C * (alpha*I + A)^{-1}
 *   D_out = D - C * (alpha*I + A)^{-1} * B
 *
 * For TYPE='C' (continuous -> discrete):
 *   A_out = alpha * (beta*I - A)^{-1} * (beta*I + A)
 *   B_out = sqrt(2*alpha*beta) * (beta*I - A)^{-1} * B
 *   C_out = sqrt(2*alpha*beta) * C * (beta*I - A)^{-1}
 *   D_out = D + C * (beta*I - A)^{-1} * B
 *
 * @param[in] type 'D' for discrete->continuous, 'C' for continuous->discrete
 * @param[in] n Order of state matrix A (n >= 0)
 * @param[in] m Number of system inputs (m >= 0)
 * @param[in] p Number of system outputs (p >= 0)
 * @param[in] alpha Bilinear transformation parameter (alpha != 0)
 * @param[in] beta Bilinear transformation parameter (beta != 0)
 * @param[in,out] a State matrix, dimension (lda, n). On exit, transformed A.
 * @param[in] lda Leading dimension of a (lda >= max(1, n))
 * @param[in,out] b Input matrix, dimension (ldb, m). On exit, transformed B.
 * @param[in] ldb Leading dimension of b (ldb >= max(1, n))
 * @param[in,out] c Output matrix, dimension (ldc, n). On exit, transformed C.
 * @param[in] ldc Leading dimension of c (ldc >= max(1, p))
 * @param[in,out] d Feedthrough matrix, dimension (ldd, m). On exit, transformed D.
 * @param[in] ldd Leading dimension of d (ldd >= max(1, p))
 * @param[out] iwork Integer workspace, dimension (n)
 * @param[out] dwork Double workspace, dimension (ldwork)
 * @param[in] ldwork Workspace size (ldwork >= max(1, n))
 * @return 0 on success, -i if parameter i is invalid, 1 if (alpha*I + A) singular,
 *         2 if (beta*I - A) singular
 */
i32 ab04md(char type, i32 n, i32 m, i32 p, f64 alpha, f64 beta,
           f64* a, i32 lda, f64* b, i32 ldb, f64* c, i32 ldc, f64* d, i32 ldd,
           i32* iwork, f64* dwork, i32 ldwork);

/**
 * @brief Cascade (series) inter-connection of two state-space systems.
 *
 * Computes the state-space model (A,B,C,D) for the cascaded connection
 * of two systems G1 and G2, where output of G1 feeds input of G2:
 * Y = G2(G1(U))
 *
 * For UPLO='L' (lower block diagonal):
 *   A = [A1,    0   ]    B = [ B1   ]    C = [D2*C1, C2]    D = D2*D1
 *       [B2*C1, A2  ]        [B2*D1 ]
 *
 * For UPLO='U' (upper block diagonal):
 *   A = [A2,    B2*C1]    B = [B2*D1]    C = [C2, D2*C1]    D = D2*D1
 *       [0,     A1   ]        [ B1  ]
 *
 * @param[in] uplo 'L' for lower block diagonal, 'U' for upper block diagonal
 * @param[in] over 'N' no overlap, 'O' overlap arrays (workspace required)
 * @param[in] n1 Number of states in first system (n1 >= 0)
 * @param[in] m1 Number of inputs to first system (m1 >= 0)
 * @param[in] p1 Number of outputs from G1 = inputs to G2 (p1 >= 0)
 * @param[in] n2 Number of states in second system (n2 >= 0)
 * @param[in] p2 Number of outputs from second system (p2 >= 0)
 * @param[in] a1 State matrix of G1, dimension (lda1, n1)
 * @param[in] lda1 Leading dimension of a1 (lda1 >= max(1, n1))
 * @param[in] b1 Input matrix of G1, dimension (ldb1, m1)
 * @param[in] ldb1 Leading dimension of b1 (ldb1 >= max(1, n1))
 * @param[in] c1 Output matrix of G1, dimension (ldc1, n1)
 * @param[in] ldc1 Leading dimension of c1 (ldc1 >= max(1, p1) if n1 > 0)
 * @param[in] d1 Feedthrough matrix of G1, dimension (ldd1, m1)
 * @param[in] ldd1 Leading dimension of d1 (ldd1 >= max(1, p1))
 * @param[in] a2 State matrix of G2, dimension (lda2, n2)
 * @param[in] lda2 Leading dimension of a2 (lda2 >= max(1, n2))
 * @param[in] b2 Input matrix of G2, dimension (ldb2, p1)
 * @param[in] ldb2 Leading dimension of b2 (ldb2 >= max(1, n2))
 * @param[in] c2 Output matrix of G2, dimension (ldc2, n2)
 * @param[in] ldc2 Leading dimension of c2 (ldc2 >= max(1, p2) if n2 > 0)
 * @param[in] d2 Feedthrough matrix of G2, dimension (ldd2, p1)
 * @param[in] ldd2 Leading dimension of d2 (ldd2 >= max(1, p2))
 * @param[out] n Total state order of cascaded system (n = n1 + n2)
 * @param[out] a State matrix of cascaded system, dimension (lda, n1+n2)
 * @param[in] lda Leading dimension of a (lda >= max(1, n1+n2))
 * @param[out] b Input matrix of cascaded system, dimension (ldb, m1)
 * @param[in] ldb Leading dimension of b (ldb >= max(1, n1+n2))
 * @param[out] c Output matrix of cascaded system, dimension (ldc, n1+n2)
 * @param[in] ldc Leading dimension of c (ldc >= max(1, p2) if n1+n2 > 0)
 * @param[out] d Feedthrough matrix of cascaded system, dimension (ldd, m1)
 * @param[in] ldd Leading dimension of d (ldd >= max(1, p2))
 * @param[out] dwork Workspace, dimension (ldwork)
 * @param[in] ldwork Workspace size (ldwork >= max(1, p1*max(n1,m1,n2,p2)) if over='O')
 * @return 0 on success, -i if parameter i is invalid
 */
i32 ab05md(char uplo, char over, i32 n1, i32 m1, i32 p1, i32 n2, i32 p2,
           const f64* a1, i32 lda1, const f64* b1, i32 ldb1,
           const f64* c1, i32 ldc1, const f64* d1, i32 ldd1,
           const f64* a2, i32 lda2, const f64* b2, i32 ldb2,
           const f64* c2, i32 ldc2, const f64* d2, i32 ldd2,
           i32* n, f64* a, i32 lda, f64* b, i32 ldb,
           f64* c, i32 ldc, f64* d, i32 ldd,
           f64* dwork, i32 ldwork);

/**
 * @brief Feedback inter-connection of two state-space systems.
 *
 * Computes the state-space model (A,B,C,D) for the feedback connection
 * of two systems G1 and G2:
 *   U = U1 + alpha*Y2,  Y = Y1 = U2
 *   alpha = +1: positive feedback
 *   alpha = -1: negative feedback
 *
 * The interconnection matrices are:
 *   E21 = (I + alpha*D1*D2)^-1
 *   E12 = I - alpha*D2*E21*D1
 *
 * Matrix A:
 *   [A1 - alpha*B1*E12*D2*C1,    -alpha*B1*E12*C2    ]
 *   [B2*E21*C1,                   A2 - alpha*B2*E21*D1*C2]
 *
 * Matrix B:  [B1*E12; B2*E21*D1]
 * Matrix C:  [E21*C1, -alpha*E21*D1*C2]
 * Matrix D:  E21*D1
 *
 * @param[in] over 'N' no overlap, 'O' overlap arrays (workspace required)
 * @param[in] n1 Number of states in first system (n1 >= 0)
 * @param[in] m1 Number of inputs to first system and outputs from G2 (m1 >= 0)
 * @param[in] p1 Number of outputs from G1 and inputs to G2 (p1 >= 0)
 * @param[in] n2 Number of states in second system (n2 >= 0)
 * @param[in] alpha Feedback coefficient (+1 positive, -1 negative feedback)
 * @param[in] a1 State matrix of G1, dimension (lda1, n1)
 * @param[in] lda1 Leading dimension of a1 (lda1 >= max(1, n1))
 * @param[in] b1 Input matrix of G1, dimension (ldb1, m1)
 * @param[in] ldb1 Leading dimension of b1 (ldb1 >= max(1, n1))
 * @param[in] c1 Output matrix of G1, dimension (ldc1, n1)
 * @param[in] ldc1 Leading dimension of c1 (ldc1 >= max(1, p1) if n1 > 0)
 * @param[in] d1 Feedthrough matrix of G1, dimension (ldd1, m1)
 * @param[in] ldd1 Leading dimension of d1 (ldd1 >= max(1, p1))
 * @param[in] a2 State matrix of G2, dimension (lda2, n2)
 * @param[in] lda2 Leading dimension of a2 (lda2 >= max(1, n2))
 * @param[in] b2 Input matrix of G2, dimension (ldb2, p1)
 * @param[in] ldb2 Leading dimension of b2 (ldb2 >= max(1, n2))
 * @param[in] c2 Output matrix of G2, dimension (ldc2, n2)
 * @param[in] ldc2 Leading dimension of c2 (ldc2 >= max(1, m1) if n2 > 0)
 * @param[in] d2 Feedthrough matrix of G2, dimension (ldd2, p1)
 * @param[in] ldd2 Leading dimension of d2 (ldd2 >= max(1, m1))
 * @param[out] n Total state order (n = n1 + n2)
 * @param[out] a State matrix, dimension (lda, n1+n2)
 * @param[in] lda Leading dimension of a (lda >= max(1, n1+n2))
 * @param[out] b Input matrix, dimension (ldb, m1)
 * @param[in] ldb Leading dimension of b (ldb >= max(1, n1+n2))
 * @param[out] c Output matrix, dimension (ldc, n1+n2)
 * @param[in] ldc Leading dimension of c (ldc >= max(1, p1) if n1+n2 > 0)
 * @param[out] d Feedthrough matrix, dimension (ldd, m1)
 * @param[in] ldd Leading dimension of d (ldd >= max(1, p1))
 * @param[out] iwork Integer workspace, dimension (p1)
 * @param[out] dwork Double workspace, dimension (ldwork)
 * @param[in] ldwork Workspace size
 * @return 0 on success, -i if parameter i is invalid, >0 if singular
 */
i32 ab05nd(char over, i32 n1, i32 m1, i32 p1, i32 n2, f64 alpha,
           const f64* a1, i32 lda1, const f64* b1, i32 ldb1,
           const f64* c1, i32 ldc1, const f64* d1, i32 ldd1,
           const f64* a2, i32 lda2, const f64* b2, i32 ldb2,
           const f64* c2, i32 ldc2, const f64* d2, i32 ldd2,
           i32* n, f64* a, i32 lda, f64* b, i32 ldb,
           f64* c, i32 ldc, f64* d, i32 ldd,
           i32* iwork, f64* dwork, i32 ldwork);

/**
 * @brief Rowwise concatenation of two state-space systems.
 *
 * Computes the state-space model (A,B,C,D) for rowwise concatenation
 * (parallel inter-connection on outputs, with separate inputs) of two
 * systems G1 and G2:
 *   Y = G1*U1 + alpha*G2*U2
 *
 * The combined system has:
 *   A = [[A1, 0], [0, A2]]     (block diagonal)
 *   B = [[B1, 0], [0, B2]]     (block diagonal)
 *   C = [C1, alpha*C2]         (rowwise concatenation)
 *   D = [D1, alpha*D2]         (rowwise concatenation)
 *
 * @param[in] over 'N' no overlap, 'O' overlap arrays A1/A, B1/B, C1/C, D1/D
 * @param[in] n1 Number of states in first system (n1 >= 0)
 * @param[in] m1 Number of inputs to first system (m1 >= 0)
 * @param[in] p1 Number of outputs from each system (p1 >= 0)
 * @param[in] n2 Number of states in second system (n2 >= 0)
 * @param[in] m2 Number of inputs to second system (m2 >= 0)
 * @param[in] alpha Coefficient multiplying second system output
 * @param[in] a1 State matrix of G1, dimension (lda1, n1)
 * @param[in] lda1 Leading dimension of a1 (lda1 >= max(1, n1))
 * @param[in] b1 Input matrix of G1, dimension (ldb1, m1)
 * @param[in] ldb1 Leading dimension of b1 (ldb1 >= max(1, n1))
 * @param[in] c1 Output matrix of G1, dimension (ldc1, n1)
 * @param[in] ldc1 Leading dimension of c1 (ldc1 >= max(1, p1) if n1 > 0)
 * @param[in] d1 Feedthrough matrix of G1, dimension (ldd1, m1)
 * @param[in] ldd1 Leading dimension of d1 (ldd1 >= max(1, p1))
 * @param[in] a2 State matrix of G2, dimension (lda2, n2)
 * @param[in] lda2 Leading dimension of a2 (lda2 >= max(1, n2))
 * @param[in] b2 Input matrix of G2, dimension (ldb2, m2)
 * @param[in] ldb2 Leading dimension of b2 (ldb2 >= max(1, n2))
 * @param[in] c2 Output matrix of G2, dimension (ldc2, n2)
 * @param[in] ldc2 Leading dimension of c2 (ldc2 >= max(1, p1) if n2 > 0)
 * @param[in] d2 Feedthrough matrix of G2, dimension (ldd2, m2)
 * @param[in] ldd2 Leading dimension of d2 (ldd2 >= max(1, p1))
 * @param[out] n Total state order (n = n1 + n2)
 * @param[out] m Total inputs (m = m1 + m2)
 * @param[out] a State matrix, dimension (lda, n1+n2)
 * @param[in] lda Leading dimension of a (lda >= max(1, n1+n2))
 * @param[out] b Input matrix, dimension (ldb, m1+m2)
 * @param[in] ldb Leading dimension of b (ldb >= max(1, n1+n2))
 * @param[out] c Output matrix, dimension (ldc, n1+n2)
 * @param[in] ldc Leading dimension of c (ldc >= max(1, p1) if n1+n2 > 0)
 * @param[out] d Feedthrough matrix, dimension (ldd, m1+m2)
 * @param[in] ldd Leading dimension of d (ldd >= max(1, p1))
 * @return 0 on success, -i if parameter i is invalid
 */
i32 ab05od(char over, i32 n1, i32 m1, i32 p1, i32 n2, i32 m2, f64 alpha,
           const f64* a1, i32 lda1, const f64* b1, i32 ldb1,
           const f64* c1, i32 ldc1, const f64* d1, i32 ldd1,
           const f64* a2, i32 lda2, const f64* b2, i32 ldb2,
           const f64* c2, i32 ldc2, const f64* d2, i32 ldd2,
           i32* n, i32* m, f64* a, i32 lda, f64* b, i32 ldb,
           f64* c, i32 ldc, f64* d, i32 ldd);

/**
 * @brief Parallel inter-connection of two state-space systems (same inputs).
 *
 * Computes the state-space model G = (A,B,C,D) corresponding to
 * G = G1 + alpha*G2, where G1 = (A1,B1,C1,D1) and G2 = (A2,B2,C2,D2)
 * are the transfer-function matrices of the corresponding state-space models.
 * Both systems share the same inputs.
 *
 * The combined system has:
 *   A = [[A1, 0], [0, A2]]      (block diagonal)
 *   B = [[B1], [B2]]            (stacked vertically)
 *   C = [C1, alpha*C2]          (concatenated horizontally)
 *   D = D1 + alpha*D2           (matrix sum)
 *
 * @param[in] over 'N' no overlap, 'O' overlap arrays A1/A, B1/B, C1/C, D1/D
 * @param[in] n1 Number of states in first system (n1 >= 0)
 * @param[in] m Number of inputs to both systems (m >= 0)
 * @param[in] p Number of outputs from each system (p >= 0)
 * @param[in] n2 Number of states in second system (n2 >= 0)
 * @param[in] alpha Coefficient multiplying second system
 * @param[in] a1 State matrix of G1, dimension (lda1, n1)
 * @param[in] lda1 Leading dimension of a1 (lda1 >= max(1, n1))
 * @param[in] b1 Input matrix of G1, dimension (ldb1, m)
 * @param[in] ldb1 Leading dimension of b1 (ldb1 >= max(1, n1))
 * @param[in] c1 Output matrix of G1, dimension (ldc1, n1)
 * @param[in] ldc1 Leading dimension of c1 (ldc1 >= max(1, p) if n1 > 0)
 * @param[in] d1 Feedthrough matrix of G1, dimension (ldd1, m)
 * @param[in] ldd1 Leading dimension of d1 (ldd1 >= max(1, p))
 * @param[in] a2 State matrix of G2, dimension (lda2, n2)
 * @param[in] lda2 Leading dimension of a2 (lda2 >= max(1, n2))
 * @param[in] b2 Input matrix of G2, dimension (ldb2, m)
 * @param[in] ldb2 Leading dimension of b2 (ldb2 >= max(1, n2))
 * @param[in] c2 Output matrix of G2, dimension (ldc2, n2)
 * @param[in] ldc2 Leading dimension of c2 (ldc2 >= max(1, p) if n2 > 0)
 * @param[in] d2 Feedthrough matrix of G2, dimension (ldd2, m)
 * @param[in] ldd2 Leading dimension of d2 (ldd2 >= max(1, p))
 * @param[out] n Total state order (n = n1 + n2)
 * @param[out] a State matrix, dimension (lda, n1+n2)
 * @param[in] lda Leading dimension of a (lda >= max(1, n1+n2))
 * @param[out] b Input matrix, dimension (ldb, m)
 * @param[in] ldb Leading dimension of b (ldb >= max(1, n1+n2))
 * @param[out] c Output matrix, dimension (ldc, n1+n2)
 * @param[in] ldc Leading dimension of c (ldc >= max(1, p) if n1+n2 > 0)
 * @param[out] d Feedthrough matrix, dimension (ldd, m)
 * @param[in] ldd Leading dimension of d (ldd >= max(1, p))
 * @return 0 on success, -i if parameter i is invalid
 */
i32 ab05pd(char over, i32 n1, i32 m, i32 p, i32 n2, f64 alpha,
           const f64* a1, i32 lda1, const f64* b1, i32 ldb1,
           const f64* c1, i32 ldc1, const f64* d1, i32 ldd1,
           const f64* a2, i32 lda2, const f64* b2, i32 ldb2,
           const f64* c2, i32 ldc2, const f64* d2, i32 ldd2,
           i32* n, f64* a, i32 lda, f64* b, i32 ldb,
           f64* c, i32 ldc, f64* d, i32 ldd);

/**
 * @brief Append two systems in state-space form (block diagonal).
 *
 * Constructs the state-space model G = (A,B,C,D) corresponding to
 * G = diag(G1, G2), where G1 = (A1,B1,C1,D1) and G2 = (A2,B2,C2,D2)
 * are given systems with separate inputs and outputs.
 *
 * The combined system has:
 *   A = [[A1, 0], [0, A2]]   (block diagonal)
 *   B = [[B1, 0], [0, B2]]   (block diagonal)
 *   C = [[C1, 0], [0, C2]]   (block diagonal)
 *   D = [[D1, 0], [0, D2]]   (block diagonal)
 *
 * @param[in] over 'N' no overlap, 'O' overlap arrays A1/A, B1/B, C1/C, D1/D
 * @param[in] n1 Number of states in first system (n1 >= 0)
 * @param[in] m1 Number of inputs to first system (m1 >= 0)
 * @param[in] p1 Number of outputs from first system (p1 >= 0)
 * @param[in] n2 Number of states in second system (n2 >= 0)
 * @param[in] m2 Number of inputs to second system (m2 >= 0)
 * @param[in] p2 Number of outputs from second system (p2 >= 0)
 * @param[in] a1 State matrix of G1, dimension (lda1, n1)
 * @param[in] lda1 Leading dimension of a1 (lda1 >= max(1, n1))
 * @param[in] b1 Input matrix of G1, dimension (ldb1, m1)
 * @param[in] ldb1 Leading dimension of b1 (ldb1 >= max(1, n1))
 * @param[in] c1 Output matrix of G1, dimension (ldc1, n1)
 * @param[in] ldc1 Leading dimension of c1 (ldc1 >= max(1, p1) if n1 > 0)
 * @param[in] d1 Feedthrough matrix of G1, dimension (ldd1, m1)
 * @param[in] ldd1 Leading dimension of d1 (ldd1 >= max(1, p1))
 * @param[in] a2 State matrix of G2, dimension (lda2, n2)
 * @param[in] lda2 Leading dimension of a2 (lda2 >= max(1, n2))
 * @param[in] b2 Input matrix of G2, dimension (ldb2, m2)
 * @param[in] ldb2 Leading dimension of b2 (ldb2 >= max(1, n2))
 * @param[in] c2 Output matrix of G2, dimension (ldc2, n2)
 * @param[in] ldc2 Leading dimension of c2 (ldc2 >= max(1, p2) if n2 > 0)
 * @param[in] d2 Feedthrough matrix of G2, dimension (ldd2, m2)
 * @param[in] ldd2 Leading dimension of d2 (ldd2 >= max(1, p2))
 * @param[out] n Total state order (n = n1 + n2)
 * @param[out] m Total inputs (m = m1 + m2)
 * @param[out] p Total outputs (p = p1 + p2)
 * @param[out] a State matrix, dimension (lda, n1+n2)
 * @param[in] lda Leading dimension of a (lda >= max(1, n1+n2))
 * @param[out] b Input matrix, dimension (ldb, m1+m2)
 * @param[in] ldb Leading dimension of b (ldb >= max(1, n1+n2))
 * @param[out] c Output matrix, dimension (ldc, n1+n2)
 * @param[in] ldc Leading dimension of c (ldc >= max(1, p1+p2) if n1+n2 > 0)
 * @param[out] d Feedthrough matrix, dimension (ldd, m1+m2)
 * @param[in] ldd Leading dimension of d (ldd >= max(1, p1+p2))
 * @return 0 on success, -i if parameter i is invalid
 */
i32 ab05qd(char over, i32 n1, i32 m1, i32 p1, i32 n2, i32 m2, i32 p2,
           const f64* a1, i32 lda1, const f64* b1, i32 ldb1,
           const f64* c1, i32 ldc1, const f64* d1, i32 ldd1,
           const f64* a2, i32 lda2, const f64* b2, i32 ldb2,
           const f64* c2, i32 ldc2, const f64* d2, i32 ldd2,
           i32* n, i32* m, i32* p,
           f64* a, i32 lda, f64* b, i32 ldb,
           f64* c, i32 ldc, f64* d, i32 ldd);

/**
 * @brief Closed-loop system for output feedback control law.
 *
 * Constructs for a given state-space system (A,B,C,D) the closed-loop
 * system (Ac,Bc,Cc,Dc) corresponding to the output feedback control law
 *
 *     u = alpha*F*y + v
 *
 * The closed-loop matrices are:
 *   Ac = A + alpha*B*F*E*C,  Bc = B + alpha*B*F*E*D
 *   Cc = E*C,                Dc = E*D
 * where E = (I - alpha*D*F)^(-1).
 *
 * @param[in] fbtype 'I' for identity feedback (F=I), 'O' for general output feedback
 * @param[in] jobd 'D' if D is present, 'Z' if D is zero matrix
 * @param[in] n Order of state matrix A (n >= 0)
 * @param[in] m Number of system inputs (m >= 0)
 * @param[in] p Number of system outputs (p >= 0, p=m if fbtype='I')
 * @param[in] alpha Feedback gain coefficient
 * @param[in,out] a State matrix, dimension (lda, n). On exit, Ac.
 * @param[in] lda Leading dimension of a (lda >= max(1, n))
 * @param[in,out] b Input matrix, dimension (ldb, m). On exit, Bc.
 * @param[in] ldb Leading dimension of b (ldb >= max(1, n))
 * @param[in,out] c Output matrix, dimension (ldc, n). On exit, Cc.
 * @param[in] ldc Leading dimension of c (ldc >= max(1, p) if n > 0, else >= 1)
 * @param[in,out] d Feedthrough matrix, dimension (ldd, m). On exit, Dc.
 *                  Not referenced if jobd='Z'.
 * @param[in] ldd Leading dimension of d (ldd >= max(1, p) if jobd='D', else >= 1)
 * @param[in] f Feedback matrix, dimension (ldf, p). Not referenced if fbtype='I' or alpha=0.
 * @param[in] ldf Leading dimension of f (ldf >= max(1, m) if fbtype='O' and alpha!=0)
 * @param[out] rcond Reciprocal condition number of (I - alpha*D*F)
 * @param[out] iwork Integer workspace, dimension (2*p) if jobd='D', else 1
 * @param[out] dwork Double workspace
 * @param[in] ldwork Workspace size (>= max(1, m, p*p+4*p) if jobd='D', else >= max(1, m))
 * @param[out] info 0=success, <0=parameter -info invalid, 1=matrix singular
 */
void ab05sd(const char* fbtype, const char* jobd, i32 n, i32 m, i32 p,
            f64 alpha, f64* a, i32 lda, f64* b, i32 ldb,
            f64* c, i32 ldc, f64* d, i32 ldd,
            const f64* f, i32 ldf, f64* rcond,
            i32* iwork, f64* dwork, i32 ldwork, i32* info);

/**
 * @brief Closed-loop system for mixed output and state feedback control law.
 *
 * Constructs the closed-loop system (Ac,Bc,Cc,Dc) corresponding to:
 *   u = alpha*F*y + beta*K*x + G*v
 *   z = H*y
 *
 * @param[in] fbtype 'I' for identity feedback (F=I), 'O' for general output feedback
 * @param[in] jobd 'D' if D is present, 'Z' if D is zero matrix
 * @param[in] n Order of state matrix (n >= 0)
 * @param[in] m Number of system inputs (m >= 0)
 * @param[in] p Number of system outputs (p >= 0, p=m if fbtype='I')
 * @param[in] mv Dimension of new input vector v (mv >= 0)
 * @param[in] pz Dimension of new output vector z (pz >= 0)
 * @param[in] alpha Coefficient in output feedback law
 * @param[in] beta Coefficient in state feedback law
 * @param[in,out] a State matrix, dimension (lda, n). On exit, Ac.
 * @param[in] lda Leading dimension of a (lda >= max(1, n))
 * @param[in,out] b Input matrix, dimension (ldb, m). On exit, B1.
 * @param[in] ldb Leading dimension of b (ldb >= max(1, n))
 * @param[in,out] c Output matrix, dimension (ldc, n). On exit, C1+beta*D1*K.
 * @param[in] ldc Leading dimension of c
 * @param[in,out] d Feedthrough matrix, dimension (ldd, m). On exit, D1. Not ref if jobd='Z'.
 * @param[in] ldd Leading dimension of d
 * @param[in] f Feedback matrix, dimension (ldf, p). Not ref if fbtype='I' or alpha=0.
 * @param[in] ldf Leading dimension of f
 * @param[in] k State feedback matrix, dimension (ldk, n). Not ref if beta=0.
 * @param[in] ldk Leading dimension of k
 * @param[in] g Input scaling matrix, dimension (ldg, mv).
 * @param[in] ldg Leading dimension of g (ldg >= max(1, m))
 * @param[in] h Output scaling matrix, dimension (ldh, p).
 * @param[in] ldh Leading dimension of h (ldh >= max(1, pz))
 * @param[out] rcond Reciprocal condition number of (I - alpha*D*F)
 * @param[out] bc Input matrix Bc of closed-loop, dimension (ldbc, mv)
 * @param[in] ldbc Leading dimension of bc (ldbc >= max(1, n))
 * @param[out] cc Output matrix Cc of closed-loop, dimension (ldcc, n)
 * @param[in] ldcc Leading dimension of cc
 * @param[out] dc Feedthrough Dc of closed-loop, dimension (lddc, mv). Not ref if jobd='Z'.
 * @param[in] lddc Leading dimension of dc
 * @param[out] iwork Integer workspace, dimension (2*p) if jobd='D', else 1
 * @param[out] dwork Double workspace
 * @param[in] ldwork Workspace size
 * @param[out] info 0=success, <0=parameter -info invalid, 1=matrix singular
 */
void ab05rd(const char* fbtype, const char* jobd, i32 n, i32 m, i32 p,
            i32 mv, i32 pz, f64 alpha, f64 beta,
            f64* a, i32 lda, f64* b, i32 ldb,
            f64* c, i32 ldc, f64* d, i32 ldd,
            const f64* f, i32 ldf, const f64* k, i32 ldk,
            const f64* g, i32 ldg, const f64* h, i32 ldh,
            f64* rcond, f64* bc, i32 ldbc, f64* cc, i32 ldcc,
            f64* dc, i32 lddc, i32* iwork, f64* dwork, i32 ldwork, i32* info);

/**
 * @brief Dual of a state-space representation.
 *
 * Finds the dual of a given state-space representation.
 * If (A,B,C,D) is M-input/P-output, its dual is P-input/M-output (A',C',B',D').
 *
 * @param[in] jobd 'D' if D is present, 'Z' if D is zero matrix
 * @param[in] n Order of state-space representation (n >= 0)
 * @param[in] m Number of system inputs (m >= 0)
 * @param[in] p Number of system outputs (p >= 0)
 * @param[in,out] a State matrix, dimension (lda, n). On exit, A' (transpose).
 * @param[in] lda Leading dimension of a (lda >= max(1, n))
 * @param[in,out] b Input matrix, dimension (ldb, max(m, p)).
 *                  On entry: N-by-M part contains B.
 *                  On exit: N-by-P part contains C' (transpose).
 * @param[in] ldb Leading dimension of b (ldb >= max(1, n))
 * @param[in,out] c Output matrix, dimension (ldc, n).
 *                  On entry: P-by-N part contains C.
 *                  On exit: M-by-N part contains B' (transpose).
 * @param[in] ldc Leading dimension of c (ldc >= max(1, m, p) if n > 0, else >= 1)
 * @param[in,out] d Feedthrough matrix, dimension (ldd, max(m, p)).
 *                  On entry with jobd='D': P-by-M part contains D.
 *                  On exit with jobd='D': M-by-P part contains D' (transpose).
 *                  Not referenced if jobd='Z'.
 * @param[in] ldd Leading dimension of d (ldd >= max(1, m, p) if jobd='D', else >= 1)
 * @return 0 on success, -i if parameter i is invalid
 */
i32 ab07md(char jobd, i32 n, i32 m, i32 p, f64* a, i32 lda,
           f64* b, i32 ldb, f64* c, i32 ldc, f64* d, i32 ldd);

/**
 * @brief Compute the inverse of a linear system.
 *
 * Computes the inverse (Ai,Bi,Ci,Di) of a given system (A,B,C,D):
 *   Ai = A - B*D^-1*C,  Bi = -B*D^-1,  Ci = D^-1*C,  Di = D^-1.
 *
 * @param[in] n Order of state matrix A (n >= 0)
 * @param[in] m Number of system inputs/outputs (m >= 0, square system)
 * @param[in,out] a State matrix, dimension (lda, n). On exit, Ai.
 * @param[in] lda Leading dimension of a (lda >= max(1, n))
 * @param[in,out] b Input matrix, dimension (ldb, m). On exit, Bi.
 * @param[in] ldb Leading dimension of b (ldb >= max(1, n))
 * @param[in,out] c Output matrix, dimension (ldc, n). On exit, Ci.
 * @param[in] ldc Leading dimension of c (ldc >= max(1, m))
 * @param[in,out] d Feedthrough matrix, dimension (ldd, m). On exit, Di.
 * @param[in] ldd Leading dimension of d (ldd >= max(1, m))
 * @param[out] rcond Reciprocal condition number of D
 * @param[out] iwork Integer workspace, dimension (2*m)
 * @param[out] dwork Double workspace, dimension (ldwork)
 * @param[in] ldwork Workspace size (ldwork >= max(1, 4*m))
 * @return 0 on success, -i if parameter i is invalid, 1..m if D is singular
 *         (i-th diagonal element zero), m+1 if D is numerically singular
 */
i32 ab07nd(i32 n, i32 m, f64* a, i32 lda, f64* b, i32 ldb,
           f64* c, i32 ldc, f64* d, i32 ldd, f64* rcond,
           i32* iwork, f64* dwork, i32 ldwork);

/**
 * @brief Compute normal rank of transfer-function matrix.
 *
 * Computes the normal rank of the transfer-function matrix of a
 * state-space model (A,B,C,D). The routine reduces the compound matrix
 *   [ B  A ]
 *   [ D  C ]
 * to one with the same invariant zeros and with D of full row rank.
 * The normal rank is the rank of the reduced D matrix.
 *
 * @param[in] equil 'S' to balance compound matrix, 'N' for no balancing
 * @param[in] n Number of state variables (n >= 0)
 * @param[in] m Number of system inputs (m >= 0)
 * @param[in] p Number of system outputs (p >= 0)
 * @param[in] a State matrix, dimension (lda, n)
 * @param[in] lda Leading dimension of a (lda >= max(1, n))
 * @param[in] b Input matrix, dimension (ldb, m)
 * @param[in] ldb Leading dimension of b (ldb >= max(1, n))
 * @param[in] c Output matrix, dimension (ldc, n)
 * @param[in] ldc Leading dimension of c (ldc >= max(1, p))
 * @param[in] d Feedthrough matrix, dimension (ldd, m)
 * @param[in] ldd Leading dimension of d (ldd >= max(1, p))
 * @param[out] rank Normal rank of the transfer-function matrix
 * @param[in] tol Tolerance for rank decisions
 * @param[out] iwork Integer workspace, dimension (2*n + max(m,p) + 1)
 * @param[out] dwork Workspace. On exit dwork[0] = optimal ldwork.
 * @param[in] ldwork Workspace size (-1 for query)
 * @param[out] info 0 = success, -i = parameter i invalid
 */
void ab08md(const char* equil, i32 n, i32 m, i32 p, f64* a, i32 lda,
            f64* b, i32 ldb, f64* c, i32 ldc, f64* d, i32 ldd,
            i32* rank, f64 tol, i32* iwork, f64* dwork, i32 ldwork,
            i32* info);

/**
 * @brief Construct regular pencil for invariant zeros of state-space system.
 *
 * Constructs a regular pencil (Af - lambda*Bf) for a state-space system
 * (A,B,C,D) whose generalized eigenvalues are the invariant zeros.
 * Also computes orders of infinite zeros and Kronecker indices.
 *
 * The algorithm extracts structural invariants under strict equivalence:
 *   - Finite invariant zeros (generalized eigenvalues of Af - lambda*Bf)
 *   - Orders of infinite zeros (in infz array)
 *   - Right Kronecker (column) indices (in kronr array)
 *   - Left Kronecker (row) indices (in kronl array)
 *
 * @param[in] equil 'S' to balance compound matrix, 'N' for no balancing
 * @param[in] n Number of state variables (n >= 0)
 * @param[in] m Number of system inputs (m >= 0)
 * @param[in] p Number of system outputs (p >= 0)
 * @param[in] a State matrix, dimension (lda, n)
 * @param[in] lda Leading dimension of a (lda >= max(1, n))
 * @param[in] b Input matrix, dimension (ldb, m)
 * @param[in] ldb Leading dimension of b (ldb >= max(1, n))
 * @param[in] c Output matrix, dimension (ldc, n)
 * @param[in] ldc Leading dimension of c (ldc >= max(1, p))
 * @param[in] d Feedthrough matrix, dimension (ldd, m)
 * @param[in] ldd Leading dimension of d (ldd >= max(1, p))
 * @param[out] nu Number of finite invariant zeros
 * @param[out] rank Normal rank of transfer function matrix
 * @param[out] dinfz Maximum degree of infinite elementary divisors
 * @param[out] nkror Number of right Kronecker indices
 * @param[out] nkrol Number of left Kronecker indices
 * @param[out] infz Integer array (n), infz[i] = number of infinite zeros of degree i+1
 * @param[out] kronr Integer array (max(n,m)+1), right Kronecker indices
 * @param[out] kronl Integer array (max(n,p)+1), left Kronecker indices
 * @param[out] af Coefficient matrix Af of reduced pencil, dimension (ldaf, n+min(p,m))
 * @param[in] ldaf Leading dimension of af (ldaf >= max(1, n+m))
 * @param[out] bf Coefficient matrix Bf of reduced pencil, dimension (ldbf, n+m)
 * @param[in] ldbf Leading dimension of bf (ldbf >= max(1, n+p))
 * @param[in] tol Tolerance for rank decisions
 * @param[out] iwork Integer workspace, dimension (max(m,p))
 * @param[out] dwork Workspace. On exit dwork[0] = optimal ldwork.
 * @param[in] ldwork Workspace size (-1 for query)
 * @param[out] info 0 = success, -i = parameter i invalid
 */
void ab08nd(const char* equil, i32 n, i32 m, i32 p, f64* a, i32 lda,
            f64* b, i32 ldb, f64* c, i32 ldc, f64* d, i32 ldd,
            i32* nu, i32* rank, i32* dinfz, i32* nkror, i32* nkrol,
            i32* infz, i32* kronr, i32* kronl, f64* af, i32 ldaf,
            f64* bf, i32 ldbf, f64 tol, i32* iwork, f64* dwork, i32 ldwork,
            i32* info);

/**
 * @brief Reduce compound matrix to one with D of full row rank.
 *
 * Extracts from the (N+P)-by-(M+N) compound system
 *   [ B  A ]
 *   [ D  C ]
 * a reduced (NU+MU)-by-(M+NU) system
 *   [ B'  A' ]
 *   [ D'  C' ]
 * having the same transmission zeros but with D' of full row rank.
 *
 * @param[in] n Number of state variables (n >= 0)
 * @param[in] m Number of system inputs (m >= 0)
 * @param[in] p Number of system outputs (p >= 0)
 * @param[in,out] ro On entry, P for original system, MAX(P-M,0) for pertransposed.
 *                   On exit, last computed rank.
 * @param[in,out] sigma On entry, 0 for original system, M for pertransposed.
 *                      On exit, last computed sigma.
 * @param[in] svlmax Estimate of max singular value for rank decisions (>= 0)
 * @param[in,out] abcd Compound matrix, dimension (ldabcd, m+n).
 *                     On exit, reduced (NU+MU)-by-(M+NU) compound matrix.
 * @param[in] ldabcd Leading dimension (ldabcd >= max(1, n+p))
 * @param[in,out] ninfz On entry/exit, number of infinite zeros
 * @param[in,out] infz Integer array (n), degrees of infinite zeros
 * @param[in,out] kronl Integer array (n+1), left Kronecker indices
 * @param[out] mu Normal rank of transfer function matrix
 * @param[out] nu Dimension of reduced system
 * @param[out] nkrol Number of left Kronecker indices
 * @param[in] tol Tolerance for rank decisions
 * @param[out] iwork Integer workspace, dimension (max(m,p))
 * @param[out] dwork Workspace. On exit dwork[0] = optimal ldwork.
 * @param[in] ldwork Workspace size (-1 for query)
 * @param[out] info 0 = success, -i = parameter i invalid
 */
void ab08nx(i32 n, i32 m, i32 p, i32* ro, i32* sigma, f64 svlmax,
            f64* abcd, i32 ldabcd, i32* ninfz, i32* infz, i32* kronl,
            i32* mu, i32* nu, i32* nkrol, f64 tol, i32* iwork,
            f64* dwork, i32 ldwork, i32* info);

/**
 * @brief Extract regular pencil for finite Smith zeros from system pencil.
 *
 * Extracts from the system pencil
 *   S(lambda) = ( A-lambda*I  B )
 *               (     C       D )
 * a regular pencil Af-lambda*Ef whose generalized eigenvalues are the finite
 * Smith zeros of S(lambda). Also computes orders of infinite Smith zeros and
 * determines the singular and infinite Kronecker structure.
 *
 * @param[in] equil 'S' to balance system matrix, 'N' for no balancing
 * @param[in] n Number of state variables (n >= 0)
 * @param[in] m Number of system inputs (m >= 0)
 * @param[in] p Number of system outputs (p >= 0)
 * @param[in,out] a State matrix, dimension (lda, n). On exit, leading nfz-by-nfz
 *                  contains Af of reduced pencil.
 * @param[in] lda Leading dimension of a (lda >= max(1, n))
 * @param[in,out] b Input matrix, dimension (ldb, m). On exit, contains no useful info.
 * @param[in] ldb Leading dimension of b (ldb >= 1, ldb >= max(1,n) if m > 0)
 * @param[in,out] c Output matrix, dimension (ldc, n). On exit, contains no useful info.
 * @param[in] ldc Leading dimension of c (ldc >= max(1, p))
 * @param[in] d Feedthrough matrix, dimension (ldd, m)
 * @param[in] ldd Leading dimension of d (ldd >= max(1, p))
 * @param[out] nfz Number of finite zeros
 * @param[out] nrank Normal rank of system pencil
 * @param[out] niz Number of infinite zeros
 * @param[out] dinfz Maximal multiplicity of infinite Smith zeros
 * @param[out] nkror Number of right Kronecker indices
 * @param[out] ninfe Number of elementary infinite blocks
 * @param[out] nkrol Number of left Kronecker indices
 * @param[out] infz Integer array (n+1), infz[i] = number of infinite zeros of degree i+1
 * @param[out] kronr Integer array (n+1), right Kronecker (column) indices
 * @param[out] infe Integer array (n+1), multiplicities of infinite eigenvalues
 * @param[out] kronl Integer array (n+1), left Kronecker (row) indices
 * @param[out] e Matrix Ef of reduced pencil, dimension (lde, n).
 *               Leading nfz-by-nfz contains Ef.
 * @param[in] lde Leading dimension of e (lde >= max(1, n))
 * @param[in] tol Tolerance for rank decisions (tol < 1, uses default if tol <= 0)
 * @param[out] iwork Integer workspace, dimension (max(m, p))
 * @param[out] dwork Workspace. On exit dwork[0] = optimal ldwork.
 * @param[in] ldwork Workspace size (-1 for query)
 * @param[out] info 0 = success, -i = parameter i invalid
 */
void ab08nw(const char* equil, i32 n, i32 m, i32 p, f64* a, i32 lda,
            f64* b, i32 ldb, f64* c, i32 ldc, f64* d, i32 ldd,
            i32* nfz, i32* nrank, i32* niz, i32* dinfz, i32* nkror,
            i32* ninfe, i32* nkrol, i32* infz, i32* kronr, i32* infe,
            i32* kronl, f64* e, i32 lde, f64 tol, i32* iwork, f64* dwork,
            i32 ldwork, i32* info);

/**
 * @brief Extract reduced system with Dr of full row rank.
 *
 * Extracts from the (N+P)-by-(M+N) system pencil
 *   ( B  A-lambda*I )
 *   ( D      C      )
 * an (NR+PR)-by-(M+NR) "reduced" system pencil
 *   ( Br Ar-lambda*I )
 *   ( Dr     Cr      )
 * having the same transmission zeros, but with Dr of full row rank.
 *
 * @param[in] first True for first call, false for subsequent calls
 *                  (if false, D must have full column rank with last M rows upper triangular)
 * @param[in] n Number of rows of B, columns of C, order of A (n >= 0)
 * @param[in] m Number of columns of B and D (m >= 0, m <= p if first=false)
 * @param[in] p Number of rows of C and D (p >= 0)
 * @param[in] svlmax Estimate of largest singular value of ABCD (svlmax >= 0)
 * @param[in,out] abcd Compound matrix (N+P)-by-(M+N) containing [B A; D C]
 *                     On exit: reduced compound matrix (NR+PR)-by-(M+NR)
 * @param[in] ldabcd Leading dimension of abcd (ldabcd >= max(1, N+P))
 * @param[in,out] ninfz On entry: current number of infinite zeros (init to 0)
 *                      On exit: total number of infinite zeros
 * @param[out] nr Order of reduced matrix Ar
 * @param[out] pr Normal rank of transfer function matrix
 * @param[out] dinfz Maximal multiplicity of infinite zeros (0 if first=false)
 * @param[out] nkronl Maximal dimension of left elementary Kronecker blocks
 * @param[out] infz Integer array (n), degrees of infinite zeros
 * @param[out] kronl Integer array (n+1), left Kronecker indices
 * @param[in] tol Tolerance for rank decisions (tol < 1, uses default if tol <= 0)
 * @param[out] iwork Integer workspace, dimension (max(m,p))
 * @param[out] dwork Workspace. On exit dwork[0] = optimal ldwork.
 * @param[in] ldwork Workspace size (-1 for query)
 * @param[out] info 0 = success, -i = parameter i invalid
 */
void ab08ny(bool first, i32 n, i32 m, i32 p, f64 svlmax,
            f64* abcd, i32 ldabcd, i32* ninfz, i32* nr, i32* pr,
            i32* dinfz, i32* nkronl, i32* infz, i32* kronl,
            f64 tol, i32* iwork, f64* dwork, i32 ldwork, i32* info);

/**
 * @brief Balance & Truncate model reduction for stable systems.
 *
 * Computes a reduced order model (Ar,Br,Cr) for a stable original
 * state-space representation (A,B,C) by using either the square-root
 * or the balancing-free square-root Balance & Truncate model
 * reduction method.
 *
 * The routine first reduces A to real Schur form using TB01WD,
 * then calls AB09AX to perform the actual model reduction.
 *
 * The method ensures:
 *     HSV(NR) <= ||G-Gr||_inf <= 2*sum(HSV(NR+1:N))
 *
 * @param[in] dico System type: 'C' = continuous-time, 'D' = discrete-time
 * @param[in] job Method: 'B' = square-root B&T, 'N' = balancing-free B&T
 * @param[in] equil Equilibration: 'S' = scale (A,B,C), 'N' = no scaling
 * @param[in] ordsel Order selection: 'F' = fixed NR, 'A' = automatic based on TOL
 * @param[in] n Order of original system (n >= 0)
 * @param[in] m Number of inputs (m >= 0)
 * @param[in] p Number of outputs (p >= 0)
 * @param[in,out] nr On entry with ordsel='F': desired order (0 <= nr <= n).
 *                   On exit: actual order of reduced system.
 * @param[in,out] a On entry: N-by-N state dynamics matrix.
 *                  On exit: NR-by-NR reduced state matrix Ar.
 * @param[in] lda Leading dimension of a (lda >= max(1, n))
 * @param[in,out] b On entry: N-by-M input matrix.
 *                  On exit: NR-by-M reduced input matrix Br.
 * @param[in] ldb Leading dimension of b (ldb >= max(1, n))
 * @param[in,out] c On entry: P-by-N output matrix.
 *                  On exit: P-by-NR reduced output matrix Cr.
 * @param[in] ldc Leading dimension of c (ldc >= max(1, p))
 * @param[out] hsv Hankel singular values, dimension (n), in decreasing order.
 * @param[in] tol Tolerance for order selection when ordsel='A'.
 * @param[out] iwork Integer workspace, dimension 0 if job='B', else n.
 * @param[out] dwork Real workspace.
 * @param[in] ldwork Workspace size (>= max(1, n*(2*n+max(n,m,p)+5)+n*(n+1)/2))
 * @param[out] iwarn Warning: 0 = none, 1 = requested NR > minimal realization
 * @param[out] info Error code: 0 = success, -i = param i invalid,
 *                  1 = Schur reduction failed, 2 = A not stable, 3 = HSV failed
 */
void ab09ad(
    const char* dico,
    const char* job,
    const char* equil,
    const char* ordsel,
    i32 n,
    i32 m,
    i32 p,
    i32* nr,
    f64* a,
    i32 lda,
    f64* b,
    i32 ldb,
    f64* c,
    i32 ldc,
    f64* hsv,
    f64 tol,
    i32* iwork,
    f64* dwork,
    i32 ldwork,
    i32* iwarn,
    i32* info
);

/**
 * @brief Balance & Truncate model reduction for stable systems (Schur form input).
 *
 * Computes a reduced order model (Ar,Br,Cr) for a stable original
 * state-space representation (A,B,C) by using either the square-root
 * or the balancing-free square-root Balance & Truncate model
 * reduction method.
 *
 * The state dynamics matrix A of the original system must be in upper
 * quasi-triangular (real Schur canonical) form. The matrices of the
 * reduced order system are computed using truncation formulas:
 *     Ar = TI * A * T,  Br = TI * B,  Cr = C * T
 *
 * The method ensures:
 *     HSV(NR) <= ||G-Gr||_inf <= 2*sum(HSV(NR+1:N))
 *
 * where G and Gr are transfer functions of original and reduced systems.
 *
 * @param[in] dico System type: 'C' = continuous-time, 'D' = discrete-time
 * @param[in] job Method: 'B' = square-root B&T, 'N' = balancing-free B&T
 * @param[in] ordsel Order selection: 'F' = fixed NR, 'A' = automatic based on TOL
 * @param[in] n Order of original system (n >= 0)
 * @param[in] m Number of inputs (m >= 0)
 * @param[in] p Number of outputs (p >= 0)
 * @param[in,out] nr On entry with ordsel='F': desired order (0 <= nr <= n).
 *                   On exit: actual order of reduced system.
 * @param[in,out] a On entry: N-by-N state matrix in real Schur form.
 *                  On exit: NR-by-NR reduced state matrix Ar.
 *                  Dimension (lda, n).
 * @param[in] lda Leading dimension of a (lda >= max(1, n))
 * @param[in,out] b On entry: N-by-M input matrix.
 *                  On exit: NR-by-M reduced input matrix Br.
 *                  Dimension (ldb, m).
 * @param[in] ldb Leading dimension of b (ldb >= max(1, n))
 * @param[in,out] c On entry: P-by-N output matrix.
 *                  On exit: P-by-NR reduced output matrix Cr.
 *                  Dimension (ldc, n).
 * @param[in] ldc Leading dimension of c (ldc >= max(1, p))
 * @param[out] hsv Hankel singular values, dimension (n), in decreasing order.
 *                 hsv[0] is the Hankel norm of the system.
 * @param[out] t Right truncation matrix, dimension (ldt, n).
 *               If NR > 0, leading N-by-NR part contains T.
 * @param[in] ldt Leading dimension of t (ldt >= max(1, n))
 * @param[out] ti Left truncation matrix, dimension (ldti, n).
 *                If NR > 0, leading NR-by-N part contains TI.
 * @param[in] ldti Leading dimension of ti (ldti >= max(1, n))
 * @param[in] tol Tolerance for order selection when ordsel='A'.
 *                If tol <= 0, uses default N*EPS*HSV[0].
 *                Ignored when ordsel='F'.
 * @param[out] iwork Integer workspace, dimension 0 if job='B', else n.
 * @param[out] dwork Real workspace, dimension (ldwork).
 *                   On exit, dwork[0] = optimal ldwork.
 * @param[in] ldwork Workspace size (>= max(1, n*(max(n,m,p)+5) + n*(n+1)/2))
 * @param[out] iwarn Warning: 0 = none, 1 = requested NR exceeds minimal realization
 * @param[out] info Error code: 0 = success, -i = param i invalid,
 *                  1 = A not stable/convergent, 2 = HSV computation failed
 */
void ab09ax(
    const char* dico,
    const char* job,
    const char* ordsel,
    i32 n,
    i32 m,
    i32 p,
    i32* nr,
    f64* a,
    i32 lda,
    f64* b,
    i32 ldb,
    f64* c,
    i32 ldc,
    f64* hsv,
    f64* t,
    i32 ldt,
    f64* ti,
    i32 ldti,
    f64 tol,
    i32* iwork,
    f64* dwork,
    i32 ldwork,
    i32* iwarn,
    i32* info);

/**
 * @brief Singular Perturbation Approximation model reduction for stable systems.
 *
 * Computes a reduced order model (Ar,Br,Cr,Dr) for a stable original
 * state-space representation (A,B,C,D) by using either the square-root or
 * the balancing-free square-root Singular Perturbation Approximation (SPA)
 * model reduction method.
 *
 * The error bound is: HSV(NR) <= ||G-Gr||_inf <= 2*[HSV(NR+1)+...+HSV(N)]
 *
 * @param[in] dico System type: 'C' = continuous-time, 'D' = discrete-time
 * @param[in] job Reduction method: 'B' = square-root SPA, 'N' = balancing-free
 * @param[in] equil Equilibration: 'S' = scale system, 'N' = no scaling
 * @param[in] ordsel Order selection: 'F' = fixed order, 'A' = automatic
 * @param[in] n Order of original system (n >= 0)
 * @param[in] m Number of inputs (m >= 0)
 * @param[in] p Number of outputs (p >= 0)
 * @param[in,out] nr On entry: desired order if ordsel='F'. On exit: actual order.
 * @param[in,out] a On entry: N-by-N state matrix. On exit: NR-by-NR reduced Ar.
 * @param[in] lda Leading dimension of a (lda >= max(1,n))
 * @param[in,out] b On entry: N-by-M input matrix. On exit: NR-by-M reduced Br.
 * @param[in] ldb Leading dimension of b (ldb >= max(1,n))
 * @param[in,out] c On entry: P-by-N output matrix. On exit: P-by-NR reduced Cr.
 * @param[in] ldc Leading dimension of c (ldc >= max(1,p))
 * @param[in,out] d On entry: P-by-M feedthrough. On exit: P-by-M reduced Dr.
 * @param[in] ldd Leading dimension of d (ldd >= max(1,p))
 * @param[out] hsv Hankel singular values, dimension (n), decreasingly ordered
 * @param[in] tol1 Tolerance for order selection (if ordsel='A'). Default if <= 0.
 * @param[in] tol2 Tolerance for minimal realization. Default if <= 0. TOL2 <= TOL1.
 * @param[out] iwork Integer workspace, dimension (2*n). iwork[0] = nmin.
 * @param[out] dwork Real workspace, dimension (ldwork)
 * @param[in] ldwork Workspace size >= max(1, n*(2*n+max(n,m,p)+5)+n*(n+1)/2)
 * @param[out] iwarn Warning: 0=ok, 1=nr > nmin (nr set to nmin)
 * @param[out] info Error: 0=ok, -i=param i invalid, 1=Schur failed,
 *                  2=A unstable/not convergent, 3=HSV computation failed
 */
void ab09bd(const char* dico, const char* job, const char* equil,
            const char* ordsel, i32 n, i32 m, i32 p, i32* nr,
            f64* a, i32 lda, f64* b, i32 ldb, f64* c, i32 ldc,
            f64* d, i32 ldd, f64* hsv, f64 tol1, f64 tol2,
            i32* iwork, f64* dwork, i32 ldwork, i32* iwarn, i32* info);

/**
 * @brief Compute reduced order model using square-root SPA method.
 *
 * Computes a reduced order model (Ar,Br,Cr,Dr) for a stable original
 * state-space representation (A,B,C,D) by using either the square-root or
 * the balancing-free square-root Singular Perturbation Approximation (SPA)
 * model reduction method.
 *
 * The state dynamics matrix A must be in real Schur canonical form.
 * The truncation matrices T and TI are computed such that:
 *   Am = TI * A * T,  Bm = TI * B,  Cm = C * T
 *
 * @param[in] dico System type: 'C' = continuous-time, 'D' = discrete-time
 * @param[in] job Reduction method: 'B' = square-root SPA, 'N' = balancing-free
 * @param[in] ordsel Order selection: 'F' = fixed order, 'A' = automatic
 * @param[in] n Order of original system (n >= 0)
 * @param[in] m Number of inputs (m >= 0)
 * @param[in] p Number of outputs (p >= 0)
 * @param[in,out] nr On entry: desired order if ordsel='F'. On exit: actual order.
 * @param[in,out] a On entry: N-by-N state matrix in Schur form.
 *                  On exit: NR-by-NR reduced state matrix Ar.
 * @param[in] lda Leading dimension of a (lda >= max(1,n))
 * @param[in,out] b On entry: N-by-M input matrix. On exit: NR-by-M reduced Br.
 * @param[in] ldb Leading dimension of b (ldb >= max(1,n))
 * @param[in,out] c On entry: P-by-N output matrix. On exit: P-by-NR reduced Cr.
 * @param[in] ldc Leading dimension of c (ldc >= max(1,p))
 * @param[in,out] d On entry: P-by-M feedthrough matrix. On exit: reduced Dr.
 * @param[in] ldd Leading dimension of d (ldd >= max(1,p))
 * @param[out] hsv Hankel singular values, dimension (n), decreasingly ordered
 * @param[out] t Right truncation matrix, dimension (ldt, n), N-by-NR used
 * @param[in] ldt Leading dimension of t (ldt >= max(1,n))
 * @param[out] ti Left truncation matrix, dimension (ldti, n), NR-by-N used
 * @param[in] ldti Leading dimension of ti (ldti >= max(1,n))
 * @param[in] tol1 Tolerance for order selection (if ordsel='A')
 * @param[in] tol2 Tolerance for minimal realization order
 * @param[out] iwork Integer workspace, dimension (2*n). iwork[0] = nmin.
 * @param[out] dwork Real workspace, dimension (ldwork)
 * @param[in] ldwork Workspace size >= max(1, n*(max(n,m,p)+5) + n*(n+1)/2)
 * @param[out] iwarn Warning indicator: 0=ok, 1=nr > nmin, nr set to nmin
 * @param[out] info Error indicator: 0=ok, 1=A unstable, 2=HSV computation failed
 */
void ab09bx(const char* dico, const char* job, const char* ordsel,
            i32 n, i32 m, i32 p, i32* nr, f64* a, i32 lda,
            f64* b, i32 ldb, f64* c, i32 ldc, f64* d, i32 ldd,
            f64* hsv, f64* t, i32 ldt, f64* ti, i32 ldti,
            f64 tol1, f64 tol2, i32* iwork, f64* dwork, i32 ldwork,
            i32* iwarn, i32* info);

/**
 * @brief Optimal Hankel-norm approximation based model reduction for stable systems.
 *
 * Computes a reduced order model (Ar,Br,Cr,Dr) for a stable original
 * state-space representation (A,B,C,D) by using the optimal Hankel-norm
 * approximation method in conjunction with square-root balancing.
 *
 * Unlike AB09CX, this routine accepts a general state matrix A (not necessarily
 * in Schur form) and optionally performs equilibration (scaling) of (A,B,C).
 *
 * The reduced system satisfies the error bound:
 *   HSV(NR) <= ||G-Gr||_inf <= 2*[HSV(NR+1) + ... + HSV(N)]
 *
 * @param[in] dico System type: 'C' = continuous-time, 'D' = discrete-time
 * @param[in] equil Equilibration: 'S' = scale (A,B,C), 'N' = no scaling
 * @param[in] ordsel Order selection: 'F' = fixed order, 'A' = automatic
 * @param[in] n Order of original system (n >= 0)
 * @param[in] m Number of inputs (m >= 0)
 * @param[in] p Number of outputs (p >= 0)
 * @param[in,out] nr On entry: desired order if ordsel='F'. On exit: actual order.
 * @param[in,out] a On entry: N-by-N state matrix (general).
 *                  On exit: NR-by-NR reduced state matrix Ar in Schur form.
 * @param[in] lda Leading dimension of a (lda >= max(1,n))
 * @param[in,out] b On entry: N-by-M input matrix. On exit: NR-by-M reduced Br.
 * @param[in] ldb Leading dimension of b (ldb >= max(1,n))
 * @param[in,out] c On entry: P-by-N output matrix. On exit: P-by-NR reduced Cr.
 * @param[in] ldc Leading dimension of c (ldc >= max(1,p))
 * @param[in,out] d On entry: P-by-M feedthrough matrix. On exit: P-by-M reduced Dr.
 * @param[in] ldd Leading dimension of d (ldd >= max(1,p))
 * @param[out] hsv Hankel singular values, dimension (n), decreasingly ordered
 * @param[in] tol1 Tolerance for order selection (if ordsel='A'). Default if <= 0.
 * @param[in] tol2 Tolerance for minimal realization. Default if <= 0. TOL2 <= TOL1.
 * @param[out] iwork Integer workspace, dimension max(1,n,m) for discrete,
 *                   max(1,m) for continuous. iwork[0] = nmin on exit.
 * @param[out] dwork Real workspace, dimension (ldwork)
 * @param[in] ldwork Workspace size >= max(LDW1, LDW2) where:
 *                   LDW1 = N*(2*N+max(N,M,P)+5) + N*(N+1)/2
 *                   LDW2 = N*(M+P+2) + 2*M*P + min(N,M) + max(3*M+1, min(N,M)+P)
 * @param[out] iwarn Warning: 0=ok, 1=nr > nmin (nr set to nmin)
 * @param[out] info Error: 0=ok, 1=Schur reduction failed, 2=A unstable,
 *                  3=HSV computation failed, 4=stable projection failed,
 *                  5=order mismatch
 */
void ab09cd(const char* dico, const char* equil, const char* ordsel,
            i32 n, i32 m, i32 p, i32* nr, f64* a, i32 lda,
            f64* b, i32 ldb, f64* c, i32 ldc, f64* d, i32 ldd,
            f64* hsv, f64 tol1, f64 tol2, i32* iwork,
            f64* dwork, i32 ldwork, i32* iwarn, i32* info);

/**
 * @brief Optimal Hankel-norm approximation model reduction for unstable systems.
 *
 * Computes a reduced order model (Ar,Br,Cr,Dr) for an original state-space
 * representation (A,B,C,D) by using the optimal Hankel-norm approximation
 * method in conjunction with square-root balancing for the ALPHA-stable part
 * of the system.
 *
 * The system is decomposed as G = G1 + G2 where G1 has ALPHA-stable poles
 * and G2 has ALPHA-unstable poles. G1 is reduced using Hankel-norm
 * approximation while G2 is kept intact.
 *
 * @param[in] dico System type: 'C' = continuous-time, 'D' = discrete-time
 * @param[in] equil Equilibration: 'S' = scale, 'N' = no scaling
 * @param[in] ordsel Order selection: 'F' = fixed order, 'A' = automatic
 * @param[in] n Order of original system (n >= 0)
 * @param[in] m Number of inputs (m >= 0)
 * @param[in] p Number of outputs (p >= 0)
 * @param[in,out] nr On entry: desired order if ordsel='F'. On exit: actual order.
 * @param[in] alpha Stability boundary. For continuous (DICO='C'): alpha <= 0.
 *                  For discrete (DICO='D'): 0 <= alpha <= 1.
 * @param[in,out] a On entry: N-by-N state matrix.
 *                  On exit: NR-by-NR reduced state matrix Ar in block-diagonal Schur form.
 * @param[in] lda Leading dimension of a (lda >= max(1,n))
 * @param[in,out] b On entry: N-by-M input matrix. On exit: NR-by-M reduced Br.
 * @param[in] ldb Leading dimension of b (ldb >= max(1,n))
 * @param[in,out] c On entry: P-by-N output matrix. On exit: P-by-NR reduced Cr.
 * @param[in] ldc Leading dimension of c (ldc >= max(1,p))
 * @param[in,out] d On entry: P-by-M feedthrough matrix. On exit: P-by-M reduced Dr.
 * @param[in] ldd Leading dimension of d (ldd >= max(1,p))
 * @param[out] ns Dimension of the ALPHA-stable subsystem
 * @param[out] hsv Hankel singular values of ALPHA-stable part, dimension (n), decreasing
 * @param[in] tol1 Tolerance for order selection (if ordsel='A'). Default if <= 0.
 * @param[in] tol2 Tolerance for minimal realization. Default if <= 0. TOL2 <= TOL1.
 * @param[out] iwork Integer workspace, dimension max(1,n,m) for discrete,
 *                   max(1,m) for continuous. iwork[0] = nmin on exit.
 * @param[out] dwork Real workspace, dimension (ldwork)
 * @param[in] ldwork Workspace size >= max(LDW1, LDW2) where:
 *                   LDW1 = N*(2*N+max(N,M,P)+5) + N*(N+1)/2
 *                   LDW2 = N*(M+P+2) + 2*M*P + min(N,M) + max(3*M+1, min(N,M)+P)
 * @param[out] iwarn Warning: 0=ok, 1=nr > nsmin, 2=nr < nu
 * @param[out] info Error: 0=ok, 1=Schur failed, 2=separation failed, 3=just stable,
 *                  4=HSV failed, 5=stable projection failed, 6=order mismatch
 */
void ab09ed(const char* dico, const char* equil, const char* ordsel,
            i32 n, i32 m, i32 p, i32* nr, f64 alpha,
            f64* a, i32 lda, f64* b, i32 ldb, f64* c, i32 ldc,
            f64* d, i32 ldd, i32* ns, f64* hsv, f64 tol1, f64 tol2,
            i32* iwork, f64* dwork, i32 ldwork, i32* iwarn, i32* info);

/**
 * @brief Balance & Truncate model reduction for unstable systems using coprime
 *        factorization.
 *
 * Computes a reduced order model (Ar,Br,Cr) for an original state-space
 * representation (A,B,C) using either the square-root or the balancing-free
 * square-root Balance & Truncate (B&T) model reduction method in conjunction
 * with stable coprime factorization techniques.
 *
 * The extended system Ge = (Q,R) (LCF) or Ge = (Q;R) (RCF) is formed, and
 * model reduction is applied to Ge. The reduced system Gr is recovered from
 * the reduced coprime factors.
 *
 * @param[in] dico System type: 'C' = continuous-time, 'D' = discrete-time
 * @param[in] jobcf Coprime factorization: 'L' = left, 'R' = right
 * @param[in] fact Factorization type: 'S' = prescribed stability degree,
 *                 'I' = inner denominator
 * @param[in] jobmr Model reduction: 'B' = square-root B&T, 'N' = balancing-free
 * @param[in] equil Equilibration: 'S' = scale, 'N' = no scaling
 * @param[in] ordsel Order selection: 'F' = fixed order, 'A' = automatic
 * @param[in] n System order (n >= 0)
 * @param[in] m Number of inputs (m >= 0)
 * @param[in] p Number of outputs (p >= 0)
 * @param[in,out] nr On entry: desired order if ordsel='F'. On exit: actual order.
 * @param[in] alpha Stability degree: < 0 for continuous, 0 <= alpha < 1 for discrete.
 *                  Not used if fact='I'.
 * @param[in,out] a On entry: N-by-N state matrix A. On exit: NR-by-NR reduced Ar.
 * @param[in] lda Leading dimension of a (lda >= max(1,n))
 * @param[in,out] b On entry: N-by-M input matrix B. On exit: NR-by-M reduced Br.
 * @param[in] ldb Leading dimension of b (ldb >= max(1,n))
 * @param[in,out] c On entry: P-by-N output matrix C. On exit: P-by-NR reduced Cr.
 * @param[in] ldc Leading dimension of c (ldc >= max(1,p))
 * @param[out] nq Order of computed coprime factorization of extended system
 * @param[out] hsv Hankel singular values of extended system, dimension (n)
 * @param[in] tol1 Tolerance for order selection (if ordsel='A'). Default if <= 0.
 * @param[in] tol2 Tolerance for controllability/observability tests. Default if <= 0.
 * @param[out] iwork Integer workspace, dimension max(n, pm) for jobmr='N', pm otherwise
 *                   where pm = p (left) or m (right)
 * @param[out] dwork Real workspace, dimension (ldwork)
 * @param[in] ldwork Workspace size. See formula in source.
 * @param[out] iwarn Warning: 10*K+I where K = stability violations, I=1 if nr > nq
 * @param[out] info Error: 0=ok, 1=Schur failed, 2=ordering failed, 3=bad eigenvalue,
 *                  4=HSV failed
 */
void ab09fd(const char* dico, const char* jobcf, const char* fact,
            const char* jobmr, const char* equil, const char* ordsel,
            i32 n, i32 m, i32 p, i32* nr, f64 alpha,
            f64* a, i32 lda, f64* b, i32 ldb, f64* c, i32 ldc,
            i32* nq, f64* hsv, f64 tol1, f64 tol2,
            i32* iwork, f64* dwork, i32 ldwork, i32* iwarn, i32* info);

/**
 * @brief SPA model reduction for unstable systems with coprime factorization.
 *
 * Computes a reduced order model (Ar,Br,Cr,Dr) for an original state-space
 * representation (A,B,C,D) by using either the square-root or the balancing-free
 * square-root Singular Perturbation Approximation (SPA) model reduction method
 * in conjunction with stable coprime factorization techniques.
 *
 * Method:
 * 1. Compute stable coprime factorization: G = R^{-1}*Q (LCF) or G = Q*R^{-1} (RCF)
 * 2. Reduce extended system Ge = (Q R) or Ge = (Q; R)
 * 3. Recover reduced system Gr from reduced factors
 *
 * Error bound: HSV(NR) <= ||Ge-Ger||_inf <= 2*sum(HSV(NR+1:NQ))
 *
 * @param[in] dico System type: 'C'=continuous, 'D'=discrete
 * @param[in] jobcf Factorization: 'L'=left coprime, 'R'=right coprime
 * @param[in] fact Type: 'S'=prescribed stability degree ALPHA, 'I'=inner denominator
 * @param[in] jobmr Method: 'B'=sqrt Balance & Truncate, 'N'=balancing-free sqrt
 * @param[in] equil Equilibration: 'S'=scale, 'N'=no scaling
 * @param[in] ordsel Order selection: 'F'=fixed, 'A'=automatic based on TOL1
 * @param[in] n System order (n >= 0)
 * @param[in] m Number of inputs (m >= 0)
 * @param[in] p Number of outputs (p >= 0)
 * @param[in,out] nr On entry (ordsel='F'): desired order. On exit: actual order.
 * @param[in] alpha Stability degree: < 0 for continuous, 0 <= alpha < 1 for discrete
 *                  (not used if fact='I')
 * @param[in,out] a State matrix A (n-by-n). On exit: reduced Ar (nr-by-nr).
 * @param[in] lda Leading dimension of a (lda >= max(1,n))
 * @param[in,out] b Input matrix B (n-by-m). On exit: reduced Br (nr-by-m).
 * @param[in] ldb Leading dimension of b (ldb >= max(1,n))
 * @param[in,out] c Output matrix C (p-by-n). On exit: reduced Cr (p-by-nr).
 * @param[in] ldc Leading dimension of c (ldc >= max(1,p))
 * @param[in,out] d Feedthrough D (p-by-m). On exit: reduced Dr (p-by-m).
 * @param[in] ldd Leading dimension of d (ldd >= max(1,p))
 * @param[out] nq Order of extended system Ge
 * @param[out] hsv Hankel singular values of Ge (nq values, decreasing)
 * @param[in] tol1 Tolerance for order selection (ordsel='A'). <= 0 uses default.
 * @param[in] tol2 Tolerance for minimal realization. <= 0 uses default. If > 0, tol2 <= tol1.
 * @param[in] tol3 Tolerance for controllability/observability tests. <= 0 uses default.
 * @param[out] iwork Integer workspace (size max(2*n, pm) where pm=p for L, m for R)
 * @param[out] dwork Real workspace (see ldwork formula)
 * @param[in] ldwork Workspace size. See formula in source.
 * @param[out] iwarn Warning: 10*K+I where K = stability violations, I=1 if nr > nq
 * @param[out] info Error: 0=ok, 1=Schur failed, 2=ordering failed, 3=bad eigenvalue,
 *                  4=HSV failed
 */
void ab09gd(const char* dico, const char* jobcf, const char* fact,
            const char* jobmr, const char* equil, const char* ordsel,
            i32 n, i32 m, i32 p, i32* nr, f64 alpha,
            f64* a, i32 lda, f64* b, i32 ldb, f64* c, i32 ldc,
            f64* d, i32 ldd, i32* nq, f64* hsv, f64 tol1, f64 tol2, f64 tol3,
            i32* iwork, f64* dwork, i32 ldwork, i32* iwarn, i32* info);

/**
 * @brief Optimal Hankel-norm approximation model reduction for stable systems.
 *
 * Computes a reduced order model (Ar,Br,Cr,Dr) for a stable original
 * state-space representation (A,B,C,D) by using the optimal Hankel-norm
 * approximation method in conjunction with square-root balancing.
 *
 * The state dynamics matrix A of the original system must be an upper
 * quasi-triangular matrix in real Schur canonical form.
 *
 * The reduced system satisfies the error bound:
 *   HSV(NR) <= ||G-Gr||_inf <= 2*[HSV(NR+1) + ... + HSV(N)]
 *
 * @param[in] dico System type: 'C' = continuous-time, 'D' = discrete-time
 * @param[in] ordsel Order selection: 'F' = fixed order, 'A' = automatic
 * @param[in] n Order of original system (n >= 0)
 * @param[in] m Number of inputs (m >= 0)
 * @param[in] p Number of outputs (p >= 0)
 * @param[in,out] nr On entry: desired order if ordsel='F'. On exit: actual order.
 * @param[in,out] a On entry: N-by-N state matrix in Schur form.
 *                  On exit: NR-by-NR reduced state matrix Ar in Schur form.
 * @param[in] lda Leading dimension of a (lda >= max(1,n))
 * @param[in,out] b On entry: N-by-M input matrix. On exit: NR-by-M reduced Br.
 * @param[in] ldb Leading dimension of b (ldb >= max(1,n))
 * @param[in,out] c On entry: P-by-N output matrix. On exit: P-by-NR reduced Cr.
 * @param[in] ldc Leading dimension of c (ldc >= max(1,p))
 * @param[in,out] d On entry: P-by-M feedthrough matrix. On exit: P-by-M reduced Dr.
 * @param[in] ldd Leading dimension of d (ldd >= max(1,p))
 * @param[out] hsv Hankel singular values, dimension (n), decreasingly ordered
 * @param[in] tol1 Tolerance for order selection (if ordsel='A'). Default if <= 0.
 * @param[in] tol2 Tolerance for minimal realization. Default if <= 0. TOL2 <= TOL1.
 * @param[out] iwork Integer workspace, dimension max(1,n,m) for discrete,
 *                   max(1,m) for continuous. iwork[0] = nmin on exit.
 * @param[out] dwork Real workspace, dimension (ldwork)
 * @param[in] ldwork Workspace size >= max(LDW1, LDW2) where:
 *                   LDW1 = N*(2*N+max(N,M,P)+5) + N*(N+1)/2
 *                   LDW2 = N*(M+P+2) + 2*M*P + min(N,M) + max(3*M+1, min(N,M)+P)
 * @param[out] iwarn Warning: 0=ok, 1=nr > nmin (nr set to nmin)
 * @param[out] info Error: 0=ok, 1=A unstable, 2=HSV computation failed,
 *                  3=stable projection failed, 4=order mismatch
 */
void ab09cx(const char* dico, const char* ordsel,
            i32 n, i32 m, i32 p, i32* nr, f64* a, i32 lda,
            f64* b, i32 ldb, f64* c, i32 ldc, f64* d, i32 ldd,
            f64* hsv, f64 tol1, f64 tol2, i32* iwork,
            f64* dwork, i32 ldwork, i32* iwarn, i32* info);

/**
 * @brief Compute reduced order model using singular perturbation approximation.
 *
 * Given a state-space system (A, B, C, D) with state dimension N, this routine
 * computes a reduced order model (Ar, Br, Cr, Dr) of order NR using singular
 * perturbation approximation (residualization) formulas.
 *
 * The system matrices are partitioned as:
 *         ( A11 A12 )        ( B1 )
 *     A = (         ) ,  B = (    ) ,  C = ( C1  C2 )
 *         ( A21 A22 )        ( B2 )
 *
 * where A11 is NR-by-NR, and the reduced system is computed as:
 *     Ar = A11 + A12*(g*I-A22)^{-1}*A21
 *     Br = B1  + A12*(g*I-A22)^{-1}*B2
 *     Cr = C1  + C2*(g*I-A22)^{-1}*A21
 *     Dr = D   + C2*(g*I-A22)^{-1}*B2
 *
 * where g = 0 for continuous-time (DICO='C') and g = 1 for discrete-time (DICO='D').
 *
 * @param[in] dico System type: 'C' = continuous-time, 'D' = discrete-time
 * @param[in] n Order of original system (n >= 0)
 * @param[in] m Number of inputs (m >= 0)
 * @param[in] p Number of outputs (p >= 0)
 * @param[in] nr Order of reduced system (0 <= nr <= n)
 * @param[in,out] a On entry: N-by-N state matrix. On exit: NR-by-NR reduced Ar.
 *                  Dimension (lda, n).
 * @param[in] lda Leading dimension of a (lda >= max(1, n))
 * @param[in,out] b On entry: N-by-M input matrix. On exit: NR-by-M reduced Br.
 *                  Dimension (ldb, m).
 * @param[in] ldb Leading dimension of b (ldb >= max(1, n))
 * @param[in,out] c On entry: P-by-N output matrix. On exit: P-by-NR reduced Cr.
 *                  Dimension (ldc, n).
 * @param[in] ldc Leading dimension of c (ldc >= max(1, p))
 * @param[in,out] d On entry: P-by-M feedthrough matrix. On exit: P-by-M reduced Dr.
 *                  If nr=0 and system is stable, contains steady-state gain.
 *                  Dimension (ldd, m).
 * @param[in] ldd Leading dimension of d (ldd >= max(1, p))
 * @param[out] rcond Reciprocal condition number of (A22 - g*I)
 * @param[out] iwork Integer workspace, dimension 2*(n-nr)
 * @param[out] dwork Real workspace, dimension 4*(n-nr)
 * @return 0 on success, -i if parameter i is invalid, 1 if (A22-g*I) is singular
 */
i32 ab09dd(const char* dico, i32 n, i32 m, i32 p, i32 nr,
           f64* a, i32 lda, f64* b, i32 ldb,
           f64* c, i32 ldc, f64* d, i32 ldd,
           f64* rcond, i32* iwork, f64* dwork);

/**
 * @brief Accuracy enhanced balancing related model reduction.
 *
 * Computes a reduced order model (Ar,Br,Cr,Dr) for an original state-space
 * representation (A,B,C,D) using square-root or balancing-free square-root
 * Balance & Truncate (B&T) or Singular Perturbation Approximation (SPA)
 * model reduction methods.
 *
 * The computation uses Cholesky factors S and R of controllability and
 * observability Grammians. Hankel singular values are computed as singular
 * values of R*S.
 *
 * For B&T: Ar = TI*A*T, Br = TI*B, Cr = C*T
 * For SPA: Same formulas applied to minimal realization, then SPA formulas.
 *
 * @param[in] dico System type: 'C'=continuous, 'D'=discrete
 * @param[in] job Reduction method: 'B'=square-root B&T, 'F'=balancing-free B&T,
 *                'S'=square-root SPA, 'P'=balancing-free SPA
 * @param[in] fact A matrix form: 'S'=real Schur form, 'N'=general matrix
 * @param[in] ordsel Order selection: 'F'=fixed order NR, 'A'=automatic based on TOL1
 * @param[in] n Order of original system (n >= 0)
 * @param[in] m Number of system inputs (m >= 0)
 * @param[in] p Number of system outputs (p >= 0)
 * @param[in,out] nr On entry (ordsel='F'): desired order. On exit: actual reduced order.
 * @param[in] scalec Scaling factor for S (controllability Grammian Cholesky factor)
 * @param[in] scaleo Scaling factor for R (observability Grammian Cholesky factor)
 * @param[in,out] a On entry: N-by-N state matrix (Schur form if fact='S').
 *                  On exit: NR-by-NR reduced Ar.
 * @param[in] lda Leading dimension of a (lda >= max(1,n))
 * @param[in,out] b On entry: N-by-M input matrix. On exit: NR-by-M reduced Br.
 * @param[in] ldb Leading dimension of b (ldb >= max(1,n))
 * @param[in,out] c On entry: P-by-N output matrix. On exit: P-by-NR reduced Cr.
 * @param[in] ldc Leading dimension of c (ldc >= max(1,p))
 * @param[in,out] d On entry (SPA): P-by-M feedthrough matrix.
 *                  On exit (SPA): P-by-M reduced Dr.
 *                  Not referenced for B&T methods.
 * @param[in] ldd Leading dimension of d (ldd >= 1, or max(1,p) for SPA)
 * @param[in,out] ti On entry: N-by-N upper triangular Cholesky factor S (P=S*S').
 *                   On exit (if NR > 0): NMINR-by-N left truncation matrix.
 * @param[in] ldti Leading dimension of ti (ldti >= max(1,n))
 * @param[in,out] t On entry: N-by-N upper triangular Cholesky factor R (Q=R'*R).
 *                  On exit (if NR > 0): N-by-NMINR right truncation matrix.
 * @param[in] ldt Leading dimension of t (ldt >= max(1,n))
 * @param[out] nminr Order of minimal realization (HSV > max(TOL2, N*EPS*HSV(1)))
 * @param[out] hsv Hankel singular values, descending order, dimension (n)
 * @param[in] tol1 Tolerance for order selection (ordsel='A'). Default if <= 0.
 * @param[in] tol2 Tolerance for minimal realization. Default N*EPS*HSV(1) if <= 0.
 * @param[out] iwork Integer workspace (0 if job='B', n if 'F', 2*n if 'S'/'P')
 * @param[out] dwork Real workspace, dimension (ldwork)
 * @param[in] ldwork Workspace size >= max(1, 2*n*n+5*n, n*max(m,p))
 * @param[out] iwarn Warning: 0=ok, 1=nr > nminr, 2=repeated singular values at cut-off
 * @return 0 on success, -i if parameter i is invalid, 1 if HSV computation failed
 */
i32 ab09ix(const char* dico, const char* job, const char* fact,
           const char* ordsel, i32 n, i32 m, i32 p, i32* nr,
           f64 scalec, f64 scaleo,
           f64* a, i32 lda, f64* b, i32 ldb,
           f64* c, i32 ldc, f64* d, i32 ldd,
           f64* ti, i32 ldti, f64* t, i32 ldt,
           i32* nminr, f64* hsv, f64 tol1, f64 tol2,
           i32* iwork, f64* dwork, i32 ldwork, i32* iwarn);

/**
 * @brief Frequency-weighted model reduction based on balancing techniques.
 *
 * AB09ID computes a reduced order model (Ar,Br,Cr,Dr) for an original
 * state-space representation (A,B,C,D) using frequency weighted square-root
 * or balancing-free square-root Balance & Truncate (B&T) or Singular
 * Perturbation Approximation (SPA) model reduction methods.
 *
 * The algorithm minimizes ||V*(G-Gr)*W|| where G and Gr are transfer-function
 * matrices of original/reduced systems, and V/W are frequency weighting TFMs.
 *
 * @param[in] dico System type: 'C'=continuous, 'D'=discrete
 * @param[in] jobc Controllability Grammian: 'S'=standard, 'E'=enhanced
 * @param[in] jobo Observability Grammian: 'S'=standard, 'E'=enhanced
 * @param[in] job Reduction method: 'B'=sqrt B&T, 'F'=balfree B&T, 'S'=sqrt SPA, 'P'=balfree SPA
 * @param[in] weight Weighting type: 'N'=none, 'L'=left V, 'R'=right W, 'B'=both
 * @param[in] equil Equilibration: 'S'=scale, 'N'=no scaling
 * @param[in] ordsel Order selection: 'F'=fixed, 'A'=automatic based on TOL1
 * @param[in] n Order of system (n >= 0)
 * @param[in] m Number of inputs (m >= 0)
 * @param[in] p Number of outputs (p >= 0)
 * @param[in] nv Order of left weighting V (nv >= 0)
 * @param[in] pv Number of outputs of V (pv >= 0)
 * @param[in] nw Order of right weighting W (nw >= 0)
 * @param[in] mw Number of inputs of W (mw >= 0)
 * @param[in,out] nr On entry (ordsel='F'): desired order. On exit: actual order.
 * @param[in] alpha Stability boundary: <= 0 for continuous, 0 <= alpha <= 1 for discrete
 * @param[in] alphac Combination parameter for controllability (|alphac| <= 1)
 * @param[in] alphao Combination parameter for observability (|alphao| <= 1)
 * @param[in,out] a State matrix A (n-by-n). On exit: reduced Ar (nr-by-nr).
 * @param[in] lda Leading dimension of a
 * @param[in,out] b Input matrix B (n-by-m). On exit: reduced Br (nr-by-m).
 * @param[in] ldb Leading dimension of b
 * @param[in,out] c Output matrix C (p-by-n). On exit: reduced Cr (p-by-nr).
 * @param[in] ldc Leading dimension of c
 * @param[in,out] d Feedthrough matrix D (p-by-m). On exit: reduced Dr (p-by-m).
 * @param[in] ldd Leading dimension of d
 * @param[in,out] av State matrix of V (nv-by-nv). On exit: minimal realization in Schur form.
 * @param[in] ldav Leading dimension of av
 * @param[in,out] bv Input matrix of V (nv-by-p). On exit: minimal realization.
 * @param[in] ldbv Leading dimension of bv
 * @param[in,out] cv Output matrix of V (pv-by-nv). On exit: minimal realization.
 * @param[in] ldcv Leading dimension of cv
 * @param[in] dv Feedthrough matrix of V (pv-by-p)
 * @param[in] lddv Leading dimension of dv
 * @param[in,out] aw State matrix of W (nw-by-nw). On exit: minimal realization in Schur form.
 * @param[in] ldaw Leading dimension of aw
 * @param[in,out] bw Input matrix of W (nw-by-mw). On exit: minimal realization.
 * @param[in] ldbw Leading dimension of bw
 * @param[in,out] cw Output matrix of W (m-by-nw). On exit: minimal realization.
 * @param[in] ldcw Leading dimension of cw
 * @param[in] dw Feedthrough matrix of W (m-by-mw)
 * @param[in] lddw Leading dimension of dw
 * @param[out] ns Dimension of ALPHA-stable subsystem
 * @param[out] hsv Hankel singular values of ALPHA-stable part, dimension (n)
 * @param[in] tol1 Tolerance for order selection (ordsel='A'). Default if <= 0.
 * @param[in] tol2 Tolerance for minimal realization. Default if <= 0.
 * @param[out] iwarn Warning: 0=ok, 1=nr>nsmin, 2=repeated singular values, 3=nr<nu, 10+K=stability
 * @param[out] info Error: 0=ok, 1-2=Schur failed, 3-5=V coprime factorization,
 *                  6-8=W coprime factorization, 9=eigenvalues, 10=Hankel SVD
 */
void ab09id(
    const char* dico, const char* jobc, const char* jobo, const char* job,
    const char* weight, const char* equil, const char* ordsel,
    i32 n, i32 m, i32 p, i32 nv, i32 pv, i32 nw, i32 mw, i32* nr,
    f64 alpha, f64 alphac, f64 alphao,
    f64* a, i32 lda, f64* b, i32 ldb, f64* c, i32 ldc, f64* d, i32 ldd,
    f64* av, i32 ldav, f64* bv, i32 ldbv, f64* cv, i32 ldcv, const f64* dv, i32 lddv,
    f64* aw, i32 ldaw, f64* bw, i32 ldbw, f64* cw, i32 ldcw, const f64* dw, i32 lddw,
    i32* ns, f64* hsv, f64 tol1, f64 tol2, i32* iwarn, i32* info);

/**
 * @brief Cholesky factors of controllability and observability Grammians.
 *
 * Computes the Cholesky factors Su and Ru of the controllability
 * Grammian P = Su*Su' and observability Grammian Q = Ru'*Ru,
 * respectively, satisfying:
 *
 *     A*P + P*A' + scalec^2*B*B' = 0           (1) continuous Lyapunov
 *     A'*Q + Q*A + scaleo^2*Cw'*Cw = 0         (2) continuous Lyapunov
 *
 * where:
 *     Cw = Hw - Bw'*X
 *     Hw = inv(Dw)*C
 *     Bw = (B*D' + P*C')*inv(Dw')
 *     D*D' = Dw*Dw' (Dw upper triangular from RQ factorization)
 *
 * and X is the stabilizing solution of the Riccati equation:
 *     Aw'*X + X*Aw + Hw'*Hw + X*Bw*Bw'*X = 0   (3)
 *
 * with Aw = A - Bw*Hw.
 *
 * @param[in] n Order of state-space representation (n >= 0)
 * @param[in] m Number of system inputs (m >= 0)
 * @param[in] p Number of system outputs (m >= p >= 0)
 * @param[in] a N-by-N stable state dynamics matrix in real Schur form
 * @param[in] lda Leading dimension of a (lda >= max(1,n))
 * @param[in] b N-by-M input/state matrix
 * @param[in] ldb Leading dimension of b (ldb >= max(1,n))
 * @param[in] c P-by-N state/output matrix
 * @param[in] ldc Leading dimension of c (ldc >= max(1,p))
 * @param[in] d P-by-M full row rank feedthrough matrix
 * @param[in] ldd Leading dimension of d (ldd >= max(1,p))
 * @param[out] scalec Scaling factor for controllability Grammian
 * @param[out] scaleo Scaling factor for observability Grammian
 * @param[out] s N-by-N upper triangular Cholesky factor Su (P = Su*Su')
 * @param[in] lds Leading dimension of s (lds >= max(1,n))
 * @param[out] r N-by-N upper triangular Cholesky factor Ru (Q = Ru'*Ru)
 * @param[in] ldr Leading dimension of r (ldr >= max(1,n))
 * @param[out] iwork Integer workspace, dimension (2*n)
 * @param[out] dwork Real workspace, dimension (ldwork).
 *                   On exit: dwork[0]=optimal ldwork, dwork[1]=RCOND
 * @param[in] ldwork Workspace size >= max(2, n*(max(n,m,p)+5),
 *                   2*n*p + max(p*(m+2), 10*n*(n+1)))
 * @param[out] bwork Logical workspace, dimension (2*n)
 * @param[out] info Error code:
 *                  0 = success
 *                  1 = A is not stable or not in real Schur form
 *                  2 = Hamiltonian reduction to Schur form failed
 *                  3 = Hamiltonian reordering failed
 *                  4 = Hamiltonian has < N stable eigenvalues
 *                  5 = U11 singular in Riccati solver
 *                  6 = D does not have full row rank
 */
void ab09hy(
    const i32 n,
    const i32 m,
    const i32 p,
    const f64* a,
    const i32 lda,
    const f64* b,
    const i32 ldb,
    const f64* c,
    const i32 ldc,
    const f64* d,
    const i32 ldd,
    f64* scalec,
    f64* scaleo,
    f64* s,
    const i32 lds,
    f64* r,
    const i32 ldr,
    i32* iwork,
    f64* dwork,
    const i32 ldwork,
    i32* bwork,
    i32* info);

/**
 * @brief Compute Cholesky factors of frequency-weighted controllability and
 *        observability Grammians using Enns/Lin-Chiu combination method.
 *
 * AB09IY computes the Cholesky factors S and R of the frequency-weighted
 * controllability Grammian P = S*S' and observability Grammian Q = R'*R.
 *
 * @param[in] dico System type: 'C'=continuous, 'D'=discrete
 * @param[in] jobc Controllability Grammian choice: 'S'=standard, 'E'=enhanced
 * @param[in] jobo Observability Grammian choice: 'S'=standard, 'E'=enhanced
 * @param[in] weight Weighting type: 'N'=none, 'L'=left, 'R'=right, 'B'=both
 * @param[in] n Order of system G (n >= 0)
 * @param[in] m Number of inputs of G (m >= 0)
 * @param[in] p Number of outputs of G (p >= 0)
 * @param[in] nv Order of left weighting V (nv >= 0)
 * @param[in] pv Number of outputs of V (pv >= 0)
 * @param[in] nw Order of right weighting W (nw >= 0)
 * @param[in] mw Number of inputs of W (mw >= 0)
 * @param[in] alphac Combination parameter for controllability (|alphac| <= 1)
 * @param[in] alphao Combination parameter for observability (|alphao| <= 1)
 * @param[in] a State matrix of G in Schur form (n-by-n)
 * @param[in] lda Leading dimension of a
 * @param[in] b Input matrix of G (n-by-m)
 * @param[in] ldb Leading dimension of b
 * @param[in] c Output matrix of G (p-by-n)
 * @param[in] ldc Leading dimension of c
 * @param[in] av State matrix of V in Schur form (nv-by-nv)
 * @param[in] ldav Leading dimension of av
 * @param[in] bv Input matrix of V (nv-by-p)
 * @param[in] ldbv Leading dimension of bv
 * @param[in] cv Output matrix of V (pv-by-nv)
 * @param[in] ldcv Leading dimension of cv
 * @param[in] dv Feedthrough matrix of V (pv-by-p)
 * @param[in] lddv Leading dimension of dv
 * @param[in] aw State matrix of W in Schur form (nw-by-nw)
 * @param[in] ldaw Leading dimension of aw
 * @param[in] bw Input matrix of W (nw-by-mw)
 * @param[in] ldbw Leading dimension of bw
 * @param[in] cw Output matrix of W (m-by-nw)
 * @param[in] ldcw Leading dimension of cw
 * @param[in] dw Feedthrough matrix of W (m-by-mw)
 * @param[in] lddw Leading dimension of dw
 * @param[out] scalec Scaling factor for controllability Grammian
 * @param[out] scaleo Scaling factor for observability Grammian
 * @param[out] s Upper triangular Cholesky factor of P (n-by-n)
 * @param[in] lds Leading dimension of s
 * @param[out] r Upper triangular Cholesky factor of Q (n-by-n)
 * @param[in] ldr Leading dimension of r
 * @param[out] dwork Workspace
 * @param[in] ldwork Workspace size
 * @param[out] info Error code: 0=success, 1=A/AV unstable, 2=A/AW unstable, 3=eigenvalue failure
 */
void ab09iy(
    const char* dico,
    const char* jobc,
    const char* jobo,
    const char* weight,
    const i32 n,
    const i32 m,
    const i32 p,
    const i32 nv,
    const i32 pv,
    const i32 nw,
    const i32 mw,
    const f64 alphac,
    const f64 alphao,
    const f64* a, const i32 lda,
    const f64* b, const i32 ldb,
    const f64* c, const i32 ldc,
    const f64* av, const i32 ldav,
    const f64* bv, const i32 ldbv,
    const f64* cv, const i32 ldcv,
    const f64* dv, const i32 lddv,
    const f64* aw, const i32 ldaw,
    const f64* bw, const i32 ldbw,
    const f64* cw, const i32 ldcw,
    const f64* dw, const i32 lddw,
    f64* scalec,
    f64* scaleo,
    f64* s, const i32 lds,
    f64* r, const i32 ldr,
    f64* dwork,
    const i32 ldwork,
    i32* info);

/**
 * @brief Check stability/antistability of finite eigenvalues.
 *
 * Checks whether all finite eigenvalues (or their reciprocals) lie within
 * a specified stability domain.
 *
 * Domain definitions:
 * - Continuous (DICO='C'), stable (STDOM='S'): Re(lambda) < ALPHA
 * - Continuous (DICO='C'), unstable (STDOM='U'): Re(lambda) > ALPHA
 * - Discrete (DICO='D'), stable (STDOM='S'): |lambda| < ALPHA
 * - Discrete (DICO='D'), unstable (STDOM='U'): |lambda| > ALPHA
 *
 * For EVTYPE='R', the same conditions apply to 1/lambda.
 *
 * @param[in] dico System type: 'C'=continuous, 'D'=discrete
 * @param[in] stdom Domain type: 'S'=stability, 'U'=instability
 * @param[in] evtype Eigenvalue type: 'S'=standard (ED=1), 'G'=generalized,
 *                   'R'=reciprocal generalized
 * @param[in] n Dimension of eigenvalue vectors (n >= 0)
 * @param[in] alpha Boundary value for real parts (continuous) or moduli (discrete).
 *                  For discrete-time, alpha >= 0.
 * @param[in] er Real parts of eigenvalues, dimension (n)
 * @param[in] ei Imaginary parts of eigenvalues, dimension (n)
 * @param[in] ed Denominators for generalized eigenvalues, dimension (n).
 *               Not referenced if evtype='S'. ED(j)=0 means infinite eigenvalue.
 * @param[in] tolinf Tolerance for detecting infinite eigenvalues. 0 <= tolinf < 1.
 * @return 0 if all eigenvalues in domain, 1 if some outside, -i if param i invalid
 */
i32 ab09jx(const char* dico, const char* stdom, const char* evtype,
           i32 n, f64 alpha, f64* er, f64* ei, f64* ed, f64 tolinf);

/**
 * @brief Projection of V*G or conj(V)*G containing poles of G.
 *
 * Constructs state-space representation (A,BS,CS,DS) of the projection
 * of V*G or conj(V)*G containing the poles of G, from the state-space
 * representations (A,B,C,D) and (AV-lambda*EV,BV,CV,DV) of G and V.
 *
 * G is assumed stable with A in real Schur form.
 * For V*G: G and V must have distinct poles.
 * For conj(V)*G: G and conj(V) must have distinct poles.
 *
 * For JOB='V': BS=B, CS=CV*X+DV*C, DS=DV*D where X satisfies AV*X-EV*X*A+BV*C=0
 * For JOB='C' continuous: BS=B, CS=BV'*X+DV'*C, DS=DV'*D
 * For JOB='C' discrete: BS=B, CS=BV'*X*A+DV'*C, DS=DV'*D+BV'*X*B
 *
 * @param[in] job 'V'=compute V*G projection, 'C'=compute conj(V)*G projection
 * @param[in] dico 'C'=continuous-time, 'D'=discrete-time
 * @param[in] jobev 'G'=general EV, 'I'=EV is identity
 * @param[in] stbchk 'C'=check stability/antistability, 'N'=no check
 * @param[in] n Order of G state matrix (n >= 0)
 * @param[in] m Number of inputs of G (m >= 0)
 * @param[in] p Number of outputs of G (p >= 0)
 * @param[in] nv Order of V state matrix (nv >= 0)
 * @param[in] pv Number of outputs (job='V') or inputs (job='C') of V (pv >= 0)
 * @param[in] a N-by-N state matrix of G in real Schur form, dimension (lda,n)
 * @param[in] lda Leading dimension of a (lda >= max(1,n))
 * @param[in] b N-by-M input matrix of G, dimension (ldb,m)
 * @param[in] ldb Leading dimension of b (ldb >= max(1,n))
 * @param[in,out] c On entry: P-by-N output matrix of G. On exit: PV-by-N output CS.
 * @param[in] ldc Leading dimension of c (ldc >= max(1,p,pv))
 * @param[in,out] d On entry: P-by-M feedthrough of G. On exit: PV-by-M feedthrough DS.
 * @param[in] ldd Leading dimension of d (ldd >= max(1,p,pv))
 * @param[in,out] av NV-by-NV state matrix of V, on exit: condensed form
 * @param[in] ldav Leading dimension of av (ldav >= max(1,nv))
 * @param[in,out] ev NV-by-NV descriptor matrix of V (if jobev='G'), on exit: condensed
 * @param[in] ldev Leading dimension of ev (ldev >= max(1,nv) if jobev='G', else >= 1)
 * @param[in,out] bv NV-by-MBV input matrix of V (MBV=P if job='V', PV if job='C')
 * @param[in] ldbv Leading dimension of bv (ldbv >= max(1,nv))
 * @param[in,out] cv PCV-by-NV output matrix of V (PCV=PV if job='V', P if job='C')
 * @param[in] ldcv Leading dimension (ldcv >= max(1,pv) if job='V', max(1,p) if job='C')
 * @param[in] dv PCV-by-MBV feedthrough matrix of V
 * @param[in] lddv Leading dimension (lddv >= max(1,pv) if job='V', max(1,p) if job='C')
 * @param[out] iwork Integer workspace (0 if jobev='I', nv+n+6 if jobev='G')
 * @param[out] dwork Real workspace, dimension (ldwork)
 * @param[in] ldwork Workspace size >= LW1 if jobev='I', >= LW2 if jobev='G'
 * @param[out] info 0=ok, 1=Schur form failed, 2=Sylvester (general) failed,
 *                  3=Sylvester (standard) failed, 4=stability check failed
 */
void ab09jv(
    const char* job, const char* dico, const char* jobev, const char* stbchk,
    i32 n, i32 m, i32 p, i32 nv, i32 pv,
    const f64* a, i32 lda,
    const f64* b, i32 ldb,
    f64* c, i32 ldc,
    f64* d, i32 ldd,
    f64* av, i32 ldav,
    f64* ev, i32 ldev,
    f64* bv, i32 ldbv,
    f64* cv, i32 ldcv,
    const f64* dv, i32 lddv,
    i32* iwork,
    f64* dwork, i32 ldwork,
    i32* info);

/**
 * @brief Projection of right weighted transfer-function matrix G*W or G*conj(W).
 *
 * Constructs the state-space representation (A,BS,CS,DS) for the projection of
 * G*W or G*conj(W) containing the poles of the transfer-function matrix G,
 * where W is a descriptor system weight.
 *
 * Used in coprime factorization-based model reduction algorithms.
 *
 * @param[in] job 'W'=compute G*W, 'C'=compute G*conj(W)
 * @param[in] dico 'C'=continuous-time, 'D'=discrete-time
 * @param[in] jobew 'G'=general EW, 'I'=identity EW
 * @param[in] stbchk 'C'=check stability/antistability, 'N'=no check
 * @param[in] n Order of G state matrix (n >= 0)
 * @param[in] m Number of inputs of G (m >= 0)
 * @param[in] p Number of outputs of G (p >= 0)
 * @param[in] nw Order of W state matrix (nw >= 0)
 * @param[in] mw Number of inputs (job='W') or outputs (job='C') of W (mw >= 0)
 * @param[in] a N-by-N state matrix of G in real Schur form, dimension (lda,n)
 * @param[in] lda Leading dimension of a (lda >= max(1,n))
 * @param[in,out] b On entry: N-by-M input matrix of G. On exit: N-by-MW input BS.
 * @param[in] ldb Leading dimension of b (ldb >= max(1,n))
 * @param[in] c P-by-N output matrix of G (unchanged, CS = C)
 * @param[in] ldc Leading dimension of c (ldc >= max(1,p))
 * @param[in,out] d On entry: P-by-M feedthrough of G. On exit: P-by-MW feedthrough DS.
 * @param[in] ldd Leading dimension of d (ldd >= max(1,p))
 * @param[in,out] aw NW-by-NW state matrix of W, on exit: condensed form
 * @param[in] ldaw Leading dimension of aw (ldaw >= max(1,nw))
 * @param[in,out] ew NW-by-NW descriptor matrix of W (if jobew='G'), on exit: condensed
 * @param[in] ldew Leading dimension of ew (ldew >= max(1,nw) if jobew='G', else >= 1)
 * @param[in,out] bw NW-by-MBW input matrix of W (MBW=MW if job='W', M if job='C')
 * @param[in] ldbw Leading dimension of bw (ldbw >= max(1,nw))
 * @param[in,out] cw PCW-by-NW output matrix of W (PCW=M if job='W', MW if job='C')
 * @param[in] ldcw Leading dimension (ldcw >= max(1,m) if job='W', max(1,mw) if job='C')
 * @param[in] dw PCW-by-MBW feedthrough matrix of W
 * @param[in] lddw Leading dimension (lddw >= max(1,m) if job='W', max(1,mw) if job='C')
 * @param[out] iwork Integer workspace (0 if jobew='I', nw+n+6 if jobew='G')
 * @param[out] dwork Real workspace, dimension (ldwork)
 * @param[in] ldwork Workspace size >= LW1 if jobew='I', >= LW2 if jobew='G'
 * @param[out] info 0=ok, 1=Schur form failed, 2=Sylvester (general) failed,
 *                  3=Sylvester (standard) failed, 4=stability check failed
 */
void ab09jw(
    const char* job, const char* dico, const char* jobew, const char* stbchk,
    i32 n, i32 m, i32 p, i32 nw, i32 mw,
    const f64* a, i32 lda,
    f64* b, i32 ldb,
    const f64* c, i32 ldc,
    f64* d, i32 ldd,
    f64* aw, i32 ldaw,
    f64* ew, i32 ldew,
    f64* bw, i32 ldbw,
    f64* cw, i32 ldcw,
    const f64* dw, i32 lddw,
    i32* iwork,
    f64* dwork, i32 ldwork,
    i32* info);

/**
 * @brief Stable projection of V*G*W or conj(V)*G*conj(W).
 *
 * Constructs state-space representation (A,BS,CS,DS) of the stable projection
 * of V*G*W or conj(V)*G*conj(W) from the state-space representations (A,B,C,D),
 * (AV,BV,CV,DV), and (AW,BW,CW,DW) of transfer-function matrices G, V and W.
 *
 * G must be stable and A must be in real Schur form.
 * For JOB='N': V and W must be completely unstable.
 * For JOB='C': V and W must be stable.
 *
 * @param[in] job 'N'=V*G*W, 'C'=conj(V)*G*conj(W)
 * @param[in] dico 'C'=continuous-time, 'D'=discrete-time
 * @param[in] weight 'N'=no weights, 'L'=left only, 'R'=right only, 'B'=both
 * @param[in] n Order of G state matrix (n >= 0)
 * @param[in] nv Order of V state matrix (nv >= 0)
 * @param[in] nw Order of W state matrix (nw >= 0)
 * @param[in] m Number of G inputs and W I/O dimension (m >= 0)
 * @param[in] p Number of G outputs and V I/O dimension (p >= 0)
 * @param[in] a N-by-N state matrix of G in real Schur form
 * @param[in] lda Leading dimension of a
 * @param[in,out] b On entry: N-by-M input matrix. On exit: BS.
 * @param[in] ldb Leading dimension of b
 * @param[in,out] c On entry: P-by-N output matrix. On exit: CS.
 * @param[in] ldc Leading dimension of c
 * @param[in,out] d On entry: P-by-M feedthrough. On exit: DS.
 * @param[in] ldd Leading dimension of d
 * @param[in,out] av NV-by-NV state matrix of V. On exit: real Schur form.
 * @param[in] ldav Leading dimension of av
 * @param[in,out] bv NV-by-P input matrix of V. On exit: transformed.
 * @param[in] ldbv Leading dimension of bv
 * @param[in,out] cv P-by-NV output matrix of V. On exit: transformed.
 * @param[in] ldcv Leading dimension of cv
 * @param[in] dv P-by-P feedthrough matrix of V
 * @param[in] lddv Leading dimension of dv
 * @param[in,out] aw NW-by-NW state matrix of W. On exit: real Schur form.
 * @param[in] ldaw Leading dimension of aw
 * @param[in,out] bw NW-by-M input matrix of W. On exit: transformed.
 * @param[in] ldbw Leading dimension of bw
 * @param[in,out] cw M-by-NW output matrix of W. On exit: transformed.
 * @param[in] ldcw Leading dimension of cw
 * @param[in] dw M-by-M feedthrough matrix of W
 * @param[in] lddw Leading dimension of dw
 * @param[out] dwork Workspace, dimension (ldwork)
 * @param[in] ldwork Workspace size (see documentation for formula)
 * @param[out] iwarn Warning: 0=ok, 1=AV issue, 2=AW issue, 3=both
 * @param[out] info Error: 0=ok, 1=AV Schur failed, 2=AW Schur failed,
 *                  3=Sylvester (V) failed, 4=Sylvester (W) failed
 */
void ab09kx(
    const char* job, const char* dico, const char* weight,
    i32 n, i32 nv, i32 nw, i32 m, i32 p,
    const f64* a, i32 lda,
    f64* b, i32 ldb,
    f64* c, i32 ldc,
    f64* d, i32 ldd,
    f64* av, i32 ldav,
    f64* bv, i32 ldbv,
    f64* cv, i32 ldcv,
    const f64* dv, i32 lddv,
    f64* aw, i32 ldaw,
    f64* bw, i32 ldbw,
    f64* cw, i32 ldcw,
    const f64* dw, i32 lddw,
    f64* dwork, i32 ldwork,
    i32* iwarn, i32* info);

/**
 * @brief Frequency-weighted Hankel-norm approximation with invertible weights.
 *
 * Computes a reduced order model (Ar,Br,Cr,Dr) for an original state-space
 * representation (A,B,C,D) by using the frequency weighted optimal Hankel-norm
 * approximation method. The Hankel norm of the weighted error
 *
 *       op(V)*(G-Gr)*op(W)
 *
 * is minimized, where G and Gr are the transfer-function matrices of the
 * original and reduced systems, V and W are invertible transfer-function
 * matrices, and op(X) denotes X, inv(X), conj(X), or conj(inv(X)).
 *
 * When minimizing ||V*(G-Gr)*W||, V and W must be antistable.
 * When minimizing ||inv(V)*(G-Gr)*inv(W)||, V and W must have antistable zeros.
 * When minimizing ||conj(V)*(G-Gr)*conj(W)||, V and W must be stable.
 * When minimizing ||conj(inv(V))*(G-Gr)*conj(inv(W))||, V and W must be minimum-phase.
 *
 * @param[in] jobv Left frequency-weighting:
 *                 'N' = V = I
 *                 'V' = op(V) = V
 *                 'I' = op(V) = inv(V)
 *                 'C' = op(V) = conj(V)
 *                 'R' = op(V) = conj(inv(V))
 * @param[in] jobw Right frequency-weighting:
 *                 'N' = W = I
 *                 'W' = op(W) = W
 *                 'I' = op(W) = inv(W)
 *                 'C' = op(W) = conj(W)
 *                 'R' = op(W) = conj(inv(W))
 * @param[in] jobinv Computational approach:
 *                   'N' = inverse-free descriptor system approach
 *                   'I' = inversion-based standard approach
 *                   'A' = automatic (switch to inverse-free if ill-conditioned)
 * @param[in] dico System type: 'C' = continuous, 'D' = discrete
 * @param[in] equil Equilibration: 'S' = scale (A,B,C), 'N' = no scaling
 * @param[in] ordsel Order selection: 'F' = fixed order NR, 'A' = automatic
 * @param[in] n Order of original system (n >= 0)
 * @param[in] nv Order of left weighting V (nv >= 0)
 * @param[in] nw Order of right weighting W (nw >= 0)
 * @param[in] m Number of system inputs (m >= 0)
 * @param[in] p Number of system outputs (p >= 0)
 * @param[in,out] nr On entry (ordsel='F'): desired order. On exit: actual order.
 * @param[in] alpha Stability boundary. Continuous: alpha <= 0, Discrete: 0 <= alpha <= 1.
 * @param[in,out] a On entry: N-by-N state matrix. On exit: NR-by-NR reduced Ar.
 * @param[in] lda Leading dimension of a (lda >= max(1,n))
 * @param[in,out] b On entry: N-by-M input matrix. On exit: NR-by-M reduced Br.
 * @param[in] ldb Leading dimension of b (ldb >= max(1,n))
 * @param[in,out] c On entry: P-by-N output matrix. On exit: P-by-NR reduced Cr.
 * @param[in] ldc Leading dimension of c (ldc >= max(1,p))
 * @param[in,out] d On entry: P-by-M feedthrough matrix. On exit: P-by-M reduced Dr.
 * @param[in] ldd Leading dimension of d (ldd >= max(1,p))
 * @param[in,out] av On entry (JOBV<>'N'): NV-by-NV state matrix of V.
 *                   On exit: Schur form. Not referenced if JOBV='N'.
 * @param[in] ldav Leading dimension of av
 * @param[in,out] bv On entry (JOBV<>'N'): NV-by-P input matrix of V.
 *                   On exit: transformed. Not referenced if JOBV='N'.
 * @param[in] ldbv Leading dimension of bv
 * @param[in,out] cv On entry (JOBV<>'N'): P-by-NV output matrix of V.
 *                   On exit: transformed. Not referenced if JOBV='N'.
 * @param[in] ldcv Leading dimension of cv
 * @param[in] dv Feedthrough matrix P-by-P of V (JOBV<>'N')
 * @param[in] lddv Leading dimension of dv
 * @param[in,out] aw On entry (JOBW<>'N'): NW-by-NW state matrix of W.
 *                   On exit: Schur form. Not referenced if JOBW='N'.
 * @param[in] ldaw Leading dimension of aw
 * @param[in,out] bw On entry (JOBW<>'N'): NW-by-M input matrix of W.
 *                   On exit: transformed. Not referenced if JOBW='N'.
 * @param[in] ldbw Leading dimension of bw
 * @param[in,out] cw On entry (JOBW<>'N'): M-by-NW output matrix of W.
 *                   On exit: transformed. Not referenced if JOBW='N'.
 * @param[in] ldcw Leading dimension of cw
 * @param[in] dw Feedthrough matrix M-by-M of W (JOBW<>'N')
 * @param[in] lddw Leading dimension of dw
 * @param[out] ns Dimension of ALPHA-stable subsystem
 * @param[out] hsv Hankel singular values of ALPHA-stable part, dimension (n)
 * @param[in] tol1 Tolerance for order selection (ordsel='A'). Default if <= 0.
 * @param[in] tol2 Tolerance for minimal realization. Default if <= 0. TOL2 <= TOL1.
 * @param[out] iwork Integer workspace
 * @param[out] dwork Real workspace, dimension (ldwork)
 * @param[in] ldwork Workspace size (see SLICOT documentation for formula)
 * @param[out] iwarn Warning: 0=ok, 1=nr > nsmin, 2=nr < nu
 * @param[out] info Error: 0=ok, 1-2=Schur failed, 3-4=AV/AW Schur failed,
 *                  5-6=generalized Schur for inv failed, 7-9=Hankel/projection,
 *                  10-11=inv(DV)/inv(DW) Schur failed, 12-17=Sylvester failed,
 *                  18-19=op(V)/op(W) not antistable, 20-21=V/W not invertible
 */
void ab09jd(const char* jobv, const char* jobw, const char* jobinv,
            const char* dico, const char* equil, const char* ordsel,
            i32 n, i32 nv, i32 nw, i32 m, i32 p, i32* nr, f64 alpha,
            f64* a, i32 lda, f64* b, i32 ldb, f64* c, i32 ldc, f64* d, i32 ldd,
            f64* av, i32 ldav, f64* bv, i32 ldbv, f64* cv, i32 ldcv, f64* dv, i32 lddv,
            f64* aw, i32 ldaw, f64* bw, i32 ldbw, f64* cw, i32 ldcw, f64* dw, i32 lddw,
            i32* ns, f64* hsv, f64 tol1, f64 tol2,
            i32* iwork, f64* dwork, i32 ldwork, i32* iwarn, i32* info);

/**
 * @brief Frequency-weighted Hankel-norm approximation model reduction.
 *
 * Computes a reduced order model (Ar,Br,Cr,Dr) for an original state-space
 * representation (A,B,C,D) by using the frequency weighted optimal Hankel-norm
 * approximation method. The Hankel norm of the weighted error
 *
 *       V*(G-Gr)*W    or    conj(V)*(G-Gr)*conj(W)
 *
 * is minimized, where G and Gr are the transfer-function matrices of the
 * original and reduced systems, V and W are left/right frequency weights.
 *
 * For V*(G-Gr)*W: V and W must be antistable.
 * For conj(V)*(G-Gr)*conj(W): V and W must be stable.
 * Additionally, DV and DW must be invertible.
 *
 * @param[in] job Frequency-weighting problem:
 *                'N' = minimize ||V*(G-Gr)*W||_H
 *                'C' = minimize ||conj(V)*(G-Gr)*conj(W)||_H
 * @param[in] dico System type: 'C' = continuous, 'D' = discrete
 * @param[in] weight Weighting type:
 *                   'N' = no weights (V=I, W=I)
 *                   'L' = left weight only (W=I)
 *                   'R' = right weight only (V=I)
 *                   'B' = both weights
 * @param[in] equil Equilibration: 'S' = scale (A,B,C), 'N' = no scaling
 * @param[in] ordsel Order selection: 'F' = fixed order NR, 'A' = automatic
 * @param[in] n Order of original system (n >= 0)
 * @param[in] nv Order of left weighting V (nv >= 0)
 * @param[in] nw Order of right weighting W (nw >= 0)
 * @param[in] m Number of system inputs (m >= 0)
 * @param[in] p Number of system outputs (p >= 0)
 * @param[in,out] nr On entry (ordsel='F'): desired order. On exit: actual order.
 * @param[in] alpha Stability boundary. Continuous: alpha <= 0, Discrete: 0 <= alpha <= 1.
 * @param[in,out] a On entry: N-by-N state matrix. On exit: NR-by-NR reduced Ar.
 * @param[in] lda Leading dimension of a (lda >= max(1,n))
 * @param[in,out] b On entry: N-by-M input matrix. On exit: NR-by-M reduced Br.
 * @param[in] ldb Leading dimension of b (ldb >= max(1,n))
 * @param[in,out] c On entry: P-by-N output matrix. On exit: P-by-NR reduced Cr.
 * @param[in] ldc Leading dimension of c (ldc >= max(1,p))
 * @param[in,out] d On entry: P-by-M feedthrough matrix. On exit: P-by-M reduced Dr.
 * @param[in] ldd Leading dimension of d (ldd >= max(1,p))
 * @param[in,out] av On entry (WEIGHT='L'/'B'): NV-by-NV state matrix of V.
 *                   On exit: Schur form of state matrix of inv(V).
 * @param[in] ldav Leading dimension of av
 * @param[in,out] bv On entry (WEIGHT='L'/'B'): NV-by-P input matrix of V.
 *                   On exit: input matrix of inv(V).
 * @param[in] ldbv Leading dimension of bv
 * @param[in,out] cv On entry (WEIGHT='L'/'B'): P-by-NV output matrix of V.
 *                   On exit: output matrix of inv(V).
 * @param[in] ldcv Leading dimension of cv
 * @param[in,out] dv On entry (WEIGHT='L'/'B'): P-by-P feedthrough matrix of V.
 *                   On exit: feedthrough matrix of inv(V).
 * @param[in] lddv Leading dimension of dv
 * @param[in,out] aw On entry (WEIGHT='R'/'B'): NW-by-NW state matrix of W.
 *                   On exit: Schur form of state matrix of inv(W).
 * @param[in] ldaw Leading dimension of aw
 * @param[in,out] bw On entry (WEIGHT='R'/'B'): NW-by-M input matrix of W.
 *                   On exit: input matrix of inv(W).
 * @param[in] ldbw Leading dimension of bw
 * @param[in,out] cw On entry (WEIGHT='R'/'B'): M-by-NW output matrix of W.
 *                   On exit: output matrix of inv(W).
 * @param[in] ldcw Leading dimension of cw
 * @param[in,out] dw On entry (WEIGHT='R'/'B'): M-by-M feedthrough matrix of W.
 *                   On exit: feedthrough matrix of inv(W).
 * @param[in] lddw Leading dimension of dw
 * @param[out] ns Dimension of ALPHA-stable subsystem
 * @param[out] hsv Hankel singular values of ALPHA-stable part, dimension (n)
 * @param[in] tol1 Tolerance for order selection (ordsel='A'). Default if <= 0.
 * @param[in] tol2 Tolerance for minimal realization. Default if <= 0. TOL2 <= TOL1.
 * @param[out] iwork Integer workspace. iwork[0] = nmin on exit.
 * @param[out] dwork Real workspace, dimension (ldwork)
 * @param[in] ldwork Workspace size (see documentation for formula)
 * @param[out] iwarn Warning: 0=ok, 1=nr > nsmin, 2=nr < nu
 * @param[out] info Error: 0=ok, 1-2=Schur failed, 3-4=weight Schur failed,
 *                  5-6=weight stability, 7-9=Hankel/projection, 10-11=DV/DW singular,
 *                  12-13=Sylvester failed
 */
void ab09kd(const char* job, const char* dico, const char* weight,
            const char* equil, const char* ordsel,
            i32 n, i32 nv, i32 nw, i32 m, i32 p, i32* nr, f64 alpha,
            f64* a, i32 lda, f64* b, i32 ldb, f64* c, i32 ldc, f64* d, i32 ldd,
            f64* av, i32 ldav, f64* bv, i32 ldbv, f64* cv, i32 ldcv, f64* dv, i32 lddv,
            f64* aw, i32 ldaw, f64* bw, i32 ldbw, f64* cw, i32 ldcw, f64* dw, i32 lddw,
            i32* ns, f64* hsv, f64 tol1, f64 tol2,
            i32* iwork, f64* dwork, i32 ldwork, i32* iwarn, i32* info);

/**
 * @brief Balance & Truncate model reduction for ALPHA-stable part.
 *
 * Computes a reduced order model (Ar,Br,Cr) for an original state-space
 * representation (A,B,C) by using either the square-root or the balancing-free
 * square-root Balance & Truncate (B&T) model reduction method for the
 * ALPHA-stable part of the system.
 *
 * The procedure:
 * 1. Decompose G = G1 + G2 where G1 has ALPHA-stable poles, G2 has unstable poles
 * 2. Reduce G1 to G1r using B&T method
 * 3. Assemble Gr = G1r + G2
 *
 * Error bound: HSV(NR+NS-N) <= ||G-Gr||_inf <= 2*sum(HSV(NR+NS-N+1:NS))
 *
 * @param[in] dico System type: 'C'=continuous, 'D'=discrete
 * @param[in] job Model reduction approach: 'B'=square-root B&T, 'N'=balancing-free B&T
 * @param[in] equil Equilibration: 'S'=scale (A,B,C), 'N'=no scaling
 * @param[in] ordsel Order selection: 'F'=fixed order NR, 'A'=automatic based on TOL
 * @param[in] n Order of original system (n >= 0)
 * @param[in] m Number of system inputs (m >= 0)
 * @param[in] p Number of system outputs (p >= 0)
 * @param[in,out] nr On entry (ordsel='F'): desired order. On exit: actual reduced order.
 * @param[in] alpha Stability boundary. Continuous: alpha <= 0, Discrete: 0 <= alpha <= 1.
 * @param[in,out] a On entry: N-by-N state matrix. On exit: NR-by-NR reduced Ar.
 * @param[in] lda Leading dimension of a (lda >= max(1,n))
 * @param[in,out] b On entry: N-by-M input matrix. On exit: NR-by-M reduced Br.
 * @param[in] ldb Leading dimension of b (ldb >= max(1,n))
 * @param[in,out] c On entry: P-by-N output matrix. On exit: P-by-NR reduced Cr.
 * @param[in] ldc Leading dimension of c (ldc >= max(1,p))
 * @param[out] ns Dimension of ALPHA-stable subsystem
 * @param[out] hsv Hankel singular values of ALPHA-stable part, dimension (n)
 * @param[in] tol Tolerance for order selection (if ordsel='A'). If tol <= 0,
 *                uses default NS*EPS*HSV(1) for minimal realization.
 * @param[out] iwork Integer workspace, dimension (n) if job='N', else 0.
 * @param[out] dwork Real workspace, dimension (ldwork)
 * @param[in] ldwork Workspace size >= max(1, n*(2*n+max(n,m,p)+5) + n*(n+1)/2)
 * @param[out] iwarn Warning: 0=ok, 1=nr > nsmin, 2=nr < nu (order of unstable part)
 * @param[out] info Error: 0=ok, 1=Schur form failed, 2=separation failed, 3=HSV failed
 */
void ab09md(const char* dico, const char* job, const char* equil,
            const char* ordsel, i32 n, i32 m, i32 p, i32* nr, f64 alpha,
            f64* a, i32 lda, f64* b, i32 ldb, f64* c, i32 ldc,
            i32* ns, f64* hsv, f64 tol,
            i32* iwork, f64* dwork, i32 ldwork, i32* iwarn, i32* info);

/**
 * @brief Singular perturbation approximation model reduction for ALPHA-stable part.
 *
 * Computes a reduced order model (Ar,Br,Cr,Dr) for an original state-space
 * representation (A,B,C,D) by using either the square-root or the balancing-free
 * square-root Singular Perturbation Approximation (SPA) model reduction method
 * for the ALPHA-stable part of the system.
 *
 * @param[in] dico System type: 'C'=continuous, 'D'=discrete
 * @param[in] job Model reduction approach: 'B'=square-root SPA, 'N'=balancing-free SPA
 * @param[in] equil Equilibration: 'S'=scale (A,B,C), 'N'=no scaling
 * @param[in] ordsel Order selection: 'F'=fixed order NR, 'A'=automatic based on TOL1
 * @param[in] n Order of original system (n >= 0)
 * @param[in] m Number of system inputs (m >= 0)
 * @param[in] p Number of system outputs (p >= 0)
 * @param[in,out] nr On entry (ordsel='F'): desired order. On exit: actual reduced order.
 * @param[in] alpha Stability boundary. Continuous: alpha <= 0, Discrete: 0 <= alpha <= 1.
 * @param[in,out] a On entry: N-by-N state matrix. On exit: NR-by-NR reduced Ar.
 * @param[in] lda Leading dimension of a (lda >= max(1,n))
 * @param[in,out] b On entry: N-by-M input matrix. On exit: NR-by-M reduced Br.
 * @param[in] ldb Leading dimension of b (ldb >= max(1,n))
 * @param[in,out] c On entry: P-by-N output matrix. On exit: P-by-NR reduced Cr.
 * @param[in] ldc Leading dimension of c (ldc >= max(1,p))
 * @param[in,out] d On entry: P-by-M feedthrough matrix. On exit: reduced Dr.
 * @param[in] ldd Leading dimension of d (ldd >= max(1,p))
 * @param[out] ns Dimension of ALPHA-stable subsystem
 * @param[out] hsv Hankel singular values of ALPHA-stable part, dimension (n)
 * @param[in] tol1 Tolerance for order selection (if ordsel='A')
 * @param[in] tol2 Tolerance for minimal realization order (tol2 <= tol1 if tol2 > 0)
 * @param[out] iwork Integer workspace, dimension (2*n). iwork[0] = nmin.
 * @param[out] dwork Real workspace, dimension (ldwork)
 * @param[in] ldwork Workspace size >= max(1, n*(2*n+max(n,m,p)+5) + n*(n+1)/2)
 * @param[out] iwarn Warning: 0=ok, 1=nr > nsmin, 2=nr < nu (order of unstable part)
 * @param[out] info Error: 0=ok, 1=Schur form failed, 2=separation failed, 3=HSV failed
 */
void ab09nd(const char* dico, const char* job, const char* equil,
            const char* ordsel, i32 n, i32 m, i32 p, i32* nr, f64 alpha,
            f64* a, i32 lda, f64* b, i32 ldb, f64* c, i32 ldc,
            f64* d, i32 ldd, i32* ns, f64* hsv, f64 tol1, f64 tol2,
            i32* iwork, f64* dwork, i32 ldwork, i32* iwarn, i32* info);

/**
 * @brief Stochastic balancing based model reduction.
 *
 * Computes a reduced order model (Ar,Br,Cr,Dr) for an original
 * state-space representation (A,B,C,D) by using the stochastic
 * balancing approach in conjunction with the square-root or
 * the balancing-free square-root Balance & Truncate (B&T) or
 * Singular Perturbation Approximation (SPA) model reduction methods
 * for the ALPHA-stable part of the system.
 *
 * @param[in] dico System type: 'C'=continuous, 'D'=discrete
 * @param[in] job Model reduction approach:
 *                'B' = square-root Balance & Truncate
 *                'F' = balancing-free square-root B&T
 *                'S' = square-root Singular Perturbation Approximation
 *                'P' = balancing-free square-root SPA
 * @param[in] equil Equilibration: 'S'=scale (A,B,C), 'N'=no scaling
 * @param[in] ordsel Order selection: 'F'=fixed order NR, 'A'=automatic based on TOL1
 * @param[in] n Order of original system (n >= 0)
 * @param[in] m Number of system inputs (m >= 0)
 * @param[in] p Number of system outputs (p >= 0, p <= m if beta = 0)
 * @param[in,out] nr On entry (ordsel='F'): desired order. On exit: actual reduced order.
 * @param[in] alpha Stability boundary. Continuous: alpha <= 0, Discrete: 0 <= alpha <= 1.
 * @param[in] beta Error weighting parameter. beta > 0 for absolute/relative balance,
 *                 beta = 0 for pure relative error method (requires rank(D) = P).
 * @param[in,out] a On entry: N-by-N state matrix. On exit: NR-by-NR reduced Ar.
 * @param[in] lda Leading dimension of a (lda >= max(1,n))
 * @param[in,out] b On entry: N-by-M input matrix. On exit: NR-by-M reduced Br.
 * @param[in] ldb Leading dimension of b (ldb >= max(1,n))
 * @param[in,out] c On entry: P-by-N output matrix. On exit: P-by-NR reduced Cr.
 * @param[in] ldc Leading dimension of c (ldc >= max(1,p))
 * @param[in,out] d On entry: P-by-M feedthrough matrix. On exit: reduced Dr.
 * @param[in] ldd Leading dimension of d (ldd >= max(1,p))
 * @param[out] ns Dimension of ALPHA-stable subsystem
 * @param[out] hsv Hankel singular values of phase system, dimension (n)
 * @param[in] tol1 Tolerance for order selection (if ordsel='A'). TOL1 < 1.
 * @param[in] tol2 Tolerance for minimal realization order. TOL2 < 1.
 * @param[out] iwork Integer workspace, dimension (2*n). iwork[0] = nmin.
 * @param[out] dwork Real workspace. dwork[0] = optimal ldwork, dwork[1] = RCOND.
 * @param[in] ldwork Workspace size >= 2*N*N + MB*(N+P) + MAX(...)
 *                   where MB = M if BETA = 0, MB = M+P if BETA > 0.
 * @param[out] bwork Logical workspace, dimension (2*n)
 * @param[out] iwarn Warning: 0=ok, 1=nr > nsmin, 2=repeated singular values, 3=nr < nu
 * @param[out] info Error indicator (0=success, 1-9 various failures)
 */
void ab09hd(const char* dico, const char* job, const char* equil,
            const char* ordsel, i32 n, i32 m, i32 p, i32* nr,
            f64 alpha, f64 beta,
            f64* a, i32 lda, f64* b, i32 ldb, f64* c, i32 ldc,
            f64* d, i32 ldd, i32* ns, f64* hsv, f64 tol1, f64 tol2,
            i32* iwork, f64* dwork, i32 ldwork, i32* bwork,
            i32* iwarn, i32* info);

/**
 * @brief Stochastic balancing model reduction of stable systems.
 *
 * Computes a reduced order model (Ar,Br,Cr,Dr) for an original stable
 * state-space representation (A,B,C,D) using the stochastic balancing
 * approach with Balance & Truncate or Singular Perturbation Approximation.
 *
 * The state dynamics matrix A must be in real Schur canonical form and
 * D must have full row rank.
 *
 * For B&T: Ar = TI * A * T,  Br = TI * B,  Cr = C * T
 * For SPA: Am = TI * A * T,  Bm = TI * B,  Cm = C * T, then SPA computed
 *
 * @param[in] dico 'C' for continuous-time, 'D' for discrete-time
 * @param[in] job 'B' = sqrt B&T, 'F' = balancing-free sqrt B&T,
 *                'S' = sqrt SPA, 'P' = balancing-free sqrt SPA
 * @param[in] ordsel 'F' = fixed order, 'A' = automatic order selection
 * @param[in] n Order of original system (n >= 0)
 * @param[in] m Number of inputs (m >= 0)
 * @param[in] p Number of outputs (m >= p >= 0)
 * @param[in,out] nr On entry with ORDSEL='F': desired order (0 <= nr <= n).
 *                   On exit: actual order of reduced model.
 * @param[in,out] a N-by-N state matrix in Schur form. On exit: NR-by-NR Ar.
 * @param[in] lda Leading dimension of A (>= max(1,n))
 * @param[in,out] b N-by-M input matrix. On exit: NR-by-M Br.
 * @param[in] ldb Leading dimension of B (>= max(1,n))
 * @param[in,out] c P-by-N output matrix. On exit: P-by-NR Cr.
 * @param[in] ldc Leading dimension of C (>= max(1,p))
 * @param[in,out] d P-by-M feedthrough matrix. On exit: Dr.
 * @param[in] ldd Leading dimension of D (>= max(1,p))
 * @param[out] hsv Hankel singular values (length n), decreasing order
 * @param[out] t N-by-NR right truncation matrix
 * @param[in] ldt Leading dimension of T (>= max(1,n))
 * @param[out] ti NR-by-N left truncation matrix
 * @param[in] ldti Leading dimension of TI (>= max(1,n))
 * @param[in] tol1 Tolerance for order selection (ORDSEL='A'). Default: n*eps.
 * @param[in] tol2 Tolerance for minimal realization. Default: n*eps.
 *                 If TOL2 > 0 and ORDSEL='A', must have TOL2 <= TOL1.
 * @param[out] iwork Integer workspace (2*n). iwork[0] = nmin on exit.
 * @param[out] dwork Real workspace. dwork[0] = optimal ldwork, dwork[1] = RCOND.
 * @param[in] ldwork Workspace size >= MAX(2, N*(MAX(N,M,P)+5),
 *                   2*N*P+MAX(P*(M+2),10*N*(N+1)))
 * @param[out] bwork Logical workspace (2*n)
 * @param[out] iwarn Warning: 0=ok, 1=nr > nmin
 * @param[out] info Error: 0=success, 1=A not stable/Schur, 2-5=Hamiltonian errors,
 *                  6=D rank deficient, 7=SVD failed
 */
void ab09hx(const char* dico, const char* job, const char* ordsel,
            i32 n, i32 m, i32 p, i32* nr,
            f64* a, i32 lda, f64* b, i32 ldb, f64* c, i32 ldc,
            f64* d, i32 ldd, f64* hsv, f64* t, i32 ldt, f64* ti, i32 ldti,
            f64 tol1, f64 tol2,
            i32* iwork, f64* dwork, i32 ldwork, i32* bwork,
            i32* iwarn, i32* info);

/**
 * @brief Compute Hankel-norm of ALPHA-stable projection.
 *
 * Computes the Hankel-norm of the ALPHA-stable projection of the
 * transfer-function matrix G of the state-space system (A,B,C).
 *
 * The routine decomposes G = G1 + G2 where G1 has only ALPHA-stable poles,
 * then returns the Hankel-norm of G1 (max Hankel singular value).
 *
 * @param[in] dico 'C' for continuous-time, 'D' for discrete-time
 * @param[in] equil 'S' to scale (A,B,C), 'N' for no scaling
 * @param[in] n Order of matrix A (n >= 0)
 * @param[in] m Number of inputs (m >= 0)
 * @param[in] p Number of outputs (p >= 0)
 * @param[in] alpha Stability boundary:
 *                  Continuous: alpha <= 0, eigenvalues with Re < alpha are stable
 *                  Discrete: 0 <= alpha <= 1, eigenvalues with |.| < alpha are stable
 * @param[in,out] a N-by-N state matrix. On exit, block diagonal Schur form
 *                  with leading NS-by-NS block having ALPHA-stable eigenvalues.
 * @param[in] lda Leading dimension of A (>= max(1,n))
 * @param[in,out] b N-by-M input matrix, transformed on exit
 * @param[in] ldb Leading dimension of B (>= max(1,n))
 * @param[in,out] c P-by-N output matrix, transformed on exit
 * @param[in] ldc Leading dimension of C (>= max(1,p))
 * @param[out] ns Dimension of ALPHA-stable subsystem
 * @param[out] hsv Hankel singular values of stable part (length N, leading NS used)
 * @param[out] dwork Workspace array (length ldwork)
 * @param[in] ldwork Workspace size >= max(1, N*(max(N,M,P)+5)+N*(N+1)/2)
 * @param[out] info Error code: 0=success, 1=Schur failed, 2=separation failed,
 *                  3=marginally stable, 4=HSV computation failed
 * @return The Hankel-norm of the ALPHA-stable projection (0 if error)
 */
f64 ab13ad(
    const char* dico,
    const char* equil,
    i32 n,
    i32 m,
    i32 p,
    f64 alpha,
    f64* a,
    i32 lda,
    f64* b,
    i32 ldb,
    f64* c,
    i32 ldc,
    i32* ns,
    f64* hsv,
    f64* dwork,
    i32 ldwork,
    i32* info);

/**
 * @brief Compute Hankel-norm of stable system (internal).
 *
 * Computes the Hankel-norm of a stable system where A is already in
 * real Schur canonical form. Used internally by AB13AD.
 *
 * @param[in] dico 'C' for continuous-time, 'D' for discrete-time
 * @param[in] n Order of matrix A (n >= 0)
 * @param[in] m Number of inputs (m >= 0)
 * @param[in] p Number of outputs (p >= 0)
 * @param[in] a N-by-N state matrix in Schur form
 * @param[in] lda Leading dimension of A
 * @param[in] b N-by-M input matrix
 * @param[in] ldb Leading dimension of B
 * @param[in] c P-by-N output matrix
 * @param[in] ldc Leading dimension of C
 * @param[out] hsv Hankel singular values (length N)
 * @param[out] dwork Workspace (length ldwork)
 * @param[in] ldwork Workspace size >= max(1, N*(max(N,M,P)+5)+N*(N+1)/2)
 * @param[out] info Error: 0=success, 1=unstable A, 2=SVD failed
 * @return Hankel-norm (max HSV), 0 on error
 */
f64 ab13ax(
    const char* dico,
    i32 n,
    i32 m,
    i32 p,
    const f64* a,
    i32 lda,
    const f64* b,
    i32 ldb,
    const f64* c,
    i32 ldc,
    f64* hsv,
    f64* dwork,
    i32 ldwork,
    i32* info);

/**
 * @brief Compute H2 or L2 norm of a transfer-function matrix.
 *
 * Computes the H2-norm (continuous) or L2-norm (discrete) of
 * transfer-function matrix G(lambda) = C*inv(lambda*I - A)*B + D.
 *
 * @param[in] dico 'C' for continuous-time, 'D' for discrete-time
 * @param[in] jobn 'H' for H-infinity controllability Gramian approach,
 *                 'L' for L2 gain approach
 * @param[in] n Order of matrix A (n >= 0)
 * @param[in] m Number of inputs (m >= 0)
 * @param[in] p Number of outputs (p >= 0)
 * @param[in,out] a N-by-N state matrix, modified on exit
 * @param[in] lda Leading dimension of A
 * @param[in,out] b N-by-M input matrix, modified on exit
 * @param[in] ldb Leading dimension of B
 * @param[in,out] c P-by-N output matrix, modified on exit
 * @param[in] ldc Leading dimension of C
 * @param[in] d P-by-M feedthrough matrix
 * @param[in] ldd Leading dimension of D
 * @param[out] nq Order of minimal realization
 * @param[in] tol Tolerance for rank determination
 * @param[out] dwork Workspace array
 * @param[in] ldwork Length of dwork
 * @param[out] iwarn Warning indicator
 * @param[out] info Exit code:
 *                  0 = success
 *                  3 = unstable A matrix
 *                  4 = Lyapunov equation failed
 *                  5 = continuous-time with D != 0
 *                  6 = system has zeros on stability boundary
 * @return The H2/L2 norm of G, or 0 on error
 */
f64 ab13bd(
    const char* dico,
    const char* jobn,
    const i32 n,
    const i32 m,
    const i32 p,
    f64* a,
    const i32 lda,
    f64* b,
    const i32 ldb,
    f64* c,
    const i32 ldc,
    f64* d,
    const i32 ldd,
    i32* nq,
    const f64 tol,
    f64* dwork,
    const i32 ldwork,
    i32* iwarn,
    i32* info);

/**
 * @brief Compute L-infinity norm of state-space system.
 *
 * Computes the L-infinity norm of a continuous-time or discrete-time
 * system in descriptor form:
 *     G(lambda) = C*inv(lambda*E - A)*B + D
 *
 * The norm is finite if and only if the matrix pair (A,E) has no
 * eigenvalue on the boundary of the stability domain.
 *
 * @param[in] dico Time domain: 'C' = continuous, 'D' = discrete
 * @param[in] jobe E matrix type: 'I' = identity, 'G' = general
 * @param[in] equil Equilibration: 'S' = scale, 'N' = no scaling
 * @param[in] jobd D matrix type: 'Z' = zero, 'D' = general
 * @param[in] n System order (n >= 0)
 * @param[in] m Number of inputs (m >= 0)
 * @param[in] p Number of outputs (p >= 0)
 * @param[in,out] fpeak On entry: frequency estimate, on exit: peak frequency
 * @param[in] a N-by-N state matrix
 * @param[in] lda Leading dimension of A (lda >= max(1,n))
 * @param[in] e N-by-N descriptor matrix (if JOBE='G')
 * @param[in] lde Leading dimension of E (lde >= 1, lde >= n if JOBE='G')
 * @param[in] b N-by-M input matrix
 * @param[in] ldb Leading dimension of B (ldb >= max(1,n))
 * @param[in] c P-by-N output matrix
 * @param[in] ldc Leading dimension of C (ldc >= max(1,p))
 * @param[in] d P-by-M feedthrough matrix (if JOBD='D')
 * @param[in] ldd Leading dimension of D (ldd >= 1, ldd >= p if JOBD='D')
 * @param[out] gpeak L-infinity norm (peak gain) encoded as [value, scale]
 * @param[in] tol Tolerance for convergence (0 <= tol < 1)
 * @param[out] iwork Integer workspace, dimension (n)
 * @param[out] dwork Real workspace
 * @param[in] ldwork Size of dwork
 * @param[out] cwork Complex workspace
 * @param[in] lcwork Size of cwork
 * @param[out] info Exit code:
 *                  0 = success
 *                  1 = E is singular
 *                  2 = QR/QZ failed to converge
 *                  3 = SVD failed to converge
 *                  4 = iteration did not converge
 */
void ab13dd(const char *dico, const char *jobe, const char *equil,
            const char *jobd, i32 n, i32 m, i32 p, f64 *fpeak,
            f64 *a, i32 lda, f64 *e, i32 lde, f64 *b, i32 ldb,
            f64 *c, i32 ldc, f64 *d, i32 ldd, f64 *gpeak, f64 tol,
            i32 *iwork, f64 *dwork, i32 ldwork, c128 *cwork, i32 lcwork,
            i32 *info);

/**
 * @brief H-infinity norm of a continuous-time stable system.
 *
 * Computes the H-infinity norm of the continuous-time stable system
 *
 *                 | A | B |
 *          G(s) = |---|---|
 *                 | C | D |
 *
 * The H-infinity norm is the peak gain of the frequency response, measured
 * by the largest singular value in the MIMO case:
 *
 *   ||G||_inf = max_omega sigma_max(G(j*omega))
 *
 * Uses the algorithm from Bruinsma & Steinbuch (1990).
 *
 * @param[in] n System order (n >= 0)
 * @param[in] m Number of inputs (m >= 0)
 * @param[in] np Number of outputs (np >= 0)
 * @param[in] a N-by-N state matrix A, dimension (lda, n)
 * @param[in] lda Leading dimension of a (lda >= max(1, n))
 * @param[in] b N-by-M input matrix B, dimension (ldb, m)
 * @param[in] ldb Leading dimension of b (ldb >= max(1, n))
 * @param[in] c NP-by-N output matrix C, dimension (ldc, n)
 * @param[in] ldc Leading dimension of c (ldc >= max(1, np))
 * @param[in] d NP-by-M feedthrough matrix D, dimension (ldd, m)
 * @param[in] ldd Leading dimension of d (ldd >= max(1, np))
 * @param[in] tol Tolerance for accuracy in determining the norm
 * @param[out] iwork Integer workspace, dimension (n)
 * @param[out] dwork Real workspace, dimension (ldwork). dwork[0] returns optimal
 *                   size, dwork[1] returns peak frequency.
 * @param[in] ldwork Real workspace size >= max(2, 4*n*n + 2*m*m + 3*m*n + m*np
 *                   + 2*(n+np)*np + 10*n + 6*max(m,np))
 * @param[out] cwork Complex workspace, dimension (lcwork). cwork[0] returns optimal.
 * @param[in] lcwork Complex workspace size >= max(1, (n+m)*(n+np) + 3*max(m,np))
 * @param[out] bwork Logical workspace, dimension (2*n)
 * @param[out] fpeak Peak frequency where norm is achieved
 * @param[out] info Error indicator:
 *                  0 = success
 *                  -i = parameter i had illegal value
 *                  1 = system is unstable
 *                  2 = tolerance too small (algorithm did not converge)
 *                  3 = eigenvalue computation failed
 *                  4 = SVD computation failed
 * @return H-infinity norm of the system (0 if error)
 */
f64 ab13cd(i32 n, i32 m, i32 np, const f64 *a, i32 lda, const f64 *b, i32 ldb,
           const f64 *c, i32 ldc, const f64 *d, i32 ldd, f64 tol,
           i32 *iwork, f64 *dwork, i32 ldwork, c128 *cwork, i32 lcwork,
           i32 *bwork, f64 *fpeak, i32 *info);

/**
 * @brief Compute transfer function gain at specific frequency.
 *
 * Computes the maximum singular value of the transfer function
 * G(lambda) = C * inv(lambda*E - A) * B + D at a given frequency.
 *
 * For continuous-time systems, lambda = j*omega.
 * For discrete-time systems, lambda = exp(j*omega).
 *
 * @param[in] dico Time domain: 'C' = continuous, 'D' = discrete
 * @param[in] jobe E matrix type: 'I' = identity, 'G' = general
 * @param[in] jobd D matrix type: 'Z' = zero, 'D' = general
 * @param[in] n System order (n >= 0)
 * @param[in] m Number of inputs (m >= 0)
 * @param[in] p Number of outputs (p >= 0)
 * @param[in] omega Frequency value
 * @param[in,out] a N-by-N state matrix (overwritten)
 * @param[in] lda Leading dimension of A (lda >= max(1,n))
 * @param[in] e N-by-N descriptor matrix (if JOBE='G')
 * @param[in] lde Leading dimension of E (lde >= 1, lde >= n if JOBE='G')
 * @param[in,out] b N-by-M input matrix (overwritten)
 * @param[in] ldb Leading dimension of B (ldb >= max(1,n))
 * @param[in] c P-by-N output matrix
 * @param[in] ldc Leading dimension of C (ldc >= max(1,p))
 * @param[in,out] d P-by-M feedthrough matrix (overwritten if JOBD='D')
 * @param[in] ldd Leading dimension of D (ldd >= 1, ldd >= p if JOBD='D')
 * @param[out] iwork Integer workspace, dimension (n)
 * @param[out] dwork Real workspace, dimension (ldwork)
 * @param[in] ldwork Size of dwork
 * @param[out] zwork Complex workspace, dimension (lzwork)
 * @param[in] lzwork Size of zwork
 * @param[out] info Exit code:
 *                  0 = success
 *                  1..n = A is singular at given frequency
 *                  n+1 = SVD failed to converge
 * @return Maximum singular value of G(lambda)
 */
f64 ab13dx(const char *dico, const char *jobe, const char *jobd,
           i32 n, i32 m, i32 p, f64 omega,
           f64 *a, i32 lda, const f64 *e, i32 lde,
           f64 *b, i32 ldb, const f64 *c, i32 ldc,
           f64 *d, i32 ldd,
           i32 *iwork, f64 *dwork, i32 ldwork,
           c128 *zwork, i32 lzwork, i32 *info);

/**
 * @brief Estimate distance to instability (complex stability radius).
 *
 * Estimates beta(A), the 2-norm distance from a real matrix A to the nearest
 * complex matrix with an eigenvalue on the imaginary axis. The estimate is:
 *   LOW <= beta(A) <= HIGH
 * where either (1 + TOL) * LOW >= HIGH, or LOW = 0 and HIGH = delta
 * (delta ~ sqrt(eps) * ||A||_F).
 *
 * If A is stable (eigenvalues in open left half-plane), beta(A) is the
 * complex stability radius.
 *
 * @param[in] n Order of matrix A (n >= 0)
 * @param[in] a N-by-N real matrix A, dimension (lda, n)
 * @param[in] lda Leading dimension of a (lda >= max(1, n))
 * @param[out] low Lower bound for beta(A)
 * @param[out] high Upper bound for beta(A)
 * @param[in] tol Accuracy tolerance. Recommended: 9 (order of magnitude).
 *                If < sqrt(eps), sqrt(eps) is used.
 * @param[out] dwork Workspace, dimension (ldwork). dwork[0] returns optimal size.
 * @param[in] ldwork Workspace size (>= max(1, 3*n*(n+1)))
 * @param[out] info 0=success, -i=param i invalid, 1=DHSEQR failed to converge
 */
void ab13ed(i32 n, f64 *a, i32 lda, f64 *low, f64 *high, f64 tol, f64 *dwork, i32 ldwork, i32 *info);

/**
 * @brief Compute complex stability radius using SVD method.
 *
 * Computes beta(A), the 2-norm distance from a real matrix A to the nearest
 * complex matrix with an eigenvalue on the imaginary axis. If A is stable
 * (eigenvalues in open left half-plane), beta(A) is the complex stability
 * radius.
 *
 * beta(A) = min_w sigma_min(A - jwI)
 *
 * Uses combined bisection method from Byers [1988] and Boyd-Balakrishnan [1990]
 * for provably reliable, quadratically convergent algorithm.
 *
 * @param[in] n Order of matrix A (n >= 0)
 * @param[in] a N-by-N real matrix A, dimension (lda, n)
 * @param[in] lda Leading dimension of a (lda >= max(1, n))
 * @param[out] beta Computed beta(A) (upper bound)
 * @param[out] omega Frequency w minimizing sigma_min(A - jwI)
 * @param[in] tol Accuracy tolerance. If < eps, eps is used.
 *                Recommended: greater than sqrt(eps).
 * @param[out] dwork Real workspace, dimension (ldwork). dwork[0] returns optimal size.
 * @param[in] ldwork Real workspace size (>= max(1, 3*n*(n+2)))
 * @param[out] cwork Complex workspace, dimension (lcwork). cwork[0] returns optimal size.
 * @param[in] lcwork Complex workspace size (>= max(1, n*(n+3)))
 * @param[out] info 0=success, -i=param i invalid, 1=failed to meet tolerance,
 *                  2=QR or SVD failed to converge
 */
void ab13fd(i32 n, const f64 *a, i32 lda, f64 *beta, f64 *omega, f64 tol,
            f64 *dwork, i32 ldwork, c128 *cwork, i32 lcwork, i32 *info);

/**
 * @brief Compute upper bound on structured singular value (mu).
 */
i32 ab13md(char fact, i32 n, c128* z, i32 ldz, i32 m,
           const i32* nblock, const i32* itype, f64* x, f64* bound,
           f64* d, f64* g, i32* iwork, f64* dwork, i32 ldwork,
           c128* zwork, i32 lzwork);

/**
 * @brief Construction of regular pencil for invariant zeros (complex case).
 *
 * To construct for a linear multivariable system described by a state-space
 * model (A,B,C,D) a regular pencil (Af - lambda*Bf) which has the invariant
 * zeros of the system as generalized eigenvalues. The routine also computes
 * the orders of the infinite zeros and the right and left Kronecker indices.
 *
 * @param[in] equil 'S' = perform balancing, 'N' = no balancing
 * @param[in] n Number of state variables (n >= 0)
 * @param[in] m Number of system inputs (m >= 0)
 * @param[in] p Number of system outputs (p >= 0)
 * @param[in] a State dynamics matrix A (n x n, column-major)
 * @param[in] lda Leading dimension of a (lda >= max(1,n))
 * @param[in] b Input/state matrix B (n x m, column-major)
 * @param[in] ldb Leading dimension of b (ldb >= max(1,n))
 * @param[in] c State/output matrix C (p x n, column-major)
 * @param[in] ldc Leading dimension of c (ldc >= max(1,p))
 * @param[in] d Direct transmission matrix D (p x m, column-major)
 * @param[in] ldd Leading dimension of d (ldd >= max(1,p))
 * @param[out] nu Number of (finite) invariant zeros
 * @param[out] rank Normal rank of the transfer function matrix
 * @param[out] dinfz Maximum degree of infinite elementary divisors
 * @param[out] nkror Number of right Kronecker indices
 * @param[out] nkrol Number of left Kronecker indices
 * @param[out] infz Infinite zero information array (size n)
 * @param[out] kronr Right Kronecker indices array (size max(n,m)+1)
 * @param[out] kronl Left Kronecker indices array (size max(n,p)+1)
 * @param[out] af Coefficient matrix Af of reduced pencil (nu x nu)
 * @param[in] ldaf Leading dimension of af (ldaf >= max(1,n+m))
 * @param[out] bf Coefficient matrix Bf of reduced pencil (nu x nu)
 * @param[in] ldbf Leading dimension of bf (ldbf >= max(1,n+p))
 * @param[in] tol Tolerance for rank decisions
 * @param[out] iwork Integer workspace array (size max(m,p))
 * @param[out] dwork Real workspace array (size max(n,2*max(p,m)))
 * @param[out] zwork Complex workspace array (size lzwork)
 * @param[in] lzwork Length of zwork (-1 for workspace query)
 * @return info Error indicator:
 *         = 0: successful exit
 *         < 0: -i means i-th argument invalid
 */
i32 slicot_ab08nz(char equil, i32 n, i32 m, i32 p, c128* a, i32 lda,
                  c128* b, i32 ldb, c128* c, i32 ldc, c128* d, i32 ldd,
                  i32* nu, i32* rank, i32* dinfz, i32* nkror, i32* nkrol,
                  i32* infz, i32* kronr, i32* kronl, c128* af, i32 ldaf,
                  c128* bf, i32 ldbf, f64 tol, i32* iwork, f64* dwork,
                  c128* zwork, i32 lzwork);

/**
 * @brief AB08MZ - Normal rank of transfer-function matrix (complex case)
 *
 * Computes the normal rank of the transfer-function matrix of a
 * complex state-space model (A,B,C,D). The routine reduces the compound matrix
 *   [ B  A ]
 *   [ D  C ]
 * to one with the same invariant zeros and with D of full row rank.
 * The normal rank is the rank of the reduced D matrix.
 *
 * @param[in] equil 'S' to balance compound matrix, 'N' for no balancing
 * @param[in] n Number of state variables (n >= 0)
 * @param[in] m Number of system inputs (m >= 0)
 * @param[in] p Number of system outputs (p >= 0)
 * @param[in] a State matrix, dimension (lda, n), complex
 * @param[in] lda Leading dimension of a (lda >= max(1, n))
 * @param[in] b Input matrix, dimension (ldb, m), complex
 * @param[in] ldb Leading dimension of b (ldb >= max(1, n))
 * @param[in] c Output matrix, dimension (ldc, n), complex
 * @param[in] ldc Leading dimension of c (ldc >= max(1, p))
 * @param[in] d Feedthrough matrix, dimension (ldd, m), complex
 * @param[in] ldd Leading dimension of d (ldd >= max(1, p))
 * @param[out] rank Normal rank of the transfer-function matrix
 * @param[in] tol Tolerance for rank decisions
 * @param[out] iwork Integer workspace, dimension (2*n + max(m,p) + 1)
 * @param[out] dwork Real workspace, dimension (2*max(m,p))
 * @param[out] zwork Complex workspace, dimension (lzwork)
 * @param[in] lzwork Workspace size (-1 for query)
 * @return info Error indicator: 0=success, <0=-i means i-th argument invalid
 */
i32 slicot_ab08mz(char equil, i32 n, i32 m, i32 p, c128* a, i32 lda,
                  c128* b, i32 ldb, c128* c, i32 ldc, c128* d, i32 ldd,
                  i32* rank, f64 tol, i32* iwork, f64* dwork, c128* zwork,
                  i32 lzwork);

/**
 * @brief AB8NXZ - Extract reduced system with same transmission zeros (complex)
 *
 * Extracts from the (N+P)-by-(M+N) system [ B  A ; D  C ] a reduced system
 * [ B'  A' ; D'  C' ] having the same transmission zeros but with D' of full row rank.
 *
 * @param[in] n Number of state variables (n >= 0)
 * @param[in] m Number of system inputs (m >= 0)
 * @param[in] p Number of system outputs (p >= 0)
 * @param[in,out] ro On entry, p for original system, max(p-m,0) for pertransposed.
 *                   On exit, the last computed rank.
 * @param[in,out] sigma On entry, 0 for original system, m for pertransposed.
 *                      On exit, the last computed value sigma.
 * @param[in] svlmax Estimate of largest singular value (svlmax >= 0)
 * @param[in,out] abcd On entry, (N+P)-by-(M+N) compound matrix [B A; D C].
 *                     On exit, (NU+MU)-by-(M+NU) reduced compound matrix.
 * @param[in] ldabcd Leading dimension of abcd (ldabcd >= max(1,n+p))
 * @param[in,out] ninfz On entry, current count of infinite zeros. On exit, total count.
 * @param[in,out] infz Array of infinite zero degrees (size n)
 * @param[in,out] kronl Left Kronecker (row) indices (size n+1)
 * @param[out] mu Row dimension of reduced system
 * @param[out] nu Column dimension of reduced system (state size)
 * @param[out] nkrol Number of left Kronecker indices
 * @param[in] tol Tolerance for rank decisions
 * @param[out] iwork Integer workspace (size max(m,p))
 * @param[out] dwork Real workspace (size 2*max(m,p))
 * @param[out] zwork Complex workspace (size lzwork)
 * @param[in] lzwork Length of zwork (-1 for workspace query)
 * @return info Error indicator (0=success, <0=-i means i-th argument invalid)
 */
i32 slicot_ab8nxz(i32 n, i32 m, i32 p, i32* ro, i32* sigma, f64 svlmax,
                  c128* abcd, i32 ldabcd, i32* ninfz, i32* infz, i32* kronl,
                  i32* mu, i32* nu, i32* nkrol, f64 tol, i32* iwork,
                  f64* dwork, c128* zwork, i32 lzwork);

/**
 * @brief AB13HD - L-infinity norm of standard/descriptor state-space system
 *
 * Computes the L-infinity norm of a proper continuous-time or causal
 * discrete-time system, either standard or in the descriptor form:
 *
 *     G(lambda) = C * (lambda*E - A)^(-1) * B + D
 *
 * @param[in] dico System type: 'C' continuous-time, 'D' discrete-time
 * @param[in] jobe E matrix type: 'I' identity, 'G' general, 'C' compressed
 * @param[in] equil Equilibration: 'S' perform scaling, 'N' no scaling
 * @param[in] jobd D matrix: 'D' present, 'Z' zero, 'F' full rank (DICO='C',JOBE='I')
 * @param[in] ckprop Check properness: 'C' check, 'N' no check
 * @param[in] reduce Reduce order: 'R' reduce, 'N' no reduction
 * @param[in] poles Use poles: 'A' all, 'P' partial
 * @param[in] n System order (n >= 0)
 * @param[in] m Number of inputs (m >= 0)
 * @param[in] p Number of outputs (p >= 0)
 * @param[in] ranke Rank of E if JOBE='C' (0 <= ranke <= n)
 * @param[in,out] fpeak On entry: frequency estimate [freq,flag]. On exit: peak frequency
 * @param[in,out] a State matrix A (n x n)
 * @param[in] lda Leading dimension of A
 * @param[in,out] e Descriptor matrix E (lde x k, k depends on JOBE)
 * @param[in] lde Leading dimension of E
 * @param[in,out] b Input matrix B (n x m)
 * @param[in] ldb Leading dimension of B
 * @param[in,out] c Output matrix C (p x n)
 * @param[in] ldc Leading dimension of C
 * @param[in] d Feedthrough matrix D (p x m)
 * @param[in] ldd Leading dimension of D
 * @param[out] nr Reduced system order
 * @param[out] gpeak L-infinity norm [norm,flag]
 * @param[in] tol Tolerances array (2 or 4 elements)
 * @param[out] iwork Integer workspace
 * @param[out] dwork Real workspace
 * @param[in] ldwork Length of dwork (-1 for query)
 * @param[out] zwork Complex workspace
 * @param[in] lzwork Length of zwork (-1 for query)
 * @param[out] bwork Logical workspace (size n)
 * @param[out] iwarn Warning indicator
 * @param[out] info Error indicator
 */
void ab13hd(const char *dico, const char *jobe, const char *equil,
            const char *jobd, const char *ckprop, const char *reduce,
            const char *poles, i32 n, i32 m, i32 p, i32 ranke, f64 *fpeak,
            f64 *a, i32 lda, f64 *e, i32 lde, f64 *b, i32 ldb,
            f64 *c, i32 ldc, f64 *d, i32 ldd, i32 *nr, f64 *gpeak, f64 *tol,
            i32 *iwork, f64 *dwork, i32 ldwork, c128 *zwork, i32 lzwork,
            bool *bwork, i32 *iwarn, i32 *info);

/**
 * @brief AB13ID - Check properness of transfer function of descriptor system
 *
 * To check whether the transfer function
 *     G(lambda) := C*(lambda*E - A)^(-1)*B
 * of a given linear time-invariant descriptor system with
 * generalized state space realization (lambda*E-A,B,C) is proper.
 *
 * @param[in] jobsys 'R' reduce system, 'N' already reduced
 * @param[in] jobeig 'A' remove all, 'I' remove infinite eigenvalues only
 * @param[in] equil 'S' scale, 'N' no scaling
 * @param[in] cksing 'C' check singularity, 'N' no check
 * @param[in] restor 'R' save/restore, 'N' no save
 * @param[in] update 'U' update matrices, 'N' no update
 * @param[in] n State dimension (n >= 0)
 * @param[in] m Number of inputs (m >= 0)
 * @param[in] p Number of outputs (p >= 0)
 * @param[in,out] a State matrix A (n x n)
 * @param[in] lda Leading dimension of A
 * @param[in,out] e Descriptor matrix E (n x n)
 * @param[in] lde Leading dimension of E
 * @param[in,out] b Input matrix B (n x max(m,p))
 * @param[in] ldb Leading dimension of B
 * @param[in,out] c Output matrix C (max(m,p) x n)
 * @param[in] ldc Leading dimension of C
 * @param[out] nr Reduced system order
 * @param[out] ranke Effective rank of Er
 * @param[in] tol Tolerances array (3 elements)
 * @param[out] iwork Integer workspace
 * @param[out] dwork Real workspace
 * @param[in] ldwork Length of dwork (-1 for query)
 * @param[out] iwarn Warning indicator
 * @param[out] info Error indicator
 * @return true if transfer function is proper, false if improper
 */
bool ab13id(const char *jobsys, const char *jobeig, const char *equil,
            const char *cksing, const char *restor, const char *update,
            i32 n, i32 m, i32 p,
            f64 *a, i32 lda, f64 *e, i32 lde, f64 *b, i32 ldb, f64 *c, i32 ldc,
            i32 *nr, i32 *ranke, const f64 *tol,
            i32 *iwork, f64 *dwork, i32 ldwork, i32 *iwarn, i32 *info);

#ifdef __cplusplus
}
#endif

#endif /* SLICOT_AB_H */
