/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#ifndef SLICOT_FB_H
#define SLICOT_FB_H

#include "../slicot_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Time-varying square root information Kalman filter (dense matrices).
 *
 * Calculates a combined measurement and time update of one iteration of the
 * time-varying Kalman filter using square root information filter with dense
 * matrices. The routine performs one recursion of the square root information
 * filter algorithm using Householder transformations to triangularize the
 * pre-array.
 *
 * The algorithm triangularizes:
 *   [ Q^{-1/2}_i         0       Q^{-1/2}_i Z_i    ]
 *   [ S^{-1}_i A^{-1}_i B_i  S^{-1}_i A^{-1}_i  S^{-1}_i X_i ]
 *   [ 0        R^{-1/2}_{i+1} C_{i+1}  R^{-1/2}_{i+1} Y_{i+1} ]
 *
 * to recover the updated information square roots and state estimate.
 *
 * @param[in] jobx Mode parameter:
 *                 'X' = X_{i+1} is computed and stored in X
 *                 'N' = X_{i+1} is not required
 * @param[in] multab Mode parameter for A^{-1}_i and B_i:
 *                   'P' = AINV contains A^{-1}_i, B contains A^{-1}_i B_i
 *                   'N' = AINV contains A^{-1}_i, B contains B_i
 * @param[in] multrc Mode parameter for R^{-1/2}_{i+1} and C_{i+1}:
 *                   'P' = C contains R^{-1/2}_{i+1} C_{i+1}, RINV not used
 *                   'N' = RINV contains R^{-1/2}_{i+1}, C contains C_{i+1}
 * @param[in] n State dimension (n >= 0)
 * @param[in] m Input dimension (m >= 0)
 * @param[in] p Output dimension (p >= 0)
 * @param[in,out] sinv Information square root of state covariance, dimension (ldsinv, n).
 *                     On entry: S^{-1}_i (N-by-N upper triangular).
 *                     On exit: S^{-1}_{i+1} (N-by-N upper triangular).
 *                     Only upper triangular part used/modified.
 * @param[in] ldsinv Leading dimension of sinv (ldsinv >= max(1, n))
 * @param[in] ainv Inverse of state transition matrix A^{-1}_i, dimension (ldainv, n)
 * @param[in] ldainv Leading dimension of ainv (ldainv >= max(1, n))
 * @param[in] b Input weight matrix B_i (or A^{-1}_i B_i if MULTAB='P'),
 *              dimension (ldb, m)
 * @param[in] ldb Leading dimension of b (ldb >= max(1, n))
 * @param[in] rinv Information square root of measurement noise R^{-1/2}_{i+1},
 *                 dimension (ldrinv, p). Only used if MULTRC='N'.
 *                 Only upper triangular part used. If MULTRC='P', can be dummy.
 * @param[in] ldrinv Leading dimension of rinv
 *                    (ldrinv >= max(1,p) if MULTRC='N', else ldrinv >= 1)
 * @param[in] c Output weight matrix C_{i+1} (or R^{-1/2}_{i+1} C_{i+1} if MULTRC='P'),
 *              dimension (ldc, n)
 * @param[in] ldc Leading dimension of c (ldc >= max(1, p))
 * @param[in,out] qinv Information square root of process noise Q^{-1/2}_i,
 *                     dimension (ldqinv, m). On entry and exit: M-by-M upper triangular.
 *                     Only upper triangular part used/modified.
 * @param[in] ldqinv Leading dimension of qinv (ldqinv >= max(1, m))
 * @param[in,out] x Filtered state estimate X_i, dimension (n).
 *                  On entry: X_i (filtered state at instant i).
 *                  On exit if JOBX='X' and INFO=0: X_{i+1} (filtered state at instant i+1).
 *                  On exit if JOBX='N' or INFO=1: S^{-1}_{i+1} X_{i+1}.
 * @param[in] rinvy Product R^{-1/2}_{i+1} Y_{i+1}, dimension (p)
 * @param[in] z Mean value of process noise Z_i, dimension (m)
 * @param[out] e Estimated error E_{i+1}, dimension (p)
 * @param[in] tol Tolerance for singularity test of S^{-1}_{i+1} (only if JOBX='X').
 *                If TOL > 0: used as lower bound for reciprocal condition number.
 *                If TOL <= 0: uses default TOLDEF = N*N*EPS.
 * @param[in] iwork Integer workspace, dimension (liwork).
 *                   liwork = n if JOBX='X', else liwork = 1
 * @param[in,out] dwork Real workspace, dimension (ldwork).
 *                       On exit: dwork[0] = optimal ldwork.
 *                       If JOBX='X' and INFO=0: dwork[1] = reciprocal condition number estimate.
 * @param[in] ldwork Workspace size.
 *                   If JOBX='N': ldwork >= max(1, n*(n+2*m)+3*m, (n+p)*(n+1)+2*n)
 *                   If JOBX='X': ldwork >= max(2, n*(n+2*m)+3*m, (n+p)*(n+1)+2*n, 3*n)
 * @param[out] info Error indicator:
 *                  0 = successful exit
 *                  <0 = if INFO=-i, the i-th argument had an illegal value
 *                  1 = if JOBX='X' and S^{-1}_{i+1} is singular (condition exceeds 1/TOL)
 *
 * @return void (error status returned via info parameter)
 */
void fb01sd(const char* jobx, const char* multab, const char* multrc,
            i32 n, i32 m, i32 p,
            f64* sinv, i32 ldsinv,
            const f64* ainv, i32 ldainv,
            const f64* b, i32 ldb,
            const f64* rinv, i32 ldrinv,
            const f64* c, i32 ldc,
            f64* qinv, i32 ldqinv,
            f64* x, const f64* rinvy, const f64* z, f64* e,
            f64 tol,
            i32* iwork,
            f64* dwork, i32 ldwork,
            i32* info);

/**
 * @brief Time-invariant square root information Kalman filter (controller Hessenberg form).
 *
 * Calculates a combined measurement and time update of one iteration of the
 * time-invariant Kalman filter using square root information filter with condensed
 * controller Hessenberg form. The routine performs one recursion of the square root
 * information filter algorithm using Householder transformations to triangularize
 * the pre-array, exploiting the sparsity of the controller Hessenberg form.
 *
 * The algorithm triangularizes:
 *   [ Q^{-1/2}_i         0       Q^{-1/2}_i Z_i       ]
 *   [ 0        R^{-1/2}_{i+1} C_{i+1}  R^{-1/2}_{i+1} Y_{i+1} ]
 *   [ S^{-1}_i A^{-1}_i B  S^{-1}_i A^{-1}_i  S^{-1}_i X_i ]
 *
 * to recover the updated information square roots and state estimate.
 *
 * @param[in] jobx Mode parameter:
 *                 'X' = X_{i+1} is computed and stored in X
 *                 'N' = X_{i+1} is not required
 * @param[in] multrc Mode parameter for R^{-1/2}_{i+1} and C_{i+1}:
 *                   'P' = C contains R^{-1/2}_{i+1} C_{i+1}, RINV not used
 *                   'N' = RINV contains R^{-1/2}_{i+1}, C contains C_{i+1}
 * @param[in] n State dimension (n >= 0)
 * @param[in] m Input dimension (m >= 0)
 * @param[in] p Output dimension (p >= 0)
 * @param[in,out] sinv Information square root of state covariance, dimension (ldsinv, n).
 *                     On entry: S^{-1}_i (N-by-N upper triangular).
 *                     On exit: S^{-1}_{i+1} (N-by-N upper triangular).
 *                     Only upper triangular part used/modified.
 * @param[in] ldsinv Leading dimension of sinv (ldsinv >= max(1, n))
 * @param[in] ainv Inverse of state transition matrix A^{-1}_i in controller Hessenberg form,
 *                 dimension (ldainv, n)
 * @param[in] ldainv Leading dimension of ainv (ldainv >= max(1, n))
 * @param[in] ainvb Product A^{-1}_i B in upper controller Hessenberg form,
 *                  dimension (ldainb, m)
 * @param[in] ldainb Leading dimension of ainvb (ldainb >= max(1, n))
 * @param[in] rinv Information square root of measurement noise R^{-1/2}_{i+1},
 *                 dimension (ldrinv, p). Only used if MULTRC='N'.
 *                 Only upper triangular part used. If MULTRC='P', can be dummy.
 * @param[in] ldrinv Leading dimension of rinv
 *                    (ldrinv >= max(1,p) if MULTRC='N', else ldrinv >= 1)
 * @param[in] c Output weight matrix C_{i+1} (or R^{-1/2}_{i+1} C_{i+1} if MULTRC='P'),
 *              dimension (ldc, n)
 * @param[in] ldc Leading dimension of c (ldc >= max(1, p))
 * @param[in,out] qinv Information square root of process noise Q^{-1/2}_i,
 *                     dimension (ldqinv, m). On entry and exit: M-by-M upper triangular.
 *                     On exit contains (Q_{i,ov})^{-1/2}.
 *                     Only upper triangular part used/modified.
 * @param[in] ldqinv Leading dimension of qinv (ldqinv >= max(1, m))
 * @param[in,out] x Filtered state estimate X_i, dimension (n).
 *                  On entry: X_i (filtered state at instant i).
 *                  On exit if JOBX='X' and INFO=0: X_{i+1} (filtered state at instant i+1).
 *                  On exit if JOBX='N' or INFO=1: S^{-1}_{i+1} X_{i+1}.
 * @param[in] rinvy Product R^{-1/2}_{i+1} Y_{i+1}, dimension (p)
 * @param[in] z Mean value of process noise Z_i, dimension (m)
 * @param[out] e Estimated error E_{i+1}, dimension (p)
 * @param[in] tol Tolerance for singularity test of S^{-1}_{i+1} (only if JOBX='X').
 *                If TOL > 0: used as lower bound for reciprocal condition number.
 *                If TOL <= 0: uses default TOLDEF = N*N*EPS.
 * @param[in] iwork Integer workspace, dimension (liwork).
 *                   liwork = n if JOBX='X', else liwork = 1
 * @param[in,out] dwork Real workspace, dimension (ldwork).
 *                       On exit: dwork[0] = optimal ldwork.
 *                       If JOBX='X' and INFO=0: dwork[1] = reciprocal condition number estimate.
 * @param[in] ldwork Workspace size.
 *                   If JOBX='N': ldwork >= max(1, n*(n+2*m)+3*m, (n+p)*(n+1)+n+max(n-1,m+1))
 *                   If JOBX='X': ldwork >= max(2, n*(n+2*m)+3*m, (n+p)*(n+1)+n+max(n-1,m+1), 3*n)
 * @param[out] info Error indicator:
 *                  0 = successful exit
 *                  <0 = if INFO=-i, the i-th argument had an illegal value
 *                  1 = if JOBX='X' and S^{-1}_{i+1} is singular (condition exceeds 1/TOL)
 *
 * @return void (error status returned via info parameter)
 */
void fb01td(const char* jobx, const char* multrc,
            i32 n, i32 m, i32 p,
            f64* sinv, i32 ldsinv,
            const f64* ainv, i32 ldainv,
            const f64* ainvb, i32 ldainb,
            const f64* rinv, i32 ldrinv,
            const f64* c, i32 ldc,
            f64* qinv, i32 ldqinv,
            f64* x, const f64* rinvy, const f64* z, f64* e,
            f64 tol,
            i32* iwork,
            f64* dwork, i32 ldwork,
            i32* info);

/**
 * @brief One recursion of the conventional Kalman filter.
 *
 * Computes one update of the Riccati difference equation and the Kalman
 * filter gain. The conventional Kalman filter gain used at the i-th recursion
 * step is of the form:
 *
 *     K_i = P_{i|i-1} C_i' RINOV_i^{-1}
 *
 * where RINOV_i = C_i P_{i|i-1} C_i' + R_i, and the state covariance matrix
 * P_{i|i-1} is updated by the discrete-time difference Riccati equation:
 *
 *     P_{i+1|i} = A_i (P_{i|i-1} - K_i C_i P_{i|i-1}) A_i' + B_i Q_i B_i'
 *
 * @param[in] n State dimension (n >= 0)
 * @param[in] m Input dimension (m >= 0)
 * @param[in] l Output dimension (l >= 0)
 * @param[in,out] p State covariance matrix P_{i|i-1}, dimension (ldp, n).
 *                  On entry: N-by-N upper triangular part contains P_{i|i-1}.
 *                  On exit if INFO=0: N-by-N upper triangular part contains P_{i+1|i}.
 *                  Strictly lower triangular part is not set.
 * @param[in] ldp Leading dimension of p (ldp >= max(1, n))
 * @param[in] a State transition matrix A_i, dimension (lda, n)
 * @param[in] lda Leading dimension of a (lda >= max(1, n))
 * @param[in] b Input weight matrix B_i, dimension (ldb, m)
 * @param[in] ldb Leading dimension of b (ldb >= max(1, n))
 * @param[in] c Output weight matrix C_i, dimension (ldc, n)
 * @param[in] ldc Leading dimension of c (ldc >= max(1, l))
 * @param[in,out] q Process noise covariance Q_i, dimension (ldq, m).
 *                  Diagonal elements are modified temporarily but restored on exit.
 * @param[in] ldq Leading dimension of q (ldq >= max(1, m))
 * @param[in,out] r Measurement noise covariance R_i, dimension (ldr, l).
 *                  On entry: L-by-L symmetric R_i.
 *                  On exit if INFO=0 or INFO=L+1: L-by-L upper triangular
 *                  Cholesky factor (RINOV_i)^{1/2}.
 * @param[in] ldr Leading dimension of r (ldr >= max(1, l))
 * @param[out] k Kalman filter gain K_i, dimension (ldk, l).
 *               If INFO=0: N-by-L Kalman gain matrix.
 *               If INFO>0: N-by-L product P_{i|i-1} C_i'.
 * @param[in] ldk Leading dimension of k (ldk >= max(1, n))
 * @param[in] tol Tolerance for singularity test of RINOV_i.
 *                If TOL > 0: used as lower bound for reciprocal condition number.
 *                If TOL <= 0: uses default TOLDEF = L*L*EPS.
 * @param[in] iwork Integer workspace, dimension (l)
 * @param[in,out] dwork Real workspace, dimension (ldwork).
 *                       On exit if INFO=0 or INFO=L+1: dwork[0] = reciprocal
 *                       condition number estimate of RINOV_i.
 * @param[in] ldwork Workspace size.
 *                   ldwork >= max(1, l*n+3*l, n*n, n*m)
 * @param[out] info Error indicator:
 *                  0 = successful exit
 *                  <0 = if INFO=-i, the i-th argument had an illegal value
 *                  1 <= k <= L: leading minor of order k of RINOV_i is not
 *                               positive-definite (Cholesky failed)
 *                  L+1 = RINOV_i is singular (condition number exceeds 1/TOL)
 */
void fb01vd(i32 n, i32 m, i32 l,
            f64* p, i32 ldp,
            const f64* a, i32 lda,
            const f64* b, i32 ldb,
            const f64* c, i32 ldc,
            f64* q, i32 ldq,
            f64* r, i32 ldr,
            f64* k, i32 ldk,
            f64 tol,
            i32* iwork,
            f64* dwork, i32 ldwork,
            i32* info);

/**
 * @brief Time-varying square root covariance Kalman filter (dense matrices).
 *
 * Performs one combined measurement and time update iteration of the
 * time-varying Kalman filter in square root covariance form. The algorithm
 * triangularizes the pre-array:
 *
 *   | R^{1/2}_i   C_i*S_{i-1}    0       |     | (RINOV_i)^{1/2}   0    0 |
 *   |                                    | T = |                         |
 *   |   0        A_i*S_{i-1}  B_i*Q^{1/2}_i |   |     AK_i        S_i   0 |
 *
 * where T is an orthogonal transformation and S_i is the square root (left
 * Cholesky factor) of the state covariance matrix P_{i|i-1} = S_i * S_i'.
 *
 * The Kalman gain is computed as K_i = AK_i * (RINOV_i)^{-1/2}.
 *
 * @param[in] jobk Mode parameter:
 *                 'K' = Kalman gain K_i is computed
 *                 'N' = K_i not required (AK_i returned instead)
 * @param[in] multbq Mode parameter for B_i and Q^{1/2}_i:
 *                   'P' = B contains B_i * Q^{1/2}_i, Q not used
 *                   'N' = B contains B_i, Q contains Q^{1/2}_i
 * @param[in] n State dimension (n >= 0)
 * @param[in] m Input dimension (m >= 0)
 * @param[in] p Output dimension (p >= 0)
 * @param[in,out] s State covariance square root S_{i-1}, dimension (lds, n).
 *                  On entry: N-by-N lower triangular S_{i-1}.
 *                  On exit: N-by-N lower triangular S_i.
 *                  Strict upper triangular part not referenced.
 * @param[in] lds Leading dimension of s (lds >= max(1, n))
 * @param[in] a State transition matrix A_i, dimension (lda, n)
 * @param[in] lda Leading dimension of a (lda >= max(1, n))
 * @param[in] b Input weight matrix B_i (or B_i*Q^{1/2}_i if MULTBQ='P'),
 *              dimension (ldb, m)
 * @param[in] ldb Leading dimension of b (ldb >= max(1, n))
 * @param[in] q Process noise square root Q^{1/2}_i, dimension (ldq, m).
 *              M-by-M lower triangular. Only used if MULTBQ='N'.
 *              If MULTBQ='P', can be dummy (1x1).
 * @param[in] ldq Leading dimension of q
 *                (ldq >= max(1,m) if MULTBQ='N', else ldq >= 1)
 * @param[in] c Output weight matrix C_i, dimension (ldc, n)
 * @param[in] ldc Leading dimension of c (ldc >= max(1, p))
 * @param[in,out] r Measurement noise square root R^{1/2}_i, dimension (ldr, p).
 *                  On entry: P-by-P lower triangular R^{1/2}_i.
 *                  On exit: P-by-P lower triangular (RINOV_i)^{1/2}.
 *                  Strict upper triangular part not referenced.
 * @param[in] ldr Leading dimension of r (ldr >= max(1, p))
 * @param[out] k Kalman gain K_i (if JOBK='K') or AK_i (if JOBK='N'),
 *               dimension (ldk, p). N-by-P matrix.
 * @param[in] ldk Leading dimension of k (ldk >= max(1, n))
 * @param[in] tol Tolerance for singularity test (only if JOBK='K').
 *                If TOL > 0: used as lower bound for reciprocal condition number.
 *                If TOL <= 0: uses default TOLDEF = P*P*EPS.
 * @param[in] iwork Integer workspace, dimension (liwork).
 *                  liwork = p if JOBK='K', else liwork = 1.
 * @param[in,out] dwork Real workspace, dimension (ldwork).
 *                      On exit: dwork[0] = optimal ldwork.
 *                      If JOBK='K' and INFO=0: dwork[1] = reciprocal condition number.
 * @param[in] ldwork Workspace size.
 *                   LDWORK >= MAX(1,N*(P+N)+2*P,N*(N+M+2))     if JOBK='N';
 *                   LDWORK >= MAX(2,N*(P+N)+2*P,N*(N+M+2),3*P) if JOBK='K'.
 * @param[out] info Error indicator:
 *                  0 = successful exit
 *                  <0 = if INFO=-i, the i-th argument had an illegal value
 *                  1 = JOBK='K' and (RINOV_i)^{1/2} is singular
 */
void fb01qd(const char* jobk, const char* multbq,
            i32 n, i32 m, i32 p,
            f64* s, i32 lds,
            const f64* a, i32 lda,
            const f64* b, i32 ldb,
            const f64* q, i32 ldq,
            const f64* c, i32 ldc,
            f64* r, i32 ldr,
            f64* k, i32 ldk,
            f64 tol,
            i32* iwork,
            f64* dwork, i32 ldwork,
            i32* info);

/**
 * @brief Time-invariant square root covariance Kalman filter (observer Hessenberg form).
 *
 * Calculates a combined measurement and time update of one iteration of the
 * time-invariant Kalman filter using square root covariance filter with condensed
 * observer Hessenberg form.
 *
 * The routine performs one recursion of the square root covariance filter algorithm,
 * summarized as follows:
 *
 *   | R^{1/2}_i      0        C x S_{i-1} |     | (RINOV)^{1/2}_i  0    0 |
 *   |                                     | T = |                        |
 *   | 0        B x Q^{1/2}_i  A x S_{i-1} |     |     AK_i         S_i  0 |
 *
 *         (Pre-array)                              (Post-array)
 *
 * where T is unitary and (A,C) is in lower observer Hessenberg form.
 * The triangularization is done entirely via Householder transformations
 * exploiting the zero pattern of the pre-array.
 *
 * @param[in] jobk Mode parameter:
 *                 'K' = K_i is computed and stored in array K
 *                 'N' = K_i is not required
 * @param[in] multbq Mode parameter for B_i and Q^{1/2}_i:
 *                   'P' = Array Q is not used, B contains B_i * Q^{1/2}_i
 *                   'N' = Arrays B and Q contain the matrices as described
 * @param[in] n State dimension (n >= 0)
 * @param[in] m Input dimension (m >= 0)
 * @param[in] p Output dimension (p >= 0)
 * @param[in,out] s State covariance square root S_{i-1}, dimension (lds, n).
 *                  On entry: N-by-N lower triangular S_{i-1}.
 *                  On exit: N-by-N lower triangular S_i.
 *                  Only lower triangular part used/modified.
 * @param[in] lds Leading dimension of s (lds >= max(1, n))
 * @param[in] a State transition matrix A in lower observer Hessenberg form,
 *              dimension (lda, n)
 * @param[in] lda Leading dimension of a (lda >= max(1, n))
 * @param[in] b Input weight matrix B_i (or B_i * Q^{1/2}_i if MULTBQ='P'),
 *              dimension (ldb, m)
 * @param[in] ldb Leading dimension of b (ldb >= max(1, n))
 * @param[in] q Process noise covariance square root Q^{1/2}_i,
 *              dimension (ldq, m). Only used if MULTBQ='N'.
 *              M-by-M lower triangular. If MULTBQ='P', can be dummy.
 * @param[in] ldq Leading dimension of q
 *                (ldq >= max(1,m) if MULTBQ='N', else ldq >= 1)
 * @param[in] c Output weight matrix C in lower observer Hessenberg form,
 *              dimension (ldc, n)
 * @param[in] ldc Leading dimension of c (ldc >= max(1, p))
 * @param[in,out] r Measurement noise covariance square root R^{1/2}_i,
 *                  dimension (ldr, p).
 *                  On entry: P-by-P lower triangular R^{1/2}_i.
 *                  On exit: P-by-P lower triangular (RINOV)^{1/2}_i.
 *                  Only lower triangular part used/modified.
 * @param[in] ldr Leading dimension of r (ldr >= max(1, p))
 * @param[out] k Kalman filter gain matrix, dimension (ldk, p).
 *               If JOBK='K' and INFO=0: N-by-P Kalman gain K_i.
 *               If JOBK='N' or INFO=1: N-by-P intermediate matrix AK_i.
 * @param[in] ldk Leading dimension of k (ldk >= max(1, n))
 * @param[in] tol Tolerance for singularity test of (RINOV)^{1/2}_i (only if JOBK='K').
 *                If TOL > 0: used as lower bound for reciprocal condition number.
 *                If TOL <= 0: uses default TOLDEF = P*P*EPS.
 * @param[in] iwork Integer workspace, dimension (liwork).
 *                  liwork = p if JOBK='K', else liwork = 1
 * @param[in,out] dwork Real workspace, dimension (ldwork).
 *                      On exit: dwork[0] = optimal ldwork.
 *                      If JOBK='K' and INFO=0: dwork[1] = reciprocal condition number estimate.
 * @param[in] ldwork Workspace size.
 *                   If JOBK='N': ldwork >= max(1, n*(p+n+1), n*(p+n)+2*p, n*(n+m+2))
 *                   If JOBK='K': ldwork >= max(2, n*(p+n+1), n*(p+n)+2*p, n*(n+m+2), 3*p)
 * @param[out] info Error indicator:
 *                  0 = successful exit
 *                  <0 = if INFO=-i, the i-th argument had an illegal value
 *                  1 = if JOBK='K' and (RINOV)^{1/2}_i is singular
 *                      (condition number exceeds 1/TOL)
 */
void fb01rd(const char* jobk, const char* multbq,
            i32 n, i32 m, i32 p,
            f64* s, i32 lds,
            const f64* a, i32 lda,
            const f64* b, i32 ldb,
            const f64* q, i32 ldq,
            const f64* c, i32 ldc,
            f64* r, i32 ldr,
            f64* k, i32 ldk,
            f64 tol,
            i32* iwork,
            f64* dwork, i32 ldwork,
            i32* info);

#ifdef __cplusplus
}
#endif

#endif  /* SLICOT_FB_H */
