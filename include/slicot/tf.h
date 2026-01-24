/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#ifndef SLICOT_TF_H
#define SLICOT_TF_H

#include "../slicot_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Output response sequence of discrete-time state-space system.
 *
 * Computes output sequence y(1),...,y(NY) of discrete-time state-space model
 * (A,B,C,D) given initial state x(1) and input sequence u(1),...,u(NY).
 *
 * Implements:
 *   x(k+1) = A*x(k) + B*u(k)
 *   y(k)   = C*x(k) + D*u(k)
 * for k = 1,...,NY.
 *
 * @param[in] n Order of state matrix A (n >= 0)
 * @param[in] m Number of system inputs (m >= 0)
 * @param[in] p Number of system outputs (p >= 0)
 * @param[in] ny Number of output vectors to compute (ny >= 0)
 * @param[in] a State matrix, dimension (lda,n), column-major
 * @param[in] lda Leading dimension of A (lda >= max(1,n))
 * @param[in] b Input matrix, dimension (ldb,m), column-major
 * @param[in] ldb Leading dimension of B (ldb >= max(1,n))
 * @param[in] c Output matrix, dimension (ldc,n), column-major
 * @param[in] ldc Leading dimension of C (ldc >= max(1,p))
 * @param[in] d Feedthrough matrix, dimension (ldd,m), column-major
 * @param[in] ldd Leading dimension of D (ldd >= max(1,p))
 * @param[in] u Input sequence, dimension (ldu,ny), column k is u(k)
 * @param[in] ldu Leading dimension of U (ldu >= max(1,m))
 * @param[in,out] x State vector, dimension (n). On entry: x(1). On exit: x(ny+1)
 * @param[out] y Output sequence, dimension (ldy,ny), column k is y(k)
 * @param[in] ldy Leading dimension of Y (ldy >= max(1,p))
 * @param[out] dwork Workspace array, dimension (n)
 * @param[out] info Exit code (0 = success, <0 = invalid parameter)
 */
void tf01md(const i32 n, const i32 m, const i32 p, const i32 ny,
            const f64* a, const i32 lda, const f64* b, const i32 ldb,
            const f64* c, const i32 ldc, const f64* d, const i32 ldd,
            const f64* u, const i32 ldu, f64* x, f64* y, const i32 ldy,
            f64* dwork, i32* info);

/**
 * @brief Output response sequence of discrete-time system with Hessenberg A.
 *
 * Computes output sequence y(1),...,y(NY) of discrete-time state-space model
 * (A,B,C,D) given initial state x(1) and input sequence u(1),...,u(NY),
 * where A is an N-by-N upper or lower Hessenberg matrix.
 *
 * Implements:
 *   x(k+1) = A*x(k) + B*u(k)
 *   y(k)   = C*x(k) + D*u(k)
 * for k = 1,...,NY.
 *
 * Processing time is approximately half that of TF01MD due to Hessenberg
 * structure exploitation.
 *
 * @param[in] uplo 'U' for upper Hessenberg A, 'L' for lower Hessenberg A
 * @param[in] n Order of state matrix A (n >= 0)
 * @param[in] m Number of system inputs (m >= 0)
 * @param[in] p Number of system outputs (p >= 0)
 * @param[in] ny Number of output vectors to compute (ny >= 0)
 * @param[in] a State matrix (upper or lower Hessenberg), dimension (lda,n)
 * @param[in] lda Leading dimension of A (lda >= max(1,n))
 * @param[in] b Input matrix, dimension (ldb,m), column-major
 * @param[in] ldb Leading dimension of B (ldb >= max(1,n))
 * @param[in] c Output matrix, dimension (ldc,n), column-major
 * @param[in] ldc Leading dimension of C (ldc >= max(1,p))
 * @param[in] d Feedthrough matrix, dimension (ldd,m), column-major
 * @param[in] ldd Leading dimension of D (ldd >= max(1,p))
 * @param[in] u Input sequence, dimension (ldu,ny), column k is u(k)
 * @param[in] ldu Leading dimension of U (ldu >= max(1,m))
 * @param[in,out] x State vector, dimension (n). On entry: x(1). On exit: x(ny+1)
 * @param[out] y Output sequence, dimension (ldy,ny), column k is y(k)
 * @param[in] ldy Leading dimension of Y (ldy >= max(1,p))
 * @param[out] dwork Workspace array, dimension (n)
 * @param[out] info Exit code (0 = success, <0 = invalid parameter)
 */
void tf01nd(const char* uplo, const i32 n, const i32 m, const i32 p, const i32 ny,
            const f64* a, const i32 lda, const f64* b, const i32 ldb,
            const f64* c, const i32 ldc, const f64* d, const i32 ldd,
            const f64* u, const i32 ldu, f64* x, f64* y, const i32 ldy,
            f64* dwork, i32* info);

/**
 * @brief Output sequence of linear time-invariant open-loop system.
 *
 * Computes output sequence y(1),...,y(NY) of discrete-time state-space model
 * with system matrix S = [A B; C D] given initial state x(1) and input
 * sequence u(1),...,u(NY).
 *
 * Implements: [x(k+1); y(k)] = S * [x(k); u(k)] for k = 1,...,NY
 *
 * @param[in] n Order of matrix A (n >= 0)
 * @param[in] m Number of system inputs (m >= 0)
 * @param[in] p Number of system outputs (p >= 0)
 * @param[in] ny Number of output vectors to compute (ny >= 0)
 * @param[in] s System matrix, dimension (lds,n+m), column-major
 * @param[in] lds Leading dimension of S (lds >= max(1,n+p))
 * @param[in] u Input sequence, dimension (ldu,m), row u(k) contains u(k)'
 * @param[in] ldu Leading dimension of U (ldu >= max(1,ny))
 * @param[in,out] x State vector, dimension (n). On entry: x(1). On exit: x(ny+1)
 * @param[out] y Output sequence, dimension (ldy,p), row y(k) contains y(k)'
 * @param[in] ldy Leading dimension of Y (ldy >= max(1,ny))
 * @param[out] dwork Workspace array, dimension (ldwork)
 * @param[in] ldwork Length of dwork (ldwork >= 2*n+m+p if m>0, n+p if m=0, 0 otherwise)
 * @param[out] info Exit code (0 = success, <0 = invalid parameter)
 */
void tf01mx(const i32 n, const i32 m, const i32 p, const i32 ny,
            const f64* s, const i32 lds, const f64* u, const i32 ldu,
            f64* x, f64* y, const i32 ldy, f64* dwork, const i32 ldwork,
            i32* info);

/**
 * @brief Output sequence of linear time-invariant open-loop system (variant).
 *
 * Computes output sequence y(1),...,y(NY) of discrete-time state-space model
 * (A,B,C,D) given initial state x(1) and input sequence u(1),...,u(NY).
 *
 * This routine differs from TF01MD in the way input and output trajectories
 * are stored: U is NY-by-M (rows are u(k)'), Y is NY-by-P (rows are y(k)').
 *
 * Implements:
 *   x(k+1) = A*x(k) + B*u(k)
 *   y(k)   = C*x(k) + D*u(k)
 * for k = 1,...,NY.
 *
 * @param[in] n Order of state matrix A (n >= 0)
 * @param[in] m Number of system inputs (m >= 0)
 * @param[in] p Number of system outputs (p >= 0)
 * @param[in] ny Number of output vectors to compute (ny >= 0)
 * @param[in] a State matrix, dimension (lda,n), column-major
 * @param[in] lda Leading dimension of A (lda >= max(1,n))
 * @param[in] b Input matrix, dimension (ldb,m), column-major
 * @param[in] ldb Leading dimension of B (ldb >= max(1,n))
 * @param[in] c Output matrix, dimension (ldc,n), column-major
 * @param[in] ldc Leading dimension of C (ldc >= max(1,p))
 * @param[in] d Feedthrough matrix, dimension (ldd,m), column-major
 * @param[in] ldd Leading dimension of D (ldd >= max(1,p))
 * @param[in] u Input sequence, dimension (ldu,m), row k is u(k)'
 * @param[in] ldu Leading dimension of U (ldu >= max(1,ny))
 * @param[in,out] x State vector, dimension (n). On entry: x(1). On exit: x(ny+1)
 * @param[out] y Output sequence, dimension (ldy,p), row k is y(k)'
 * @param[in] ldy Leading dimension of Y (ldy >= max(1,ny))
 * @param[out] dwork Workspace array, dimension (ldwork). dwork[0] returns optimal ldwork.
 * @param[in] ldwork Length of dwork (ldwork >= max(1,n)). If ldwork=-1, workspace query.
 * @param[out] info Exit code (0 = success, <0 = invalid parameter)
 */
void tf01my(const i32 n, const i32 m, const i32 p, const i32 ny,
            const f64* a, const i32 lda, const f64* b, const i32 ldb,
            const f64* c, const i32 ldc, const f64* d, const i32 ldd,
            const f64* u, const i32 ldu, f64* x, f64* y, const i32 ldy,
            f64* dwork, const i32 ldwork, i32* info);

/**
 * @brief Compute Markov parameters from state-space representation.
 *
 * Computes N Markov parameters M(1), M(2), ..., M(N) from system matrices
 * (A, B, C), where M(k) = C * A^(k-1) * B for k = 1, 2, ..., N.
 *
 * For linear time-invariant discrete-time system:
 *   x(k+1) = A*x(k) + B*u(k)
 *   y(k)   = C*x(k) + D*u(k)
 *
 * The transfer function G(z) = D + sum_{k=1}^inf M(k) * z^(-k),
 * where M(k) = C * A^(k-1) * B are the Markov parameters.
 *
 * @param[in] na Order of matrix A (na >= 0)
 * @param[in] nb Number of system inputs (nb >= 0)
 * @param[in] nc Number of system outputs (nc >= 0)
 * @param[in] n Number of Markov parameters to compute (n >= 0)
 * @param[in] a State matrix, dimension (lda,na), column-major
 * @param[in] lda Leading dimension of A (lda >= max(1,na))
 * @param[in] b Input matrix, dimension (ldb,nb), column-major
 * @param[in] ldb Leading dimension of B (ldb >= max(1,na))
 * @param[in] c Output matrix, dimension (ldc,na), column-major
 * @param[in] ldc Leading dimension of C (ldc >= max(1,nc))
 * @param[out] h Markov parameters, dimension (ldh,n*nb). M(k) stored in
 *               columns (k-1)*nb+1 to k*nb.
 * @param[in] ldh Leading dimension of H (ldh >= max(1,nc))
 * @param[out] dwork Workspace, dimension (ldwork)
 * @param[in] ldwork Length of dwork (ldwork >= max(1,2*na*nc))
 * @param[out] info Exit code (0 = success, <0 = invalid parameter)
 */
void tf01rd(const i32 na, const i32 nb, const i32 nc, const i32 n,
            const f64* a, const i32 lda, const f64* b, const i32 ldb,
            const f64* c, const i32 ldc, f64* h, const i32 ldh,
            f64* dwork, const i32 ldwork, i32* info);

/**
 * @brief Block Hankel expansion of multivariable parameter sequence.
 *
 * Constructs the block Hankel expansion T of a multivariable parameter
 * sequence M(1),...,M(NR+NC-1), where each parameter M(k) is an
 * NH1-by-NH2 block matrix.
 *
 * The resulting matrix T has Hankel structure:
 *   T = | M(1)   M(2)    ...  M(NC)     |
 *       | M(2)   M(3)    ...  M(NC+1)   |
 *       |  ...    ...    ...   ...      |
 *       | M(NR)  M(NR+1) ...  M(NR+NC-1)|
 *
 * @param[in] nh1 Number of rows in each parameter M(k) (nh1 >= 0)
 * @param[in] nh2 Number of columns in each parameter M(k) (nh2 >= 0)
 * @param[in] nr Number of block rows in Hankel expansion (nr >= 0)
 * @param[in] nc Number of block columns in Hankel expansion (nc >= 0)
 * @param[in] h Parameter sequence, dimension (ldh,(nr+nc-1)*nh2).
 *              M(k) is stored in columns (k-1)*nh2+1 to k*nh2.
 * @param[in] ldh Leading dimension of H (ldh >= max(1,nh1))
 * @param[out] t Block Hankel matrix, dimension (ldt,nh2*nc)
 * @param[in] ldt Leading dimension of T (ldt >= max(1,nh1*nr))
 * @param[out] info Exit code (0 = success, <0 = invalid parameter)
 */
void tf01od(const i32 nh1, const i32 nh2, const i32 nr, const i32 nc,
            const f64* h, const i32 ldh, f64* t, const i32 ldt, i32* info);

/**
 * @brief Block Toeplitz expansion of multivariable parameter sequence.
 *
 * Constructs the block Toeplitz expansion T of a multivariable parameter
 * sequence M(1),...,M(NR+NC-1), where each parameter M(k) is an
 * NH1-by-NH2 block matrix.
 *
 * The resulting matrix T has Toeplitz structure:
 *   T = | M(NC)     M(NC-1)   ...  M(1)      |
 *       | M(NC+1)   M(NC)     ...  M(2)      |
 *       |  ...       ...      ...  ...       |
 *       | M(NR+NC-1) M(NR+NC-2) ... M(NR)    |
 *
 * @param[in] nh1 Number of rows in each parameter M(k) (nh1 >= 0)
 * @param[in] nh2 Number of columns in each parameter M(k) (nh2 >= 0)
 * @param[in] nr Number of block rows in Toeplitz expansion (nr >= 0)
 * @param[in] nc Number of block columns in Toeplitz expansion (nc >= 0)
 * @param[in] h Parameter sequence, dimension (ldh,(nr+nc-1)*nh2).
 *              M(k) is stored in columns (k-1)*nh2+1 to k*nh2.
 * @param[in] ldh Leading dimension of H (ldh >= max(1,nh1))
 * @param[out] t Block Toeplitz matrix, dimension (ldt,nh2*nc)
 * @param[in] ldt Leading dimension of T (ldt >= max(1,nh1*nr))
 * @param[out] info Exit code (0 = success, <0 = invalid parameter)
 */
void tf01pd(const i32 nh1, const i32 nh2, const i32 nr, const i32 nc,
            const f64* h, const i32 ldh, f64* t, const i32 ldt, i32* info);

/**
 * @brief Markov parameters from transfer function matrix.
 *
 * Computes N Markov parameters M(1), M(2), ..., M(N) from a multivariable
 * system whose transfer function matrix G(z) is given in ARMA form.
 *
 * The (i,j)-th element of G(z) has the form:
 *           MA(1)z^{-1} + MA(2)z^{-2} + ... + MA(r)z^{-r}
 *   G_ij = ----------------------------------------------
 *          1 + AR(1)z^{-1} + AR(2)z^{-2} + ... + AR(r)z^{-r}
 *
 * where r is the order of the element, MA are numerator coefficients,
 * and AR are denominator coefficients (constant term = 1 in denominator).
 *
 * Markov parameter recurrence:
 *   M_ij(1) = MA(1)
 *   M_ij(k) = MA(k) - sum_{p=1}^{k-1} AR(p)*M_ij(k-p)  for 1 < k <= r
 *   M_ij(k+r) = -sum_{p=1}^{r} AR(p)*M_ij(k+r-p)       for k > 0
 *
 * @param[in] nc Number of outputs (rows in G(z)), nc >= 0
 * @param[in] nb Number of inputs (columns in G(z)), nb >= 0
 * @param[in] n Number of Markov parameters to compute, n >= 0
 * @param[in] iord Orders of transfer function elements, dimension (nc*nb).
 *                 Order of G_ij is iord[(i-1)*nb + j - 1] (0-based C index)
 * @param[in] ar Denominator coefficients, dimension (sum of iord).
 *               Coefficients stored row-wise by G(z) element in decreasing
 *               powers of z. Highest order term (constant) assumed = 1.
 * @param[in] ma Numerator coefficients, dimension (sum of iord).
 *               Coefficients stored row-wise by G(z) element in decreasing
 *               powers of z.
 * @param[out] h Markov parameters, dimension (ldh, n*nb).
 *               M(k) is nc-by-nb matrix stored in columns (k-1)*nb to k*nb-1.
 *               Element (i,j) of M(k) is h[i + ((k-1)*nb + j) * ldh].
 * @param[in] ldh Leading dimension of H, ldh >= max(1, nc)
 * @param[out] info Exit code (0 = success, <0 = invalid parameter)
 */
void tf01qd(const i32 nc, const i32 nb, const i32 n, const i32* iord,
            const f64* ar, const f64* ma, f64* h, const i32 ldh, i32* info);

#ifdef __cplusplus
}
#endif

#endif /* SLICOT_TF_H */
