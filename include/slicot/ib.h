/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#ifndef SLICOT_IB_H
#define SLICOT_IB_H

#include "../slicot_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief SVD system order via block Hankel.
 *
 * Computes singular value decomposition (SVD) of triangular factor R from
 * QR factorization of concatenated block Hankel matrices to determine system
 * order. Related preliminary calculations for computing system matrices are
 * also performed.
 *
 * @param[in] meth Subspace identification method:
 *                 'M' = MOESP with past inputs/outputs
 *                 'N' = N4SID algorithm
 * @param[in] jobd MOESP BD computation mode (not relevant for N4SID):
 *                 'M' = compute B,D using MOESP approach
 *                 'N' = don't compute B,D using MOESP
 * @param[in] nobr Number of block rows s (nobr > 0)
 * @param[in] m Number of system inputs (m >= 0)
 * @param[in] l Number of system outputs (l > 0)
 * @param[in,out] r DOUBLE PRECISION array, dimension (ldr, 2*(m+l)*nobr)
 *                  In: upper triangular factor R from QR factorization
 *                  Out: processed matrix S for subsequent routines
 * @param[in] ldr Leading dimension of R
 *                MOESP/JOBD='M': ldr >= max(2*(m+l)*nobr, 3*m*nobr)
 *                Otherwise: ldr >= 2*(m+l)*nobr
 * @param[out] sv Singular values array, dimension (l*nobr), descending
 * @param[in] tol Tolerance for rank estimation (N4SID only):
 *                tol > 0: lower bound for reciprocal condition number
 *                tol <= 0: use default m*n*eps
 * @param[out] iwork INTEGER array, dimension ((m+l)*nobr)
 *                   Not referenced for METH='M'
 * @param[out] dwork DOUBLE PRECISION array, dimension (ldwork)
 *                   dwork[0] = optimal ldwork
 *                   For N4SID: dwork[1], dwork[2] = reciprocal cond numbers
 * @param[in] ldwork Length of dwork:
 *                   MOESP/JOBD='M': max((2*m-1)*nobr, (m+l)*nobr, 5*l*nobr)
 *                   MOESP/JOBD='N': 5*l*nobr
 *                   N4SID: 5*(m+l)*nobr+1
 * @param[out] iwarn Warning indicator:
 *                   0 = no warning
 *                   4 = U_f rank-deficient (N4SID)
 *                   5 = r_1 rank-deficient (N4SID)
 * @param[out] info Exit code:
 *                  0 = success
 *                  -i = parameter i had illegal value
 *                  2 = SVD did not converge
 * @return info value
 */
i32 SLC_IB01ND(char meth, char jobd, i32 nobr, i32 m, i32 l,
               f64 *r, i32 ldr, f64 *sv, f64 tol,
               i32 *iwork, f64 *dwork, i32 ldwork,
               i32 *iwarn, i32 *info);

/**
 * @brief Estimate system order from Hankel singular values.
 *
 * Estimates the system order based on singular values of the relevant part
 * of the triangular factor from QR factorization of concatenated block
 * Hankel matrices.
 *
 * @param[in] ctrl Control mode:
 *                 'C' = call IB01OY for user confirmation
 *                 'N' = no confirmation
 * @param[in] nobr Number of block rows s in Hankel matrices (nobr > 0)
 * @param[in] l Number of system outputs (l > 0)
 * @param[in] sv Singular values array, dimension (l*nobr), descending order
 * @param[out] n Estimated system order
 * @param[in] tol Tolerance for order estimation:
 *                tol >= 0: n = index of last SV >= tol
 *                tol = 0: default tol = nobr * eps * sv[0]
 *                tol < 0: n = index of largest logarithmic gap
 * @param[out] iwarn Warning indicator:
 *                   0 = no warning
 *                   3 = all SVs zero, n = 0
 * @param[out] info Exit code:
 *                  = 0: success
 *                  < 0: if info = -i, parameter i had illegal value
 * @return info value
 */
i32 SLC_IB01OD(char ctrl, i32 nobr, i32 l, const f64 *sv, i32 *n,
               f64 tol, i32 *iwarn, i32 *info);

/**
 * @brief User's confirmation of the system order.
 *
 * Non-interactive version for library use. Validates parameters and allows
 * programmatic modification of the estimated system order.
 *
 * In the original Fortran version, this routine provides interactive user
 * confirmation via terminal I/O. For library use, this version validates
 * parameters and ensures N <= NMAX.
 *
 * @param[in] ns Number of singular values (ns > 0)
 * @param[in] nmax Maximum value of system order (0 <= nmax <= ns)
 * @param[in,out] n On entry: estimated system order (0 <= n <= ns)
 *                  On exit: validated order (n <= nmax)
 * @param[in] sv Singular values array, dimension (ns), descending order
 * @param[out] info Exit code:
 *                  = 0: successful exit
 *                  < 0: if info = -i, the i-th argument had an illegal value
 *
 * @note This routine is typically called by IB01OD for system order validation.
 */
i32 SLC_IB01OY(i32 ns, i32 nmax, i32 *n, const f64 *sv, i32 *info);

/**
 * @brief System identification driver - MOESP/N4SID preprocessing and order estimation.
 *
 * Preprocesses input-output data for estimating state-space matrices and finds
 * an estimate of the system order using MOESP or N4SID subspace identification.
 * This driver calls IB01MD (R factor), IB01ND (SVD), IB01OD (order estimation).
 *
 * @param[in] meth Method: 'M' = MOESP, 'N' = N4SID
 * @param[in] alg Algorithm: 'C' = Cholesky, 'F' = Fast QR, 'Q' = QR
 * @param[in] jobd MOESP B/D mode: 'M' = compute via MOESP, 'N' = don't (N4SID: not used)
 * @param[in] batch Processing: 'F' = first, 'I' = intermediate, 'L' = last, 'O' = one block
 * @param[in] conct Connection: 'C' = connected, 'N' = not connected (unused if BATCH='O')
 * @param[in] ctrl Confirmation: 'C' = user confirmation via IB01OY, 'N' = no confirmation
 * @param[in] nobr Number of block rows (nobr > 0)
 * @param[in] m Number of inputs (m >= 0)
 * @param[in] l Number of outputs (l > 0)
 * @param[in] nsmp Number of samples (nsmp >= 2*nobr for sequential,
 *                 nsmp >= 2*(m+l+1)*nobr - 1 for non-sequential)
 * @param[in] u NSMP-by-M input data, dimension (ldu,m)
 * @param[in] ldu Leading dimension of U (ldu >= nsmp if m>0, else >= 1)
 * @param[in] y NSMP-by-L output data, dimension (ldy,l)
 * @param[in] ldy Leading dimension of Y (ldy >= nsmp)
 * @param[out] n Estimated system order
 * @param[out] r Upper triangular R/S factor, dimension (ldr, 2*(m+l)*nobr)
 * @param[in] ldr Leading dimension of R (ldr >= max(2*(m+l)*nobr, 3*m*nobr) for MOESP/JOBD='M')
 * @param[out] sv Singular values, dimension (l*nobr)
 * @param[in] rcond Rank tolerance for N4SID (rcond > 0, or <= 0 for default)
 * @param[in] tol Order estimation tolerance:
 *                tol >= 0: n = last SV >= tol; tol = 0: default tol = nobr*eps*sv[0]
 *                tol < 0: n = index of largest logarithmic gap
 * @param[in,out] iwork INTEGER workspace, dimension >= max(3,(m+l)*nobr) for N4SID
 *                      For sequential: iwork[0:2] preserves state
 * @param[out] dwork DOUBLE PRECISION workspace
 * @param[in] ldwork Workspace size (see IB01MD for requirements)
 * @param[out] iwarn Warning: 0=none, 1=100 cycles, 2=fast failed, 3=all SV zero,
 *                   4=U_f rank-deficient, 5=r_1 rank-deficient
 * @param[out] info Exit code: 0=success, -i=param i invalid, 1=fast failed, 2=SVD failed
 */
void ib01ad(const char *meth, const char *alg, const char *jobd,
            const char *batch, const char *conct, const char *ctrl,
            i32 nobr, i32 m, i32 l, i32 nsmp,
            const f64 *u, i32 ldu, const f64 *y, i32 ldy,
            i32 *n, f64 *r, i32 ldr, f64 *sv, f64 rcond, f64 tol,
            i32 *iwork, f64 *dwork, i32 ldwork,
            i32 *iwarn, i32 *info);

/**
 * @brief Estimate state-space matrices from N4SID/MOESP triangular factor.
 *
 * Estimates system matrices (A,C,B,D), optionally noise covariance matrices
 * (Q,Ry,S), and Kalman gain K from the triangular factor R computed by IB01AD.
 * Supports N4SID, MOESP, or combined subspace identification methods.
 *
 * @param[in] meth Method: 'M' MOESP, 'N' N4SID, 'C' combined
 * @param[in] job Job: 'A' all matrices, 'C' A,C only, 'B' B,D only, 'D' D only
 * @param[in] jobck Covariance: 'K' Kalman gain, 'C' covariances, 'N' neither
 * @param[in] nobr Number of block rows s >= 2
 * @param[in] n System order (0 < n < nobr)
 * @param[in] m Number of inputs (m >= 0)
 * @param[in] l Number of outputs (l > 0)
 * @param[in] nsmpl Number of samples (nsmpl >= 2*(m+l)*nobr for covariances)
 * @param[in,out] r Triangular factor from IB01AD, dimension (ldr, 2*(m+l)*nobr)
 * @param[in] ldr Leading dimension of R
 * @param[out] a N-by-N state matrix, dimension (lda,n)
 * @param[in] lda Leading dimension of A
 * @param[out] c L-by-N output matrix, dimension (ldc,n)
 * @param[in] ldc Leading dimension of C
 * @param[out] b N-by-M input matrix, dimension (ldb,m)
 * @param[in] ldb Leading dimension of B
 * @param[out] d L-by-M feedthrough matrix, dimension (ldd,m)
 * @param[in] ldd Leading dimension of D
 * @param[out] q N-by-N state covariance, dimension (ldq,n)
 * @param[in] ldq Leading dimension of Q
 * @param[out] ry L-by-L output covariance, dimension (ldry,l)
 * @param[in] ldry Leading dimension of RY
 * @param[out] s N-by-L state-output cross-covariance, dimension (lds,l)
 * @param[in] lds Leading dimension of S
 * @param[out] k N-by-L Kalman gain, dimension (ldk,l)
 * @param[in] ldk Leading dimension of K
 * @param[in] tol Tolerance for rank estimation
 * @param[out] iwork INTEGER workspace
 * @param[out] dwork DOUBLE PRECISION workspace
 * @param[in] ldwork Workspace size
 * @param[out] iwarn Warning indicator
 * @param[out] info Exit code
 */
void ib01bd(const char *meth, const char *job, const char *jobck,
            i32 nobr, i32 n, i32 m, i32 l, i32 nsmpl,
            f64 *r, i32 ldr, f64 *a, i32 lda, f64 *c, i32 ldc,
            f64 *b, i32 ldb, f64 *d, i32 ldd, f64 *q, i32 ldq,
            f64 *ry, i32 ldry, f64 *s, i32 lds, f64 *k, i32 ldk,
            f64 tol, i32 *iwork, f64 *dwork, i32 ldwork, i32 *bwork,
            i32 *iwarn, i32 *info);

/**
 * @brief Upper triangular factor R of concatenated block Hankel matrices.
 *
 * Constructs the upper triangular factor R of the concatenated block
 * Hankel matrices using input-output data. Used in subspace identification
 * methods (MOESP and N4SID). Data can optionally be processed sequentially.
 *
 * For MOESP (METH='M'): H = [Uf' Up' Y']
 * For N4SID (METH='N'): H = [U' Y']
 *
 * @param[in] meth Method: 'M' = MOESP, 'N' = N4SID
 * @param[in] alg Algorithm: 'C' = Cholesky, 'F' = Fast QR, 'Q' = QR
 * @param[in] batch Processing mode: 'F' = first, 'I' = intermediate,
 *                  'L' = last, 'O' = one block only
 * @param[in] conct Connection: 'C' = connected blocks, 'N' = not connected
 * @param[in] nobr Number of block rows s in Hankel matrices (nobr > 0)
 * @param[in] m Number of system inputs (m >= 0)
 * @param[in] l Number of system outputs (l > 0)
 * @param[in] nsmp Number of samples (nsmp >= 2*nobr for sequential,
 *                 nsmp >= 2*(m+l+1)*nobr - 1 for non-sequential)
 * @param[in] u NSMP-by-M input data, dimension (ldu,m)
 * @param[in] ldu Leading dimension of U (ldu >= nsmp if m>0, else >= 1)
 * @param[in] y NSMP-by-L output data, dimension (ldy,l)
 * @param[in] ldy Leading dimension of Y (ldy >= nsmp)
 * @param[in,out] r On exit: 2*(m+l)*nobr-by-2*(m+l)*nobr upper triangular R
 *                  On entry for sequential: previous R matrix
 * @param[in] ldr Leading dimension of R (ldr >= 2*(m+l)*nobr)
 * @param[in,out] iwork INTEGER workspace, dimension >= max(3,m+l)
 *                      For sequential: iwork[0:2] preserves state between calls
 * @param[out] dwork DOUBLE PRECISION workspace
 *                   dwork[0] = optimal ldwork on exit
 * @param[in] ldwork Workspace size (use -1 for query)
 * @param[out] iwarn Warning: 0=none, 1=100 cycles exhausted, 2=fast alg failed
 * @param[out] info Exit code: 0=success, -i=param i invalid, 1=fast alg failed
 */
void ib01md(const char *meth, const char *alg, const char *batch,
            const char *conct, i32 nobr, i32 m, i32 l, i32 nsmp,
            const f64 *u, i32 ldu, const f64 *y, i32 ldy,
            f64 *r, i32 ldr, i32 *iwork, f64 *dwork, i32 ldwork,
            i32 *iwarn, i32 *info);

/**
 * @brief Fast QR factorization for block Hankel matrices.
 *
 * Constructs the upper triangular factor R of concatenated block Hankel
 * matrices using input-output data via fast QR based on displacement rank.
 * This is a helper routine called by IB01MD when ALG='F'.
 *
 * @param[in] meth Method: 'M' = MOESP, 'N' = N4SID
 * @param[in] batch Processing mode: 'F' = first, 'I' = intermediate,
 *                  'L' = last, 'O' = one block only
 * @param[in] conct Connection: 'C' = connected blocks, 'N' = not connected
 * @param[in] nobr Number of block rows s in Hankel matrices (nobr > 0)
 * @param[in] m Number of system inputs (m >= 0)
 * @param[in] l Number of system outputs (l > 0)
 * @param[in] nsmp Number of samples
 * @param[in] u NSMP-by-M input data, dimension (ldu,m)
 * @param[in] ldu Leading dimension of U
 * @param[in] y NSMP-by-L output data, dimension (ldy,l)
 * @param[in] ldy Leading dimension of Y
 * @param[out] r Upper triangular R factor, dimension (ldr,2*(m+l)*nobr)
 * @param[in] ldr Leading dimension of R (ldr >= 2*(m+l)*nobr)
 * @param[in,out] iwork INTEGER workspace
 * @param[out] dwork DOUBLE PRECISION workspace
 * @param[in] ldwork Workspace size (use -1 for query)
 * @param[out] iwarn Warning indicator
 * @param[out] info Exit code: 0=success, -i=param i invalid, 1=H'H not pos def
 */
void ib01my(const char *meth, const char *batch, const char *conct,
            i32 nobr, i32 m, i32 l, i32 nsmp,
            const f64 *u, i32 ldu, const f64 *y, i32 ldy,
            f64 *r, i32 ldr, i32 *iwork, f64 *dwork, i32 ldwork,
            i32 *iwarn, i32 *info);

/**
 * @brief Estimate system matrices from R factor (subspace identification).
 *
 * Estimates state-space matrices A, C, B, D from the R factor produced by
 * IB01MD. Optionally computes covariance matrices for Kalman gain.
 *
 * @param[in] meth Method: 'M' = MOESP, 'N' = N4SID
 * @param[in] job Matrices to compute: 'A' = all, 'C' = A,C only,
 *                'B' = B only, 'D' = B,D only
 * @param[in] jobcv Covariances: 'C' = compute, 'N' = do not compute
 * @param[in] nobr Block rows (nobr > 1)
 * @param[in] n System order (0 < n < nobr)
 * @param[in] m Number of inputs (m >= 0)
 * @param[in] l Number of outputs (l > 0)
 * @param[in] nsmpl Number of samples (nsmpl >= 2*(m+l)*nobr if jobcv='C')
 * @param[in,out] r R factor from IB01MD, dimension (ldr, 2*(m+l)*nobr)
 * @param[in] ldr Leading dimension of R (ldr >= 2*(m+l)*nobr)
 * @param[in,out] a N-by-N state matrix, dimension (lda,n)
 * @param[in] lda Leading dimension of A
 * @param[in,out] c L-by-N output matrix, dimension (ldc,n)
 * @param[in] ldc Leading dimension of C
 * @param[out] b N-by-M input matrix, dimension (ldb,m)
 * @param[in] ldb Leading dimension of B
 * @param[out] d L-by-M feedthrough matrix, dimension (ldd,m)
 * @param[in] ldd Leading dimension of D
 * @param[out] q N-by-N state covariance, dimension (ldq,n)
 * @param[in] ldq Leading dimension of Q
 * @param[out] ry L-by-L output covariance, dimension (ldry,l)
 * @param[in] ldry Leading dimension of RY
 * @param[out] s N-by-L state-output cross-covariance, dimension (lds,l)
 * @param[in] lds Leading dimension of S
 * @param[out] o L*nobr-by-N extended observability matrix, dimension (ldo,n)
 * @param[in] ldo Leading dimension of O
 * @param[in] tol Tolerance for rank estimation
 * @param[out] iwork INTEGER workspace
 * @param[out] dwork DOUBLE PRECISION workspace
 * @param[in] ldwork Workspace size
 * @param[out] iwarn Warning indicator
 * @param[out] info Exit code
 */
void ib01pd(const char *meth, const char *job, const char *jobcv,
            i32 nobr, i32 n, i32 m, i32 l, i32 nsmpl,
            f64 *r, i32 ldr, f64 *a, i32 lda, f64 *c, i32 ldc,
            f64 *b, i32 ldb, f64 *d, i32 ldd, f64 *q, i32 ldq,
            f64 *ry, i32 ldry, f64 *s, i32 lds, f64 *o, i32 ldo,
            f64 tol, i32 *iwork, f64 *dwork, i32 ldwork,
            i32 *iwarn, i32 *info);

/**
 * @brief Estimate system matrices B and D using Kronecker products.
 *
 * Builds and solves the least squares problem T*X = Kv to estimate
 * the matrices B and D of a linear time-invariant state space model,
 * using the solution X and the singular value decomposition information
 * provided by other routines.
 *
 * The matrix T is computed as a sum of Kronecker products:
 *   T = T + kron(Uf(:,(i-1)*m+1:i*m), N_i) for i = 1:s
 *
 * @param[in] job Specifies which matrices to compute:
 *                'B' = compute matrix B only, not D
 *                'D' = compute both matrices B and D
 * @param[in] nobr Number of block rows s in Hankel matrices (nobr > 1)
 * @param[in] n System order (0 < n < nobr)
 * @param[in] m Number of system inputs (m >= 0)
 * @param[in] l Number of system outputs (l > 0)
 * @param[in,out] uf Upper triangular factor of QR factorization of future
 *                   input part from IB01ND, dimension (lduf, m*nobr).
 *                   On exit: strict lower triangle set to zero.
 * @param[in] lduf Leading dimension of UF (lduf >= max(1, m*nobr))
 * @param[in] un Matrix GaL (first n columns of singular vectors Un),
 *               dimension (ldun, n). Leading l*(nobr-1)-by-n part used.
 * @param[in] ldun Leading dimension of UN (ldun >= l*(nobr-1))
 * @param[in,out] ul Matrix L, dimension (ldul, l*nobr).
 *                   On entry: leading (n+l)-by-l*nobr part contains L.
 *                   On exit if m > 0: overwritten by Q_1i matrices.
 * @param[in] ldul Leading dimension of UL (ldul >= n+l)
 * @param[in] pgal Pseudoinverse of GaL from IB01PD, dimension (ldpgal, l*(nobr-1))
 * @param[in] ldpgal Leading dimension of PGAL (ldpgal >= n)
 * @param[in] k Matrix K, dimension (ldk, m*nobr).
 *              Leading (n+l)-by-m*nobr part contains given matrix K.
 * @param[in] ldk Leading dimension of K (ldk >= n+l)
 * @param[out] r Complete orthogonal factorization details of T,
 *               dimension (ldr, m*(n+l))
 * @param[in] ldr Leading dimension of R (ldr >= max(1, (n+l)*m*nobr))
 * @param[out] x Least squares solution, dimension ((n+l)*m*nobr).
 *               First m*(n+l) elements contain solution of T*X = Kv.
 * @param[out] b System input matrix, dimension (ldb, m).
 *               Leading n-by-m part contains estimated B matrix.
 * @param[in] ldb Leading dimension of B (ldb >= n)
 * @param[out] d System input-output matrix, dimension (ldd, m).
 *               If job='D': leading l-by-m part contains estimated D.
 *               Not referenced if job='B'.
 * @param[in] ldd Leading dimension of D (ldd >= l if job='D', else >= 1)
 * @param[in] tol Tolerance for rank estimation.
 *                tol > 0: lower bound for reciprocal condition number.
 *                tol <= 0: uses default m*n*eps.
 * @param[out] iwork INTEGER workspace, dimension (m*(n+l))
 * @param[out] dwork DOUBLE PRECISION workspace, dimension (ldwork).
 *                   On exit: dwork[0] = optimal ldwork,
 *                            dwork[1] = reciprocal condition of T (if m > 0).
 * @param[in] ldwork Workspace size (ldwork >= max((n+l)^2, 4*m*(n+l)+1))
 * @param[out] iwarn Warning: 0 = none, 4 = rank-deficient coefficient matrix
 * @param[out] info Exit code: 0 = success, < 0 = -i means param i invalid
 */
void ib01px(const char *job, i32 nobr, i32 n, i32 m, i32 l,
            f64 *uf, i32 lduf, const f64 *un, i32 ldun,
            f64 *ul, i32 ldul, const f64 *pgal, i32 ldpgal,
            const f64 *k, i32 ldk, f64 *r, i32 ldr, f64 *x,
            f64 *b, i32 ldb, f64 *d, i32 ldd, f64 tol,
            i32 *iwork, f64 *dwork, i32 ldwork, i32 *iwarn, i32 *info);

/**
 * @brief Estimate system matrices B and D using structure-exploiting QR.
 *
 * Computes the triangular (QR) factor of a structured p-by-L*s matrix Q
 * and applies transformations to matrix Kexpand. Optionally estimates
 * the matrices B and D of a linear time-invariant state space model.
 *
 * Intended for speed and efficient memory use. Generally not recommended
 * for METH='N' as IB01PX can produce more accurate results.
 *
 * @param[in] meth Subspace identification method:
 *                 'M' = MOESP with past inputs/outputs
 *                 'N' = N4SID algorithm
 * @param[in] job Specifies which matrices to compute:
 *                'B' = compute matrix B only, not D
 *                'D' = compute both matrices B and D
 *                'N' = compute only R factor of Q and transformed Kexpand
 * @param[in] nobr Number of block rows s in Hankel matrices (nobr > 1)
 * @param[in] n System order (0 < n < nobr)
 * @param[in] m Number of system inputs (m >= 0)
 * @param[in] l Number of system outputs (l > 0)
 * @param[in] rankr1 Effective rank of triangular factor r1 (QR factor of GaL)
 *                   from IB01PD. 0 <= rankr1 <= n. Not used if job='N',
 *                   m=0, or meth='N'.
 * @param[in,out] ul For meth='M': l*nobr-by-l*nobr matrix Un.
 *                   For meth='N': (n+l)-by-l*nobr matrix L.
 *                   On exit: overwritten by matrix F.
 * @param[in] ldul Leading dimension of UL
 *                 (ldul >= l*nobr if meth='M', ldul >= n+l if meth='N')
 * @param[in] r1 QR factorization details of GaL from IB01PD,
 *               dimension (ldr1, n). Not used if job='N', m=0, meth='N',
 *               or meth='M' and rankr1 < n.
 * @param[in] ldr1 Leading dimension of R1
 * @param[in] tau1 Scalar factors of elementary reflectors from IB01PD,
 *                 dimension (n). Same usage conditions as r1.
 * @param[in] pgal Pseudoinverse of GaL from IB01PD, dimension (ldpgal, l*(nobr-1)).
 *                 Used if meth='N', or job != 'N' and m > 0 and meth='M'
 *                 and rankr1 < n.
 * @param[in] ldpgal Leading dimension of PGAL
 * @param[in,out] k Matrix K, dimension (ldk, m*nobr).
 *                  On entry: leading (p/s)-by-m*nobr part contains K.
 *                  On exit: transformed matrix K.
 * @param[in] ldk Leading dimension of K (ldk >= p/s)
 * @param[out] r QR factor of matrix Q, dimension (ldr, l*nobr)
 * @param[in] ldr Leading dimension of R (ldr >= l*nobr)
 * @param[out] h Updated Kexpand or least squares solution,
 *               dimension (ldh, m)
 * @param[in] ldh Leading dimension of H (ldh >= l*nobr)
 * @param[out] b System input matrix, dimension (ldb, m).
 *               If m > 0 and job='B' or 'D': leading n-by-m part
 *               contains estimated B matrix.
 * @param[in] ldb Leading dimension of B
 * @param[out] d System input-output matrix, dimension (ldd, m).
 *               If m > 0 and job='D': leading l-by-m part contains
 *               estimated D matrix.
 * @param[in] ldd Leading dimension of D
 * @param[in] tol Tolerance for rank estimation.
 *                tol > 0: lower bound for reciprocal condition number.
 *                tol <= 0: uses default m*n*eps.
 *                Not used if m=0 or job='N'.
 * @param[out] iwork INTEGER workspace (dimension l*nobr if job != 'N' and m > 0)
 * @param[out] dwork DOUBLE PRECISION workspace, dimension (ldwork).
 *                   On exit: dwork[0] = optimal ldwork,
 *                            dwork[1] = reciprocal condition of R (if job != 'N' and m > 0).
 * @param[in] ldwork Workspace size
 * @param[out] iwarn Warning: 0 = none, 4 = rank-deficient coefficient matrix
 * @param[out] info Exit code: 0 = success, < 0 = -i means param i invalid,
 *                  3 = singular upper triangular matrix found
 */
void ib01py(const char *meth, const char *job, i32 nobr, i32 n, i32 m, i32 l,
            i32 rankr1, f64 *ul, i32 ldul, const f64 *r1, i32 ldr1,
            const f64 *tau1, const f64 *pgal, i32 ldpgal,
            f64 *k, i32 ldk, f64 *r, i32 ldr, f64 *h, i32 ldh,
            f64 *b, i32 ldb, f64 *d, i32 ldd, f64 tol,
            i32 *iwork, f64 *dwork, i32 ldwork, i32 *iwarn, i32 *info);

/**
 * @brief Wiener system identification using neural networks and L-M.
 *
 * Computes parameters for approximating a Wiener system in a least-squares
 * sense using a neural network approach and Levenberg-Marquardt algorithm.
 *
 * @param[in] init 'L'=init linear only, 'S'=init nonlinear only,
 *                 'B'=init both, 'N'=use given parameters
 * @param[in] alg 'D'=direct (Cholesky), 'I'=iterative (CG)
 * @param[in] stor 'F'=full storage, 'P'=packed storage (for alg='D')
 * @param[in] nobr Number of block rows for linear system ID
 * @param[in] m Number of system inputs
 * @param[in] l Number of system outputs
 * @param[in] nsmp Number of input/output samples
 * @param[in,out] n System order (input if >=0, computed if <0)
 * @param[in] nn Number of neurons
 * @param[in] itmax1 Max iterations for nonlinear init
 * @param[in] itmax2 Max iterations for main optimization
 * @param[in] nprint Print interval (0=no print)
 * @param[in] u Input samples (nsmp x m)
 * @param[in] ldu Leading dimension of u
 * @param[in] y Output samples (nsmp x l)
 * @param[in] ldy Leading dimension of y
 * @param[in,out] x Parameter vector
 * @param[in,out] lx Length of x
 * @param[in] tol1 Tolerance for nonlinear init
 * @param[in] tol2 Tolerance for main optimization
 * @param[out] iwork Integer workspace
 * @param[out] dwork Double workspace
 * @param[in] ldwork Length of dwork
 * @param[out] iwarn Warning indicator
 * @param[out] info Exit code
 */
void ib03ad(const char *init, const char *alg, const char *stor,
            i32 nobr, i32 m, i32 l, i32 nsmp, i32 *n, i32 nn,
            i32 itmax1, i32 itmax2, i32 nprint,
            const f64 *u, i32 ldu, const f64 *y, i32 ldy,
            f64 *x, i32 *lx, f64 tol1, f64 tol2,
            i32 *iwork, f64 *dwork, i32 ldwork,
            i32 *iwarn, i32 *info);

/**
 * @brief Wiener system identification using Levenberg-Marquardt algorithm.
 *
 * Computes parameters for approximating a Wiener system consisting of a
 * linear state-space part and a static nonlinearity (neural network):
 *   x(t+1) = A*x(t) + B*u(t)       (linear state-space)
 *   z(t)   = C*x(t) + D*u(t)
 *   y(t)   = f(z(t), wb(1:L))      (nonlinear function)
 *
 * The parameter vector X = (wb(1),...,wb(L), theta) where wb(i) are neural
 * network weights and theta are linear part parameters in output normal form.
 *
 * @param[in] init Initialization mode:
 *                 'L' = initialize linear part only
 *                 'S' = initialize static nonlinearity only
 *                 'B' = initialize both parts
 *                 'N' = no initialization (use given X)
 * @param[in] nobr Block rows for MOESP/N4SID (used if INIT='L' or 'B')
 * @param[in] m Number of system inputs (m >= 0)
 * @param[in] l Number of system outputs (l >= 0, l > 0 if INIT='L' or 'B')
 * @param[in] nsmp Number of input/output samples
 * @param[in,out] n System order. If n < 0 and INIT='L' or 'B', order is estimated.
 * @param[in] nn Number of neurons for nonlinear approximation (nn >= 0)
 * @param[in] itmax1 Max iterations for nonlinear initialization (ignored if INIT='L' or 'N')
 * @param[in] itmax2 Max iterations for whole optimization (itmax2 >= 0)
 * @param[in] nprint Print control (> 0 enables iteration printing)
 * @param[in] u Input samples, dimension (ldu, m)
 * @param[in] ldu Leading dimension of U (ldu >= max(1, nsmp))
 * @param[in] y Output samples, dimension (ldy, l)
 * @param[in] ldy Leading dimension of Y (ldy >= max(1, nsmp))
 * @param[in,out] x Parameter vector, dimension (lx).
 *                  On entry: initial parameters (depending on INIT mode)
 *                  On exit: optimized parameters
 * @param[in,out] lx Length of X. On exit, may be updated if N was auto-detected.
 * @param[in] tol1 Tolerance for nonlinear initialization (tol1 < 0 uses sqrt(eps))
 * @param[in] tol2 Tolerance for whole optimization (tol2 < 0 uses sqrt(eps))
 * @param[out] iwork INTEGER workspace
 *                   On exit: iwork[0]=fcn evals, iwork[1]=jac evals,
 *                   iwork[2]=number of condition numbers in dwork
 * @param[in,out] dwork DOUBLE PRECISION workspace
 *                      On entry: dwork[0:3] = seed for random initialization
 *                      On exit: dwork[0]=opt workspace, dwork[1]=residual,
 *                      dwork[2]=iterations, dwork[3]=final Levenberg factor
 * @param[in] ldwork Workspace size
 * @param[out] iwarn Warning indicator (k*100 + j*10 + i)
 * @param[out] info Exit code (0=success, <0=invalid param, >0=algorithm error)
 */
void ib03bd(
    const char* init,
    i32 nobr, i32 m, i32 l, i32 nsmp,
    i32* n,
    i32 nn, i32 itmax1, i32 itmax2, i32 nprint,
    const f64* u, i32 ldu,
    const f64* y, i32 ldy,
    f64* x, i32* lx,
    f64 tol1, f64 tol2,
    i32* iwork, f64* dwork, i32 ldwork,
    i32* iwarn, i32* info);

/**
 * @brief Estimate initial state and system matrices B, D (driver routine).
 *
 * Estimates initial state x(0) and optionally B and D for discrete-time LTI:
 *   x(k+1) = A*x(k) + B*u(k)
 *   y(k)   = C*x(k) + D*u(k)
 *
 * Driver routine that:
 * 1. Transforms A to real Schur form via TB01WD (A = V*At*V')
 * 2. Calls IB01QD (COMUSE='C') or IB01RD for estimation
 * 3. Back-transforms results to original coordinates
 *
 * @param[in] jobx0 Initial state computation:
 *                  'X' = compute initial state x(0)
 *                  'N' = do not compute x(0), set to zero
 * @param[in] comuse How to handle B and D matrices:
 *                   'C' = compute B (and D if JOB='D')
 *                   'U' = use given B (and D if JOB='D')
 *                   'N' = do not compute/use B and D
 * @param[in] job Matrix computation extent:
 *                'B' = compute B only (D is zero)
 *                'D' = compute B and D
 * @param[in] n System order (n >= 0)
 * @param[in] m Number of inputs (m >= 0)
 * @param[in] l Number of outputs (l > 0)
 * @param[in] nsmp Number of samples
 * @param[in] a N-by-N state matrix A, dimension (lda,n)
 * @param[in] lda Leading dimension of A (lda >= max(1,n))
 * @param[in,out] b N-by-M input matrix B, dimension (ldb,m)
 *                  If COMUSE='U': on entry, given B matrix
 *                  If COMUSE='C': on exit, estimated B matrix
 * @param[in] ldb Leading dimension of B (ldb >= n if m>0, else >= 1)
 * @param[in] c L-by-N output matrix C, dimension (ldc,n)
 * @param[in] ldc Leading dimension of C (ldc >= l)
 * @param[in,out] d L-by-M feedthrough matrix D, dimension (ldd,m)
 *                  If COMUSE='U' and JOB='D': on entry, given D matrix
 *                  If COMUSE='C' and JOB='D': on exit, estimated D matrix
 * @param[in] ldd Leading dimension of D (ldd >= l if m>0 and JOB='D', else >= 1)
 * @param[in] u NSMP-by-M input data U, dimension (ldu,m)
 * @param[in] ldu Leading dimension of U (ldu >= nsmp if m>0, else >= 1)
 * @param[in] y NSMP-by-L output data Y, dimension (ldy,l)
 * @param[in] ldy Leading dimension of Y (ldy >= nsmp)
 * @param[out] x0 Estimated initial state, dimension (n)
 * @param[out] v N-by-N orthogonal transformation matrix, dimension (ldv,n)
 *               Satisfies A = V * At * V', where At is in Schur form
 * @param[in] ldv Leading dimension of V (ldv >= max(1,n))
 * @param[in] tol Tolerance for rank estimation (tol <= 0 uses default)
 * @param[out] iwork INTEGER workspace
 * @param[out] dwork DOUBLE PRECISION workspace
 *                   dwork[0] = optimal workspace
 *                   dwork[1] = reciprocal condition number
 *                   dwork[2] = rcond of U (if COMUSE='C', JOB='D', M>0)
 * @param[in] ldwork Workspace size
 * @param[out] iwarn Warning: 0=none, 4=rank-deficient, 6=A not stable
 * @param[out] info Exit code: 0=success, 1=Schur failed, -i=param i invalid
 */
void slicot_ib01cd(
    const char* jobx0, const char* comuse, const char* job,
    i32 n, i32 m, i32 l, i32 nsmp,
    const f64* a, i32 lda,
    f64* b, i32 ldb,
    const f64* c, i32 ldc,
    f64* d, i32 ldd,
    f64* u, i32 ldu,
    const f64* y, i32 ldy,
    f64* x0,
    f64* v, i32 ldv,
    f64 tol,
    i32* iwork, f64* dwork, i32 ldwork,
    i32* iwarn, i32* info
);

/**
 * @brief Estimate initial state and system matrices B, D.
 *
 * Given (A, C) and input/output trajectories, estimates the system matrices
 * B and D, and optionally the initial state x(0), for a discrete-time LTI
 * system: x(k+1) = A*x(k) + B*u(k), y(k) = C*x(k) + D*u(k).
 * Matrix A is assumed to be in real Schur form.
 *
 * @param[in] jobx0 'X' to compute initial state, 'N' if x(0) known to be zero
 * @param[in] job 'B' to compute B only (D known zero), 'D' to compute B and D
 * @param[in] n System order (n >= 0)
 * @param[in] m Number of inputs (m >= 0)
 * @param[in] l Number of outputs (l > 0)
 * @param[in] nsmp Number of samples (rows of U and Y)
 * @param[in] a N-by-N state matrix A in real Schur form, dimension (lda,n)
 * @param[in] lda Leading dimension of A (lda >= max(1,n))
 * @param[in] c L-by-N output matrix C, dimension (ldc,n)
 * @param[in] ldc Leading dimension of C (ldc >= l)
 * @param[in,out] u NSMP-by-M input data, dimension (ldu,m)
 *                  If JOB='D': on exit contains QR factorization details
 * @param[in] ldu Leading dimension of U (ldu >= max(1,nsmp) if m>0, else >= 1)
 * @param[in] y NSMP-by-L output data, dimension (ldy,l)
 * @param[in] ldy Leading dimension of Y (ldy >= max(1,nsmp))
 * @param[out] x0 Estimated initial state, dimension (n)
 * @param[out] b Estimated N-by-M input matrix B, dimension (ldb,m)
 * @param[in] ldb Leading dimension of B (ldb >= n if n>0 and m>0, else >= 1)
 * @param[out] d Estimated L-by-M direct transmission matrix D, dimension (ldd,m)
 * @param[in] ldd Leading dimension of D (ldd >= l if m>0 and JOB='D', else >= 1)
 * @param[in] tol Tolerance for rank estimation (tol <= 0 uses machine precision)
 * @param[out] iwork INTEGER workspace
 * @param[out] dwork DOUBLE PRECISION workspace
 *                   dwork[0] = optimal ldwork
 *                   dwork[1] = rcond of W2 triangular factor
 *                   dwork[2] = rcond of U triangular factor (if JOB='D' and m>0)
 * @param[in] ldwork Workspace size
 * @param[out] iwarn Warning indicator (4 = rank-deficient coefficient matrix)
 * @param[out] info Exit code (0=success, -i=param i invalid, 2=SVD failed)
 */
void slicot_ib01qd(
    const char* jobx0, const char* job,
    i32 n, i32 m, i32 l, i32 nsmp,
    const f64* a, i32 lda,
    const f64* c, i32 ldc,
    f64* u, i32 ldu,
    const f64* y, i32 ldy,
    f64* x0,
    f64* b, i32 ldb,
    f64* d, i32 ldd,
    f64 tol,
    i32* iwork, f64* dwork, i32 ldwork,
    i32* iwarn, i32* info
);

/**
 * @brief Estimate initial state for discrete-time LTI system.
 *
 * Given system matrices (A,B,C,D) and input/output trajectories, estimates
 * the initial state x(0) of the discrete-time LTI system:
 *   x(k+1) = A*x(k) + B*u(k)
 *   y(k)   = C*x(k) + D*u(k)
 *
 * Matrix A is assumed to be in real Schur form.
 *
 * @param[in] job 'Z' if D matrix is zero, 'N' if D is not zero
 * @param[in] n System order (n >= 0)
 * @param[in] m Number of inputs (m >= 0)
 * @param[in] l Number of outputs (l > 0)
 * @param[in] nsmp Number of samples (nsmp >= n)
 * @param[in] a N-by-N state matrix A in real Schur form, dimension (lda,n)
 * @param[in] lda Leading dimension of A (lda >= max(1,n))
 * @param[in] b N-by-M input matrix B, dimension (ldb,m)
 * @param[in] ldb Leading dimension of B (ldb >= n if n>0 and m>0, else >= 1)
 * @param[in] c L-by-N output matrix C, dimension (ldc,n)
 * @param[in] ldc Leading dimension of C (ldc >= l)
 * @param[in] d L-by-M direct transmission matrix D, dimension (ldd,m)
 * @param[in] ldd Leading dimension of D (ldd >= l if m>0 and job='N', else >= 1)
 * @param[in] u NSMP-by-M input data U, dimension (ldu,m)
 * @param[in] ldu Leading dimension of U (ldu >= max(1,nsmp) if m>0, else >= 1)
 * @param[in] y NSMP-by-L output data Y, dimension (ldy,l)
 * @param[in] ldy Leading dimension of Y (ldy >= max(1,nsmp))
 * @param[out] x0 Estimated initial state, dimension (n)
 * @param[in] tol Tolerance for rank estimation (tol <= 0 uses machine precision)
 * @param[out] iwork INTEGER workspace, dimension (n)
 * @param[out] dwork DOUBLE PRECISION workspace
 *                   dwork[0] = optimal ldwork
 *                   dwork[1] = reciprocal condition number of triangular factor
 * @param[in] ldwork Workspace size
 * @param[out] iwarn Warning indicator (4 = rank-deficient coefficient matrix)
 * @param[out] info Exit code (0=success, -i=param i invalid, 2=SVD failed)
 */
void slicot_ib01rd(
    const char* job,
    i32 n, i32 m, i32 l, i32 nsmp,
    const f64* a, i32 lda,
    const f64* b, i32 ldb,
    const f64* c, i32 ldc,
    const f64* d, i32 ldd,
    const f64* u, i32 ldu,
    const f64* y, i32 ldy,
    f64* x0,
    f64 tol,
    i32* iwork, f64* dwork, i32 ldwork,
    i32* iwarn, i32* info
);

#ifdef __cplusplus
}
#endif

#endif /* SLICOT_IB_H */
