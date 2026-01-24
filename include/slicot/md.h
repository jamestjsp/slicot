/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#ifndef SLICOT_MD_H
#define SLICOT_MD_H

#include "../slicot_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief FCN callback type for MD03AD.
 *
 * User-provided function that computes error functions and Jacobian.
 *
 * @param[in,out] iflag Operation mode:
 *                0: print intermediate results
 *                1: compute error functions e
 *                2: compute Jacobian J and J'*e
 *                3: return workspace requirements in ipar/ldj
 *                On exit: set to negative to terminate
 * @param[in] m Number of functions
 * @param[in] n Number of variables
 * @param[in,out] ipar Integer parameters (output if iflag=3)
 * @param[in] lipar Length of ipar
 * @param[in] dpar1 First parameter array
 * @param[in] ldpar1 Length/leading dimension of dpar1
 * @param[in] dpar2 Second parameter array
 * @param[in] ldpar2 Length/leading dimension of dpar2
 * @param[in] x Variables (n elements)
 * @param[in,out] nfevl Number of function evals for Jacobian (output if iflag=2)
 * @param[in,out] e Error vector (m elements, output if iflag=1)
 * @param[in,out] j Jacobian matrix (output if iflag=2)
 * @param[in,out] ldj Leading dimension of J (output if iflag=3)
 * @param[out] jte Product J'*e (n elements, output if iflag=2)
 * @param[out] dwork Workspace
 * @param[in] ldwork Length of dwork
 * @param[out] info Error indicator
 */
typedef void (*md03ad_fcn)(
    i32* iflag, i32 m, i32 n,
    i32* ipar, i32 lipar,
    f64* dpar1, i32 ldpar1,
    f64* dpar2, i32 ldpar2,
    const f64* x, i32* nfevl,
    f64* e, f64* j, i32* ldj, f64* jte,
    f64* dwork, i32 ldwork, i32* info
);

/**
 * @brief JPJ callback type for MD03AD direct method (ALG='D').
 *
 * Computes J'*J + par*I.
 *
 * @param[in] stor Storage scheme ('F'=full, 'P'=packed)
 * @param[in] uplo Triangle stored ('U'=upper, 'L'=lower)
 * @param[in] n Number of columns of J
 * @param[in] ipar Integer parameters
 * @param[in] lipar Length of ipar
 * @param[in] dpar Real parameters. dpar[0] = Levenberg parameter par
 * @param[in] ldpar Length of dpar
 * @param[in] j Jacobian matrix
 * @param[in] ldj Leading dimension of J
 * @param[out] jtj Matrix J'*J + par*I
 * @param[in] ldjtj Leading dimension of jtj
 * @param[out] dwork Workspace
 * @param[in] ldwork Length of dwork
 * @param[out] info Error indicator
 */
typedef void (*md03ad_jpj_direct)(
    const char* stor, const char* uplo,
    const i32* n, const i32* ipar, const i32* lipar,
    const f64* dpar, const i32* ldpar,
    const f64* j, const i32* ldj,
    f64* jtj, const i32* ldjtj,
    f64* dwork, const i32* ldwork, i32* info
);

/**
 * @brief JPJ callback type for MD03AD iterative method (ALG='I').
 *
 * Computes (J'*J + par*I)*x in-place.
 *
 * @param[in] n Number of columns of J
 * @param[in] ipar Integer parameters
 * @param[in] lipar Length of ipar
 * @param[in] dpar Real parameters. dpar[0] = Levenberg parameter par
 * @param[in] ldpar Length of dpar
 * @param[in] j Jacobian matrix
 * @param[in] ldj Leading dimension of J
 * @param[in,out] x Input: vector x. Output: (J'*J + par*I)*x
 * @param[in] incx Increment for x elements
 * @param[out] dwork Workspace
 * @param[in] ldwork Length of dwork
 * @param[out] info Error indicator
 */
typedef void (*md03ad_jpj_iter)(
    i32 n, i32* ipar, i32 lipar,
    f64* dpar, i32 ldpar,
    f64* j, i32 ldj,
    f64* x, i32 incx,
    f64* dwork, i32 ldwork, i32* info
);

/**
 * @brief Levenberg-Marquardt optimizer with Cholesky or conjugate gradients solver.
 *
 * Minimizes sum of squares of m nonlinear functions in n variables using
 * modified Levenberg-Marquardt algorithm. Uses either Cholesky-based direct
 * method (ALG='D') or conjugate gradients iterative method (ALG='I').
 *
 * Solves: min ||e(x)||^2 where e: R^n -> R^m
 *
 * Uses linear system (J'*J + par*I)*p = J'*e at each iteration.
 *
 * @param[in] xinit 'R'=random initialization, 'G'=use given X
 * @param[in] alg 'D'=direct (Cholesky via MB02XD), 'I'=iterative (CG via MB02WD)
 * @param[in] stor If ALG='D': 'F'=full storage, 'P'=packed storage
 * @param[in] uplo If ALG='D': 'U'=upper triangle, 'L'=lower triangle
 * @param[in] fcn User function for error functions and Jacobian
 * @param[in] jpj User function for J'*J+par*I (ALG='D') or (J'*J+par*I)*x (ALG='I')
 * @param[in] m Number of functions (m >= 0)
 * @param[in] n Number of variables (m >= n >= 0)
 * @param[in] itmax Maximum iterations (itmax >= 0)
 * @param[in] nprint Print frequency. If > 0, FCN called with iflag=0 every nprint iters
 * @param[in] ipar INTEGER array, dimension (lipar). Problem parameters
 * @param[in] lipar Length of ipar (lipar >= 5)
 * @param[in] dpar1 First parameter array for FCN
 * @param[in] ldpar1 Length/leading dimension of dpar1
 * @param[in] dpar2 Second parameter array for FCN
 * @param[in] ldpar2 Length/leading dimension of dpar2
 * @param[in,out] x Solution vector, dimension (n).
 *                  Input: initial guess (if xinit='G')
 *                  Output: solution minimizing sum of squares
 * @param[out] nfev Number of FCN calls with iflag=1
 * @param[out] njev Number of FCN calls with iflag=2
 * @param[in] tol Relative error tolerance (tol < 0 uses sqrt(eps))
 * @param[in] cgtol CG tolerance if ALG='I' (cgtol <= 0 uses sqrt(eps))
 * @param[out] dwork Workspace, dimension (ldwork).
 *                   On exit: dwork[0]=optimal ldwork, dwork[1]=residual norm,
 *                   dwork[2]=iterations, dwork[3]=CG iterations, dwork[4]=final par
 * @param[in] ldwork Length of dwork
 * @param[out] iwarn Warning indicator:
 *                   <0: user set iflag=iwarn in FCN
 *                   0: no warning
 *                   1: did not converge in itmax iterations
 *                   2: CG did not finish in 3*n iterations (ALG='I')
 *                   3: gradient nearly orthogonal to columns of J
 *                   4: tol too small
 * @param[out] info Exit code:
 *                  0: success
 *                  <0: invalid parameter -info
 *                  1: FCN returned info != 0 for iflag=1
 *                  2: FCN returned info != 0 for iflag=2
 *                  3: MB02XD or MB02WD returned info != 0
 */
void md03ad(
    const char* xinit,
    const char* alg,
    const char* stor,
    const char* uplo,
    md03ad_fcn fcn,
    md03ad_jpj_direct jpj,
    i32 m,
    i32 n,
    i32 itmax,
    i32 nprint,
    i32* ipar,
    i32 lipar,
    f64* dpar1,
    i32 ldpar1,
    f64* dpar2,
    i32 ldpar2,
    f64* x,
    i32* nfev,
    i32* njev,
    f64 tol,
    f64 cgtol,
    f64* dwork,
    i32 ldwork,
    i32* iwarn,
    i32* info
);

/**
 * @brief QR factorization with column pivoting for Levenberg-Marquardt.
 *
 * This routine is an interface to SLICOT Library routine MD03BX.
 *
 * @param[in] n Number of columns of Jacobian matrix J.
 * @param[in] ipar Integer parameters. ipar[0] must contain M (rows of J).
 * @param[in] lipar Length of ipar.
 * @param[in] fnorm Euclidean norm of error vector e.
 * @param[in,out] j Jacobian matrix (M x N). On exit, upper triangular R.
 * @param[in,out] ldj Leading dimension of J.
 * @param[in,out] e Error vector (M). On exit, Q'*e.
 * @param[out] jnorms Euclidean norms of columns of J.
 * @param[out] gnorm 1-norm of scaled gradient.
 * @param[out] ipvt Permutation matrix P.
 * @param[out] dwork Workspace.
 * @param[in] ldwork Length of dwork.
 * @param[out] info Exit code.
 */
void md03ba(i32 n, const i32 *ipar, i32 lipar, f64 fnorm, f64 *j, i32 *ldj, 
            f64 *e, f64 *jnorms, f64 *gnorm, i32 *ipvt, f64 *dwork, 
            i32 ldwork, i32 *info);

/**
 * @brief Compute Levenberg-Marquardt parameter for compressed Jacobian.
 *
 * This routine is an interface to SLICOT Library routine MD03BY.
 *
 * @param[in] cond Condition estimation mode ('E', 'N', 'U').
 * @param[in] n Order of matrix R.
 * @param[in] ipar Integer parameters (unused, for compatibility).
 * @param[in] lipar Length of ipar.
 * @param[in,out] r Upper triangular matrix R.
 * @param[in] ldr Leading dimension of R.
 * @param[in] ipvt Permutation matrix P.
 * @param[in] diag Diagonal scaling matrix D.
 * @param[in] qtb First n elements of Q'*b.
 * @param[in] delta Trust region radius.
 * @param[in,out] par Levenberg-Marquardt parameter.
 * @param[in,out] ranks Numerical rank.
 * @param[out] x Least squares solution.
 * @param[out] rx Residual -R*P'*x.
 * @param[in] tol Tolerance for rank estimation.
 * @param[out] dwork Workspace.
 * @param[in] ldwork Length of dwork.
 * @param[out] info Exit code.
 */
void md03bb(const char *cond, i32 n, const i32 *ipar, i32 lipar, f64 *r, 
            i32 ldr, const i32 *ipvt, const f64 *diag, const f64 *qtb, 
            f64 delta, f64 *par, i32 *ranks, f64 *x, f64 *rx, f64 tol, 
            f64 *dwork, i32 ldwork, i32 *info);

/**
 * @brief Levenberg-Marquardt nonlinear least squares optimizer.
 *
 * Minimize sum of squares of m nonlinear functions in n variables
 * using modified Levenberg-Marquardt algorithm with trust region.
 * Requires user-provided FCN (function/Jacobian), QRFACT (QR factorization),
 * and LMPARM (L-M parameter computation) subroutines.
 *
 * @param[in] xinit 'R'=random initialization, 'G'=use given X
 * @param[in] scale 'I'=internal scaling, 'S'=use specified DIAG
 * @param[in] cond 'E'=use condition estimation, 'N'=check diagonal only
 * @param[in] fcn Function pointer for error functions and Jacobian
 * @param[in] qrfact Function pointer for QR factorization with pivoting
 * @param[in] lmparm Function pointer for Levenberg-Marquardt parameter
 * @param[in] m Number of functions (m >= 0)
 * @param[in] n Number of variables (m >= n >= 0)
 * @param[in] itmax Maximum iterations (itmax >= 0)
 * @param[in] factor Initial step bound factor (factor > 0, typically 100)
 * @param[in] nprint Print frequency (IFLAG=0 calls). If <= 0, no printing
 * @param[in] ipar INTEGER array, dimension (lipar). Problem parameters
 * @param[in] lipar Length of ipar (lipar >= 5)
 * @param[in] dpar1 DOUBLE PRECISION array. First parameter set
 * @param[in] ldpar1 Leading dimension/length of dpar1
 * @param[in] dpar2 DOUBLE PRECISION array. Second parameter set
 * @param[in] ldpar2 Leading dimension/length of dpar2
 * @param[in,out] x DOUBLE PRECISION array, dimension (n)
 *                  Input: initial guess (if xinit='G')
 *                  Output: solution minimizing sum of squares
 * @param[in,out] diag DOUBLE PRECISION array, dimension (n)
 *                     Input: scaling factors (if scale='S')
 *                     Output: final scaling factors used
 * @param[out] nfev Number of function evaluations (IFLAG=1)
 * @param[out] njev Number of Jacobian evaluations (IFLAG=2)
 * @param[in] ftol Relative error tolerance for sum of squares
 *                 (ftol < 0 uses sqrt(eps))
 * @param[in] xtol Relative error tolerance for solution
 *                 (xtol < 0 uses sqrt(eps))
 * @param[in] gtol Orthogonality tolerance between e and J columns
 *                 (gtol < 0 uses eps)
 * @param[in] tol Tolerance for rank determination if cond='E'
 *                (tol <= 0 uses n*eps)
 * @param[out] iwork INTEGER array, dimension (n+r)
 *                   iwork[0:n-1]: permutation defining J*P = Q*R
 *                   iwork[n:n+r-1]: ranks of submatrices
 * @param[out] dwork DOUBLE PRECISION array, dimension (ldwork)
 *                   dwork[0]: optimal ldwork
 *                   dwork[1]: final residual norm
 *                   dwork[2]: iterations performed
 *                   dwork[3]: final Levenberg-Marquardt parameter
 * @param[in] ldwork Length of dwork (see MD03BD documentation)
 * @param[out] iwarn Warning indicator
 *                   <0: user set IFLAG=iwarn
 *                   1: both actual/predicted reductions <= ftol
 *                   2: relative error between iterates <= xtol
 *                   3: conditions 1 and 2 both hold
 *                   4: cosine(e,J) <= gtol
 *                   5: iterations reached itmax
 *                   6: ftol too small
 *                   7: xtol too small
 *                   8: gtol too small
 * @param[out] info Exit code
 *                  0: success
 *                  <0: invalid parameter -info
 *                  1: FCN returned info != 0 for IFLAG=1
 *                  2: FCN returned info != 0 for IFLAG=2
 *                  3: QRFACT returned info != 0
 *                  4: LMPARM returned info != 0
 */
void md03bd(
    const char* xinit,
    const char* scale,
    const char* cond,
    void (*fcn)(i32*, i32, i32, i32*, i32, const f64*, i32, const f64*, i32,
                const f64*, i32*, f64*, f64*, i32*, f64*, i32, i32*),
    void (*qrfact)(i32, const i32*, i32, f64, f64*, i32*, f64*, f64*, f64*,
                   i32*, f64*, i32, i32*),
    void (*lmparm)(const char*, i32, const i32*, i32, f64*, i32, const i32*,
                   const f64*, const f64*, f64, f64*, i32*, f64*, f64*, f64,
                   f64*, i32, i32*),
    i32 m,
    i32 n,
    i32 itmax,
    f64 factor,
    i32 nprint,
    i32* ipar,
    i32 lipar,
    const f64* dpar1,
    i32 ldpar1,
    const f64* dpar2,
    i32 ldpar2,
    f64* x,
    f64* diag,
    i32* nfev,
    i32* njev,
    f64 ftol,
    f64 xtol,
    f64 gtol,
    f64 tol,
    i32* iwork,
    f64* dwork,
    i32 ldwork,
    i32* iwarn,
    i32* info
);

/**
 * @brief QR factorization with column pivoting for Levenberg-Marquardt
 *
 * Computes QR factorization with column pivoting of m-by-n matrix J (m >= n):
 * J*P = Q*R, where Q has orthogonal columns, P is permutation, R is upper
 * triangular with diagonal elements of nonincreasing magnitude.
 * Applies Q' to error vector e in-place.
 *
 * @param[in] m Number of rows of Jacobian matrix J (m >= 0)
 * @param[in] n Number of columns of J (m >= n >= 0)
 * @param[in] fnorm Euclidean norm of error vector e (fnorm >= 0)
 * @param[in,out] j Jacobian matrix, dimension (ldj, n)
 *                  In: m-by-n Jacobian matrix
 *                  Out: n-by-n upper triangular R with ldj=n
 * @param[in,out] ldj Leading dimension of J
 *                    In: ldj >= max(1,m)
 *                    Out: ldj = max(1,n)
 * @param[in,out] e Error vector, dimension (m)
 *                  In: Error vector e
 *                  Out: Transformed vector Q'*e
 * @param[out] jnorms Column norms of J (original order), dimension (n)
 * @param[out] gnorm 1-norm of scaled gradient J'*Q'*e/fnorm
 *                   (each element divided by jnorms)
 * @param[out] ipvt Permutation indices, dimension (n)
 *                  Column j of P is column ipvt[j] of identity
 * @param[out] dwork Workspace, dimension (ldwork)
 *                   dwork[0] returns optimal ldwork
 * @param[in] ldwork Workspace size
 *                   ldwork >= 1 if n=0 or m=1
 *                   ldwork >= 4*n+1 if n>1
 * @param[out] info Exit code (0=success, <0=invalid parameter)
 */
void md03bx(
    i32 m, i32 n, f64 fnorm,
    f64* j, i32* ldj, f64* e,
    f64* jnorms, f64* gnorm, i32* ipvt,
    f64* dwork, i32 ldwork, i32* info
);

/**
 * @brief Compute Levenberg-Marquardt parameter for trust region subproblem.
 *
 * Determines parameter PAR such that if x solves the system
 *   A*x = b, sqrt(PAR)*D*x = 0
 * in the least squares sense, then ||D*x|| satisfies the trust region constraint:
 *   either PAR=0 and (||D*x|| - DELTA) <= 0.1*DELTA,
 *   or PAR>0 and abs(||D*x|| - DELTA) <= 0.1*DELTA.
 *
 * Assumes QR factorization A*P = Q*R is available (R, IPVT, Q'*b).
 * Provides upper triangular S such that P'*(A'*A + PAR*D*D)*P = S'*S.
 *
 * @param[in] cond Condition estimation mode:
 *                 'E' = estimate condition of R and S
 *                 'N' = check diagonal entries for zeros only
 *                 'U' = use rank already in RANK parameter
 * @param[in] n Order of matrix R (n >= 0)
 * @param[in,out] r Upper triangular matrix, dimension (ldr,n)
 *                  In: QR factor from A*P=Q*R
 *                  Out: strict lower triangle contains S' (transposed)
 * @param[in] ldr Leading dimension of R (ldr >= max(1,n))
 * @param[in] ipvt Permutation from QR, dimension (n)
 *                 Column j of P is column IPVT(j) of identity
 * @param[in] diag Diagonal scaling matrix D, dimension (n)
 *                 All elements must be nonzero
 * @param[in] qtb First n elements of Q'*b, dimension (n)
 * @param[in] delta Trust region radius (delta > 0)
 * @param[in,out] par Levenberg-Marquardt parameter
 *                    In: initial estimate (par >= 0)
 *                    Out: final estimate
 * @param[in,out] rank Numerical rank
 *                     In: rank of R if COND='U'
 *                     Out: rank of S
 * @param[out] x Least squares solution, dimension (n)
 * @param[out] rx Residual -R*P'*x, dimension (n)
 * @param[in] tol Tolerance for rank estimation if COND='E'
 *                If tol <= 0, use n*eps (machine precision)
 *                Not used if COND='N' or 'U'
 * @param[out] dwork Workspace, dimension (ldwork)
 *                   First n elements contain diagonal of S on exit
 * @param[in] ldwork Workspace size
 *                   ldwork >= 4*n if COND='E'
 *                   ldwork >= 2*n if COND='N' or 'U'
 * @param[out] info Exit code: 0=success, <0=invalid parameter
 */
void md03by(
    const char* cond,
    const i32 n,
    f64* r,
    const i32 ldr,
    const i32* ipvt,
    const f64* diag,
    const f64* qtb,
    const f64 delta,
    f64* par,
    i32* rank,
    f64* x,
    f64* rx,
    const f64 tol,
    f64* dwork,
    const i32 ldwork,
    i32* info
);

/**
 * @brief MD03BF - FCN callback for Kowalik-Osborne nonlinear least squares
 *
 * Example FCN routine for MD03BD implementing the Kowalik-Osborne test
 * function from MINPACK with M=15 observations and N=3 parameters.
 *
 * @param[in,out] iflag Flag controlling operation:
 *                      1: compute error function E
 *                      2: compute Jacobian J
 *                      3: return workspace requirements
 * @param[in] m Number of error function values (15)
 * @param[in] n Number of parameters (3)
 * @param[in,out] ipar Integer parameters (workspace sizes on iflag=3)
 * @param[in] lipar Length of ipar
 * @param[in] dpar1 Not used
 * @param[in] ldpar1 Not used
 * @param[in] dpar2 Not used
 * @param[in] ldpar2 Not used
 * @param[in] x Parameter vector, dimension (n)
 * @param[out] nfevl Function evaluations (set to 0 on iflag=2)
 * @param[out] e Error function values, dimension (m)
 * @param[out] j Jacobian matrix, dimension (ldj, n)
 * @param[in,out] ldj Leading dimension of J (set to M on iflag=3)
 * @param[in] dwork Workspace (not used)
 * @param[in] ldwork Length of dwork
 * @param[out] info Exit code (always 0)
 */
void md03bf(
    i32* iflag,
    i32 m,
    i32 n,
    i32* ipar,
    i32 lipar,
    const f64* dpar1,
    i32 ldpar1,
    const f64* dpar2,
    i32 ldpar2,
    const f64* x,
    i32* nfevl,
    f64* e,
    f64* j,
    i32* ldj,
    f64* dwork,
    i32 ldwork,
    i32* info
);

#ifdef __cplusplus
}
#endif

#endif /* SLICOT_MD_H */
