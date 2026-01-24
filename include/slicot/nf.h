/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#ifndef SLICOT_NF_H
#define SLICOT_NF_H

#include "../slicot_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Calculate the output of the Wiener system.
 *
 * @param[in] nsmp Number of training samples.
 * @param[in] m Length of each input sample.
 * @param[in] l Length of each output sample.
 * @param[in] ipar Integer parameters (n, nn).
 * @param[in] lipar Length of ipar.
 * @param[in] x Parameter vector (wb, theta).
 * @param[in] lx Length of x.
 * @param[in] u Input samples.
 * @param[in] ldu Leading dimension of u.
 * @param[out] y Simulated output.
 * @param[in] ldy Leading dimension of y.
 * @param[out] dwork Workspace.
 * @param[in] ldwork Length of dwork.
 * @param[out] info Exit code.
 */
void nf01ad(i32 nsmp, i32 m, i32 l, i32 *ipar, i32 lipar, f64 *x, i32 lx, 
            f64 *u, i32 ldu, f64 *y, i32 ldy, f64 *dwork, i32 ldwork, i32 *info);

/**
 * @brief Calculate the output of a set of neural networks.
 *
 * Calculates the output of a set of neural networks with the structure:
 *
 *          - tanh(w1'*z+b1) -
 *        /      :             \
 *      z ---    :           --- sum(ws(i)*...)+ b(n+1)  --- y,
 *        \      :             /
 *          - tanh(wn'*z+bn) -
 *
 * given the input z and the parameter vectors wi, ws, and b.
 *
 * @param[in] nsmp The number of training samples. NSMP >= 0.
 * @param[in] nz The length of each input sample. NZ >= 0.
 * @param[in] l The length of each output sample. L >= 0.
 * @param[in] ipar Integer parameters. ipar[0] must contain NN (neurons).
 * @param[in] lipar Length of ipar.
 * @param[in] wb Weights and biases vector.
 * @param[in] lwb Length of wb.
 * @param[in] z Input samples matrix (nsmp x nz).
 * @param[in] ldz Leading dimension of z.
 * @param[out] y Output samples matrix (nsmp x l).
 * @param[in] ldy Leading dimension of y.
 * @param[out] dwork Workspace.
 * @param[in] ldwork Length of dwork.
 * @param[out] info Exit code.
 */
void nf01ay(i32 nsmp, i32 nz, i32 l, const i32 *ipar, i32 lipar,
            const f64 *wb, i32 lwb, const f64 *z, i32 ldz,
            f64 *y, i32 ldy, f64 *dwork, i32 ldwork, i32 *info);

/**
 * @brief FCN routine for optimizing nonlinear part of Wiener system.
 *
 * This is the FCN routine for the nonlinear part initialization phase.
 * It is called for each output of the Wiener system.
 *
 * @param[in,out] iflag Operation flag (1=error, 2=Jacobian, 3=J'*e)
 * @param[in] nsmp Number of training samples.
 * @param[in] n Number of parameters.
 * @param[in,out] ipar Integer parameters.
 * @param[in] lipar Length of ipar.
 * @param[in] z Linear system output (nsmp).
 * @param[in] ldz Leading dimension of z.
 * @param[in] y Observed output (nsmp).
 * @param[in] ldy Leading dimension of y.
 * @param[in,out] x Parameter vector.
 * @param[out] nfevl Function evaluation count.
 * @param[out] e Error vector (nsmp).
 * @param[out] j Jacobian matrix.
 * @param[in,out] ldj Leading dimension of j.
 * @param[out] jte Vector J'*e.
 * @param[out] dwork Workspace.
 * @param[in] ldwork Length of dwork.
 * @param[out] info Exit code.
 */
void nf01ba(i32 *iflag, i32 nsmp, i32 n, i32 *ipar, i32 lipar,
            const f64 *z, i32 ldz, const f64 *y, i32 ldy, f64 *x,
            i32 *nfevl, f64 *e, f64 *j, i32 *ldj, f64 *jte,
            f64 *dwork, i32 ldwork, i32 *info);

/**
 * @brief FCN callback for optimizing all parameters of a Wiener system.
 *
 * Used with MD03AD to optimize both linear and nonlinear parameters.
 */
void nf01bb(i32 *iflag, i32 nfun, i32 lx, i32 *ipar, i32 lipar,
            const f64 *u, i32 ldu, const f64 *y, i32 ldy, f64 *x,
            i32 *nfevl, f64 *e, f64 *j, i32 *ldj, f64 *jte,
            f64 *dwork, i32 ldwork, i32 *info);

/**
 * @brief Calculate the Jacobian of the Wiener system.
 *
 * @param[in] cjte 'C' to compute J'*e, 'N' to skip.
 * @param[in] nsmp Number of training samples.
 * @param[in] m Length of each input sample.
 * @param[in] l Length of each output sample.
 * @param[in,out] ipar Integer parameters (n, nn).
 * @param[in] lipar Length of ipar.
 * @param[in,out] x Parameter vector.
 * @param[in] lx Length of x.
 * @param[in] u Input samples.
 * @param[in] ldu Leading dimension of u.
 * @param[in] e Error vector (if cjte='C').
 * @param[out] j Jacobian matrix.
 * @param[in,out] ldj Leading dimension of j.
 * @param[out] jte J'*e product (if cjte='C').
 * @param[out] dwork Workspace.
 * @param[in] ldwork Length of dwork.
 * @param[out] info Exit code.
 */
void nf01bd(const char *cjte, i32 nsmp, i32 m, i32 l, i32 *ipar, i32 lipar, 
            f64 *x, i32 lx, f64 *u, i32 ldu, f64 *e, f64 *j, i32 *ldj, 
            f64 *jte, f64 *dwork, i32 ldwork, i32 *info);

/**
 * @brief Error function for Wiener system identification (FCN for MD03BD).
 *
 * @param[in,out] iflag Integer indicating the action to be performed.
 * @param[in] nsmp Number of training samples.
 * @param[in] n Number of variables.
 * @param[in,out] ipar Integer parameters.
 * @param[in] lipar Length of ipar.
 * @param[in] z Input samples.
 * @param[in] ldz Leading dimension of Z.
 * @param[in] y Output samples.
 * @param[in] ldy Leading dimension of Y.
 * @param[in] x Current estimate of parameters.
 * @param[out] nfevl Number of function evaluations.
 * @param[out] e Error vector.
 * @param[out] j Jacobian matrix.
 * @param[in,out] ldj Leading dimension of J.
 * @param[out] dwork Workspace.
 * @param[in] ldwork Length of dwork.
 * @param[out] info Exit code.
 */
void nf01be(i32 *iflag, i32 nsmp, i32 n, i32 *ipar, i32 lipar, 
            f64 *z, i32 ldz, f64 *y, i32 ldy, f64 *x, 
            i32 *nfevl, f64 *e, f64 *j, i32 *ldj, 
            f64 *dwork, i32 ldwork, i32 *info);

/**
 * @brief Error function for Wiener system identification (Full parameter optimization).
 *
 * @param[in,out] iflag Integer indicating the action to be performed.
 * @param[in] nfun Number of functions.
 * @param[in] lx Number of variables.
 * @param[in,out] ipar Integer parameters.
 * @param[in] lipar Length of ipar.
 * @param[in] u Input samples.
 * @param[in] ldu Leading dimension of U.
 * @param[in] y Output samples.
 * @param[in] ldy Leading dimension of Y.
 * @param[in] x Current estimate of parameters.
 * @param[out] nfevl Number of function evaluations.
 * @param[out] e Error vector.
 * @param[out] j Jacobian matrix.
 * @param[in,out] ldj Leading dimension of J.
 * @param[out] dwork Workspace.
 * @param[in] ldwork Length of dwork.
 * @param[out] info Exit code.
 */
void nf01bf(i32 *iflag, i32 nfun, i32 lx, i32 *ipar, i32 lipar, 
            f64 *u, i32 ldu, f64 *y, i32 ldy, f64 *x, 
            i32 *nfevl, f64 *e, f64 *j, i32 *ldj, 
            f64 *dwork, i32 ldwork, i32 *info);

/**
 * @brief Compute Levenberg-Marquardt parameter for Wiener system.
 *
 * @param[in] cond Condition estimation mode.
 * @param[in] n Order of matrix R.
 * @param[in] ipar Integer parameters.
 * @param[in] lipar Length of ipar.
 * @param[in,out] r Matrix R.
 * @param[in] ldr Leading dimension of R.
 * @param[in] ipvt Permutation matrix.
 * @param[in] diag Diagonal scaling.
 * @param[in] qtb Q'*b.
 * @param[in] delta Trust region radius.
 * @param[in,out] par LM parameter.
 * @param[in,out] ranks Ranks.
 * @param[out] x Solution.
 * @param[out] rx Residual.
 * @param[in] tol Tolerance.
 * @param[out] dwork Workspace.
 * @param[in] ldwork Length of dwork.
 * @param[out] info Exit code.
 */
void nf01bp(const char *cond, i32 n, const i32 *ipar, i32 lipar, f64 *r, i32 ldr,
            const i32 *ipvt, const f64 *diag, const f64 *qtb, f64 delta,
            f64 *par, i32 *ranks, f64 *x, f64 *rx, f64 tol, f64 *dwork,
            i32 ldwork, i32 *info);

/**
 * @brief Solve linear system J*x = b, D*x = 0 in least squares sense.
 *
 * @param[in] cond Condition estimation mode.
 * @param[in] n Order of matrix R.
 * @param[in] ipar Integer parameters.
 * @param[in] lipar Length of ipar.
 * @param[in,out] r Matrix R.
 * @param[in] ldr Leading dimension of R.
 * @param[in] ipvt Permutation matrix.
 * @param[in] diag Diagonal scaling.
 * @param[in] qtb Q'*b.
 * @param[in,out] ranks Ranks.
 * @param[out] x Solution.
 * @param[in] tol Tolerance.
 * @param[out] dwork Workspace.
 * @param[in] ldwork Length of dwork.
 * @param[out] info Exit code.
 */
void nf01bq(const char *cond, i32 n, const i32 *ipar, i32 lipar, f64 *r, i32 ldr, 
            const i32 *ipvt, const f64 *diag, const f64 *qtb, i32 *ranks, 
            f64 *x, f64 *tol, f64 *dwork, i32 ldwork, i32 *info);

/**
 * @brief Solve system of linear equations R*x = b or R'*x = b in least squares sense.
 *
 * Solves R*x = b or R'*x = b where R is an n-by-n block upper triangular matrix.
 *
 * @param[in] cond Condition estimation mode ('E', 'N', 'U').
 * @param[in] uplo Storage scheme ('U', 'L').
 * @param[in] trans Form of system ('N', 'T', 'C').
 * @param[in] n Order of matrix R.
 * @param[in] ipar Integer parameters (st, bn, bsm, bsn).
 * @param[in] lipar Length of ipar.
 * @param[in,out] r Matrix R (ldr x nc).
 * @param[in] ldr Leading dimension of R.
 * @param[in] sdiag Diagonal elements of blocks (if uplo='L').
 * @param[in] s Transpose of last block column (if uplo='L').
 * @param[in] lds Leading dimension of S.
 * @param[in,out] b Right hand side vector b. On exit, solution x.
 * @param[in,out] ranks Numerical ranks of submatrices.
 * @param[in] tol Tolerance for rank determination.
 * @param[out] dwork Workspace.
 * @param[in] ldwork Length of dwork.
 * @param[out] info Exit code.
 */
void nf01br(const char *cond, const char *uplo, const char *trans, i32 n, 
            const i32 *ipar, i32 lipar, f64 *r, i32 ldr, f64 *sdiag, 
            f64 *s, i32 lds, f64 *b, i32 *ranks, f64 tol, f64 *dwork, 
            i32 ldwork, i32 *info);

/**
 * @brief QR factorization of Jacobian in compressed form.
 *
 * Computes QR factorization with column pivoting of Jacobian J in compressed form.
 *
 * @param[in] n Number of columns of J.
 * @param[in] ipar Integer parameters (st, bn, bsm, bsn).
 * @param[in] lipar Length of ipar.
 * @param[in] fnorm Norm of error vector.
 * @param[in,out] j Jacobian matrix (ldj x nc).
 * @param[in] ldj Leading dimension of J.
 * @param[in,out] e Error vector.
 * @param[out] jnorms Euclidean norms of columns of J.
 * @param[out] gnorm 1-norm of scaled gradient.
 * @param[out] ipvt Permutation matrix P.
 * @param[out] dwork Workspace.
 * @param[in] ldwork Length of dwork.
 * @param[out] info Exit code.
 */
void nf01bs(i32 n, const i32 *ipar, i32 lipar, f64 fnorm, f64 *j, i32 *ldj,
            f64 *e, f64 *jnorms, f64 *gnorm, i32 *ipvt, f64 *dwork,
            i32 ldwork, i32 *info);

/**
 * @brief Compute J'*J + c*I for full Wiener system Jacobian (Cholesky method).
 */
void nf01bu(const char *stor, const char *uplo, const i32 *n,
            const i32 *ipar, const i32 *lipar,
            const f64 *dpar, const i32 *ldpar,
            const f64 *j, const i32 *ldj,
            f64 *jtj, const i32 *ldjtj,
            f64 *dwork, const i32 *ldwork, i32 *info);

/**
 * @brief Compute J'*J + c*I for single output Jacobian (Cholesky method).
 */
void nf01bv(const char *stor, const char *uplo, const i32 *n,
            const i32 *ipar, const i32 *lipar,
            const f64 *dpar, const i32 *ldpar,
            const f64 *j, const i32 *ldj,
            f64 *jtj, const i32 *ldjtj,
            f64 *dwork, const i32 *ldwork, i32 *info);

/**
 * @brief Compute (J'*J + c*I)*x for full Wiener system Jacobian (CG method).
 */
void nf01bw(i32 n, i32 *ipar, i32 lipar, f64 *dpar, i32 ldpar,
            f64 *j, i32 ldj, f64 *x, i32 incx,
            f64 *dwork, i32 ldwork, i32 *info);

/**
 * @brief Compute (J'*J + c*I)*x for single output Jacobian (CG method).
 */
void nf01bx(i32 n, i32 *ipar, i32 lipar, f64 *dpar, i32 ldpar,
            f64 *j, i32 ldj, f64 *x, i32 incx,
            f64 *dwork, i32 ldwork, i32 *info);

/**
 * @brief Compute the Jacobian of the error function for a neural network.
 *
 * Computes the Jacobian of the error function for a neural network of the structure:
 *
 *          - tanh(w1*z+b1) -
 *        /      :            \
 *      z ---    :          --- sum(ws(i)*...)+ b(n+1)  --- y,
 *        \      :            /
 *          - tanh(wn*z+bn) -
 *
 * for the single-output case.
 *
 * @param[in] cjte 'C' to compute J'*e, 'N' to skip.
 * @param[in] nsmp Number of training samples.
 * @param[in] nz Length of each input sample.
 * @param[in] l Length of each output sample (must be 1).
 * @param[in,out] ipar Integer parameters. ipar[0] contains NN.
 * @param[in] lipar Length of ipar.
 * @param[in] wb Weights and biases vector.
 * @param[in] lwb Length of wb.
 * @param[in] z Input samples matrix (nsmp x nz).
 * @param[in] ldz Leading dimension of z.
 * @param[in] e Error vector (nsmp).
 * @param[out] j Jacobian matrix (nsmp x nwb).
 * @param[in] ldj Leading dimension of j.
 * @param[out] jte Vector J'*e (nwb).
 * @param[out] dwork Workspace.
 * @param[in] ldwork Length of dwork.
 * @param[out] info Exit code.
 */
void nf01by(const char *cjte, i32 nsmp, i32 nz, i32 l, i32 *ipar, i32 lipar, 
            const f64 *wb, i32 lwb, const f64 *z, i32 ldz, const f64 *e, 
            f64 *j, i32 ldj, f64 *jte, f64 *dwork, i32 ldwork, i32 *info);

#ifdef __cplusplus
}
#endif

#endif /* SLICOT_NF_H */
