/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#ifndef SLICOT_MC_H
#define SLICOT_MC_H

#include "../slicot_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Compute polynomial coefficients from zeros.
 *
 * Computes the coefficients of a real polynomial P(x) from its zeros:
 *   P(x) = (x - r(1)) * (x - r(2)) * ... * (x - r(K))
 * where r(i) = (REZ(i), IMZ(i)).
 *
 * Complex conjugate zeros must appear consecutively in the input arrays.
 *
 * @param[in] k Number of zeros (degree of polynomial). k >= 0.
 * @param[in] rez Real parts of zeros, dimension (k)
 * @param[in] imz Imaginary parts of zeros, dimension (k)
 * @param[out] p Polynomial coefficients in increasing powers of x, dimension (k+1)
 * @param[in,out] dwork Workspace, dimension (k+1). Not referenced if k = 0.
 * @param[out] info Exit code:
 *                  0 = success
 *                  -1 = k < 0
 *                  i > 0: (REZ(i), IMZ(i)) is complex but (REZ(i-1), IMZ(i-1)) is not its conjugate
 */
void mc01pd(i32 k, const f64* rez, const f64* imz, f64* p, f64* dwork, i32* info);

/**
 * @brief Compute shifted polynomial coefficients using Horner's algorithm.
 *
 * Given a real polynomial P(x) = p[0] + p[1]*x + ... + p[dp]*x^dp and a
 * scalar alpha, computes the leading K coefficients of the shifted polynomial:
 *   P(x) = q[0] + q[1]*(x-alpha) + ... + q[k-1]*(x-alpha)^(k-1) + ...
 *
 * The coefficients satisfy: q[i] = P^(i)(alpha) / i! for i = 0, 1, ..., k-1
 *
 * @param[in] dp Degree of polynomial P(x). dp >= 0.
 * @param[in] alpha Shift value
 * @param[in] k Number of shifted coefficients to compute. 1 <= k <= dp+1.
 * @param[in] p Polynomial coefficients in increasing powers of x, dimension (dp+1)
 * @param[out] q Shifted polynomial coefficients, dimension (dp+1).
 *               Leading k elements contain coefficients in increasing powers of (x-alpha).
 *               Remaining dp-k+1 elements are workspace.
 * @param[out] info Exit code:
 *                  0 = success
 *                  -1 = dp < 0
 *                  -3 = k <= 0 or k > dp+1
 */
void mc01md(i32 dp, f64 alpha, i32 k, const f64* p, f64* q, i32* info);

/**
 * @brief Evaluate real polynomial at complex point using Horner's algorithm.
 *
 * Computes P(x0) where P(x) = p[0] + p[1]*x + ... + p[dp]*x^dp is a real
 * polynomial and x0 = xr + xi*j is a complex point.
 *
 * Uses Horner's recursion:
 *   q[dp] = p[dp]
 *   q[i] = x0*q[i+1] + p[i] for i = dp-1, ..., 0
 * Result: P(x0) = q[0]
 *
 * @param[in] dp Degree of polynomial (dp >= 0)
 * @param[in] xr Real part of evaluation point x0
 * @param[in] xi Imaginary part of evaluation point x0
 * @param[in] p Polynomial coefficients, dimension (dp+1), in increasing powers of x
 * @param[out] vr Real part of P(x0)
 * @param[out] vi Imaginary part of P(x0)
 * @param[out] info Exit code: 0 = success, -1 = dp < 0
 */
void mc01nd(i32 dp, f64 xr, f64 xi, const f64* p, f64* vr, f64* vi, i32* info);

/**
 * @brief Determine polynomial stability (Routh or Schur-Cohn algorithm).
 *
 * Determines whether a polynomial P(x) with real coefficients is stable:
 * - Continuous-time (DICO='C'): all zeros in left half-plane (Routh algorithm)
 * - Discrete-time (DICO='D'): all zeros inside unit circle (Schur-Cohn algorithm)
 *
 * The polynomial is: P(x) = p[0] + p[1]*x + p[2]*x^2 + ... + p[dp]*x^dp
 *
 * @param[in] dico 'C' for continuous-time, 'D' for discrete-time
 * @param[in] dp Degree of polynomial (dp >= 0)
 * @param[in] p Polynomial coefficients, dimension (dp+1), in increasing powers of x
 * @param[out] stable true if polynomial is stable, false otherwise
 * @param[out] nz Number of unstable zeros (right half-plane for 'C', outside unit circle for 'D')
 * @param[out] dp_out Actual degree after removing leading zero coefficients
 * @param[out] iwarn Warning indicator (number of leading zeros removed from high-order coefficients)
 * @param[out] info Exit code:
 *                  0 = success
 *                  -1 = invalid DICO
 *                  -2 = dp < 0
 *                  1 = zero polynomial
 *                  2 = algorithm cannot determine stability (zero Routh coefficient or Schur transform)
 */
void mc01td(const char* dico, i32 dp, const f64* p, bool* stable,
            i32* nz, i32* dp_out, i32* iwarn, i32* info);

/**
 * @brief Compute variation of exponents in floating-point number series.
 *
 * Computes the variation V of the exponents of a series of non-zero
 * floating-point numbers: a(j) = MANT(j) * beta^E(j), where beta is
 * the base of machine representation, i.e.,
 *   V = max(E(j)) - min(E(j)), j = lb,...,ub and MANT(j) non-zero.
 *
 * @param[in] lb Lower bound index (1-based Fortran convention)
 * @param[in] ub Upper bound index (1-based Fortran convention)
 * @param[in] e Array of exponents, dimension >= ub
 * @param[in] mant Array of mantissas, dimension >= ub
 * @return Variation V = max(E(j)) - min(E(j)) for non-zero mantissas
 */
i32 mc01sx(i32 lb, i32 ub, const i32* e, const f64* mant);

/**
 * @brief Extract mantissa and exponent from a real number.
 *
 * Finds M and E such that A = M * B^E where 1 <= |M| < B.
 * If A = 0, then M = E = 0.
 *
 * @param[in] a The real number to decompose
 * @param[in] b The base of the floating-point representation (b >= 2)
 * @param[out] m The mantissa satisfying 1 <= |m| < b (or 0 if a = 0)
 * @param[out] e The exponent
 */
void mc01sw(f64 a, i32 b, f64* m, i32* e);

/**
 * @brief Reconstruct a real number from mantissa and exponent.
 *
 * Computes A = M * B^E given mantissa M and exponent E.
 * Returns A = 0 if |A| < B^(EMIN-1) (underflow).
 * Returns A = M if M = 0 or E = 0.
 * Sets OVFLOW = true if |M| * B^E >= B^EMAX (overflow).
 *
 * @param[in] m The mantissa
 * @param[in] e The exponent
 * @param[in] b The base (>= 2)
 * @param[out] a The reconstructed value
 * @param[out] ovflow Overflow flag
 */
void mc01sy(f64 m, i32 e, i32 b, f64* a, bool* ovflow);

/**
 * @brief Compute complex polynomial coefficients from zeros.
 *
 * Computes the coefficients of a complex polynomial P(x) from its zeros:
 *   P(x) = (x - r(1)) * (x - r(2)) * ... * (x - r(K))
 * where r(i) = REZ(i) + j*IMZ(i).
 *
 * Unlike MC01PD, this routine handles arbitrary complex zeros (not just
 * conjugate pairs) and produces complex polynomial coefficients.
 *
 * @param[in] k Number of zeros (degree of polynomial). k >= 0.
 * @param[in] rez Real parts of zeros, dimension (k)
 * @param[in] imz Imaginary parts of zeros, dimension (k)
 * @param[out] rep Real parts of coefficients in increasing powers of x, dimension (k+1)
 * @param[out] imp Imaginary parts of coefficients in increasing powers of x, dimension (k+1)
 * @param[in,out] dwork Workspace, dimension (2*k+2). Not referenced if k = 0.
 * @param[out] info Exit code:
 *                  0 = success
 *                  -1 = k < 0
 */
void mc01od(i32 k, const f64* rez, const f64* imz, f64* rep, f64* imp,
            f64* dwork, i32* info);

/**
 * @brief Compute real polynomial coefficients from zeros (decreasing order).
 *
 * Computes the coefficients of a real polynomial P(x) from its zeros:
 *   P(x) = (x - r(1)) * (x - r(2)) * ... * (x - r(K))
 * where r(i) = REZ(i) + j*IMZ(i).
 *
 * The coefficients are stored in DECREASING order of powers of x.
 * Complex zeros must appear as consecutive conjugate pairs.
 *
 * @param[in] k Number of zeros (degree of polynomial). k >= 0.
 * @param[in] rez Real parts of zeros, dimension (k)
 * @param[in] imz Imaginary parts of zeros, dimension (k)
 * @param[out] p Coefficients in decreasing powers of x, dimension (k+1)
 * @param[in,out] dwork Workspace, dimension (k). Not referenced if k = 0.
 * @param[out] info Exit code:
 *                  0 = success
 *                  -1 = k < 0
 *                  >0 = INFO=i means (REZ(i),IMZ(i)) is complex but
 *                       (REZ(i-1),IMZ(i-1)) is not its conjugate
 */
void mc01py(i32 k, const f64* rez, const f64* imz, f64* p, f64* dwork, i32* info);

/**
 * @brief Extract mantissa and exponent of a real number.
 *
 * Finds M and E such that A = M * B^E where 1 <= |M| < B.
 * If A = 0, then M = E = 0.
 *
 * @param[in] a The real number to decompose
 * @param[in] b The base of the floating-point representation (b >= 2)
 * @param[out] m The mantissa satisfying 1 <= |m| < b (or 0 if a = 0)
 * @param[out] e The exponent
 */
void mc01sw(f64 a, i32 b, f64* m, i32* e);

/**
 * @brief Polynomial division: quotient and remainder.
 *
 * Computes Q(x) and R(x) such that A(x) = B(x) * Q(x) + R(x)
 * where deg(R) < deg(B).
 *
 * @param[in] da Degree of polynomial A(x). If A is zero, da = -1.
 * @param[in,out] db On entry, degree of B(x). On exit, actual degree after
 *                   removing leading zeros.
 * @param[in] a Polynomial coefficients in increasing powers (length da+1).
 * @param[in] b Polynomial coefficients in increasing powers (length db+1).
 * @param[out] rq On exit, contains [R(x) | Q(x)] where R has db coefficients
 *                and Q has da-db+1 coefficients.
 * @param[out] iwarn Number of leading zeros removed from B(x).
 * @param[out] info Error indicator: 0=success, -1=da<-1, -2=db<0, 1=B is zero.
 */
void mc01qd(i32 da, i32* db, const f64* a, const f64* b, f64* rq,
            i32* iwarn, i32* info);

/**
 * @brief Compute polynomial P(x) = P1(x) * P2(x) + alpha * P3(x).
 *
 * Computes the coefficients of the polynomial P(x) = P1(x) * P2(x) + alpha * P3(x),
 * where P1(x), P2(x), and P3(x) are given real polynomials and alpha is a real scalar.
 * Each polynomial may be the zero polynomial (degree -1).
 *
 * @param[in] dp1 Degree of P1(x). dp1 >= -1. (-1 means zero polynomial)
 * @param[in] dp2 Degree of P2(x). dp2 >= -1.
 * @param[in,out] dp3 On entry: degree of P3(x). On exit: degree of result P(x).
 * @param[in] alpha Scalar multiplier for P3(x).
 * @param[in] p1 Coefficients of P1(x) in increasing powers of x. Size dp1+1.
 * @param[in] p2 Coefficients of P2(x) in increasing powers of x. Size dp2+1.
 * @param[in,out] p3 On entry: coefficients of P3(x). On exit: coefficients of P(x).
 *                   Size must be MAX(dp1+dp2, dp3, 0) + 1.
 * @param[out] info Error indicator: 0=success, -1=dp1<-1, -2=dp2<-1, -3=dp3<-1.
 */
void mc01rd(i32 dp1, i32 dp2, i32 *dp3, f64 alpha, const f64 *p1,
            const f64 *p2, f64 *p3, i32 *info);

/**
 * @brief Compute minimal polynomial basis for right nullspace of staircase pencil.
 *
 * Determines a minimal basis of the right nullspace of the subpencil
 * s*E(eps)-A(eps) using the method given in [1]. This pencil only contains
 * Kronecker column indices and must be in staircase form as supplied by MB04VD.
 *
 * NOTE: This routine is intended to be called only from MC03ND.
 *
 * @param[in] nblcks Number of full row rank blocks (>= 0)
 * @param[in] nra Number of rows = sum(nu(i)) (>= 0)
 * @param[in] nca Number of columns = sum(mu(i)) (>= 0)
 * @param[in,out] a Matrix A (nra x nca). Modified on exit.
 * @param[in] lda Leading dimension of A (>= max(1, nra))
 * @param[in,out] e Matrix E (nra x nca). Modified on exit.
 * @param[in] lde Leading dimension of E (>= max(1, nra))
 * @param[in,out] imuk Column dimensions mu(k). Restored on exit.
 * @param[in] inuk Row dimensions nu(k)
 * @param[out] veps Minimal polynomial basis (nca x ncv) where ncv = sum(i*(mu(i)-nu(i)))
 * @param[in] ldveps Leading dimension of VEPS (>= max(1, nca))
 * @param[out] info 0=success, <0=parameter error, >0=block not full row rank
 */
void mc03ny(i32 nblcks, i32 nra, i32 nca, f64 *a, i32 lda,
            f64 *e, i32 lde, i32 *imuk, const i32 *inuk,
            f64 *veps, i32 ldveps, i32 *info);

/**
 * @brief Construct companion pencil s*E-A from polynomial matrix P(s)
 *
 * Given a polynomial matrix P(s) = P(0) + P(1)*s + ... + P(dp)*s^dp,
 * constructs the companion pencil s*E - A.
 *
 * @param[in] mp Number of rows of P(s)
 * @param[in] np Number of columns of P(s)
 * @param[in] dp Degree of P(s)
 * @param[in] p Polynomial coefficients (mp x np x (dp+1))
 * @param[in] ldp1 Leading dimension of P (>= max(1, mp))
 * @param[in] ldp2 Second dimension of P (>= max(1, np))
 * @param[out] a Output matrix A (dp*mp x (dp-1)*mp+np)
 * @param[in] lda Leading dimension of A (>= max(1, dp*mp))
 * @param[out] e Output matrix E (dp*mp x (dp-1)*mp+np)
 * @param[in] lde Leading dimension of E (>= max(1, dp*mp))
 */
void mc03nx(i32 mp, i32 np, i32 dp, const f64 *p, i32 ldp1, i32 ldp2,
            f64 *a, i32 lda, f64 *e, i32 lde);

/**
 * @brief Compute polynomial matrix operation P(x) = P1(x) * P2(x) + alpha * P3(x)
 *
 * Computes the coefficients of the real polynomial matrix
 * P(x) = P1(x) * P2(x) + alpha * P3(x), where P1(x), P2(x), and P3(x)
 * are given real polynomial matrices and alpha is a real scalar.
 *
 * @param[in] rp1 Number of rows of P1 and P3
 * @param[in] cp1 Number of columns of P1 and rows of P2
 * @param[in] cp2 Number of columns of P2 and P3
 * @param[in] dp1 Degree of P1 (-1 if zero polynomial)
 * @param[in] dp2 Degree of P2 (-1 if zero polynomial)
 * @param[in,out] dp3 On entry: degree of P3. On exit: degree of result P
 * @param[in] alpha Scalar multiplier for P3
 * @param[in] p1 Coefficients of P1 (rp1 x cp1 x (dp1+1))
 * @param[in] ldp11 Leading dimension of P1
 * @param[in] ldp12 Second dimension of P1
 * @param[in] p2 Coefficients of P2 (cp1 x cp2 x (dp2+1))
 * @param[in] ldp21 Leading dimension of P2
 * @param[in] ldp22 Second dimension of P2
 * @param[in,out] p3 On entry: coefficients of P3. On exit: result P
 * @param[in] ldp31 Leading dimension of P3
 * @param[in] ldp32 Second dimension of P3
 * @param[out] dwork Workspace array of length CP1
 * @param[out] info Exit code (0 = success, < 0 = parameter error)
 */
void SLC_MC03MD(i32 rp1, i32 cp1, i32 cp2, i32 dp1, i32 dp2, i32 *dp3,
                f64 alpha, const f64 *p1, i32 ldp11, i32 ldp12,
                const f64 *p2, i32 ldp21, i32 ldp22,
                f64 *p3, i32 ldp31, i32 ldp32, f64 *dwork, i32 *info);

/**
 * @brief Compute roots of a cubic polynomial.
 *
 * Computes the roots of the polynomial
 *   P(t) = ALPHA + BETA*t + GAMMA*t^2 + DELTA*t^3.
 *
 * @param[in] alpha Constant coefficient
 * @param[in] beta Linear coefficient
 * @param[in] gamma Quadratic coefficient
 * @param[in] delta Cubic coefficient
 * @param[out] evr Real parts of eigenvalues (length 3)
 * @param[out] evi Imaginary parts of eigenvalues (length 3)
 * @param[out] evq Quotients for eigenvalues (length 3), root = (evr+i*evi)/evq
 * @param[out] dwork Workspace array of length LDWORK
 * @param[in] ldwork Workspace size (>= 42, or -1 for query)
 * @param[out] info Exit code (0 = success)
 */
void mc01xd(f64 alpha, f64 beta, f64 gamma, f64 delta,
            f64 *evr, f64 *evi, f64 *evq,
            f64 *dwork, i32 ldwork, i32 *info);

/**
 * @brief Compute polynomial quotient for quadratic divisor.
 *
 * Divides polynomial P(x) by B(x) = x^2 - u1*x + u2.
 *
 * @param[in] dp Degree of polynomial P (>= 0)
 * @param[in] p Polynomial coefficients (dp+1 elements)
 * @param[in] u1 Linear coefficient of divisor
 * @param[in] u2 Constant coefficient of divisor
 * @param[out] q Quotient polynomial coefficients (dp+1 elements)
 * @param[out] info Exit code (0 = success)
 */
void mc01wd(i32 dp, const f64 *p, f64 u1, f64 u2, f64 *q, i32 *info);

/**
 * @brief Compute roots of cubic polynomial.
 *
 * Computes roots of P(t) = ALPHA + BETA*t + GAMMA*t^2 + DELTA*t^3.
 * Uses QZ or QR algorithm depending on coefficient variation.
 *
 * @param[in] alpha Constant coefficient
 * @param[in] beta Linear coefficient
 * @param[in] gamma Quadratic coefficient
 * @param[in] delta Cubic coefficient
 * @param[out] evr Real parts of root quotients (3 elements)
 * @param[out] evi Imaginary parts of root quotients (3 elements)
 * @param[out] evq Denominators of root quotients (3 elements, >= 0)
 * @param[out] dwork Workspace (ldwork elements)
 * @param[in] ldwork Workspace size (>= 42, or -1 for query)
 * @param[out] info Exit code (0 = success)
 */
void mc01xd(f64 alpha, f64 beta, f64 gamma, f64 delta,
            f64 *evr, f64 *evi, f64 *evq,
            f64 *dwork, i32 ldwork, i32 *info);

/**
 * @brief Scale polynomial coefficients for minimal variation.
 *
 * Scales the coefficients of real polynomial P(x) such that the coefficients
 * of the scaled polynomial Q(x) = s*P(t*x) have minimal variation, where
 * s = BASE^S and t = BASE^T (BASE is the machine floating-point base).
 *
 * @param[in] dp Degree of polynomial P(x). dp >= 0.
 * @param[in,out] p On entry: coefficients in increasing powers of x, dimension (dp+1).
 *                  On exit: coefficients of scaled polynomial Q(x).
 * @param[out] s Exponent for scaling factor s = BASE^S.
 * @param[out] t Exponent for scaling factor t = BASE^T.
 * @param[out] mant Mantissas of Q(x) coefficients, dimension (dp+1).
 * @param[out] e Exponents of Q(x) coefficients, dimension (dp+1).
 * @param[out] iwork Integer workspace, dimension (dp+1).
 * @param[out] info Exit code:
 *                  0 = success
 *                  -1 = dp < 0
 *                  1 = P(x) is the zero polynomial
 */
void mc01sd(i32 dp, f64 *p, i32 *s, i32 *t, f64 *mant, i32 *e, i32 *iwork, i32 *info);

/**
 * @brief Compute roots of a quadratic equation with real coefficients.
 *
 * Computes roots of: a*x^2 + b*x + c = 0
 *
 * The roots r1 and r2 are computed as:
 *   r1 = (-b - sign(b)*sqrt(b^2 - 4*a*c)) / (2*a)
 *   r2 = c / (a*r1)
 * unless a = 0, in which case r1 = -c/b.
 *
 * @param[in] a Coefficient of x^2
 * @param[in] b Coefficient of x
 * @param[in] c Constant term
 * @param[out] z1re Real part of largest root in magnitude
 * @param[out] z1im Imaginary part of largest root in magnitude
 * @param[out] z2re Real part of smallest root in magnitude
 * @param[out] z2im Imaginary part of smallest root in magnitude
 * @param[out] info Exit code:
 *                  0 = success
 *                  1 = a=b=0 or a=0 and -c/b overflows
 *                  2 = a=0 (linear equation), z1re=BIG
 *                  3 = c=0 and -b/a overflows, or largest root overflows
 *                  4 = roots cannot be computed without overflow
 */
void mc01vd(f64 a, f64 b, f64 c, f64* z1re, f64* z1im, f64* z2re, f64* z2im, i32* info);

/**
 * @brief Compute minimal polynomial basis for right nullspace of polynomial matrix.
 *
 * Computes the coefficients of a minimal polynomial basis K(s) for the right
 * nullspace of an MP-by-NP polynomial matrix P(s) of degree DP, solving:
 *   P(s) * K(s) = 0
 *
 * K(s) = K(0) + K(1)*s + ... + K(DK)*s^DK
 *
 * @param[in] mp Number of rows of P(s). mp >= 0.
 * @param[in] np Number of columns of P(s). np >= 0.
 * @param[in] dp Degree of P(s). dp >= 1.
 * @param[in] p Polynomial matrix coefficients (ldp1 x ldp2 x (dp+1)).
 *              P(i,j,k) is the (i,j) element of P(k-1), coefficient of s^(k-1).
 * @param[in] ldp1 Leading dimension of P. ldp1 >= max(1, mp).
 * @param[in] ldp2 Second dimension of P. ldp2 >= max(1, np).
 * @param[out] dk Degree of K(s). DK=-1 if no right nullspace.
 * @param[out] gam Information about nullspace vector ordering (length dp*mp+1).
 * @param[out] nullsp Right nullspace vectors in condensed form (ldnull x (dp*mp+1)*np).
 * @param[in] ldnull Leading dimension of NULLSP. ldnull >= max(1, np).
 * @param[out] ker Minimal polynomial basis coefficients (ldker1 x ldker2 x (dp*mp+1)).
 * @param[in] ldker1 Leading dimension of KER. ldker1 >= max(1, np).
 * @param[in] ldker2 Second dimension of KER. ldker2 >= max(1, np).
 * @param[in] tol Tolerance for rank determination.
 * @param[out] iwork Integer workspace (m+2*max(n,m+1)+n where m=dp*mp, n=(dp-1)*mp+np).
 * @param[out] dwork Double workspace (ldwork).
 * @param[in] ldwork Length of dwork. ldwork >= m*n*n + 2*m*n + 2*n*n.
 * @param[out] info Exit code: 0=success, <0=parameter error, >0=rank error.
 */
void mc03nd(i32 mp, i32 np, i32 dp, const f64 *p, i32 ldp1, i32 ldp2,
            i32 *dk, i32 *gam, f64 *nullsp, i32 ldnull,
            f64 *ker, i32 ldker1, i32 ldker2, f64 tol,
            i32 *iwork, f64 *dwork, i32 ldwork, i32 *info);

#ifdef __cplusplus
}
#endif

#endif /* SLICOT_MC_H */
