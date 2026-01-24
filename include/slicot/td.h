/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#ifndef SLICOT_TD_H
#define SLICOT_TD_H

#include "../slicot_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Calculate state-space from polynomial row vectors over diagonal denominators.
 *
 * Calculates a state-space representation for a (PWORK x MWORK) transfer matrix
 * given in the form of polynomial row vectors over common (diagonal) denominators:
 *
 *     T(s) = inv(D(s)) * U(s)
 *
 * where D(s) is diagonal with (I,I)-th element of degree INDEX(I). The output
 * is in observable companion form of order N = sum(INDEX(I)).
 *
 * As D(s) is diagonal, the PWORK ordered 'non-trivial' columns of C and A are
 * diagonal and (INDEX(I) x 1) block-diagonal respectively.
 *
 * @param[in] mwork Number of inputs (columns of transfer matrix)
 * @param[in] pwork Number of outputs (rows of transfer matrix, rows of D(s))
 * @param[in] index Degrees of diagonal D(s) elements, dimension (pwork).
 *                  INDEX(I) is degree of I-th row denominator.
 * @param[in] dcoeff Denominator coefficients, dimension (lddcoe, *).
 *                   DCOEFF(I,K) = coeff of s^(INDEX(I)-K+1) in D:I(s), K=1..INDEX(I)+1
 * @param[in] lddcoe Leading dimension of dcoeff (>= max(1,pwork))
 * @param[in] ucoeff Numerator coefficients, 3D array dimension (lduco1, lduco2, *).
 *                   UCOEFF(I,J,K) = coeff of s^(INDEX(I)-K+1) in U(I,J)(s)
 * @param[in] lduco1 First dimension of ucoeff (>= max(1,pwork))
 * @param[in] lduco2 Second dimension of ucoeff (>= max(1,mwork))
 * @param[in] n State-space order = sum(INDEX(I))
 * @param[out] a State matrix, dimension (lda, n). Observable companion form.
 * @param[in] lda Leading dimension of a (>= max(1,n))
 * @param[out] b Input matrix, dimension (ldb, mwork)
 * @param[in] ldb Leading dimension of b (>= max(1,n))
 * @param[out] c Output matrix, dimension (ldc, n)
 * @param[in] ldc Leading dimension of c (>= max(1,pwork))
 * @param[out] d Feedthrough matrix, dimension (ldd, mwork)
 * @param[in] ldd Leading dimension of d (>= max(1,pwork))
 * @param[out] info Exit code: 0 = success, I > 0 = row I leading coeff near zero
 */
void td03ay(
    const i32 mwork,
    const i32 pwork,
    const i32* index,
    const f64* dcoeff,
    const i32 lddcoe,
    const f64* ucoeff,
    const i32 lduco1,
    const i32 lduco2,
    const i32 n,
    f64* a,
    const i32 lda,
    f64* b,
    const i32 ldb,
    f64* c,
    const i32 ldc,
    f64* d,
    const i32 ldd,
    i32* info
);

/**
 * @brief Minimal state-space representation from transfer function.
 *
 * Finds a minimal state-space representation (A,B,C,D) for a proper
 * transfer matrix T(s) given as either row or column polynomial vectors
 * over denominator polynomials, possibly with uncancelled common terms.
 *
 * For ROWCOL='R': T(s) = inv(D(s)) * U(s) (rows over common denominators)
 * For ROWCOL='C': T(s) = U(s) * inv(D(s)) (columns over common denominators)
 *
 * where D(s) is diagonal with (I,I)-th polynomial of degree INDEX(I).
 *
 * Uses Wolovich's Observable Structure Theorem to construct an observable
 * companion form, then TB01PD to extract a minimal realization.
 *
 * @param[in] rowcol 'R' = rows over common denominators,
 *                   'C' = columns over common denominators
 * @param[in] m Number of system inputs (M >= 0)
 * @param[in] p Number of system outputs (P >= 0)
 * @param[in] index Degrees of denominator polynomials, dimension (porm)
 *                  where porm = P if ROWCOL='R', porm = M if ROWCOL='C'
 * @param[in] dcoeff Denominator coefficients, dimension (lddcoe, kdcoef)
 *                   where kdcoef = max(INDEX(I)) + 1
 *                   DCOEFF(I,K) = coeff of s^(INDEX(I)-K+1) in I-th denominator
 * @param[in] lddcoe Leading dimension of dcoeff
 *                   >= max(1,P) if ROWCOL='R', >= max(1,M) if ROWCOL='C'
 * @param[in,out] ucoeff Numerator coefficients, dimension (lduco1, lduco2, kdcoef)
 *                       The leading P-by-M-by-kdcoef part contains U(s)
 *                       If ROWCOL='C', modified internally but restored on exit
 * @param[in] lduco1 First dimension of ucoeff
 *                   >= max(1,P) if ROWCOL='R', >= max(1,M,P) if ROWCOL='C'
 * @param[in] lduco2 Second dimension of ucoeff
 *                   >= max(1,M) if ROWCOL='R', >= max(1,M,P) if ROWCOL='C'
 * @param[out] nr Order of minimal realization (order of A matrix)
 * @param[out] a State dynamics matrix, dimension (lda, N) where N = sum(INDEX)
 *               The leading NR-by-NR part contains upper block Hessenberg A
 * @param[in] lda Leading dimension of a (>= max(1,N))
 * @param[out] b Input matrix, dimension (ldb, max(M,P))
 *               The leading NR-by-M part contains B
 * @param[in] ldb Leading dimension of b (>= max(1,N))
 * @param[out] c Output matrix, dimension (ldc, N)
 *               The leading P-by-NR part contains C
 * @param[in] ldc Leading dimension of c (>= max(1,M,P))
 * @param[out] d Direct transmission matrix, dimension (ldd, M) if ROWCOL='R',
 *               (ldd, max(M,P)) if ROWCOL='C'
 *               The leading P-by-M part contains D
 * @param[in] ldd Leading dimension of d
 *                >= max(1,P) if ROWCOL='R', >= max(1,M,P) if ROWCOL='C'
 * @param[in] tol Tolerance for rank determination. If tol <= 0, uses default.
 * @param[out] iwork Integer workspace, dimension (N + max(M,P))
 *                   On exit, first nonzero elements return diagonal block orders
 * @param[out] dwork Double workspace, dimension (ldwork)
 *                   On exit, dwork[0] = optimal ldwork
 * @param[in] ldwork Workspace size (>= max(1, N + max(N, 3*M, 3*P)))
 * @param[out] info Exit code: 0 = success, -i = i-th argument invalid,
 *                  i > 0 = row i leading coefficient near zero (overflow risk)
 */
void td04ad(
    const char* rowcol,
    const i32 m,
    const i32 p,
    const i32* index,
    f64* dcoeff,
    const i32 lddcoe,
    f64* ucoeff,
    const i32 lduco1,
    const i32 lduco2,
    i32* nr,
    f64* a,
    const i32 lda,
    f64* b,
    const i32 ldb,
    f64* c,
    const i32 ldc,
    f64* d,
    const i32 ldd,
    const f64 tol,
    i32* iwork,
    f64* dwork,
    const i32 ldwork,
    i32* info
);

/**
 * @brief Left/right polynomial matrix representation for a proper transfer matrix.
 *
 * Finds a relatively prime left or right polynomial matrix representation
 * for a proper transfer matrix T(s) given as either row or column polynomial
 * vectors over common denominator polynomials, possibly with uncancelled
 * common terms.
 *
 * For LERI='L' (left): T(s) = inv(P(s)) * Q(s)
 * For LERI='R' (right): T(s) = Q(s) * inv(P(s))
 *
 * Uses Wolovich's Observable Structure Theorem to construct a state-space
 * representation in observable companion form, then TB03AD to extract a
 * minimal realization and compute the polynomial matrix representation.
 *
 * @param[in] rowcol 'R' = T(s) factorized by rows, 'C' = by columns
 * @param[in] leri 'L' = left PMR inv(P)*Q, 'R' = right PMR Q*inv(P)
 * @param[in] equil 'S' = perform balancing, 'N' = no balancing
 * @param[in] m Number of system inputs (M >= 0)
 * @param[in] p Number of system outputs (P >= 0)
 * @param[in] indexd Degrees of denominator polynomials, dimension (pormd)
 *                   where pormd = P if ROWCOL='R', pormd = M if ROWCOL='C'
 * @param[in] dcoeff Denominator coefficients, dimension (lddcoe, kdcoef)
 *                   where kdcoef = max(INDEXD(I)) + 1
 * @param[in] lddcoe Leading dimension of dcoeff
 * @param[in,out] ucoeff Numerator coefficients, dimension (lduco1, lduco2, kdcoef)
 *                       If ROWCOL='C', modified internally but restored on exit
 * @param[in] lduco1 First dimension of ucoeff
 * @param[in] lduco2 Second dimension of ucoeff
 * @param[out] nr Order of minimal realization
 * @param[out] a State matrix, dimension (lda, N) where N = sum(INDEXD)
 * @param[in] lda Leading dimension of a
 * @param[out] b Input matrix, dimension (ldb, max(M,P))
 * @param[in] ldb Leading dimension of b
 * @param[out] c Output matrix, dimension (ldc, N)
 * @param[in] ldc Leading dimension of c
 * @param[out] d Feedthrough matrix, dimension (ldd, max(M,P))
 * @param[in] ldd Leading dimension of d
 * @param[out] indexp Row/column degrees of denominator matrix P(s)
 * @param[out] pcoeff Denominator matrix P(s) coefficients, dimension (ldpco1, ldpco2, N+1)
 * @param[in] ldpco1 First dimension of pcoeff
 * @param[in] ldpco2 Second dimension of pcoeff
 * @param[out] qcoeff Numerator matrix Q(s) coefficients, dimension (ldqco1, ldqco2, N+1)
 * @param[in] ldqco1 First dimension of qcoeff
 * @param[in] ldqco2 Second dimension of qcoeff
 * @param[out] vcoeff Intermediate matrix V(s) coefficients, dimension (ldvco1, ldvco2, N+1)
 * @param[in] ldvco1 First dimension of vcoeff
 * @param[in] ldvco2 Second dimension of vcoeff
 * @param[in] tol Tolerance for rank determination
 * @param[out] iwork Integer workspace, dimension (N + max(M,P))
 * @param[out] dwork Double workspace, dimension (ldwork)
 * @param[in] ldwork Workspace size
 * @param[out] info Exit code: 0 = success, -i = i-th argument invalid,
 *                  i > 0 (i <= pormd) = row i leading coeff near zero,
 *                  pormd+1 = singular matrix in V(s) computation,
 *                  pormd+2 = singular matrix in P(s) computation
 */
void td03ad(
    const char* rowcol,
    const char* leri,
    const char* equil,
    const i32 m,
    const i32 p,
    const i32* indexd,
    const f64* dcoeff,
    const i32 lddcoe,
    f64* ucoeff,
    const i32 lduco1,
    const i32 lduco2,
    i32* nr,
    f64* a,
    const i32 lda,
    f64* b,
    const i32 ldb,
    f64* c,
    const i32 ldc,
    f64* d,
    const i32 ldd,
    i32* indexp,
    f64* pcoeff,
    const i32 ldpco1,
    const i32 ldpco2,
    f64* qcoeff,
    const i32 ldqco1,
    const i32 ldqco2,
    f64* vcoeff,
    const i32 ldvco1,
    const i32 ldvco2,
    const f64 tol,
    i32* iwork,
    f64* dwork,
    const i32 ldwork,
    i32* info
);

/**
 * @brief Evaluate transfer function G(jW) at specified frequency.
 *
 * Given a complex valued rational function of frequency (transfer function)
 * G(jW), this routine calculates its complex value or its magnitude and phase
 * for a specified frequency value.
 *
 *         B(1)+B(2)*(jW)+B(3)*(jW)^2+...+B(MP1)*(jW)^(MP1-1)
 *  G(jW) = --------------------------------------------------
 *         A(1)+A(2)*(jW)+A(3)*(jW)^2+...+A(NP1)*(jW)^(NP1-1)
 *
 * @param[in] unitf Frequency unit: 'R' = radians/second, 'H' = hertz
 * @param[in] output Output format: 'C' = Cartesian (real/imag),
 *                                  'P' = Polar (magnitude dB/phase degrees)
 * @param[in] np1 Denominator order + 1 (>= 1)
 * @param[in] mp1 Numerator order + 1 (>= 1)
 * @param[in] w Frequency value
 * @param[in] a Denominator coefficients, dimension (np1), ascending powers
 * @param[in] b Numerator coefficients, dimension (mp1), ascending powers
 * @param[out] valr If OUTPUT='C': real part of G(jW)
 *                  If OUTPUT='P': magnitude of G(jW) in dB
 * @param[out] vali If OUTPUT='C': imaginary part of G(jW)
 *                  If OUTPUT='P': phase of G(jW) in degrees
 * @param[out] info Exit code: 0 = success, -i = i-th argument invalid,
 *                  1 = frequency W is a pole or all A coefficients zero
 */
void td05ad(
    const char* unitf,
    const char* output,
    const i32 np1,
    const i32 mp1,
    const f64 w,
    const f64* a,
    const f64* b,
    f64* valr,
    f64* vali,
    i32* info
);

#ifdef __cplusplus
}
#endif

#endif /* SLICOT_TD_H */
