/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#ifndef SLICOT_FD_H
#define SLICOT_FD_H

#include "../slicot_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Fast recursive least-squares filtering.
 *
 * Solves the least-squares filtering problem recursively in time.
 * Each subroutine call implements one time update of the solution.
 * The algorithm uses a fast QR-decomposition based approach.
 *
 * The output error EOUT at instant n is:
 *     EOUT(n) = YIN(n) - sum_{i=0}^{L-1} h_i * XIN(n-i)
 *
 * where h_0,...,h_{L-1} minimize an exponentially weighted sum of
 * successive output errors squared:
 *     sum_{k=1}^{n} [LAMBDA^{2(n-k)} * EOUT(k)^2]
 *
 * The algorithm furnishes the parameters of an equivalent normalized
 * least-squares lattice filter (reflection coefficients SALPH and
 * tap multipliers YQ).
 *
 * @param[in] jp Mode parameter:
 *               'B' = Both prediction and filtering parts applied
 *               'P' = Only prediction section applied
 * @param[in] l Length of impulse response (l >= 1)
 * @param[in] lambda Square root of forgetting factor (0 < lambda <= 1).
 *                   For tracking and stable error propagation, use lambda < 1.
 * @param[in] xin Input sample at instant n
 * @param[in] yin Reference sample at instant n (only if jp='B')
 * @param[in,out] efor Square root of exponentially weighted forward prediction
 *                     error energy. On entry: at instant (n-1). On exit: at instant n.
 * @param[in,out] xf Transformed forward prediction variables, dimension (l).
 *                   On entry: at instant (n-1). On exit: at instant n.
 * @param[in,out] epsbck Normalized a posteriori backward prediction error residuals,
 *                       dimension (l+1). Elements 0..l-1 contain residuals of orders
 *                       0..l-1. Element l contains square root of conversion factor.
 *                       On entry: at instant (n-1). On exit: at instant n.
 * @param[in,out] cteta Cosines of rotation angles, dimension (l).
 *                      On entry: at instant (n-1). On exit: at instant n.
 * @param[in,out] steta Sines of rotation angles, dimension (l).
 *                      On entry: at instant (n-1). On exit: at instant n.
 * @param[in,out] yq Orthogonally transformed reference vector (tap multipliers),
 *                   dimension (l). Only used if jp='B'.
 *                   On entry: at instant (n-1). On exit: at instant n.
 * @param[out] epos A posteriori forward prediction error residual
 * @param[out] eout A posteriori output error residual (only if jp='B')
 * @param[out] salph Opposite of reflection coefficients, dimension (l).
 *                   The i-th reflection coefficient is -salph[i].
 * @param[out] iwarn Warning indicator:
 *                   0 = no warning
 *                   1 = element to be annihilated by rotation < machine precision
 * @param[out] info Error indicator:
 *                  0 = successful exit
 *                  <0 = if info=-i, the i-th argument had illegal value
 *
 * @note Recommended initial values:
 *       - xf[i] = 0, i=0..l-1
 *       - epsbck[i] = 0, i=0..l-1; epsbck[l] = 1
 *       - cteta[i] = 1, i=0..l-1
 *       - steta[i] = 0, i=0..l-1
 *       - yq[i] = 0, i=0..l-1
 *       - efor = 0 (exact start) or small positive (soft start, more reliable)
 */
void fd01ad(const char* jp, i32 l, f64 lambda, f64 xin, f64 yin,
            f64* efor, f64* xf, f64* epsbck, f64* cteta, f64* steta,
            f64* yq, f64* epos, f64* eout, f64* salph,
            i32* iwarn, i32* info);

#ifdef __cplusplus
}
#endif

#endif  /* SLICOT_FD_H */
