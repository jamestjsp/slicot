/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#ifndef SLICOT_DK_H
#define SLICOT_DK_H

#include "../slicot_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Apply anti-aliasing window to a real signal.
 *
 * Applies a windowing function (Hamming, Hann, or Quadratic) to a real signal
 * for anti-aliasing in digital signal processing applications.
 *
 * Window functions:
 * - Hamming (TYPE='M'): A(i) = (0.54 + 0.46*cos(pi*(i-1)/(N-1)))*A(i)
 * - Hann (TYPE='N'):    A(i) = 0.5*(1 + cos(pi*(i-1)/(N-1)))*A(i)
 * - Quadratic (TYPE='Q'):
 *     For i = 1,...,(N-1)/2+1: A(i) = (1 - 2*((i-1)/(N-1))^2)*(1 - (i-1)/(N-1))*A(i)
 *     For i = (N-1)/2+2,...,N: A(i) = 2*(1 - ((i-1)/(N-1))^3)*A(i)
 *
 * @param[in] type Window type:
 *                 'M' = Hamming window
 *                 'N' = Hann window
 *                 'Q' = Quadratic window
 * @param[in] n Number of samples (n >= 1)
 * @param[in,out] a Signal array, dimension (n)
 *                  In: Signal to be processed
 *                  Out: Windowed signal
 * @param[out] info Exit code:
 *                  0 = success
 *                  -1 = invalid TYPE
 *                  -2 = N <= 0
 */
void dk01md(const char* type, i32 n, f64* a, i32* info);

#ifdef __cplusplus
}
#endif

#endif /* SLICOT_DK_H */
