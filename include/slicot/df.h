/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#ifndef SLICOT_DF_H
#define SLICOT_DF_H

#include "../slicot_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Sine transform or cosine transform of a real signal.
 *
 * Computes the sine transform or cosine transform of a real signal using
 * a Fast Fourier Transform approach. N must be a power of 2 plus 1.
 *
 * For SICO='S' (sine transform):
 * - First and last coefficients are always zero: S_1 = S_N = 0
 * - Uses the formula: S_k = DT * ([C(k) - C(N+1-k)] - [C(k) + C(N+1-k)] / [2*sin(pi*(k-1)/(N-1))])
 *
 * For SICO='C' (cosine transform):
 * - First coefficient: S_1 = 2*DT*[D(1) + A0]
 * - Last coefficient: S_N = 2*DT*[D(1) - A0]
 * - Middle coefficients: S_k = DT * ([D(k) + D(N+1-k)] - [D(k) - D(N+1-k)] / [2*sin(pi*(k-1)/(N-1))])
 *   where A0 = 2*SUM_{i=1}^{(N-1)/2} A(2i)
 *
 * @param[in] sico Transform type:
 *                 'S' = sine transform
 *                 'C' = cosine transform
 * @param[in] n Number of samples (n >= 5, n must be power of 2 plus 1)
 * @param[in] dt Sampling time of the signal
 * @param[in,out] a Signal array, dimension (n)
 *                  In: signal to be processed
 *                  Out: sine or cosine transform coefficients
 * @param[out] info Exit code:
 *                  0 = success
 *                  -1 = invalid SICO
 *                  -2 = N < 5 or N not power of 2 plus 1
 */
void df01md(const char* sico, i32 n, f64 dt, f64* a, i32* info);

#ifdef __cplusplus
}
#endif

#endif /* SLICOT_DF_H */
