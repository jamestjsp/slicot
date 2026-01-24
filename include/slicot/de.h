/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#ifndef SLICOT_DE_H
#define SLICOT_DE_H

#include "../slicot_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Convolution or deconvolution of two real signals.
 *
 * Computes the convolution or deconvolution of two real signals A and B
 * using an FFT algorithm (DG01MD). O(N*log(N)) complexity.
 *
 * Convolution: C = A * B (in frequency domain, element-wise multiplication)
 * Deconvolution: C = A / B (in frequency domain, element-wise division)
 *
 * @param[in] conv Operation type:
 *                 'C' = Convolution
 *                 'D' = Deconvolution
 * @param[in] n Number of samples (n >= 2, must be power of 2)
 * @param[in,out] a First signal array, dimension (n)
 *                  In: First signal
 *                  Out: Convolution or deconvolution result
 * @param[in,out] b Second signal array, dimension (n)
 *                  Note: This array is overwritten
 * @param[out] info Exit code:
 *                  0 = success
 *                  -1 = invalid CONV
 *                  -2 = N < 2 or N not power of 2
 */
void de01od(const char* conv, i32 n, f64* a, f64* b, i32* info);

/**
 * @brief Convolution or deconvolution of two real signals using Hartley transform.
 *
 * Computes the convolution or deconvolution of two real signals A and B
 * using three scrambled Hartley transforms (DG01OD). O(N*log(N)) complexity.
 *
 * @param[in] conv Operation type:
 *                 'C' = Convolution
 *                 'D' = Deconvolution
 * @param[in] wght Weight availability:
 *                 'A' = Weights available (from previous call)
 *                 'N' = Weights not available (will be computed)
 * @param[in] n Number of samples (n >= 0, must be power of 2)
 * @param[in,out] a First signal array, dimension (n)
 *                  In: First signal
 *                  Out: Convolution or deconvolution result
 * @param[in,out] b Second signal array, dimension (n)
 *                  Note: This array is overwritten
 * @param[in,out] w Weight vector, dimension (n - log2(n)) for n > 1
 *                  In: If WGHT='A', precomputed weights
 *                  Out: Computed weight vector
 * @param[out] info Exit code:
 *                  0 = success
 *                  -1 = invalid CONV
 *                  -2 = invalid WGHT
 *                  -3 = N < 0 or N not power of 2
 */
void de01pd(const char* conv, const char* wght, i32 n, f64* a, f64* b, f64* w, i32* info);

/**
 * @brief Void logical function for DGGES.
 *
 * Dummy eigenvalue selection function that always returns true (selects all).
 * Used as SELCTG callback for DGGES when SORT='N' (no eigenvalue ordering).
 *
 * CRITICAL: Return type is int, not bool, due to FORTRAN LOGICAL ABI (4 bytes).
 *
 * @param[in] alphar Real part of alpha (generalized eigenvalue = alpha/beta)
 * @param[in] alphai Imaginary part of alpha
 * @param[in] beta Beta value (denominator of generalized eigenvalue)
 * @return 1 (TRUE) - select all eigenvalues
 */
int delctg(const f64* alphar, const f64* alphai, const f64* beta);

/**
 * @brief Void logical function for ZGGES.
 *
 * Dummy eigenvalue selection function that always returns true (selects all).
 * Used as SELCTG callback for ZGGES when SORT='N' (no eigenvalue ordering).
 *
 * CRITICAL: Return type is int, not bool, due to FORTRAN LOGICAL ABI (4 bytes).
 *
 * @param[in] par1 Complex alpha (generalized eigenvalue = alpha/beta)
 * @param[in] par2 Complex beta (denominator of generalized eigenvalue)
 * @return 1 (TRUE) - select all eigenvalues
 */
int zelctg(const c128* par1, const c128* par2);

#ifdef __cplusplus
}
#endif

#endif /* SLICOT_DE_H */
