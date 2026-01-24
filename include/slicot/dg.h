/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#ifndef SLICOT_DG_H
#define SLICOT_DG_H

#include "../slicot_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Discrete Fourier transform or inverse transform of complex signal.
 *
 * Computes the discrete Fourier transform (DFT) or inverse DFT of a complex
 * signal using a decimation-in-time FFT algorithm. N must be a power of 2.
 *
 * Forward transform (INDI='D'):
 *   FZ(k) = SUM_{i=1}^{N} Z(i) * V^((k-1)*(i-1))
 *   where V = exp(-2*pi*j/N) and j^2 = -1
 *
 * Inverse transform (INDI='I'):
 *   Z(i) = SUM_{k=1}^{N} FZ(k) * W^((k-1)*(i-1))
 *   where W = exp(2*pi*j/N)
 *
 * Note: Forward then inverse transform scales signal by factor N.
 *
 * @param[in] indi Transform direction:
 *                 'D' = (Direct) Fourier transform
 *                 'I' = Inverse Fourier transform
 * @param[in] n Number of complex samples (n >= 2, must be power of 2)
 * @param[in,out] xr Real part of signal, dimension (n)
 *                   In: real part of z (if 'D') or f(z) (if 'I')
 *                   Out: real part of f(z) (if 'D') or z (if 'I')
 * @param[in,out] xi Imaginary part of signal, dimension (n)
 *                   In: imaginary part of z (if 'D') or f(z) (if 'I')
 *                   Out: imaginary part of f(z) (if 'D') or z (if 'I')
 * @param[out] info Exit code:
 *                  0 = success
 *                  -1 = invalid INDI
 *                  -2 = N < 2 or N not power of 2
 */
void dg01md(const char* indi, i32 n, f64* xr, f64* xi, i32* info);

/**
 * @brief Discrete Fourier transform of a real signal.
 *
 * Computes the discrete Fourier transform (DFT) or inverse DFT of a real signal
 * of 2*N samples. The signal is split into odd/even parts (for direct transform)
 * or real/imaginary frequency components (for inverse transform).
 *
 * For INDI='D' (direct transform):
 * - Input: XR contains odd samples A(1), A(3), ..., A(2*N-1)
 *          XI contains even samples A(2), A(4), ..., A(2*N)
 * - Output: N+1 complex frequency components in XR (real) and XI (imaginary)
 *
 * For INDI='I' (inverse transform):
 * - Input: N+1 complex frequency components in XR and XI
 * - Output: XR contains odd samples, XI contains even samples
 *
 * Note: A forward transform followed by inverse transform yields a signal
 * scaled by factor 2*N.
 *
 * @param[in] indi Direction indicator:
 *                 'D' = Direct Fourier transform
 *                 'I' = Inverse Fourier transform
 * @param[in] n Half the number of real samples (n >= 2, must be power of 2)
 * @param[in,out] xr Real/odd part array, dimension (n+1)
 *                   For INDI='D': In: odd samples, Out: real part of DFT
 *                   For INDI='I': In: real part of DFT, Out: odd samples
 * @param[in,out] xi Imaginary/even part array, dimension (n+1)
 *                   For INDI='D': In: even samples, Out: imaginary part of DFT
 *                   For INDI='I': In: imaginary part of DFT, Out: even samples
 * @param[out] info Exit code:
 *                  0 = success
 *                  -1 = invalid INDI
 *                  -2 = N < 2 or N not power of 2
 */
void dg01nd(const char* indi, i32 n, f64* xr, f64* xi, i32* info);

/**
 * @brief Auxiliary routine for DG01ND.
 *
 * Performs linear combination of complex FFT components to compute real signal
 * DFT components (for direct transform) or converts real signal DFT back to
 * complex FFT form (for inverse transform).
 *
 * For internal use only. No parameter validation is performed.
 *
 * @param[in] indi Direction indicator: 'D' = direct, 'I' = inverse
 * @param[in] n Half the number of real samples
 * @param[in,out] xr Real part array, dimension (n+1)
 * @param[in,out] xi Imaginary part array, dimension (n+1)
 */
void dg01ny(const char* indi, i32 n, f64* xr, f64* xi);

/**
 * @brief Scrambled discrete Hartley transform of a real signal.
 *
 * Computes the (scrambled) discrete Hartley transform of a real signal
 * using a Hartley butterfly algorithm. N must be a power of 2.
 *
 * The Hartley transform is self-inverse: applying it twice returns
 * the original signal scaled by N.
 *
 * @param[in] scr Scrambling mode:
 *                'N' = no scrambling (standard transform)
 *                'I' = input signal is bit-reversed
 *                'O' = output transform is bit-reversed
 * @param[in,out] wght Weight availability (modified if 'N' on input):
 *                     'A' = precomputed weights available in w
 *                     'N' = weights not available (will be computed)
 *                     Note: If N > 1 and WGHT = 'N', weights are computed
 *                     and WGHT effectively becomes 'A' for reuse.
 * @param[in] n Number of real samples (n >= 0, must be power of 2)
 * @param[in,out] a Signal array, dimension (n)
 *                  In: input signal (bit-reversed if SCR='I')
 *                  Out: Hartley transform (bit-reversed if SCR='O')
 * @param[in,out] w Weight vector array, dimension (n - log2(n))
 *                  In: precomputed weights if WGHT='A', ignored if 'N'
 *                  Out: computed weights for reuse
 * @param[out] info Exit code:
 *                  0 = success
 *                  -1 = invalid SCR
 *                  -2 = invalid WGHT
 *                  -3 = N < 0 or N not power of 2
 */
void dg01od(const char* scr, const char* wght, i32 n, f64* a, f64* w, i32* info);

#ifdef __cplusplus
}
#endif

#endif /* SLICOT_DG_H */
