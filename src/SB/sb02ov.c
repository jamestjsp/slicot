/*
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * SB02OV - Select unstable generalized eigenvalues for discrete-time Riccati.
 *
 * Returns true for eigenvalues with modulus >= 1.
 * Used as SELCTG callback for DGGES in discrete-time Riccati solvers.
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>

int sb02ov(const f64* alphar, const f64* alphai, const f64* beta)
{
    f64 modulus = SLC_DLAPY2(alphar, alphai);
    return modulus >= fabs(*beta);
}
