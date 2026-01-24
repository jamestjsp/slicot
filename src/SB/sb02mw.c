/*
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * SB02MW - Select stable eigenvalues for discrete-time Riccati.
 *
 * Returns true for eigenvalues with modulus < 1.
 */

#include "slicot.h"
#include "slicot_blas.h"

int sb02mw(const f64* reig, const f64* ieig)
{
    return SLC_DLAPY2(reig, ieig) < 1.0;
}
