/*
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * SB02MR - Select unstable eigenvalues for continuous-time Riccati.
 *
 * Returns true for eigenvalues with real part >= 0.
 */

#include "slicot.h"

int sb02mr(const f64* reig, const f64* ieig)
{
    (void)ieig;
    return *reig >= 0.0;
}
