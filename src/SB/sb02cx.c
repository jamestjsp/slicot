/**
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * SB02CX - Select purely imaginary eigenvalues for H-infinity norm.
 *
 * Returns true for eigenvalues with |real part| < 100*eps.
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>

int sb02cx(const f64* reig, const f64* ieig)
{
    (void)ieig;

    f64 eps = SLC_DLAMCH("Epsilon");
    f64 tol = 100.0 * eps;

    return fabs(*reig) < tol;
}
