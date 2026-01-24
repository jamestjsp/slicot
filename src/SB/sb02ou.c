/*
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * SB02OU - Select unstable generalized eigenvalues for continuous-time Riccati.
 *
 * Returns true for eigenvalues with positive real part.
 * Used as SELCTG callback for DGGES in SB02OD.
 */

#include "slicot.h"

int sb02ou(const f64* alphar, const f64* alphai, const f64* beta)
{
    (void)alphai;
    return (*alphar < 0.0 && *beta < 0.0) || (*alphar > 0.0 && *beta > 0.0);
}
