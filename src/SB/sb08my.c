/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include <math.h>

/**
 * @brief Compute B(s) = A(s) * A(-s) and accuracy norm
 *
 * Computes the coefficients of B(s) = A(s) * A(-s) where A(s) is a
 * polynomial given in increasing powers of s. B(s) is returned in
 * increasing powers of s**2.
 *
 * @param[in] da Degree of polynomials A(s) and B(s). DA >= 0.
 * @param[in] a Array of dimension DA+1 containing coefficients of A(s)
 *              in increasing powers of s.
 * @param[out] b Array of dimension DA+1 containing coefficients of B(s)
 *               in increasing powers of s**2.
 * @param[in,out] epsb On entry: machine precision (DLAMCH("E")).
 *                     On exit: updated accuracy norm 3*maxsa*epsb.
 */
void sb08my(
    const i32 da,
    const f64* a,
    f64* b,
    f64* epsb
)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const f64 TWO = 2.0;
    const f64 THREE = 3.0;

    f64 signi = ONE;
    f64 maxsa = ZERO;

    for (i32 i = 0; i <= da; i++) {
        f64 sabs = a[i] * a[i];
        f64 sa = signi * sabs;
        f64 signk = -TWO * signi;

        i32 kmax = i < (da - i) ? i : (da - i);
        for (i32 k = 1; k <= kmax; k++) {
            f64 term = signk * a[i - k] * a[i + k];
            sa = sa + term;
            sabs = sabs + fabs(term);
            signk = -signk;
        }

        b[i] = sa;
        maxsa = maxsa > sabs ? maxsa : sabs;
        signi = -signi;
    }

    *epsb = THREE * maxsa * (*epsb);
}
