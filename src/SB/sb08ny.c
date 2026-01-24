/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"

/**
 * @brief Compute B(z) = A(1/z) * A(z) and accuracy norm for discrete-time
 *
 * Computes the coefficients of B(z) = A(1/z) * A(z) where A(z) is a
 * polynomial given in increasing powers of z. The output B contains
 * the autocorrelation coefficients of A.
 *
 * @param[in] da Degree of polynomials A(z) and B(z). DA >= 0.
 * @param[in] a Array of dimension DA+1 containing coefficients of A(z)
 *              in increasing powers of z.
 * @param[out] b Array of dimension DA+1 containing coefficients of B(z).
 *               b[i] = sum_{k=0}^{da-i} a[k] * a[k+i] (autocorrelation at lag i).
 * @param[out] epsb Accuracy norm: 3 * machine_epsilon * b[0].
 */
void sb08ny(
    const i32 da,
    const f64* a,
    f64* b,
    f64* epsb
)
{
    const f64 THREE = 3.0;
    i32 int1 = 1;

    for (i32 i = 0; i <= da; i++) {
        i32 len = da - i + 1;
        b[i] = SLC_DDOT(&len, a, &int1, &a[i], &int1);
    }

    *epsb = THREE * SLC_DLAMCH("E") * b[0];
}
