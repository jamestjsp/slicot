/**
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * SB03OV - Complex plane rotation for Lyapunov solver
 *
 * Constructs a complex plane rotation such that:
 *    ( conjg(c)  s ) * ( a ) = ( d )
 *    (    -s     c )   ( b )   ( 0 )
 *
 * where d is always real and is overwritten on a.
 */

#include "slicot.h"
#include "slicot_blas.h"

void sb03ov(f64* a, const f64 b, const f64 small, f64* c, f64* s)
{
    const f64 one = 1.0;
    const f64 zero = 0.0;

    f64 d = SLC_DLAPY3(&a[0], &a[1], &b);

    if (d < small) {
        c[0] = one;
        c[1] = zero;
        *s = zero;
        if (d > zero) {
            a[0] = d;
            a[1] = zero;
        }
    } else {
        c[0] = a[0] / d;
        c[1] = a[1] / d;
        *s = b / d;
        a[0] = d;
        a[1] = zero;
    }
}
