// SPDX-License-Identifier: BSD-3-Clause

#include "slicot.h"
#include "slicot_blas.h"
#include <complex.h>

void mb03hz(c128 s11, c128 s12, c128 h11, c128 h12, f64 *co, c128 *si) {
    const f64 two = 2.0;
    c128 g, tmp;

    g = two * creal(h11 * conj(s11));
    c128 f = conj(s11) * h12 + s12 * conj(h11);
    SLC_ZLARTG(&f, &g, co, si, &tmp);
}
