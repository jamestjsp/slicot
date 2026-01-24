// SPDX-License-Identifier: BSD-3-Clause

#include "slicot.h"
#include "slicot_blas.h"
#include <complex.h>

void mb03gz(c128 z11, c128 z12, c128 z22, c128 h11, c128 h12,
            f64 *co1, c128 *si1, f64 *co2, c128 *si2) {
    const f64 two = 2.0;
    c128 d, g, tmp;

    g = two * creal(h11 * conj(z11) * z22);
    d = z22 * conj(z11) * h12 +
        (conj(z22) * z12 - conj(z12) * z22) * conj(h11);
    SLC_ZLARTG(&d, &g, co1, si1, &tmp);

    d = z11 * (*co1) - z12 * conj(*si1);
    g = -z22 * conj(*si1);
    SLC_ZLARTG(&d, &g, co2, si2, &tmp);
    *si2 = -(*si2);
}
