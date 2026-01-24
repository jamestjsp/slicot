// SPDX-License-Identifier: BSD-3-Clause

#include "slicot.h"
#include "slicot_blas.h"
#include <complex.h>

void mb03dz(const c128 *a, i32 lda, const c128 *b, i32 ldb, f64 *co1, c128 *si1,
            f64 *co2, c128 *si2) {
    c128 d, g, tmp;

    // G = A(1,1)*B(2,2) - A(2,2)*B(1,1)
    g = a[0] * b[1 + ldb] - a[1 + lda] * b[0];

    // D = A(1,2)*B(2,2) - A(2,2)*B(1,2)
    d = a[lda] * b[1 + ldb] - a[1 + lda] * b[ldb];
    SLC_ZLARTG(&d, &g, co1, si1, &tmp);

    // D = A(1,2)*B(1,1) - A(1,1)*B(1,2)
    d = a[lda] * b[0] - a[0] * b[ldb];
    SLC_ZLARTG(&d, &g, co2, si2, &tmp);
}
