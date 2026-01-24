// SPDX-License-Identifier: BSD-3-Clause

#include "slicot.h"
#include "slicot_blas.h"
#include <complex.h>

void mb03cz(const c128 *a, i32 lda, const c128 *b, i32 ldb, const c128 *d,
            i32 ldd, f64 *co1, c128 *si1, f64 *co2, c128 *si2, f64 *co3,
            c128 *si3) {
    c128 f, g, tmp;

    // G = A(1,1)*B(1,1)*D(2,2) - A(2,2)*B(2,2)*D(1,1)
    g = a[0] * b[0] * d[1 + ldd] - a[1 + lda] * b[1 + ldb] * d[0];

    // F = (A(1,1)*B(1,2) + A(1,2)*B(2,2))*D(2,2) - A(2,2)*B(2,2)*D(1,2)
    f = (a[0] * b[ldb] + a[lda] * b[1 + ldb]) * d[1 + ldd] -
        a[1 + lda] * b[1 + ldb] * d[ldd];
    SLC_ZLARTG(&f, &g, co1, si1, &tmp);

    // F = (A(1,2)*D(2,2) - A(2,2)*D(1,2))*B(1,1) + A(2,2)*D(1,1)*B(1,2)
    f = (a[lda] * d[1 + ldd] - a[1 + lda] * d[ldd]) * b[0] +
        a[1 + lda] * d[0] * b[ldb];
    SLC_ZLARTG(&f, &g, co2, si2, &tmp);

    // F = (B(1,2)*D(1,1) - B(1,1)*D(1,2))*A(1,1) + A(1,2)*B(2,2)*D(1,1)
    f = (b[ldb] * d[0] - b[0] * d[ldd]) * a[0] + a[lda] * b[1 + ldb] * d[0];
    SLC_ZLARTG(&f, &g, co3, si3, &tmp);
}
