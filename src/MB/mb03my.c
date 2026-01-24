// SPDX-License-Identifier: BSD-3-Clause

#include "slicot/mb03.h"
#include <math.h>

f64 mb03my(i32 nx, const f64 *x, i32 incx) {
    if (nx <= 0) {
        return 0.0;
    }

    f64 result = fabs(x[0]);

    for (i32 i = incx; i < nx * incx; i += incx) {
        f64 dx = fabs(x[i]);
        if (dx < result) {
            result = dx;
        }
    }

    return result;
}
