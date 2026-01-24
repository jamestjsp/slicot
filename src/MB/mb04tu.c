// SPDX-License-Identifier: BSD-3-Clause

#include "slicot.h"

void mb04tu(i32 n, f64 *x, i32 incx, f64 *y, i32 incy, f64 c, f64 s) {
    if (n <= 0) {
        return;
    }

    if (incx != 1 || incy != 1) {
        i32 ix = 0;
        i32 iy = 0;
        if (incx < 0) {
            ix = (-n + 1) * incx;
        }
        if (incy < 0) {
            iy = (-n + 1) * incy;
        }

        for (i32 i = 0; i < n; i++) {
            f64 temp = c * y[iy] - s * x[ix];
            y[iy] = c * x[ix] + s * y[iy];
            x[ix] = temp;
            ix += incx;
            iy += incy;
        }
    } else {
        for (i32 i = 0; i < n; i++) {
            f64 temp = c * y[i] - s * x[i];
            y[i] = c * x[i] + s * y[i];
            x[i] = temp;
        }
    }
}
