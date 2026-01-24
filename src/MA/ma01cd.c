// SPDX-License-Identifier: BSD-3-Clause

#include "slicot.h"
#include <math.h>

i32 ma01cd(f64 a, i32 ia, f64 b, i32 ib) {
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    f64 s, sa, sb;

    if (a == ZERO && b == ZERO) {
        return 0;
    } else if (a == ZERO) {
        return (b > ZERO) ? 1 : -1;
    } else if (b == ZERO) {
        return (a > ZERO) ? 1 : -1;
    } else if (ia == ib) {
        s = a + b;
        if (s == ZERO) {
            return 0;
        } else {
            return (s > ZERO) ? 1 : -1;
        }
    } else {
        sa = (a > ZERO) ? ONE : -ONE;
        sb = (b > ZERO) ? ONE : -ONE;

        if (sa == sb) {
            return (i32)sa;
        } else if (ia > ib) {
            if ((log(fabs(a)) + (f64)(ia - ib)) >= log(fabs(b))) {
                return (i32)sa;
            } else {
                return (i32)sb;
            }
        } else {
            if ((log(fabs(b)) + (f64)(ib - ia)) >= log(fabs(a))) {
                return (i32)sb;
            } else {
                return (i32)sa;
            }
        }
    }
}
