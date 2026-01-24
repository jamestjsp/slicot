#include "slicot.h"
#include <math.h>

void ma02fd(f64 *x1, f64 x2, f64 *c, f64 *s, i32 *info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    if (((*x1 != ZERO) || (x2 != ZERO)) && (fabs(x2) >= fabs(*x1))) {
        *info = 1;
    } else {
        *info = 0;
        if (*x1 == ZERO) {
            *s = ZERO;
            *c = ONE;
        } else {
            *s = x2 / (*x1);
            *c = copysign(sqrt(ONE - *s) * sqrt(ONE + *s), *x1);
            *x1 = (*c) * (*x1);
        }
    }
}
