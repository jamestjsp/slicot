// SPDX-License-Identifier: BSD-3-Clause

#include "slicot.h"
#include <math.h>

void ma01bd(f64 base, f64 lgbas, i32 k, const i32 *s, const f64 *a, i32 inca,
            f64 *alpha, f64 *beta, i32 *scal) {
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    *alpha = ONE;
    *beta = ONE;
    *scal = 0;

    for (i32 i = 0; i < k; i++) {
        f64 temp = a[i * inca];
        i32 sl;

        if (temp != ZERO) {
            sl = (i32)(log(fabs(temp)) / lgbas);
            temp = temp / base / pow(base, (f64)(sl - 1));
        } else {
            sl = 0;
        }

        if (s[i] == 1) {
            *alpha = (*alpha) * temp;
            *scal = (*scal) + sl;
        } else {
            *beta = (*beta) * temp;
            *scal = (*scal) - sl;
        }

        if ((i + 1) % 10 == 0) {
            if (*alpha != ZERO) {
                sl = (i32)(log(fabs(*alpha)) / lgbas);
                *scal = (*scal) + sl;
                *alpha = (*alpha) / base / pow(base, (f64)(sl - 1));
            }
            if (*beta != ZERO) {
                sl = (i32)(log(fabs(*beta)) / lgbas);
                *scal = (*scal) - sl;
                *beta = (*beta) / base / pow(base, (f64)(sl - 1));
            }
        }
    }

    if (*beta != ZERO) {
        *alpha = (*alpha) / (*beta);
        *beta = ONE;
    }

    if (*alpha == ZERO) {
        *scal = 0;
    } else {
        i32 sl = (i32)(log(fabs(*alpha)) / lgbas);
        *alpha = (*alpha) / base / pow(base, (f64)(sl - 1));
        *scal = (*scal) + sl;
    }
}
