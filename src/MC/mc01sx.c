/*
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "slicot.h"

i32 mc01sx(i32 lb, i32 ub, const i32 *e, const f64 *mant)
{
    const f64 ZERO = 0.0;

    i32 maxe = e[lb - 1];
    i32 mine = maxe;

    for (i32 j = lb; j < ub; j++) {
        if (mant[j] != ZERO) {
            if (e[j] > maxe) maxe = e[j];
            if (e[j] < mine) mine = e[j];
        }
    }

    return maxe - mine;
}
