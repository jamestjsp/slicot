// SPDX-License-Identifier: BSD-3-Clause

#include "slicot.h"
#include <complex.h>
#include <math.h>

void ma01bz(f64 base, i32 k, const i32 *s, const c128 *a, i32 inca,
            c128 *alpha, c128 *beta, i32 *scal) {
    const c128 CONE = 1.0 + 0.0*I;
    const c128 CZERO = 0.0 + 0.0*I;
    const c128 CBASE = base + 0.0*I;

    *alpha = CONE;
    *beta = CONE;
    *scal = 0;

    i32 inda = 0;

    for (i32 i = 0; i < k; i++) {
        c128 ai = a[inda];

        if (s[i] == 1) {
            *alpha = (*alpha) * ai;
        } else {
            if (ai == CZERO) {
                *beta = CZERO;
            } else {
                *alpha = (*alpha) / ai;
            }
        }

        if (cabs(*alpha) == 0.0) {
            *alpha = CZERO;
            *scal = 0;
            if (cabs(*beta) == 0.0) {
                return;
            }
        } else {
            while (cabs(*alpha) < 1.0) {
                *alpha = (*alpha) * CBASE;
                (*scal)--;
            }
            while (cabs(*alpha) >= base) {
                *alpha = (*alpha) / CBASE;
                (*scal)++;
            }
        }

        inda += inca;
    }
}
