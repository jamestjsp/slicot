/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <stdbool.h>

void tg01oa(
    const char* jobe,
    const i32 n,
    f64* dcba, const i32 lddcba,
    f64* e, const i32 lde,
    i32* info
)
{
    const f64 zero = 0.0;
    bool unite;
    i32 k, n1, nmkp1;
    f64 cs, sn, temp;
    i32 int1 = 1;

    unite = (jobe[0] == 'I' || jobe[0] == 'i');
    *info = 0;
    n1 = n + 1;

    i32 max1n = (1 > n) ? 1 : n;

    if (!unite && !(jobe[0] == 'U' || jobe[0] == 'u')) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (lddcba < n1) {
        *info = -4;
    } else if (lde < 1 || (!unite && lde < max1n)) {
        *info = -6;
    }

    if (*info != 0) {
        return;
    }

    if (n <= 1) {
        return;
    }

    for (k = n; k >= 2; k--) {
        if (dcba[k + 0 * lddcba] != zero) {
            SLC_DLARTG(&dcba[(k - 1) + 0 * lddcba], &dcba[k + 0 * lddcba], &cs, &sn, &temp);
            dcba[(k - 1) + 0 * lddcba] = temp;
            dcba[k + 0 * lddcba] = zero;

            SLC_DROT(&n, &dcba[(k - 1) + 1 * lddcba], &lddcba,
                     &dcba[k + 1 * lddcba], &lddcba, &cs, &sn);

            if (unite) {
                SLC_DROT(&n1, &dcba[0 + (k - 1) * lddcba], &int1,
                         &dcba[0 + k * lddcba], &int1, &cs, &sn);
            } else {
                e[k - 1 + (k - 2) * lde] = sn * e[(k - 2) + (k - 2) * lde];
                e[(k - 2) + (k - 2) * lde] = cs * e[(k - 2) + (k - 2) * lde];

                nmkp1 = n - k + 1;
                SLC_DROT(&nmkp1, &e[(k - 2) + (k - 1) * lde], &lde,
                         &e[(k - 1) + (k - 1) * lde], &lde, &cs, &sn);

                if (e[k - 1 + (k - 2) * lde] != zero) {
                    SLC_DLARTG(&e[(k - 1) + (k - 1) * lde], &e[(k - 1) + (k - 2) * lde],
                               &cs, &sn, &temp);
                    e[(k - 1) + (k - 1) * lde] = temp;
                    e[(k - 1) + (k - 2) * lde] = zero;

                    i32 km1 = k - 1;
                    SLC_DROT(&km1, &e[0 + (k - 2) * lde], &int1,
                             &e[0 + (k - 1) * lde], &int1, &cs, &sn);

                    SLC_DROT(&n1, &dcba[0 + (k - 1) * lddcba], &int1,
                             &dcba[0 + k * lddcba], &int1, &cs, &sn);
                }
            }
        }
    }
}
