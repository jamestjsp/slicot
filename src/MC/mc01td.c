/*
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <ctype.h>
#include <math.h>
#include <stdlib.h>

void mc01td(const char *dico, i32 dp, const f64 *p, bool *stable,
            i32 *nz, i32 *dp_out, i32 *iwarn, i32 *info)
{
    *iwarn = 0;
    *info = 0;
    *stable = false;
    *nz = 0;

    char d = (char)toupper((unsigned char)dico[0]);
    bool dicoc = (d == 'C');

    if (!dicoc && d != 'D') {
        *info = -1;
        return;
    }
    if (dp < 0) {
        *info = -2;
        return;
    }

    i32 degree = dp;
    while (degree >= 0 && p[degree] == 0.0) {
        degree--;
        (*iwarn)++;
    }

    *dp_out = degree;

    if (degree == -1) {
        *info = 1;
        return;
    }

    if (degree == 0) {
        *stable = true;
        *nz = 0;
        return;
    }

    i32 n = degree + 1;
    f64 *dwork = (f64 *)malloc((size_t)(2 * n) * sizeof(f64));
    if (!dwork) {
        *info = -99;
        return;
    }

    if (dicoc) {
        for (i32 i = 0; i < n; i++) {
            dwork[i] = p[i];
        }

        *nz = 0;
        i32 k = degree;

        while (k > 0) {
            if (dwork[k - 1] == 0.0) {
                *info = 2;
                break;
            }

            f64 alpha = dwork[k] / dwork[k - 1];
            if (alpha < 0.0) {
                (*nz)++;
            }
            k--;

            for (i32 i = k - 1; i >= 1; i -= 2) {
                dwork[i] = dwork[i] - alpha * dwork[i - 1];
            }
        }
    } else {
        for (i32 i = 0; i < n; i++) {
            dwork[n - 1 - i] = p[i];
        }

        f64 signum = 1.0;
        *nz = 0;
        i32 k = 1;

        while (k <= degree && *info == 0) {
            i32 k1 = degree - k + 2;
            i32 k2 = degree + 2;

            i32 one = 1;
            i32 idx = SLC_IDAMAX(&k1, &dwork[k - 1], &one);
            f64 alpha = dwork[k - 1 + idx - 1];

            if (alpha == 0.0) {
                *info = 2;
                break;
            }

            for (i32 i = 0; i < k1; i++) {
                dwork[k2 - 1 + i] = dwork[k - 1 + i] / alpha;
            }

            f64 p1 = dwork[k2 - 1];
            f64 pk1 = dwork[k2 - 1 + k1 - 1];

            for (i32 i = 1; i < k1; i++) {
                dwork[k - 1 + i] = p1 * dwork[degree + i] - pk1 * dwork[k2 - 1 + k1 - i];
            }

            k++;

            if (dwork[k - 1] == 0.0) {
                *info = 2;
                break;
            }

            f64 sign_val = (dwork[k - 1] > 0.0) ? 1.0 : -1.0;
            signum = signum * sign_val;
            if (signum < 0.0) {
                (*nz)++;
            }
        }
    }

    free(dwork);

    if (*info == 0 && *nz == 0) {
        *stable = true;
    } else {
        *stable = false;
    }
}
