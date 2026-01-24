/*
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * IB01OY - User's confirmation of the system order
 *
 * Non-interactive version for library use.
 * Validates parameters and returns the system order estimate.
 */

#include "slicot.h"

i32 SLC_IB01OY(i32 ns, i32 nmax, i32 *n, const f64 *sv, i32 *info)
{
    /* Parameter validation */
    *info = 0;

    if (ns <= 0) {
        *info = -1;
    } else if (nmax < 0 || nmax > ns) {
        *info = -2;
    } else if (*n < 0 || *n > ns) {
        *info = -3;
    }

    if (*info != 0) {
        return *info;
    }

    /* In non-interactive mode, we simply validate and return.
     * The caller can modify n before calling if needed.
     * We ensure n <= nmax for consistency. */
    if (*n > nmax) {
        *n = nmax;
    }

    return 0;
}
