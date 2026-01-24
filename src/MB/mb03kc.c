/*
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * MB03KC - Reduce 2x2 formal matrix product to periodic Hessenberg-triangular form
 *
 * Reduces a 2-by-2 general, formal matrix product A of length K to
 * periodic Hessenberg-triangular form using K-periodic sequence of
 * elementary reflectors (Householder matrices).
 */

#include "slicot.h"
#include "slicot_blas.h"

void mb03kc(const i32 k, const i32 khess, const i32 n, const i32 r,
            const i32 *s, f64 *a, const i32 lda, f64 *v, f64 *tau)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    i32 i, i1, i2, ic, inc, ip1, ir, ix, no;
    f64 tmp[1], work[2];
    i32 int1 = 1;
    i32 int2 = 2;

    ir = (r - 1) * lda;
    ic = ir + r - 1;
    no = n - r;
    inc = n * lda;
    i1 = khess * inc;
    ip1 = (khess % k);

    tau[ip1] = ZERO;
    v[2 * ip1] = ZERO;
    v[2 * ip1 + 1] = ZERO;

    for (i = khess + 1; i <= k; i++) {
        ip1 = (i % k);
        ix = i1 + ic;
        i2 = ip1 * inc;
        ip1 = ip1 + 1;

        if (s[i - 1] == 1) {
            work[0] = ONE;
            work[1] = a[ix + 1];
            SLC_DLARFG(&int2, &a[ix], &work[1], &int1, &tau[ip1 - 1]);
            v[2 * (ip1 - 1)] = ONE;
            v[2 * (ip1 - 1) + 1] = work[1];

            i32 no_val = no;
            SLC_DLARFX("L", &int2, &no_val, work, &tau[ip1 - 1], &a[ix + lda], &lda, tmp);
        } else {
            work[0] = a[ix + 1];
            work[1] = ONE;
            SLC_DLARFG(&int2, &a[ix + lda + 1], work, &int1, &tau[ip1 - 1]);
            v[2 * (ip1 - 1)] = work[0];
            v[2 * (ip1 - 1) + 1] = ONE;

            i32 r_val = r;
            SLC_DLARFX("R", &r_val, &int2, work, &tau[ip1 - 1], &a[i1 + ir], &lda, tmp);
        }
        a[ix + 1] = ZERO;

        if (s[ip1 - 1] == 1) {
            i32 rp1 = r + 1;
            SLC_DLARFX("R", &rp1, &int2, work, &tau[ip1 - 1], &a[i2 + ir], &lda, tmp);
        } else {
            i32 nop1 = no + 1;
            SLC_DLARFX("L", &int2, &nop1, work, &tau[ip1 - 1], &a[i2 + ic], &lda, tmp);
        }
        i1 = i1 + inc;
    }

    i1 = 0;

    for (i = 1; i <= khess - 1; i++) {
        ip1 = (i % k);
        ix = i1 + ic;
        i2 = ip1 * inc;
        ip1 = ip1 + 1;

        if (s[i - 1] == 1) {
            work[0] = ONE;
            work[1] = a[ix + 1];
            SLC_DLARFG(&int2, &a[ix], &work[1], &int1, &tau[ip1 - 1]);
            v[2 * (ip1 - 1)] = ONE;
            v[2 * (ip1 - 1) + 1] = work[1];

            i32 no_val = no;
            SLC_DLARFX("L", &int2, &no_val, work, &tau[ip1 - 1], &a[ix + lda], &lda, tmp);
        } else {
            work[0] = a[ix + 1];
            work[1] = ONE;
            SLC_DLARFG(&int2, &a[ix + lda + 1], work, &int1, &tau[ip1 - 1]);
            v[2 * (ip1 - 1)] = work[0];
            v[2 * (ip1 - 1) + 1] = ONE;

            i32 r_val = r;
            SLC_DLARFX("R", &r_val, &int2, work, &tau[ip1 - 1], &a[i1 + ir], &lda, tmp);
        }
        a[ix + 1] = ZERO;

        if (s[ip1 - 1] == 1) {
            i32 rp1 = r + 1;
            SLC_DLARFX("R", &rp1, &int2, work, &tau[ip1 - 1], &a[i2 + ir], &lda, tmp);
        } else {
            i32 nop1 = no + 1;
            SLC_DLARFX("L", &int2, &nop1, work, &tau[ip1 - 1], &a[i2 + ic], &lda, tmp);
        }
        i1 = i1 + inc;
    }
}
