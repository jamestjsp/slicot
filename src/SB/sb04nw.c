/**
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * SB04NW - Construct right-hand side for Sylvester equation solver (1 RHS case)
 *
 * Constructs right-hand side D for system in Hessenberg form solved via SB04NY.
 */

#include "slicot.h"
#include "slicot_blas.h"

void sb04nw(
    const char* abschr,
    const char* ul,
    const i32 n,
    const i32 m,
    const f64* c,
    const i32 ldc,
    const i32 indx,
    const f64* ab,
    const i32 ldab,
    f64* d)
{
    const f64 one = 1.0;
    const f64 minus_one = -1.0;
    i32 int1 = 1;
    i32 len;

    if (n == 0 || m == 0)
        return;

    bool labscb = (*abschr == 'B' || *abschr == 'b');
    bool lulu = (*ul == 'U' || *ul == 'u');

    if (labscb) {
        SLC_DCOPY(&n, &c[(indx - 1) * ldc], &int1, d, &int1);

        if (lulu) {
            if (indx > 1) {
                len = indx - 1;
                SLC_DGEMV("N", &n, &len, &minus_one, c, &ldc,
                          &ab[(indx - 1) * ldab], &int1, &one, d, &int1);
            }
        } else {
            if (indx < m) {
                len = m - indx;
                SLC_DGEMV("N", &n, &len, &minus_one, &c[indx * ldc], &ldc,
                          &ab[indx + (indx - 1) * ldab], &int1, &one, d, &int1);
            }
        }
    } else {
        SLC_DCOPY(&m, &c[indx - 1], &ldc, d, &int1);

        if (lulu) {
            if (indx < n) {
                len = n - indx;
                SLC_DGEMV("T", &len, &m, &minus_one, &c[indx], &ldc,
                          &ab[(indx - 1) + indx * ldab], &ldab, &one, d, &int1);
            }
        } else {
            if (indx > 1) {
                len = indx - 1;
                SLC_DGEMV("T", &len, &m, &minus_one, c, &ldc,
                          &ab[indx - 1], &ldab, &one, d, &int1);
            }
        }
    }
}
