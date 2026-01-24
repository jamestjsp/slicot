/**
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * SB04NV - Construct right-hand side for Sylvester equation solver (2 RHS case)
 *
 * Constructs right-hand sides D for system in Hessenberg form solved via SB04NX.
 */

#include "slicot.h"
#include "slicot_blas.h"

void sb04nv(
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
    i32 int1 = 1, int2 = 2;
    i32 len;

    if (n == 0 || m == 0)
        return;

    bool labscb = (*abschr == 'B' || *abschr == 'b');
    bool lulu = (*ul == 'U' || *ul == 'u');

    if (labscb) {
        SLC_DCOPY(&n, &c[(indx - 1) * ldc], &int1, d, &int2);
        SLC_DCOPY(&n, &c[indx * ldc], &int1, &d[1], &int2);

        if (lulu) {
            if (indx > 1) {
                len = indx - 1;
                SLC_DGEMV("N", &n, &len, &minus_one, c, &ldc,
                          &ab[(indx - 1) * ldab], &int1, &one, d, &int2);
                SLC_DGEMV("N", &n, &len, &minus_one, c, &ldc,
                          &ab[indx * ldab], &int1, &one, &d[1], &int2);
            }
        } else {
            if (indx < m - 1) {
                len = m - indx - 1;
                SLC_DGEMV("N", &n, &len, &minus_one, &c[(indx + 1) * ldc], &ldc,
                          &ab[(indx + 1) + (indx - 1) * ldab], &int1, &one, d, &int2);
                SLC_DGEMV("N", &n, &len, &minus_one, &c[(indx + 1) * ldc], &ldc,
                          &ab[(indx + 1) + indx * ldab], &int1, &one, &d[1], &int2);
            }
        }
    } else {
        SLC_DCOPY(&m, &c[indx - 1], &ldc, d, &int2);
        SLC_DCOPY(&m, &c[indx], &ldc, &d[1], &int2);

        if (lulu) {
            if (indx < n - 1) {
                len = n - indx - 1;
                SLC_DGEMV("T", &len, &m, &minus_one, &c[indx + 1], &ldc,
                          &ab[(indx - 1) + (indx + 1) * ldab], &ldab, &one, d, &int2);
                SLC_DGEMV("T", &len, &m, &minus_one, &c[indx + 1], &ldc,
                          &ab[indx + (indx + 1) * ldab], &ldab, &one, &d[1], &int2);
            }
        } else {
            if (indx > 1) {
                len = indx - 1;
                SLC_DGEMV("T", &len, &m, &minus_one, c, &ldc,
                          &ab[indx - 1], &ldab, &one, d, &int2);
                SLC_DGEMV("T", &len, &m, &minus_one, c, &ldc,
                          &ab[indx], &ldab, &one, &d[1], &int2);
            }
        }
    }
}
