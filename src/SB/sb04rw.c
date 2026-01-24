/**
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * SB04RW - Construct right-hand side for Hessenberg Sylvester solver (1 RHS case)
 *
 * Constructs right-hand side D for system of equations in Hessenberg form
 * solved via SB04RY (case with 1 right-hand side).
 */

#include "slicot.h"
#include "slicot_blas.h"

void sb04rw(
    const char* abschr,
    const char* ul,
    const i32 n,
    const i32 m,
    const f64* c,
    const i32 ldc,
    const i32 indx,
    const f64* ab,
    const i32 ldab,
    const f64* ba,
    const i32 ldba,
    f64* d,
    f64* dwork)
{
    const f64 one = 1.0;
    const f64 zero = 0.0;
    const f64 minus_one = -1.0;
    i32 int1 = 1;
    i32 len;

    if (n == 0 || m == 0)
        return;

    bool labscb = (*abschr == 'B' || *abschr == 'b');
    bool lulu = (*ul == 'U' || *ul == 'u');

    if (labscb) {
        // Construct the column of the right-hand side.
        // D = C[:, indx-1] (0-indexed: column indx-1)
        SLC_DCOPY(&n, &c[(indx - 1) * ldc], &int1, d, &int1);

        if (lulu) {
            // Upper Hessenberg: accumulate from columns 1 to indx-1
            if (indx > 1) {
                len = indx - 1;
                // dwork = C[:, 0:indx-1] @ AB[0:indx-1, indx-1]
                SLC_DGEMV("N", &n, &len, &one, c, &ldc,
                          &ab[(indx - 1) * ldab], &int1, &zero, dwork, &int1);
                // D -= BA @ dwork
                SLC_DGEMV("N", &n, &n, &minus_one, ba, &ldba, dwork, &int1, &one, d, &int1);
            }
        } else {
            // Lower Hessenberg: accumulate from columns indx+1 to m
            if (indx < m) {
                len = m - indx;
                // dwork = C[:, indx:m] @ AB[indx:m, indx-1]
                SLC_DGEMV("N", &n, &len, &one, &c[indx * ldc], &ldc,
                          &ab[indx + (indx - 1) * ldab], &int1, &zero, dwork, &int1);
                // D -= BA @ dwork
                SLC_DGEMV("N", &n, &n, &minus_one, ba, &ldba, dwork, &int1, &one, d, &int1);
            }
        }
    } else {
        // Construct the row of the right-hand side.
        // D = C[indx-1, :] (0-indexed: row indx-1)
        SLC_DCOPY(&m, &c[indx - 1], &ldc, d, &int1);

        if (lulu) {
            // Upper Hessenberg: accumulate from rows indx+1 to n
            if (indx < n) {
                len = n - indx;
                // dwork = C[indx:n, :].T @ AB[indx-1, indx:n].T
                SLC_DGEMV("T", &len, &m, &one, &c[indx], &ldc,
                          &ab[(indx - 1) + indx * ldab], &ldab, &zero, dwork, &int1);
                // D -= BA.T @ dwork
                SLC_DGEMV("T", &m, &m, &minus_one, ba, &ldba, dwork, &int1, &one, d, &int1);
            }
        } else {
            // Lower Hessenberg: accumulate from rows 1 to indx-1
            if (indx > 1) {
                len = indx - 1;
                // dwork = C[0:indx-1, :].T @ AB[indx-1, 0:indx-1].T
                SLC_DGEMV("T", &len, &m, &one, c, &ldc,
                          &ab[indx - 1], &ldab, &zero, dwork, &int1);
                // D -= BA.T @ dwork
                SLC_DGEMV("T", &m, &m, &minus_one, ba, &ldba, dwork, &int1, &one, d, &int1);
            }
        }
    }
}
