/**
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * SB04RV - Construct right-hand sides for quasi-Hessenberg Sylvester solver (2 RHS case)
 *
 * Constructs right-hand sides D for system of equations in quasi-Hessenberg form
 * solved via SB04RX (case with 2 right-hand sides).
 */

#include "slicot.h"
#include "slicot_blas.h"

void sb04rv(
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
    i32 int1 = 1, int2 = 2;
    i32 len;

    if (n == 0 || m == 0)
        return;

    bool labscb = (*abschr == 'B' || *abschr == 'b');
    bool lulu = (*ul == 'U' || *ul == 'u');

    if (labscb) {
        // Construct the 2 columns of the right-hand side.
        // D is stored as 2-row matrix (interleaved): D[0::2] = col1, D[1::2] = col2
        SLC_DCOPY(&n, &c[(indx - 1) * ldc], &int1, d, &int2);
        SLC_DCOPY(&n, &c[indx * ldc], &int1, &d[1], &int2);

        if (lulu) {
            // Upper Hessenberg: accumulate from columns 1 to indx-1
            if (indx > 1) {
                len = indx - 1;
                // dwork[0:n] = C[:,0:indx-1] @ AB[0:indx-1, indx-1]
                SLC_DGEMV("N", &n, &len, &one, c, &ldc,
                          &ab[(indx - 1) * ldab], &int1, &zero, dwork, &int1);
                // dwork[n:2n] = C[:,0:indx-1] @ AB[0:indx-1, indx]
                SLC_DGEMV("N", &n, &len, &one, c, &ldc,
                          &ab[indx * ldab], &int1, &zero, &dwork[n], &int1);
                // D[:,0] -= BA @ dwork[0:n]
                SLC_DGEMV("N", &n, &n, &minus_one, ba, &ldba, dwork, &int1, &one, d, &int2);
                // D[:,1] -= BA @ dwork[n:2n]
                SLC_DGEMV("N", &n, &n, &minus_one, ba, &ldba, &dwork[n], &int1, &one, &d[1], &int2);
            }
        } else {
            // Lower Hessenberg: accumulate from columns indx+2 to m
            if (indx < m - 1) {
                len = m - indx - 1;
                // dwork[0:n] = C[:,indx+1:m] @ AB[indx+1:m, indx-1]
                SLC_DGEMV("N", &n, &len, &one, &c[(indx + 1) * ldc], &ldc,
                          &ab[(indx + 1) + (indx - 1) * ldab], &int1, &zero, dwork, &int1);
                // dwork[n:2n] = C[:,indx+1:m] @ AB[indx+1:m, indx]
                SLC_DGEMV("N", &n, &len, &one, &c[(indx + 1) * ldc], &ldc,
                          &ab[(indx + 1) + indx * ldab], &int1, &zero, &dwork[n], &int1);
                // D[:,0] -= BA @ dwork[0:n]
                SLC_DGEMV("N", &n, &n, &minus_one, ba, &ldba, dwork, &int1, &one, d, &int2);
                // D[:,1] -= BA @ dwork[n:2n]
                SLC_DGEMV("N", &n, &n, &minus_one, ba, &ldba, &dwork[n], &int1, &one, &d[1], &int2);
            }
        }
    } else {
        // Construct the 2 rows of the right-hand side.
        // D is stored as 2-row matrix (interleaved): D[0::2] = row1, D[1::2] = row2
        SLC_DCOPY(&m, &c[indx - 1], &ldc, d, &int2);
        SLC_DCOPY(&m, &c[indx], &ldc, &d[1], &int2);

        if (lulu) {
            // Upper Hessenberg: accumulate from rows indx+2 to n
            if (indx < n - 1) {
                len = n - indx - 1;
                // dwork[0:m] = C[indx+1:n,:].T @ AB[indx-1, indx+1:n].T
                SLC_DGEMV("T", &len, &m, &one, &c[indx + 1], &ldc,
                          &ab[(indx - 1) + (indx + 1) * ldab], &ldab, &zero, dwork, &int1);
                // dwork[m:2m] = C[indx+1:n,:].T @ AB[indx, indx+1:n].T
                SLC_DGEMV("T", &len, &m, &one, &c[indx + 1], &ldc,
                          &ab[indx + (indx + 1) * ldab], &ldab, &zero, &dwork[m], &int1);
                // D[:,0] -= BA.T @ dwork[0:m]
                SLC_DGEMV("T", &m, &m, &minus_one, ba, &ldba, dwork, &int1, &one, d, &int2);
                // D[:,1] -= BA.T @ dwork[m:2m]
                SLC_DGEMV("T", &m, &m, &minus_one, ba, &ldba, &dwork[m], &int1, &one, &d[1], &int2);
            }
        } else {
            // Lower Hessenberg: accumulate from rows 1 to indx-1
            if (indx > 1) {
                len = indx - 1;
                // dwork[0:m] = C[0:indx-1,:].T @ AB[indx-1, 0:indx-1].T
                SLC_DGEMV("T", &len, &m, &one, c, &ldc,
                          &ab[indx - 1], &ldab, &zero, dwork, &int1);
                // dwork[m:2m] = C[0:indx-1,:].T @ AB[indx, 0:indx-1].T
                SLC_DGEMV("T", &len, &m, &one, c, &ldc,
                          &ab[indx], &ldab, &zero, &dwork[m], &int1);
                // D[:,0] -= BA.T @ dwork[0:m]
                SLC_DGEMV("T", &m, &m, &minus_one, ba, &ldba, dwork, &int1, &one, d, &int2);
                // D[:,1] -= BA.T @ dwork[m:2m]
                SLC_DGEMV("T", &m, &m, &minus_one, ba, &ldba, &dwork[m], &int1, &one, &d[1], &int2);
            }
        }
    }
}
