/**
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * SB04ND - Solve continuous-time Sylvester equation AX + XB = C
 *
 * Solves the continuous-time Sylvester equation AX + XB = C, where at least
 * one of the matrices A or B is in Schur form and the other is in Hessenberg
 * or Schur form (both either upper or lower).
 *
 * Uses the Hessenberg-Schur back substitution method proposed by
 * Golub, Nash and Van Loan.
 */

#include "slicot.h"
#include "slicot_blas.h"

void sb04nd(
    const char* abschu,
    const char* ula,
    const char* ulb,
    const i32 n,
    const i32 m,
    const f64* a,
    const i32 lda,
    const f64* b,
    const i32 ldb,
    f64* c,
    const i32 ldc,
    const f64 tol,
    i32* iwork,
    f64* dwork,
    const i32 ldwork,
    i32* info)
{
    const f64 one = 1.0;
    const f64 zero = 0.0;
    i32 int1 = 1;

    *info = 0;

    i32 maxmn = (m > n) ? m : n;
    bool labscb = (*abschu == 'B' || *abschu == 'b');
    bool labscs = (*abschu == 'S' || *abschu == 's');
    bool lula = (*ula == 'U' || *ula == 'u');
    bool lulb = (*ulb == 'U' || *ulb == 'u');

    bool labsca = (*abschu == 'A' || *abschu == 'a');
    if (!labscb && !labscs && !labsca) {
        *info = -1;
        return;
    }
    if (!lula && !(*ula == 'L' || *ula == 'l')) {
        *info = -2;
        return;
    }
    if (!lulb && !(*ulb == 'L' || *ulb == 'l')) {
        *info = -3;
        return;
    }
    if (n < 0) {
        *info = -4;
        return;
    }
    if (m < 0) {
        *info = -5;
        return;
    }
    if (lda < 1 || lda < n) {
        *info = -7;
        return;
    }
    if (ldb < 1 || ldb < m) {
        *info = -9;
        return;
    }
    if (ldc < 1 || ldc < n) {
        *info = -11;
        return;
    }

    i32 min_ldwork = 0;
    if (!(labscs && lula && lulb)) {
        min_ldwork = 2 * maxmn * (4 + 2 * maxmn);
    }
    if (ldwork < min_ldwork) {
        *info = -15;
        return;
    }

    if (maxmn == 0)
        return;

    if (labscs && lula && lulb) {
        f64 scale;
        i32 isgn = 1;
        SLC_DTRSYL("N", "N", &isgn, &n, &m, a, &lda, b, &ldb, c, &ldc, &scale, info);
        if (scale != one)
            *info = 1;
        return;
    }

    i32 ldw = 2 * maxmn;
    i32 jwork = ldw * ldw + 3 * ldw;

    f64 tol1 = tol;
    if (tol1 <= zero)
        tol1 = SLC_DLAMCH("E");

    char abschr = *abschu;
    if (labscs) {
        abschr = (n > m) ? 'A' : 'B';
    }

    bool labschr_b = (abschr == 'B' || abschr == 'b');

    if (labschr_b) {
        i32 ibeg, iend, fwd, incr;
        if (lulb) {
            ibeg = 1;
            iend = m;
            fwd = 1;
            incr = 0;
        } else {
            ibeg = m;
            iend = 1;
            fwd = -1;
            incr = -1;
        }

        i32 i = ibeg;
        while ((iend - i) * fwd >= 0) {
            i32 istep;
            if (i == iend) {
                istep = 1;
            } else {
                i32 i_next = i + fwd;
                if (b[(i_next - 1) + (i - 1) * ldb] == zero) {
                    istep = 1;
                } else {
                    istep = 2;
                }
            }

            if (istep == 1) {
                sb04nw(&abschr, ulb, n, m, c, ldc, i, b, ldb, &dwork[jwork]);
                sb04ny("R", ula, n, a, lda, b[(i - 1) + (i - 1) * ldb],
                       &dwork[jwork], tol1, iwork, dwork, ldw, info);
                if (*info == 1)
                    return;
                SLC_DCOPY(&n, &dwork[jwork], &int1, &c[(i - 1) * ldc], &int1);
            } else {
                i32 ipincr = i + incr;
                sb04nv(&abschr, ulb, n, m, c, ldc, ipincr, b, ldb, &dwork[jwork]);
                sb04nx("R", ula, n, a, lda,
                       b[(ipincr - 1) + (ipincr - 1) * ldb],
                       b[ipincr + (ipincr - 1) * ldb],
                       b[(ipincr - 1) + ipincr * ldb],
                       b[ipincr + ipincr * ldb],
                       &dwork[jwork], tol1, iwork, dwork, ldw, info);
                if (*info == 1)
                    return;
                i32 int2 = 2;
                SLC_DCOPY(&n, &dwork[jwork], &int2, &c[(ipincr - 1) * ldc], &int1);
                SLC_DCOPY(&n, &dwork[jwork + 1], &int2, &c[ipincr * ldc], &int1);
            }
            i = i + fwd * istep;
        }
    } else {
        i32 ibeg, iend, fwd, incr;
        if (lula) {
            ibeg = n;
            iend = 1;
            fwd = -1;
            incr = -1;
        } else {
            ibeg = 1;
            iend = n;
            fwd = 1;
            incr = 0;
        }

        i32 i = ibeg;
        while ((iend - i) * fwd >= 0) {
            i32 istep;
            if (i == iend) {
                istep = 1;
            } else {
                i32 i_next = i + fwd;
                if (a[(i - 1) + (i_next - 1) * lda] == zero) {
                    istep = 1;
                } else {
                    istep = 2;
                }
            }

            if (istep == 1) {
                sb04nw(&abschr, ula, n, m, c, ldc, i, a, lda, &dwork[jwork]);
                sb04ny("C", ulb, m, b, ldb, a[(i - 1) + (i - 1) * lda],
                       &dwork[jwork], tol1, iwork, dwork, ldw, info);
                if (*info == 1)
                    return;
                SLC_DCOPY(&m, &dwork[jwork], &int1, &c[i - 1], &ldc);
            } else {
                i32 ipincr = i + incr;
                sb04nv(&abschr, ula, n, m, c, ldc, ipincr, a, lda, &dwork[jwork]);
                sb04nx("C", ulb, m, b, ldb,
                       a[(ipincr - 1) + (ipincr - 1) * lda],
                       a[ipincr + (ipincr - 1) * lda],
                       a[(ipincr - 1) + ipincr * lda],
                       a[ipincr + ipincr * lda],
                       &dwork[jwork], tol1, iwork, dwork, ldw, info);
                if (*info == 1)
                    return;
                i32 int2 = 2;
                SLC_DCOPY(&m, &dwork[jwork], &int2, &c[ipincr - 1], &ldc);
                SLC_DCOPY(&m, &dwork[jwork + 1], &int2, &c[ipincr], &ldc);
            }
            i = i + fwd * istep;
        }
    }
}
