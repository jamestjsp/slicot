/**
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * SB04RD - Discrete-time Sylvester equation solver
 *
 * Solves: X + A*X*B = C
 *
 * with at least one of A or B in Schur form and the other in Hessenberg
 * or Schur form.
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <string.h>

void sb04rd(
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
    const i32 int1 = 1;

    char abschr;
    bool labscb, labscs, lula, lulb;
    i32 fwd, i, ibeg, iend, incr, ipincr, istep, jwork, ldw, maxmn;
    f64 scale, tol1;

    *info = 0;
    maxmn = (m > n) ? m : n;

    labscb = (*abschu == 'B' || *abschu == 'b');
    labscs = (*abschu == 'S' || *abschu == 's');
    lula = (*ula == 'U' || *ula == 'u');
    lulb = (*ulb == 'U' || *ulb == 'u');

    if (!labscb && !labscs && !(*abschu == 'A' || *abschu == 'a')) {
        *info = -1;
    } else if (!lula && !(*ula == 'L' || *ula == 'l')) {
        *info = -2;
    } else if (!lulb && !(*ulb == 'L' || *ulb == 'l')) {
        *info = -3;
    } else if (n < 0) {
        *info = -4;
    } else if (m < 0) {
        *info = -5;
    } else if (lda < 1 || lda < n) {
        *info = -7;
    } else if (ldb < 1 || ldb < m) {
        *info = -9;
    } else if (ldc < 1 || ldc < n) {
        *info = -11;
    } else if (ldwork < 2*n ||
               (ldwork < 2*maxmn*(4 + 2*maxmn) && !(labscs && lula && lulb))) {
        *info = -15;
    }

    if (*info != 0) {
        return;
    }

    if (maxmn == 0) {
        return;
    }

    if (labscs && lula && lulb) {
        sb04py('N', 'N', 1, n, m, a, lda, b, ldb, c, ldc, &scale, dwork, info);
        if (scale != one) {
            *info = 1;
        }
        return;
    }

    ldw = 2 * maxmn;
    jwork = ldw * ldw + 3 * ldw;
    tol1 = tol;
    if (tol1 <= zero) {
        tol1 = SLC_DLAMCH("Epsilon");
    }

    abschr = *abschu;
    if (labscs) {
        if (n > m) {
            abschr = 'A';
        } else {
            abschr = 'B';
        }
    }

    #define A(i,j) a[(i) + (j)*lda]
    #define B(i,j) b[(i) + (j)*ldb]
    #define C(i,j) c[(i) + (j)*ldc]

    if (abschr == 'B' || abschr == 'b') {
        if (lulb) {
            ibeg = 0;
            iend = m - 1;
            fwd = 1;
            incr = 0;
        } else {
            ibeg = m - 1;
            iend = 0;
            fwd = -1;
            incr = -1;
        }
        i = ibeg;
        while ((iend - i) * fwd >= 0) {
            if (i == iend) {
                istep = 1;
            } else {
                i32 i_next = i + fwd;
                if (B(i_next, i) == zero) {
                    istep = 1;
                } else {
                    istep = 2;
                }
            }

            if (istep == 1) {
                i32 i1 = i + 1;
                sb04rw(&abschr, ulb, n, m, c, ldc, i1, b, ldb, a, lda,
                       &dwork[jwork], dwork);
                sb04ry("R", ula, n, a, lda, B(i, i), &dwork[jwork],
                       tol1, iwork, dwork, ldw, info);
                if (*info == 1) return;
                SLC_DCOPY(&n, &dwork[jwork], &int1, &C(0, i), &int1);
            } else {
                ipincr = i + incr;
                i32 ipincr1 = ipincr + 1;
                sb04rv(&abschr, ulb, n, m, c, ldc, ipincr1, b, ldb, a, lda,
                       &dwork[jwork], dwork);
                sb04rx("R", ula, n, a, lda,
                       B(ipincr, ipincr), B(ipincr + 1, ipincr),
                       B(ipincr, ipincr + 1), B(ipincr + 1, ipincr + 1),
                       &dwork[jwork], tol1, iwork, dwork, ldw, info);
                if (*info == 1) return;
                i32 int2 = 2;
                SLC_DCOPY(&n, &dwork[jwork], &int2, &C(0, ipincr), &int1);
                SLC_DCOPY(&n, &dwork[jwork + 1], &int2, &C(0, ipincr + 1), &int1);
            }
            i = i + fwd * istep;
        }
    } else {
        if (lula) {
            ibeg = n - 1;
            iend = 0;
            fwd = -1;
            incr = -1;
        } else {
            ibeg = 0;
            iend = n - 1;
            fwd = 1;
            incr = 0;
        }
        i = ibeg;
        while ((iend - i) * fwd >= 0) {
            if (i == iend) {
                istep = 1;
            } else {
                i32 i_check = i + fwd;
                if (A(i, i_check) == zero) {
                    istep = 1;
                } else {
                    istep = 2;
                }
            }

            if (istep == 1) {
                i32 i1 = i + 1;
                sb04rw(&abschr, ula, n, m, c, ldc, i1, a, lda, b, ldb,
                       &dwork[jwork], dwork);
                sb04ry("C", ulb, m, b, ldb, A(i, i), &dwork[jwork],
                       tol1, iwork, dwork, ldw, info);
                if (*info == 1) return;
                SLC_DCOPY(&m, &dwork[jwork], &int1, &C(i, 0), &ldc);
            } else {
                ipincr = i + incr;
                i32 ipincr1 = ipincr + 1;
                sb04rv(&abschr, ula, n, m, c, ldc, ipincr1, a, lda, b, ldb,
                       &dwork[jwork], dwork);
                sb04rx("C", ulb, m, b, ldb,
                       A(ipincr, ipincr), A(ipincr + 1, ipincr),
                       A(ipincr, ipincr + 1), A(ipincr + 1, ipincr + 1),
                       &dwork[jwork], tol1, iwork, dwork, ldw, info);
                if (*info == 1) return;
                i32 int2 = 2;
                SLC_DCOPY(&m, &dwork[jwork], &int2, &C(ipincr, 0), &ldc);
                SLC_DCOPY(&m, &dwork[jwork + 1], &int2, &C(ipincr + 1, 0), &ldc);
            }
            i = i + fwd * istep;
        }
    }

    #undef A
    #undef B
    #undef C
}
