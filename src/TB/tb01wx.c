/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

/*
 * TB01WX - Orthogonal similarity transformation to upper Hessenberg form
 *
 * Purpose:
 *   To reduce the system state matrix A to an upper Hessenberg form
 *   by using an orthogonal similarity transformation A <-- U'*A*U and
 *   to apply the transformation to the matrices B and C: B <-- U'*B
 *   and C <-- C*U.
 *
 * Method:
 *   Matrix A is reduced to the Hessenberg form using DGEHRD.
 *   Then the orthogonal transformation matrix is formed (DORGHR if COMPU='I')
 *   or applied (DORMHR if COMPU='U'). The transformations are also applied
 *   to B (DORMHR from left) and C (DORMHR from right).
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <stdlib.h>

void tb01wx(
    const char* compu,
    const i32 n,
    const i32 m,
    const i32 p,
    f64* a,
    const i32 lda,
    f64* b,
    const i32 ldb,
    f64* c,
    const i32 ldc,
    f64* u,
    const i32 ldu,
    f64* dwork,
    const i32 ldwork,
    i32* info
)
{
    const f64 zero = 0.0;
    const f64 one = 1.0;
    const i32 int1 = 1;

    i32 ilu;
    i32 icompu;
    bool lquery;
    i32 itau, jwork;
    i32 minwrk, wrkopt;
    i32 loc_info = 0;
    i32 lwrk;

    *info = 0;

    char compu_char = compu[0];
    if (compu_char == 'N' || compu_char == 'n') {
        ilu = 0;
        icompu = 1;
    } else if (compu_char == 'U' || compu_char == 'u') {
        ilu = 1;
        icompu = 2;
    } else if (compu_char == 'I' || compu_char == 'i') {
        ilu = 1;
        icompu = 3;
    } else {
        icompu = 0;
    }

    if (icompu == 0) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (m < 0) {
        *info = -3;
    } else if (p < 0) {
        *info = -4;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -6;
    } else if (ldb < (n > 1 ? n : 1)) {
        *info = -8;
    } else if (ldc < (p > 1 ? p : 1)) {
        *info = -10;
    } else if (ldu < 1 || (ilu && ldu < (n > 1 ? n : 1))) {
        *info = -12;
    } else {
        lquery = ldwork < 0;
        if (n == 0) {
            minwrk = 1;
        } else {
            minwrk = n - 1 + (n > m ? (n > p ? n : p) : (m > p ? m : p));
        }

        if (lquery) {
            i32 neg1 = -1;

            SLC_DGEHRD(&n, &int1, &n, a, &lda, dwork, dwork, &neg1, &loc_info);
            wrkopt = (n > 1 ? n - 1 : 0) + (i32)dwork[0];
            if (wrkopt < minwrk) wrkopt = minwrk;

            SLC_DORMHR("L", "T", &n, &m, &int1, &n, a, &lda,
                       dwork, b, &ldb, dwork, &neg1, &loc_info);
            i32 opt = (n > 1 ? n - 1 : 0) + (i32)dwork[0];
            if (opt > wrkopt) wrkopt = opt;

            SLC_DORMHR("R", "N", &p, &n, &int1, &n, a, &lda,
                       dwork, c, &ldc, dwork, &neg1, &loc_info);
            opt = (n > 1 ? n - 1 : 0) + (i32)dwork[0];
            if (opt > wrkopt) wrkopt = opt;

            if (ilu) {
                if (icompu == 3) {
                    SLC_DORGHR(&n, &int1, &n, u, &ldu, dwork, dwork, &neg1, &loc_info);
                } else {
                    SLC_DORMHR("R", "N", &n, &n, &int1, &n, a, &lda,
                               dwork, u, &ldu, dwork, &neg1, &loc_info);
                }
                opt = (n > 1 ? n - 1 : 0) + (i32)dwork[0];
                if (opt > wrkopt) wrkopt = opt;
            }
        } else if (ldwork < minwrk) {
            *info = -14;
        }
    }

    if (*info != 0) {
        return;
    }

    if (lquery) {
        dwork[0] = (f64)wrkopt;
        return;
    }

    if (n == 0) {
        dwork[0] = one;
        return;
    }

    itau = 0;
    jwork = itau + n - 1;
    lwrk = ldwork - jwork;

    SLC_DGEHRD(&n, &int1, &n, a, &lda, &dwork[itau], &dwork[jwork], &lwrk, &loc_info);
    wrkopt = (i32)dwork[jwork] + jwork;

    SLC_DORMHR("L", "T", &n, &m, &int1, &n, a, &lda,
               &dwork[itau], b, &ldb, &dwork[jwork], &lwrk, &loc_info);
    i32 opt = (i32)dwork[jwork] + jwork;
    if (opt > wrkopt) wrkopt = opt;

    SLC_DORMHR("R", "N", &p, &n, &int1, &n, a, &lda,
               &dwork[itau], c, &ldc, &dwork[jwork], &lwrk, &loc_info);
    opt = (i32)dwork[jwork] + jwork;
    if (opt > wrkopt) wrkopt = opt;

    if (ilu) {
        if (icompu == 3) {
            SLC_DLACPY("L", &n, &n, a, &lda, u, &ldu);

            SLC_DORGHR(&n, &int1, &n, u, &ldu, &dwork[itau], &dwork[jwork], &lwrk, &loc_info);
        } else {
            SLC_DORMHR("R", "N", &n, &n, &int1, &n, a, &lda,
                       &dwork[itau], u, &ldu, &dwork[jwork], &lwrk, &loc_info);
        }
        opt = (i32)dwork[jwork] + jwork;
        if (opt > wrkopt) wrkopt = opt;
    }

    if (n > 2) {
        i32 nm2 = n - 2;
        SLC_DLASET("L", &nm2, &nm2, &zero, &zero, &a[2], &lda);
    }

    dwork[0] = (f64)wrkopt;
}
