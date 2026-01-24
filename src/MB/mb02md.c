/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

void mb02md(const char* job, i32 m, i32 n, i32 l, i32* rank, f64* c,
            i32 ldc, f64* s, f64* x, i32 ldx, f64* tol,
            i32* iwork, f64* dwork, i32 ldwork, i32* iwarn, i32* info) {
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const f64 TWO = 2.0;
    const f64 NEG_ONE = -1.0;

    char jobu = toupper(job[0]);
    bool ljobr = (jobu == 'R');
    bool ljobt = (jobu == 'T');
    bool ljobn = (jobu == 'N');
    bool crank = !ljobt && !ljobn;
    bool ctol = !ljobr && !ljobn;

    i32 nl = n + l;
    i32 k = (m > nl) ? m : nl;
    i32 p = (m < n) ? m : n;
    i32 minmnl = (m < nl) ? m : nl;
    i32 ldw = (3 * minmnl + k > 5 * minmnl) ? (3 * minmnl + k) : (5 * minmnl);

    *iwarn = 0;
    *info = 0;

    i32 minwrk;
    if (m >= nl) {
        minwrk = (2 > ldw) ? 2 : ldw;
    } else {
        i32 tmp1 = m * nl + ldw;
        i32 tmp2 = 3 * l;
        minwrk = (2 > tmp1) ? 2 : tmp1;
        minwrk = (minwrk > tmp2) ? minwrk : tmp2;
    }

    if (ctol && crank && jobu != 'B') {
        *info = -1;
    } else if (m < 0) {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    } else if (l < 0) {
        *info = -4;
    } else if (!crank && *rank > p) {
        *info = -5;
    } else if (ldc < ((1 > k) ? 1 : k)) {
        *info = -7;
    } else if (ldx < ((1 > n) ? 1 : n)) {
        *info = -10;
    } else if (ctol && *tol < ZERO) {
        *info = -11;
    } else if (ldwork < minwrk && ldwork != -1) {
        *info = -14;
    }

    bool lquery = (ldwork == -1);
    i32 wrkopt = minwrk;

    if (*info == 0 && lquery) {
        i32 int1 = 1;
        i32 query_info = 0;
        if (m >= nl) {
            SLC_DGESVD("N", "O", &m, &nl, c, &ldc, s, dwork, &int1, dwork,
                       &int1, dwork, &ldwork, &query_info);
            i32 tmp = (i32)dwork[0];
            wrkopt = (minwrk > tmp) ? minwrk : tmp;
        } else {
            SLC_DGESVD("N", "A", &m, &nl, dwork, &m, s, dwork, &int1, c,
                       &ldc, dwork, &ldwork, &query_info);
            i32 tmp = (i32)dwork[0] + m * nl;
            wrkopt = (minwrk > tmp) ? minwrk : tmp;
        }

        if (l > 0) {
            i32 nl_minus_1 = nl - 1;
            SLC_DGERQF(&l, &nl_minus_1, c, &ldc, dwork, dwork, &ldwork, &query_info);
            i32 tmp1 = (i32)dwork[0] + l;
            wrkopt = (wrkopt > tmp1) ? wrkopt : tmp1;

            SLC_DORMRQ("R", "T", &n, &nl_minus_1, &l, c, &ldc, dwork, c, &ldc,
                       dwork, &ldwork, &query_info);
            i32 tmp2 = (i32)dwork[0] + l;
            wrkopt = (wrkopt > tmp2) ? wrkopt : tmp2;
        }

        dwork[0] = (f64)wrkopt;
        return;
    }

    if (*info != 0) {
        return;
    }

    if (crank) {
        *rank = p;
    }

    if (minmnl == 0) {
        if (m == 0) {
            SLC_DLASET("F", &nl, &nl, &ZERO, &ONE, c, &ldc);
            SLC_DLASET("F", &n, &l, &ZERO, &ZERO, x, &ldx);
        }
        dwork[0] = TWO;
        dwork[1] = ONE;
        return;
    }

    i32 jwork;
    i32 int1 = 1;

    if (m >= nl) {
        jwork = 0;
        i32 lwork_avail = ldwork - jwork;
        SLC_DGESVD("N", "O", &m, &nl, c, &ldc, s, dwork, &int1, dwork,
                   &int1, &dwork[jwork], &lwork_avail, info);
    } else {
        SLC_DLACPY("F", &m, &nl, c, &ldc, dwork, &m);
        jwork = m * nl;
        i32 lwork_avail = ldwork - jwork;
        SLC_DGESVD("N", "A", &m, &nl, dwork, &m, s, dwork, &int1, c,
                   &ldc, &dwork[jwork], &lwork_avail, info);
    }

    if (*info > 0) {
        for (i32 j = 0; j < minmnl - 1; j++) {
            dwork[j] = dwork[jwork + j + 1];
        }
        return;
    }
    wrkopt = (2 > (i32)dwork[jwork] + jwork) ? 2 : ((i32)dwork[jwork] + jwork);

    for (i32 j = 1; j < nl; j++) {
        i32 jm1 = j;
        SLC_DSWAP(&jm1, &c[j], &ldc, &c[j * ldc], &int1);
    }

    f64 toltmp;
    f64 smax;
    if (ctol) {
        toltmp = sqrt(TWO * (f64)k) * (*tol);
        smax = toltmp;
    } else {
        toltmp = *tol;
        if (toltmp <= ZERO) {
            toltmp = SLC_DLAMCH("P");
        }
        f64 safe_min = SLC_DLAMCH("S");
        smax = toltmp * s[0];
        smax = (smax > safe_min) ? smax : safe_min;
    }

    if (crank) {
        while (*rank > 0 && s[*rank - 1] <= smax) {
            (*rank)--;
        }
    }

    if (l == 0) {
        dwork[0] = (f64)wrkopt;
        dwork[1] = ONE;
        return;
    }

    i32 n1 = n;
    i32 itau = 0;
    jwork = itau + l;

    f64 rcond = ONE;
    i32 r1;

repeat_step3:
    r1 = *rank;
    if (*rank < minmnl) {
        while (*rank > 0) {
            f64 ratio = s[r1] / s[*rank - 1];
            f64 tol_ratio = toltmp / s[*rank - 1];
            if (ONE - ratio * ratio <= tol_ratio * tol_ratio) {
                (*rank)--;
                *iwarn = 1;
                r1 = *rank;
            } else {
                break;
            }
        }
    }

    if (*rank == 0) {
        SLC_DLASET("F", &n, &l, &ZERO, &ZERO, x, &ldx);
        dwork[0] = (f64)wrkopt;
        dwork[1] = ONE;
        return;
    }

    r1 = *rank;
    i32 nl_minus_rank = nl - *rank;
    i32 n_minus_rank = n - *rank;

    i32 lwork_avail = ldwork - jwork;
    SLC_DGERQF(&l, &nl_minus_rank, &c[n1 + r1 * ldc], &ldc, &dwork[itau],
               &dwork[jwork], &lwork_avail, info);
    i32 tmp = (i32)dwork[jwork] + jwork;
    wrkopt = (wrkopt > tmp) ? wrkopt : tmp;

    SLC_DORMRQ("R", "T", &n, &nl_minus_rank, &l, &c[n1 + r1 * ldc], &ldc,
               &dwork[itau], &c[r1 * ldc], &ldc, &dwork[jwork], &lwork_avail, info);
    tmp = (i32)dwork[jwork] + jwork;
    wrkopt = (wrkopt > tmp) ? wrkopt : tmp;

    if (n_minus_rank > 0) {
        SLC_DLASET("F", &l, &n_minus_rank, &ZERO, &ZERO, &c[n1 + r1 * ldc], &ldc);
    }
    if (l > 1) {
        i32 l_minus_1 = l - 1;
        SLC_DLASET("L", &l_minus_1, &l_minus_1, &ZERO, &ZERO, &c[n1 + 1 + n1 * ldc], &ldc);
    }

    SLC_DTRCON("1", "U", "N", &l, &c[n1 + n1 * ldc], &ldc, &rcond, dwork, iwork, info);
    i32 tmp3l = 3 * l;
    wrkopt = (wrkopt > tmp3l) ? wrkopt : tmp3l;

    f64 fnorm = SLC_DLANTR("1", "U", "N", &l, &l, &c[n1 + n1 * ldc], &ldc, dwork);

    if (rcond <= toltmp * fnorm) {
        (*rank)--;
        *iwarn = 2;
        goto repeat_step3;
    } else {
        f64 ynorm = SLC_DLANGE("1", &n, &l, &c[n1 * ldc], &ldc, dwork);
        if (fnorm <= toltmp * ynorm) {
            *rank -= l;
            *iwarn = 2;
            goto repeat_step3;
        }
    }

    SLC_DLACPY("F", &n, &l, &c[n1 * ldc], &ldc, x, &ldx);
    SLC_DTRSM("R", "U", "N", "N", &n, &l, &NEG_ONE, &c[n1 + n1 * ldc], &ldc, x, &ldx);

    dwork[0] = (f64)wrkopt;
    dwork[1] = rcond;
}
