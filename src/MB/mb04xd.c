/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"

void mb04xd(const char *jobu, const char *jobv, const i32 m, const i32 n,
            i32 *rank, f64 *theta, f64 *a, const i32 lda,
            f64 *u, const i32 ldu, f64 *v, const i32 ldv,
            f64 *q, bool *inul, const f64 tol, const f64 reltol,
            f64 *dwork, const i32 ldwork, i32 *iwarn, i32 *info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    char jobuy, jobvy;
    bool all, ljobua, ljobus, ljobva, ljobvs, lquery, qr, wantu, wantv;
    i32 i, ihoush, ij, itau, itaup, itauq, j, ju, jv, jwork, k;
    i32 ldw, ldy, ma, minwrk, p, pp1, wrkopt;
    f64 cs, sn, temp;

    i32 int1 = 1, int0 = 0, intm1 = -1;

    *iwarn = 0;
    *info = 0;
    p = (m < n) ? m : n;
    k = (m > n) ? m : n;

    ljobua = (jobu[0] == 'A' || jobu[0] == 'a');
    ljobus = (jobu[0] == 'S' || jobu[0] == 's');
    ljobva = (jobv[0] == 'A' || jobv[0] == 'a');
    ljobvs = (jobv[0] == 'S' || jobv[0] == 's');
    wantu = ljobua || ljobus;
    wantv = ljobva || ljobvs;
    lquery = (ldwork == -1);

    if (!wantu && !(jobu[0] == 'N' || jobu[0] == 'n')) {
        *info = -1;
    } else if (!wantv && !(jobv[0] == 'N' || jobv[0] == 'n')) {
        *info = -2;
    } else if (m < 0) {
        *info = -3;
    } else if (n < 0) {
        *info = -4;
    } else if (*rank > p) {
        *info = -5;
    } else if (*rank < 0 && *theta < ZERO) {
        *info = -6;
    } else if (lda < (1 > m ? 1 : m)) {
        *info = -8;
    } else if ((wantu && ldu < (1 > m ? 1 : m)) ||
               (!wantu && ldu < 1)) {
        *info = -10;
    } else if ((wantv && ldv < (1 > n ? 1 : n)) ||
               (!wantv && ldv < 1)) {
        *info = -12;
    } else {
        i32 ilaenv_result = SLC_ILAENV(&(i32){6}, "DGESVD", "NN", &m, &n, &int0, &int0);
        qr = (m >= ilaenv_result);
        if (qr && wantu) {
            i32 t1 = 2 * n;
            i32 t2 = n * (n + 1) / 2;
            ldw = (t1 > t2) ? t1 : t2;
        } else {
            ldw = 0;
        }
        if (wantu || wantv) {
            ldy = 8 * p - 5;
        } else {
            ldy = 6 * p - 3;
        }
        i32 t1 = 2 * p + k;
        i32 t2 = (t1 > ldy) ? t1 : ldy;
        minwrk = ldw + t2;
        if (minwrk < 1) minwrk = 1;

        if (ldwork < minwrk && !lquery) {
            *info = -18;
        } else if (lquery) {
            if (qr) {
                SLC_DGEQRF(&m, &n, a, &lda, dwork, dwork, &intm1, info);
                i32 t = n + (i32)dwork[0];
                wrkopt = (minwrk > t) ? minwrk : t;
                ma = n;
            } else {
                wrkopt = minwrk;
                ma = m;
            }
            SLC_DGEBRD(&ma, &n, a, &lda, q, q, dwork, dwork, dwork, &intm1, info);
            i32 t = ldw + 2 * p + (i32)dwork[0];
            wrkopt = (wrkopt > t) ? wrkopt : t;
        }
    }

    if (*info != 0) {
        return;
    } else if (lquery) {
        dwork[0] = (f64)wrkopt;
        return;
    }

    if (p == 0) {
        if (*rank >= 0) {
            *theta = ZERO;
        }
        *rank = 0;
        return;
    }

    pp1 = p + 1;
    all = (ljobua && m > n) || (ljobva && m < n);

    if (all && !qr) {
        for (i = 0; i < p; i++) {
            inul[i] = false;
        }
        for (i = p; i < k; i++) {
            inul[i] = true;
        }
    } else {
        for (i = 0; i < k; i++) {
            inul[i] = false;
        }
    }

    if (qr) {
        itau = 0;
        jwork = itau + n;
        i32 lwork_qr = ldwork - jwork;
        SLC_DGEQRF(&m, &n, a, &lda, &dwork[itau], &dwork[jwork], &lwork_qr, info);
        wrkopt = (i32)dwork[jwork] + jwork;

        if (wantu) {
            ihoush = jwork;
            k = ihoush;
            i = n;
        } else {
            k = 0;
        }

        for (j = 0; j < n - 1; j++) {
            if (wantu) {
                i = i - 1;
                SLC_DCOPY(&i, &a[(j + 1) + j * lda], &int1, &dwork[k], &int1);
                k = k + i;
            }
            for (ij = j + 1; ij < n; ij++) {
                a[ij + j * lda] = ZERO;
            }
        }

        ma = n;
        wrkopt = (wrkopt > k) ? wrkopt : k;
    } else {
        k = 0;
        ma = m;
        wrkopt = 1;
    }

    itauq = k;
    itaup = itauq + p;
    jwork = itaup + p;
    i32 lwork_brd = ldwork - jwork;
    SLC_DGEBRD(&ma, &n, a, &lda, q, &q[p], &dwork[itauq], &dwork[itaup],
               &dwork[jwork], &lwork_brd, info);
    i32 t = (i32)dwork[jwork] + jwork;
    wrkopt = (wrkopt > t) ? wrkopt : t;

    if (wantu) {
        if (all) {
            ju = m;
        } else {
            ju = p;
        }
        SLC_DLASET("Full", &m, &ju, &ZERO, &ONE, u, &ldu);
        jobuy = 'U';
    } else {
        jobuy = 'N';
    }
    if (wantv) {
        if (all) {
            jv = n;
        } else {
            jv = p;
        }
        SLC_DLASET("Full", &n, &jv, &ZERO, &ONE, v, &ldv);
        jobvy = 'U';
    } else {
        jobvy = 'N';
    }

    if (m < n) {
        for (i = 0; i < p - 1; i++) {
            SLC_DLARTG(&q[i], &q[p + i], &cs, &sn, &temp);
            q[i] = temp;
            q[p + i] = sn * q[i + 1];
            q[i + 1] = cs * q[i + 1];
            if (wantu) {
                dwork[jwork + i] = cs;
                dwork[jwork + p - 1 + i] = sn;
            }
        }

        if (wantu) {
            SLC_DLASR("Right", "Variable pivot", "Forward", &m, &ju,
                      &dwork[jwork], &dwork[jwork + p - 1], u, &ldu);
        }
    }

    char jobuy_str[2] = {jobuy, '\0'};
    char jobvy_str[2] = {jobvy, '\0'};
    i32 lwork_yd = ldwork - jwork;
    mb04yd(jobuy_str, jobvy_str, m, n, rank, theta, q, &q[p], u, ldu,
           v, ldv, inul, tol, reltol, &dwork[jwork], lwork_yd, iwarn, info);

    if (wantu || wantv) {
        t = jwork - 6 + 8 * p;
        wrkopt = (wrkopt > t) ? wrkopt : t;
    } else {
        t = jwork - 4 + 6 * p;
        wrkopt = (wrkopt > t) ? wrkopt : t;
    }
    if (*info > 0) {
        return;
    }

    mb04xy(jobu, jobv, ma, n, a, lda, &dwork[itauq], &dwork[itaup],
           u, ldu, v, ldv, inul, info);

    if (qr && wantu) {
        if (all) {
            for (i = p; i < m; i++) {
                inul[i] = true;
            }
        }
        k = ihoush;
        i = n;

        for (j = 0; j < n - 1; j++) {
            i = i - 1;
            SLC_DCOPY(&i, &dwork[k], &int1, &a[(j + 1) + j * lda], &int1);
            k = k + i;
        }

        jwork = p;
        f64 dummy;
        mb04xy(jobu, "N", m, n, a, lda, &dwork[itau], &dwork[itau],
               u, ldu, &dummy, 1, inul, info);
        wrkopt = (wrkopt > pp1) ? wrkopt : pp1;
    }

    dwork[0] = (f64)wrkopt;
}
