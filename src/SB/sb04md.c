/**
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * SB04MD - Solve continuous-time Sylvester equation AX + XB = C
 *
 * Uses Hessenberg-Schur method: A is reduced to upper Hessenberg form,
 * B is reduced to real Schur form.
 */

#include "slicot.h"
#include "slicot_blas.h"

static int select_none(const f64* wr, const f64* wi)
{
    (void)wr;
    (void)wi;
    return 0;
}

void sb04md(i32 n, i32 m, f64* a, i32 lda, f64* b, i32 ldb,
            f64* c, i32 ldc, f64* z, i32 ldz,
            i32* iwork, f64* dwork, i32 ldwork, i32* info)
{
    const f64 zero = 0.0;
    const f64 one = 1.0;

    i32 i, ieig, ifail, ilo, ihi, ind, itau, jwork, mindw, sdim, wrkopt;
    bool lquery;
    i32 bwork_dummy = 0;

    *info = 0;
    lquery = (ldwork == -1);

    if (n < 0) {
        *info = -1;
    } else if (m < 0) {
        *info = -2;
    } else if (lda < 1 || lda < n) {
        *info = -4;
    } else if (ldb < 1 || ldb < m) {
        *info = -6;
    } else if (ldc < 1 || ldc < n) {
        *info = -8;
    } else if (ldz < 1 || ldz < m) {
        *info = -10;
    } else {
        ilo = 1;
        ihi = n;

        i32 t1 = 2 * n * n + 8 * n;
        i32 t2 = 5 * m;
        i32 t3 = n + m;
        mindw = 1;
        if (t1 > mindw) mindw = t1;
        if (t2 > mindw) mindw = t2;
        if (t3 > mindw) mindw = t3;

        if (lquery) {
            i32 neg1 = -1;
            SLC_DGEES("V", "N", select_none, &m, b, &ldb, &sdim,
                      dwork, dwork, z, &ldz, dwork, &neg1, &bwork_dummy, &ifail);
            wrkopt = mindw;
            i32 opt1 = 2 * m + (i32)dwork[0];
            if (opt1 > wrkopt) wrkopt = opt1;

            SLC_DGEHRD(&n, &ilo, &ihi, a, &lda, dwork, dwork, &neg1, &ifail);
            i32 opt2 = n + (i32)dwork[0];
            if (opt2 > wrkopt) wrkopt = opt2;

            SLC_DORMHR("L", "T", &n, &m, &ilo, &ihi, a, &lda, dwork, c, &ldc,
                       dwork, &neg1, &ifail);
            i32 opt3 = n + (i32)dwork[0];
            if (opt3 > wrkopt) wrkopt = opt3;

            SLC_DORMHR("L", "N", &n, &m, &ilo, &ihi, a, &lda, dwork, c, &ldc,
                       dwork, &neg1, &ifail);
            i32 opt4 = n + (i32)dwork[0];
            if (opt4 > wrkopt) wrkopt = opt4;

            dwork[0] = (f64)wrkopt;
            return;
        } else if (ldwork < mindw) {
            *info = -13;
        }
    }

    if (*info != 0) {
        return;
    }

    if (n == 0 || m == 0) {
        dwork[0] = one;
        return;
    }

    wrkopt = 1;

    i32 int1 = 1;
    for (i = 2; i <= m; i++) {
        i32 len = i - 1;
        SLC_DSWAP(&len, &b[(i - 1) * ldb], &int1, &b[i - 1], &ldb);
    }

    ieig = m + 1;
    jwork = ieig + m;
    i32 lwork_dgees = ldwork - jwork + 1;
    SLC_DGEES("V", "N", select_none, &m, b, &ldb, &sdim,
              dwork, &dwork[ieig - 1], z, &ldz, &dwork[jwork - 1],
              &lwork_dgees, &bwork_dummy, info);
    if (*info != 0) return;
    i32 opt1 = (i32)dwork[jwork - 1] + jwork - 1;
    if (opt1 > wrkopt) wrkopt = opt1;

    itau = 2;
    jwork = itau + n - 1;
    i32 lwork_dgehrd = ldwork - jwork + 1;
    SLC_DGEHRD(&n, &ilo, &ihi, a, &lda, &dwork[itau - 1], &dwork[jwork - 1],
               &lwork_dgehrd, &ifail);
    i32 opt2 = (i32)dwork[jwork - 1] + jwork - 1;
    if (opt2 > wrkopt) wrkopt = opt2;

    i32 lwork_dormhr = ldwork - jwork + 1;
    SLC_DORMHR("L", "T", &n, &m, &ilo, &ihi, a, &lda, &dwork[itau - 1],
               c, &ldc, &dwork[jwork - 1], &lwork_dormhr, &ifail);
    i32 opt3 = (i32)dwork[jwork - 1] + jwork - 1;
    if (opt3 > wrkopt) wrkopt = opt3;

    if (ldwork >= jwork - 1 + n * m) {
        SLC_DGEMM("N", "N", &n, &m, &m, &one, c, &ldc, z, &ldz,
                  &zero, &dwork[jwork - 1], &n);
        SLC_DLACPY("F", &n, &m, &dwork[jwork - 1], &n, c, &ldc);
        i32 opt4 = jwork - 1 + n * m;
        if (opt4 > wrkopt) wrkopt = opt4;
    } else {
        for (i = 1; i <= n; i++) {
            SLC_DGEMV("T", &m, &m, &one, z, &ldz, &c[(i - 1)], &ldc,
                      &zero, &dwork[jwork - 1], &int1);
            SLC_DCOPY(&m, &dwork[jwork - 1], &int1, &c[(i - 1)], &ldc);
        }
    }

    ind = m;
    while (ind > 1) {
        if (b[(ind - 1) + (ind - 2) * ldb] == zero) {
            sb04my(m, n, ind, a, lda, b, ldb, c, ldc, &dwork[jwork - 1], iwork, info);
            if (*info != 0) {
                *info = *info + m;
                return;
            }
            i32 opt5 = jwork + n * (n + 1) / 2 + 2 * n - 1;
            if (opt5 > wrkopt) wrkopt = opt5;
            ind = ind - 1;
        } else {
            sb04mu(m, n, ind, a, lda, b, ldb, c, ldc, &dwork[jwork - 1], iwork, info);
            if (*info != 0) {
                *info = *info + m;
                return;
            }
            i32 opt6 = jwork + 2 * n * n + 7 * n - 1;
            if (opt6 > wrkopt) wrkopt = opt6;
            ind = ind - 2;
        }
    }

    if (ind == 1) {
        sb04my(m, n, ind, a, lda, b, ldb, c, ldc, &dwork[jwork - 1], iwork, info);
        if (*info != 0) {
            *info = *info + m;
            return;
        }
        i32 opt7 = jwork + n * (n + 1) / 2 + 2 * n - 1;
        if (opt7 > wrkopt) wrkopt = opt7;
    }

    lwork_dormhr = ldwork - jwork + 1;
    SLC_DORMHR("L", "N", &n, &m, &ilo, &ihi, a, &lda, &dwork[itau - 1],
               c, &ldc, &dwork[jwork - 1], &lwork_dormhr, &ifail);
    i32 opt8 = (i32)dwork[jwork - 1] + jwork - 1;
    if (opt8 > wrkopt) wrkopt = opt8;

    if (ldwork >= jwork - 1 + n * m) {
        SLC_DGEMM("N", "T", &n, &m, &m, &one, c, &ldc, z, &ldz,
                  &zero, &dwork[jwork - 1], &n);
        SLC_DLACPY("F", &n, &m, &dwork[jwork - 1], &n, c, &ldc);
    } else {
        for (i = 1; i <= n; i++) {
            SLC_DGEMV("N", &m, &m, &one, z, &ldz, &c[(i - 1)], &ldc,
                      &zero, &dwork[jwork - 1], &int1);
            SLC_DCOPY(&m, &dwork[jwork - 1], &int1, &c[(i - 1)], &ldc);
        }
    }

    dwork[0] = (f64)wrkopt;
}
