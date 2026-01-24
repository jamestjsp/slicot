/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot/tg.h"
#include "slicot/ab.h"
#include "slicot/ma.h"
#include "slicot/tb.h"
#include "slicot_blas.h"

#include <stdbool.h>

static void tg01ly_internal(
    const bool compq, const bool compz,
    const i32 n, const i32 m, const i32 p,
    const i32 ranke, const i32 rnka22,
    f64* a, const i32 lda,
    f64* e, const i32 lde,
    f64* b, const i32 ldb,
    f64* c, const i32 ldc,
    f64* q, const i32 ldq,
    f64* z, const i32 ldz,
    i32* nf, i32* niblck, i32* iblck,
    const f64 tol,
    i32* iwork, f64* dwork, const i32 ldwork,
    i32* info
);

void tg01ld(
    const char* job, const char* joba, const char* compq, const char* compz,
    const i32 n, const i32 m, const i32 p,
    f64* a, const i32 lda,
    f64* e, const i32 lde,
    f64* b, const i32 ldb,
    f64* c, const i32 ldc,
    f64* q, const i32 ldq,
    f64* z, const i32 ldz,
    i32* nf, i32* nd, i32* niblck, i32* iblck,
    const f64 tol,
    i32* iwork, f64* dwork, const i32 ldwork,
    i32* info
)
{
    const f64 one = 1.0;

    bool ilq, ilz, lquery, reda, redif;
    i32 icompq, icompz, ihi = 0, ilo = 1, minwrk, ranke = 0, rnka22 = 0, wrkopt = 1;
    char jobq[2], jobz[2];
    f64 dum[1];
    i32 i;
    i32 int1 = 1;
    i32 max_mp = (m > p) ? m : p;

    if (compq[0] == 'N' || compq[0] == 'n') {
        ilq = false;
        icompq = 1;
    } else if (compq[0] == 'U' || compq[0] == 'u') {
        ilq = true;
        icompq = 2;
    } else if (compq[0] == 'I' || compq[0] == 'i') {
        ilq = true;
        icompq = 3;
    } else {
        icompq = 0;
    }

    if (compz[0] == 'N' || compz[0] == 'n') {
        ilz = false;
        icompz = 1;
    } else if (compz[0] == 'U' || compz[0] == 'u') {
        ilz = true;
        icompz = 2;
    } else if (compz[0] == 'I' || compz[0] == 'i') {
        ilz = true;
        icompz = 3;
    } else {
        icompz = 0;
    }

    redif = (job[0] == 'I' || job[0] == 'i');
    reda = (joba[0] == 'H' || joba[0] == 'h');

    *info = 0;

    if (!(job[0] == 'F' || job[0] == 'f') && !redif) {
        *info = -1;
    } else if (!(joba[0] == 'N' || joba[0] == 'n') && !reda) {
        *info = -2;
    } else if (icompq <= 0) {
        *info = -3;
    } else if (icompz <= 0) {
        *info = -4;
    } else if (n < 0) {
        *info = -5;
    } else if (m < 0) {
        *info = -6;
    } else if (p < 0) {
        *info = -7;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -9;
    } else if (lde < (n > 1 ? n : 1)) {
        *info = -11;
    } else if (ldb < (n > 1 ? n : 1)) {
        *info = -13;
    } else if ((!redif && ldc < (p > 1 ? p : 1)) ||
               (redif && ldc < (max_mp > 1 ? max_mp : 1))) {
        *info = -15;
    } else if ((ilq && ldq < n) || ldq < 1) {
        *info = -17;
    } else if ((ilz && ldz < n) || ldz < 1) {
        *info = -19;
    } else if (tol >= one) {
        *info = -24;
    } else {
        lquery = (ldwork == -1);
        if (n == 0) {
            minwrk = 1;
        } else {
            i32 temp1 = (3 * n > m) ? 3 * n : m;
            temp1 = (temp1 > p) ? temp1 : p;
            minwrk = n + temp1;
        }

        if (lquery) {
            ranke = (n > 2) ? (n / 2) : 1;
            if (n == 0) ranke = 0;
            rnka22 = n - ranke;

            if (redif) {
                tg01fd(compz, compq, "T", n, n, p, m,
                       a, lda, e, lde, b, ldb, c, ldc, z, ldz, q, ldq,
                       &ranke, &rnka22, tol, iwork, dwork, -1, info);
                wrkopt = (minwrk > (i32)dwork[0]) ? minwrk : (i32)dwork[0];

                tg01ly_internal(ilz, ilq, n, p, m, ranke, rnka22,
                               a, lda, e, lde, b, ldb, c, ldc, z, ldz, q, ldq,
                               nf, niblck, iblck, tol, iwork, dwork, -1, info);
            } else {
                tg01fd(compq, compz, "T", n, n, m, p,
                       a, lda, e, lde, b, ldb, c, ldc, q, ldq, z, ldz,
                       &ranke, &rnka22, tol, iwork, dwork, -1, info);
                wrkopt = (minwrk > (i32)dwork[0]) ? minwrk : (i32)dwork[0];

                tg01ly_internal(ilq, ilz, n, m, p, ranke, rnka22,
                               a, lda, e, lde, b, ldb, c, ldc, q, ldq, z, ldz,
                               nf, niblck, iblck, tol, iwork, dwork, -1, info);
            }
            wrkopt = (wrkopt > (i32)dwork[0]) ? wrkopt : (i32)dwork[0];
        }

        if (ldwork < minwrk && !lquery) {
            *info = -27;
        }
    }

    if (*info != 0) {
        i32 neg_info = -(*info);
        SLC_XERBLA("TG01LD", &neg_info);
        return;
    } else if (lquery) {
        dwork[0] = (f64)wrkopt;
        return;
    }

    if (n == 0) {
        *nf = 0;
        *nd = 0;
        *niblck = 0;
        dwork[0] = one;
        return;
    }

    if (redif) {
        ab07md('Z', n, m, p, a, lda, b, ldb, c, ldc, dum, 1);

        for (i = 1; i < n; i++) {
            SLC_DSWAP(&i, &e[i], &lde, &e[i * lde], &int1);
        }

        tg01fd(compz, compq, "T", n, n, p, m,
               a, lda, e, lde, b, ldb, c, ldc, z, ldz, q, ldq,
               &ranke, &rnka22, tol, iwork, dwork, ldwork, info);
        wrkopt = (minwrk > (i32)dwork[0]) ? minwrk : (i32)dwork[0];

        tg01ly_internal(ilz, ilq, n, p, m, ranke, rnka22,
                       a, lda, e, lde, b, ldb, c, ldc, z, ldz, q, ldq,
                       nf, niblck, iblck, tol, iwork, dwork, ldwork, info);
        ilo = n - *nf + 1;
        ihi = n;
    } else {
        tg01fd(compq, compz, "T", n, n, m, p,
               a, lda, e, lde, b, ldb, c, ldc, q, ldq, z, ldz,
               &ranke, &rnka22, tol, iwork, dwork, ldwork, info);
        wrkopt = (minwrk > (i32)dwork[0]) ? minwrk : (i32)dwork[0];

        tg01ly_internal(ilq, ilz, n, m, p, ranke, rnka22,
                       a, lda, e, lde, b, ldb, c, ldc, q, ldq, z, ldz,
                       nf, niblck, iblck, tol, iwork, dwork, ldwork, info);
        ilo = 1;
        ihi = *nf;
    }

    if (*info != 0)
        return;
    wrkopt = (wrkopt > (i32)dwork[0]) ? wrkopt : (i32)dwork[0];

    *nd = n - ranke;

    if (redif) {
        i32 nfm1 = *nf - 1;
        i32 nm1 = n - 1;
        i32 max_nfm1 = (nfm1 > 0) ? nfm1 : 0;
        i32 max_nm1 = (nm1 > 0) ? nm1 : 0;
        i32 tb01xd_info;

        tb01xd("Z", n, p, m, max_nfm1, max_nm1,
               a, lda, b, ldb, c, ldc, dum, 1, &tb01xd_info);

        ma02cd(n, 0, max_nm1, e, lde);

        if (ilq)
            ma02bd('R', n, n, q, ldq);
        if (ilz)
            ma02bd('R', n, n, z, ldz);
    }

    if (reda) {
        if (ilq) {
            jobq[0] = 'V'; jobq[1] = '\0';
        } else {
            jobq[0] = 'N'; jobq[1] = '\0';
        }
        if (ilz) {
            jobz[0] = 'V'; jobz[1] = '\0';
        } else {
            jobz[0] = 'N'; jobz[1] = '\0';
        }

        i32 tg01bd_info;
        tg01bd("U", jobq, jobz, n, m, p, ilo, ihi,
               a, lda, e, lde, b, ldb, c, ldc, q, ldq, z, ldz,
               dwork, ldwork, &tg01bd_info);
    }

    dwork[0] = (f64)wrkopt;
}

static void tg01ly_internal(
    const bool compq, const bool compz,
    const i32 n, const i32 m, const i32 p,
    const i32 ranke, const i32 rnka22,
    f64* a, const i32 lda,
    f64* e, const i32 lde,
    f64* b, const i32 ldb,
    f64* c, const i32 ldc,
    f64* q, const i32 ldq,
    f64* z, const i32 ldz,
    i32* nf, i32* niblck, i32* iblck,
    const f64 tol,
    i32* iwork, f64* dwork, const i32 ldwork,
    i32* info
)
{
    tg01ly(compq, compz, n, m, p, ranke, rnka22,
           a, lda, e, lde, b, ldb, c, ldc, q, ldq, z, ldz,
           nf, niblck, iblck, tol, iwork, dwork, ldwork, info);
}
