/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <stdbool.h>

void tg01hd(
    const char* jobcon, const char* compq, const char* compz,
    const i32 n, const i32 m, const i32 p,
    f64* a, const i32 lda,
    f64* e, const i32 lde,
    f64* b, const i32 ldb,
    f64* c, const i32 ldc,
    f64* q, const i32 ldq,
    f64* z, const i32 ldz,
    i32* ncont, i32* niucon, i32* nrblck, i32* rtau,
    const f64 tol,
    i32* iwork, f64* dwork,
    i32* info
)
{
    const f64 one = 1.0;

    bool fincon, infcon, ilq, ilz;
    i32 icompq, icompz, lba, nr;
    char jobq[2], jobz[2];

    if (jobcon[0] == 'C' || jobcon[0] == 'c') {
        fincon = true;
        infcon = true;
    } else if (jobcon[0] == 'F' || jobcon[0] == 'f') {
        fincon = true;
        infcon = false;
    } else if (jobcon[0] == 'I' || jobcon[0] == 'i') {
        fincon = false;
        infcon = true;
    } else {
        fincon = false;
        infcon = false;
    }

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

    *info = 0;
    if (!fincon && !infcon) {
        *info = -1;
    } else if (icompq <= 0) {
        *info = -2;
    } else if (icompz <= 0) {
        *info = -3;
    } else if (n < 0) {
        *info = -4;
    } else if (m < 0) {
        *info = -5;
    } else if (p < 0) {
        *info = -6;
    } else if (lda < ((1 > n) ? 1 : n)) {
        *info = -8;
    } else if (lde < ((1 > n) ? 1 : n)) {
        *info = -10;
    } else if (ldb < ((1 > n) ? 1 : n)) {
        *info = -12;
    } else if (ldc < ((1 > p) ? 1 : p)) {
        *info = -14;
    } else if ((ilq && ldq < n) || ldq < 1) {
        *info = -16;
    } else if ((ilz && ldz < n) || ldz < 1) {
        *info = -18;
    } else if (tol >= one) {
        *info = -23;
    }

    if (*info != 0) {
        return;
    }

    jobq[0] = compq[0];
    jobq[1] = '\0';
    jobz[0] = compz[0];
    jobz[1] = '\0';

    if (fincon) {
        i32 n1_lbe = (n > 0) ? n - 1 : 0;
        tg01hx(jobq, jobz, n, n, m, p, n, n1_lbe, a, lda,
               e, lde, b, ldb, c, ldc, q, ldq, z, ldz, &nr,
               nrblck, rtau, tol, iwork, dwork, info);

        if (*nrblck > 1) {
            lba = rtau[0] + rtau[1] - 1;
        } else if (*nrblck == 1) {
            lba = rtau[0] - 1;
        } else {
            lba = 0;
        }

        if (ilq) {
            jobq[0] = 'U';
        }
        if (ilz) {
            jobz[0] = 'U';
        }
    } else {
        nr = n;
        lba = (n > 0) ? n - 1 : 0;
    }

    if (infcon) {
        tg01hx(jobq, jobz, n, n, m, p, nr, lba, e, lde,
               a, lda, b, ldb, c, ldc, q, ldq, z, ldz, ncont,
               nrblck, rtau, tol, iwork, dwork, info);

        if (fincon) {
            *niucon = nr - *ncont;
        } else {
            *niucon = 0;
        }
    } else {
        *ncont = nr;
        *niucon = 0;
    }
}
