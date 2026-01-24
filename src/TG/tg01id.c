/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <stdbool.h>

/**
 * @brief Orthogonal reduction of descriptor system to observability staircase form.
 *
 * TG01ID computes orthogonal transformation matrices Q and Z which
 * reduce the N-th order descriptor system (A-lambda*E,B,C) to the form:
 *
 *   Q'*A*Z = ( Ano  * )    Q'*E*Z = ( Eno  * )    Q'*B = ( Bno )
 *            ( 0   Ao )             ( 0   Eo )           ( Bo  )
 *
 *   C*Z = ( 0   Co )
 *
 * where the NOBSV-th order descriptor system (Ao-lambda*Eo,Bo,Co)
 * is finite and/or infinite observable.
 *
 * @param[in] jobobs Specifies observability form:
 *                   'O': separate both finite and infinite unobservable eigenvalues
 *                   'F': separate only finite unobservable eigenvalues
 *                   'I': separate only nonzero finite and infinite unobservable eigenvalues
 * @param[in] compq  Specifies whether Q is computed:
 *                   'N': do not compute Q
 *                   'I': initialize Q to identity, return orthogonal Q
 *                   'U': on entry Q contains orthogonal Q1, return Q1*Q
 * @param[in] compz  Specifies whether Z is computed:
 *                   'N': do not compute Z
 *                   'I': initialize Z to identity, return orthogonal Z
 *                   'U': on entry Z contains orthogonal Z1, return Z1*Z
 * @param[in] n      Order of matrices A and E (N >= 0)
 * @param[in] m      Number of columns of B (M >= 0)
 * @param[in] p      Number of rows of C (P >= 0)
 * @param[in,out] a  N-by-N state matrix A (transformed to Q'*A*Z on exit)
 * @param[in] lda    Leading dimension of A (LDA >= max(1,N))
 * @param[in,out] e  N-by-N descriptor matrix E (transformed to Q'*E*Z on exit)
 * @param[in] lde    Leading dimension of E (LDE >= max(1,N))
 * @param[in,out] b  N-by-M input matrix B (transformed to Q'*B on exit)
 * @param[in] ldb    Leading dimension of B
 * @param[in,out] c  P-by-N output matrix C (transformed to C*Z on exit)
 * @param[in] ldc    Leading dimension of C (LDC >= max(1,M,P))
 * @param[in,out] q  N-by-N left transformation matrix
 * @param[in] ldq    Leading dimension of Q
 * @param[in,out] z  N-by-N right transformation matrix
 * @param[in] ldz    Leading dimension of Z
 * @param[out] nobsv Order of observable part
 * @param[out] niuobs Number of unobservable infinite eigenvalues (JOBOBS='O')
 * @param[out] nlblck Number of full column rank blocks in staircase form
 * @param[out] ctau  Column dimensions of full rank blocks (dimension N)
 * @param[in] tol    Tolerance for rank determination (TOL < 1)
 * @param[out] iwork Integer workspace (dimension P)
 * @param[out] dwork Double workspace (dimension max(N, 2*P))
 * @param[out] info  Error indicator (0 = success, -i = i-th argument invalid)
 */
void tg01id(
    const char* jobobs, const char* compq, const char* compz,
    const i32 n, const i32 m, const i32 p,
    f64* a, const i32 lda,
    f64* e, const i32 lde,
    f64* b, const i32 ldb,
    f64* c, const i32 ldc,
    f64* q, const i32 ldq,
    f64* z, const i32 ldz,
    i32* nobsv, i32* niuobs, i32* nlblck, i32* ctau,
    const f64 tol,
    i32* iwork, f64* dwork,
    i32* info
)
{
    const f64 one = 1.0;
    bool finobs, infobs, ilq, ilz;
    i32 i, icompq, icompz, lba, lbe, nr;
    f64 dum[1];
    char jobq[2], jobz[2];
    const i32 int1 = 1;

    *info = 0;

    if (jobobs[0] == 'O' || jobobs[0] == 'o') {
        finobs = true;
        infobs = true;
    } else if (jobobs[0] == 'F' || jobobs[0] == 'f') {
        finobs = true;
        infobs = false;
    } else if (jobobs[0] == 'I' || jobobs[0] == 'i') {
        finobs = false;
        infobs = true;
    } else {
        finobs = false;
        infobs = false;
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

    i32 max1n = (1 > n) ? 1 : n;
    i32 max1mp = (1 > m) ? 1 : m;
    if (p > max1mp) max1mp = p;

    if (!finobs && !infobs) {
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
    } else if (lda < max1n) {
        *info = -8;
    } else if (lde < max1n) {
        *info = -10;
    } else if (ldb < 1 || (m > 0 && ldb < n)) {
        *info = -12;
    } else if (ldc < max1mp) {
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

    ab07md('Z', n, m, p, a, lda, b, ldb, c, ldc, dum, 1);

    for (i = 1; i < n; i++) {
        i32 len = i;
        SLC_DSWAP(&len, &e[i + 0 * lde], &lde, &e[0 + i * lde], &int1);
    }

    i32 n1_param = (n > 0) ? n : 0;
    i32 lbe_param = (n > 1) ? n - 1 : 0;

    if (finobs) {
        tg01hx(jobz, jobq, n, n, p, m, n1_param, lbe_param,
               a, lda, e, lde, b, ldb, c, ldc, z, ldz, q, ldq,
               &nr, nlblck, ctau, tol, iwork, dwork, info);

        if (*nlblck > 1) {
            lba = ctau[0] + ctau[1] - 1;
        } else if (*nlblck == 1) {
            lba = ctau[0] - 1;
        } else {
            lba = 0;
        }
        if (ilq) {
            jobq[0] = 'U';
        }
        if (ilz) {
            jobz[0] = 'U';
        }
        lbe = 0;
    } else {
        nr = n;
        lba = (n > 1) ? n - 1 : 0;
        lbe = lba;
    }

    if (infobs) {
        tg01hx(jobz, jobq, n, n, p, m, nr, lba,
               e, lde, a, lda, b, ldb, c, ldc, z, ldz, q, ldq,
               nobsv, nlblck, ctau, tol, iwork, dwork, info);

        if (finobs) {
            *niuobs = nr - *nobsv;
        } else {
            *niuobs = 0;
        }
        if (*nlblck > 1) {
            lbe = ctau[0] + ctau[1] - 1;
        } else if (*nlblck == 1) {
            lbe = ctau[0] - 1;
        } else {
            lbe = 0;
        }
        lba = 0;
    } else {
        *nobsv = nr;
        *niuobs = 0;
    }

    i32 temp1 = *niuobs - 1;
    i32 temp2 = n - *nobsv - *niuobs - 1;
    if (temp1 > lba) lba = temp1;
    if (temp2 > lba) lba = temp2;

    if (p == 0 || nr == 0) {
        lbe = (n > 1) ? n - 1 : 0;
    }

    i32 ku_param = (n > 1) ? n - 1 : 0;
    tb01xd("Z", n, p, m, lba, ku_param, a, lda, b, ldb, c, ldc, dum, 1, info);

    ma02cd(n, lbe, ku_param, e, lde);

    if (ilz) {
        ma02bd('R', n, n, z, ldz);
    }
    if (ilq) {
        ma02bd('R', n, n, q, ldq);
    }
}
