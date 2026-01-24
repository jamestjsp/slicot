/**
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * MB02OD - Solve triangular matrix equation with condition estimation
 *
 * Solves op(A)*X = alpha*B or X*op(A) = alpha*B where A is triangular,
 * only if the reciprocal condition number of A exceeds a given tolerance.
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <ctype.h>

void mb02od(
    const char* side_str,
    const char* uplo_str,
    const char* trans_str,
    const char* diag_str,
    const char* norm_str,
    i32 m,
    i32 n,
    f64 alpha,
    const f64* a,
    i32 lda,
    f64* b,
    i32 ldb,
    f64* rcond,
    f64 tol,
    i32* iwork,
    f64* dwork,
    i32* info)
{
    const f64 zero = 0.0;
    const f64 one = 1.0;

    char side = toupper((unsigned char)side_str[0]);
    char uplo = toupper((unsigned char)uplo_str[0]);
    char trans = toupper((unsigned char)trans_str[0]);
    char diag = toupper((unsigned char)diag_str[0]);
    char norm = toupper((unsigned char)norm_str[0]);

    bool lside = (side == 'L');
    i32 nrowa = lside ? m : n;
    bool onenrm = (norm == '1' || norm == 'O');

    *info = 0;

    if (!lside && side != 'R') {
        *info = -1;
    } else if (uplo != 'U' && uplo != 'L') {
        *info = -2;
    } else if (trans != 'N' && trans != 'T' && trans != 'C') {
        *info = -3;
    } else if (diag != 'U' && diag != 'N') {
        *info = -4;
    } else if (!onenrm && norm != 'I') {
        *info = -5;
    } else if (m < 0) {
        *info = -6;
    } else if (n < 0) {
        *info = -7;
    } else if (lda < (nrowa > 1 ? nrowa : 1)) {
        *info = -10;
    } else if (ldb < (m > 1 ? m : 1)) {
        *info = -12;
    }

    if (*info != 0) {
        return;
    }

    if (nrowa == 0) {
        *rcond = one;
        return;
    }

    f64 toldef = tol;
    if (toldef <= zero) {
        f64 eps = SLC_DLAMCH("Epsilon");
        toldef = (f64)(nrowa * nrowa) * eps;
    }

    SLC_DTRCON(&norm, &uplo, &diag, &nrowa, a, &lda, rcond, dwork, iwork, info);

    if (*rcond > toldef) {
        SLC_DTRSM(&side, &uplo, &trans, &diag, &m, &n, &alpha, a, &lda, b, &ldb);
    } else {
        *info = 1;
    }
}
