// SPDX-License-Identifier: BSD-3-Clause

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>

void mb04tb(const char *trana, const char *tranb, i32 n, i32 ilo,
            f64 *a, i32 lda, f64 *b, i32 ldb, f64 *g, i32 ldg,
            f64 *q, i32 ldq, f64 *csl, f64 *csr,
            f64 *taul, f64 *taur, f64 *dwork, i32 ldwork, i32 *info) {

    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    bool ltra = (trana[0] == 'T' || trana[0] == 't' ||
                 trana[0] == 'C' || trana[0] == 'c');
    bool ltrb = (tranb[0] == 'T' || tranb[0] == 't' ||
                 tranb[0] == 'C' || tranb[0] == 'c');

    *info = 0;
    i32 minwrk = (n > 1) ? n : 1;

    if (!ltra && !(trana[0] == 'N' || trana[0] == 'n')) {
        *info = -1;
    } else if (!ltrb && !(tranb[0] == 'N' || tranb[0] == 'n')) {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    } else if (ilo < 1 || ilo > n + 1) {
        *info = -4;
    } else if (lda < minwrk) {
        *info = -6;
    } else if (ldb < minwrk) {
        *info = -8;
    } else if (ldg < minwrk) {
        *info = -10;
    } else if (ldq < minwrk) {
        *info = -12;
    }

    if (*info != 0) {
        return;
    }

    bool lquery = (ldwork == -1);

    if (ldwork < minwrk && !lquery) {
        dwork[0] = (f64)minwrk;
        *info = -18;
        return;
    }

    i32 wrkopt = 1;
    if (n == 0) {
        wrkopt = 1;
    } else {
        i32 nb = 32;
        wrkopt = 16 * n * nb + 5 * nb;
        if (wrkopt < minwrk) wrkopt = minwrk;
    }

    if (lquery) {
        dwork[0] = (f64)wrkopt;
        return;
    }

    for (i32 i = 0; i < ilo - 1; i++) {
        csl[2 * i] = ONE;
        csl[2 * i + 1] = ZERO;
        taul[i] = ZERO;
        if (i < n - 1) {
            csr[2 * i] = ONE;
            csr[2 * i + 1] = ZERO;
            taur[i] = ZERO;
        }
    }

    i32 nh = n - ilo + 1;
    if (nh == 0) {
        dwork[0] = ONE;
        return;
    }

    i32 ierr;
    mb04ts(trana, tranb, n, ilo, a, lda, b, ldb, g, ldg, q, ldq,
           csl, csr, taul, taur, dwork, ldwork, &ierr);

    dwork[0] = (f64)wrkopt;
}
