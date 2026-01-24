/**
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * SB10VD - State feedback and output injection matrices for H2 optimal controller
 *
 * Computes the state feedback matrix F and output injection matrix H
 * for an H2 optimal n-state controller, by solving two Riccati equations.
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <stdlib.h>

void sb10vd(
    const i32 n,
    const i32 m,
    const i32 np,
    const i32 ncon,
    const i32 nmeas,
    const f64* a,
    const i32 lda,
    const f64* b,
    const i32 ldb,
    const f64* c,
    const i32 ldc,
    f64* f,
    const i32 ldf,
    f64* h,
    const i32 ldh,
    f64* x,
    const i32 ldx,
    f64* y,
    const i32 ldy,
    f64* xycond,
    i32* iwork,
    f64* dwork,
    const i32 ldwork,
    i32* bwork,
    i32* info)
{
    const f64 zero = 0.0;
    const f64 one = 1.0;
    const f64 mone = -1.0;

    i32 m1 = m - ncon;
    i32 m2 = ncon;
    i32 np1 = np - nmeas;
    i32 np2 = nmeas;

    *info = 0;

    if (n < 0) {
        *info = -1;
    } else if (m < 0) {
        *info = -2;
    } else if (np < 0) {
        *info = -3;
    } else if (ncon < 0 || m1 < 0 || m2 > np1) {
        *info = -4;
    } else if (nmeas < 0 || np1 < 0 || np2 > m1) {
        *info = -5;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -7;
    } else if (ldb < (n > 1 ? n : 1)) {
        *info = -9;
    } else if (ldc < (np > 1 ? np : 1)) {
        *info = -11;
    } else if (ldf < (ncon > 1 ? ncon : 1)) {
        *info = -13;
    } else if (ldh < (n > 1 ? n : 1)) {
        *info = -15;
    } else if (ldx < (n > 1 ? n : 1)) {
        *info = -17;
    } else if (ldy < (n > 1 ? n : 1)) {
        *info = -19;
    } else {
        i32 minwrk = 13 * n * n + 12 * n + 5;
        if (ldwork < minwrk) {
            *info = -23;
        }
    }

    if (*info != 0) {
        return;
    }

    if (n == 0 || m == 0 || np == 0 || m1 == 0 || m2 == 0 || np1 == 0 || np2 == 0) {
        dwork[0] = one;
        xycond[0] = one;
        xycond[1] = one;
        return;
    }

    i32 nd1 = np1 - m2;
    i32 nd2 = m1 - np2;
    i32 n2 = 2 * n;
    i32 nn = n * n;

    i32 iwq = nn;
    i32 iwg = iwq + nn;
    i32 iwt = iwg + nn;
    i32 iwv = iwt + nn;
    i32 iwr = iwv + nn;
    i32 iwi = iwr + n2;
    i32 iws = iwi + n2;
    i32 iwrk = iws + 4 * nn;

    SLC_DLACPY("F", &n, &n, a, &lda, dwork, &n);
    SLC_DGEMM("N", "N", &n, &n, &m2, &mone, &b[m1 * ldb], &ldb,
              &c[nd1], &ldc, &one, dwork, &n);

    if (nd1 > 0) {
        SLC_DSYRK("L", "T", &n, &nd1, &one, c, &ldc, &zero, &dwork[iwq], &n);
    } else {
        SLC_DLASET("L", &n, &n, &zero, &zero, &dwork[iwq], &n);
    }

    SLC_DSYRK("L", "N", &n, &m2, &one, &b[m1 * ldb], &ldb, &zero, &dwork[iwg], &n);

    f64 sep, ferr;
    i32 info2;
    i32 ldwork_sb02rd = ldwork - iwrk;

    sb02rd("A", "C", "D", "N", "L", "G", "S", "N", "O",
           n, dwork, n, &dwork[iwt], n, &dwork[iwv], n,
           &dwork[iwg], n, &dwork[iwq], n, x, ldx,
           &sep, &xycond[0], &ferr,
           &dwork[iwr], &dwork[iwi], &dwork[iws], n2,
           iwork, &dwork[iwrk], ldwork_sb02rd, bwork, &info2);

    if (info2 > 0) {
        *info = 1;
        return;
    }

    i32 lwamax = (i32)dwork[iwrk] + iwrk;

    SLC_DLACPY("F", &m2, &n, &c[nd1], &ldc, f, &ldf);
    SLC_DGEMM("T", "N", &m2, &n, &n, &mone, &b[m1 * ldb], &ldb, x, &ldx,
              &mone, f, &ldf);

    SLC_DLACPY("F", &n, &n, a, &lda, dwork, &n);
    SLC_DGEMM("N", "N", &n, &n, &np2, &mone, &b[nd2 * ldb], &ldb,
              &c[np1], &ldc, &one, dwork, &n);

    if (nd2 > 0) {
        SLC_DSYRK("U", "N", &n, &nd2, &one, b, &ldb, &zero, &dwork[iwq], &n);
    } else {
        SLC_DLASET("U", &n, &n, &zero, &zero, &dwork[iwq], &n);
    }

    SLC_DSYRK("U", "T", &n, &np2, &one, &c[np1], &ldc, &zero, &dwork[iwg], &n);

    sb02rd("A", "C", "D", "T", "U", "G", "S", "N", "O",
           n, dwork, n, &dwork[iwt], n, &dwork[iwv], n,
           &dwork[iwg], n, &dwork[iwq], n, y, ldy,
           &sep, &xycond[1], &ferr,
           &dwork[iwr], &dwork[iwi], &dwork[iws], n2,
           iwork, &dwork[iwrk], ldwork_sb02rd, bwork, &info2);

    if (info2 > 0) {
        *info = 2;
        return;
    }

    i32 lwamax2 = (i32)dwork[iwrk] + iwrk;
    lwamax = lwamax > lwamax2 ? lwamax : lwamax2;

    SLC_DLACPY("F", &n, &np2, &b[nd2 * ldb], &ldb, h, &ldh);
    SLC_DGEMM("N", "T", &n, &np2, &n, &mone, y, &ldy, &c[np1], &ldc,
              &mone, h, &ldh);

    dwork[0] = (f64)lwamax;
}
