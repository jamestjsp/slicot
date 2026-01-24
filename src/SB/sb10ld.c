/**
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * SB10LD - Compute closed-loop system matrices
 *
 * Computes the matrices of the closed-loop system G = (AC, BC, CC, DC) from
 * the open-loop plant P = (A, B, C, D) and controller K = (AK, BK, CK, DK).
 *
 * Based on SLICOT routine SB10LD.
 */

#include "slicot.h"
#include "slicot_blas.h"

void sb10ld(
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
    const f64* d,
    const i32 ldd,
    const f64* ak,
    const i32 ldak,
    const f64* bk,
    const i32 ldbk,
    const f64* ck,
    const i32 ldck,
    const f64* dk,
    const i32 lddk,
    f64* ac,
    const i32 ldac,
    f64* bc,
    const i32 ldbc,
    f64* cc,
    const i32 ldcc,
    f64* dc,
    const i32 lddc,
    i32* iwork,
    f64* dwork,
    const i32 ldwork,
    i32* info)
{
    const f64 zero = 0.0;
    const f64 one = 1.0;
    const f64 mone = -1.0;

    i32 n2 = 2 * n;
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
    } else if (ldd < (np > 1 ? np : 1)) {
        *info = -13;
    } else if (ldak < (n > 1 ? n : 1)) {
        *info = -15;
    } else if (ldbk < (n > 1 ? n : 1)) {
        *info = -17;
    } else if (ldck < (m2 > 1 ? m2 : 1)) {
        *info = -19;
    } else if (lddk < (m2 > 1 ? m2 : 1)) {
        *info = -21;
    } else if (ldac < (n2 > 1 ? n2 : 1)) {
        *info = -23;
    } else if (ldbc < (n2 > 1 ? n2 : 1)) {
        *info = -25;
    } else if (ldcc < (np1 > 1 ? np1 : 1)) {
        *info = -27;
    } else if (lddc < (np1 > 1 ? np1 : 1)) {
        *info = -29;
    } else {
        i32 minwrk = 2*m*m + np*np + 2*m*n + m*np + 2*n*np;
        if (ldwork < minwrk) {
            *info = -32;
        }
    }

    if (*info != 0) {
        return;
    }

    if (n == 0 || m == 0 || np == 0 || m1 == 0 || m2 == 0 ||
        np1 == 0 || np2 == 0) {
        dwork[0] = one;
        return;
    }

    f64 eps = SLC_DLAMCH("Epsilon");

    i32 iw2 = np2 * np2;
    i32 iw3 = iw2 + m2 * m2;
    i32 iw4 = iw3 + np2 * n;
    i32 iw5 = iw4 + m2 * n;
    i32 iw6 = iw5 + np2 * m1;
    i32 iw7 = iw6 + m2 * m1;
    i32 iw8 = iw7 + m2 * n;
    i32 iwrk = iw8 + np2 * n;

    SLC_DLASET("Full", &np2, &np2, &zero, &one, dwork, &np2);
    SLC_DGEMM("N", "N", &np2, &np2, &m2, &mone,
              &d[np1 + m1 * ldd], &ldd, dk, &lddk, &one, dwork, &np2);

    f64 anorm = SLC_DLANGE("1", &np2, &np2, dwork, &np2, &dwork[iwrk]);

    i32 info2 = 0;
    SLC_DGETRF(&np2, &np2, dwork, &np2, iwork, &info2);
    if (info2 > 0) {
        *info = 1;
        return;
    }

    f64 rcond;
    SLC_DGECON("1", &np2, dwork, &np2, &anorm, &rcond, &dwork[iwrk],
               &iwork[np2], &info2);
    i32 lwamax = (i32)dwork[iwrk] + iwrk;

    if (rcond < eps) {
        *info = 1;
        return;
    }

    i32 lwork = ldwork - iwrk;
    SLC_DGETRI(&np2, dwork, &np2, iwork, &dwork[iwrk], &lwork, &info2);
    i32 tmp = (i32)dwork[iwrk] + iwrk;
    lwamax = tmp > lwamax ? tmp : lwamax;

    SLC_DLASET("Full", &m2, &m2, &zero, &one, &dwork[iw2], &m2);
    SLC_DGEMM("N", "N", &m2, &m2, &np2, &mone, dk, &lddk,
              &d[np1 + m1 * ldd], &ldd, &one, &dwork[iw2], &m2);

    anorm = SLC_DLANGE("1", &m2, &m2, &dwork[iw2], &m2, &dwork[iwrk]);
    SLC_DGETRF(&m2, &m2, &dwork[iw2], &m2, iwork, &info2);
    if (info2 > 0) {
        *info = 2;
        return;
    }

    SLC_DGECON("1", &m2, &dwork[iw2], &m2, &anorm, &rcond, &dwork[iwrk],
               &iwork[m2], &info2);
    tmp = (i32)dwork[iwrk] + iwrk;
    lwamax = tmp > lwamax ? tmp : lwamax;

    if (rcond < eps) {
        *info = 2;
        return;
    }

    lwork = ldwork - iwrk;
    SLC_DGETRI(&m2, &dwork[iw2], &m2, iwork, &dwork[iwrk], &lwork, &info2);
    tmp = (i32)dwork[iwrk] + iwrk;
    lwamax = tmp > lwamax ? tmp : lwamax;

    SLC_DGEMM("N", "N", &np2, &n, &np2, &one, dwork, &np2,
              &c[np1], &ldc, &zero, &dwork[iw3], &np2);

    SLC_DGEMM("N", "N", &m2, &n, &np2, &one, dk, &lddk,
              &dwork[iw3], &np2, &zero, &dwork[iw4], &m2);

    SLC_DGEMM("N", "N", &np2, &m1, &np2, &one, dwork, &np2,
              &d[np1], &ldd, &zero, &dwork[iw5], &np2);

    SLC_DGEMM("N", "N", &m2, &m1, &np2, &one, dk, &lddk,
              &dwork[iw5], &np2, &zero, &dwork[iw6], &m2);

    SLC_DGEMM("N", "N", &m2, &n, &m2, &one, &dwork[iw2], &m2,
              ck, &ldck, &zero, &dwork[iw7], &m2);

    SLC_DGEMM("N", "N", &np2, &n, &m2, &one, &d[np1 + m1 * ldd], &ldd,
              &dwork[iw7], &m2, &zero, &dwork[iw8], &np2);

    SLC_DLACPY("Full", &n, &n, a, &lda, ac, &ldac);
    SLC_DGEMM("N", "N", &n, &n, &m2, &one, &b[m1 * ldb], &ldb,
              &dwork[iw4], &m2, &one, ac, &ldac);

    SLC_DGEMM("N", "N", &n, &n, &m2, &one, &b[m1 * ldb], &ldb,
              &dwork[iw7], &m2, &zero, &ac[n * ldac], &ldac);

    SLC_DGEMM("N", "N", &n, &n, &np2, &one, bk, &ldbk,
              &dwork[iw3], &np2, &zero, &ac[n], &ldac);

    SLC_DLACPY("Full", &n, &n, ak, &ldak, &ac[n + n * ldac], &ldac);
    SLC_DGEMM("N", "N", &n, &n, &np2, &one, bk, &ldbk,
              &dwork[iw8], &np2, &one, &ac[n + n * ldac], &ldac);

    SLC_DLACPY("Full", &n, &m1, b, &ldb, bc, &ldbc);
    SLC_DGEMM("N", "N", &n, &m1, &m2, &one, &b[m1 * ldb], &ldb,
              &dwork[iw6], &m2, &one, bc, &ldbc);

    SLC_DGEMM("N", "N", &n, &m1, &np2, &one, bk, &ldbk,
              &dwork[iw5], &np2, &zero, &bc[n], &ldbc);

    SLC_DLACPY("Full", &np1, &n, c, &ldc, cc, &ldcc);
    SLC_DGEMM("N", "N", &np1, &n, &m2, &one, &d[m1 * ldd], &ldd,
              &dwork[iw4], &m2, &one, cc, &ldcc);

    SLC_DGEMM("N", "N", &np1, &n, &m2, &one, &d[m1 * ldd], &ldd,
              &dwork[iw7], &m2, &zero, &cc[n * ldcc], &ldcc);

    SLC_DLACPY("Full", &np1, &m1, d, &ldd, dc, &lddc);
    SLC_DGEMM("N", "N", &np1, &m1, &m2, &one, &d[m1 * ldd], &ldd,
              &dwork[iw6], &m2, &one, dc, &lddc);

    dwork[0] = (f64)lwamax;
}
