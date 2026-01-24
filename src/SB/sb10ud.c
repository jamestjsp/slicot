/**
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * SB10UD - Normalization of D12 and D21 for H2 controller design
 *
 * Reduces D12 and D21 matrices of a partitioned system to unit diagonal
 * form and transforms B and C matrices for H2 optimal controller computation.
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <stdlib.h>
#include <math.h>

void sb10ud(
    const i32 n,
    const i32 m,
    const i32 np,
    const i32 ncon,
    const i32 nmeas,
    f64* b,
    const i32 ldb,
    f64* c,
    const i32 ldc,
    f64* d,
    const i32 ldd,
    f64* tu,
    const i32 ldtu,
    f64* ty,
    const i32 ldty,
    f64* rcond,
    const f64 tol,
    f64* dwork,
    const i32 ldwork,
    i32* info)
{
    const f64 zero = 0.0;
    const f64 one = 1.0;
    const i32 int1 = 1;

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
    } else if (ldb < (n > 1 ? n : 1)) {
        *info = -7;
    } else if (ldc < (np > 1 ? np : 1)) {
        *info = -9;
    } else if (ldd < (np > 1 ? np : 1)) {
        *info = -11;
    } else if (ldtu < (m2 > 1 ? m2 : 1)) {
        *info = -13;
    } else if (ldty < (np2 > 1 ? np2 : 1)) {
        *info = -15;
    } else {
        i32 t1 = np1 * n > 3 * m2 + np1 ? np1 * n : 3 * m2 + np1;
        t1 = t1 > 5 * m2 ? t1 : 5 * m2;
        i32 part1 = m2 + np1 * np1 + t1;

        i32 t2 = m1 * n > 3 * np2 + m1 ? m1 * n : 3 * np2 + m1;
        t2 = t2 > 5 * np2 ? t2 : 5 * np2;
        i32 part2 = np2 + m1 * m1 + t2;

        i32 minwrk = part1 > part2 ? part1 : part2;
        minwrk = minwrk > n * m2 ? minwrk : n * m2;
        minwrk = minwrk > np2 * n ? minwrk : np2 * n;
        minwrk = minwrk > np2 * m2 ? minwrk : np2 * m2;
        minwrk = minwrk > 1 ? minwrk : 1;

        if (ldwork < minwrk) {
            *info = -19;
        }
    }

    if (*info != 0) {
        return;
    }

    if (n == 0 || m == 0 || np == 0 || m1 == 0 || m2 == 0 ||
        np1 == 0 || np2 == 0) {
        rcond[0] = one;
        rcond[1] = one;
        dwork[0] = one;
        return;
    }

    i32 nd1 = np1 - m2;
    i32 nd2 = m1 - np2;
    f64 toll = tol;
    if (toll <= zero) {
        toll = sqrt(SLC_DLAMCH("Epsilon"));
    }

    i32 iq = m2;
    i32 iwrk = iq + np1 * np1;
    i32 lwork = ldwork - iwrk;
    i32 info2 = 0;
    i32 lwamax;

    SLC_DGESVD("A", "A", &np1, &m2, &d[m1 * ldd], &ldd, dwork,
               &dwork[iq], &np1, tu, &ldtu, &dwork[iwrk], &lwork, &info2);

    if (info2 != 0) {
        *info = 3;
        return;
    }

    if (dwork[0] <= zero) {
        rcond[0] = zero;
    } else {
        rcond[0] = dwork[m2 - 1] / dwork[0];
    }
    if (rcond[0] <= toll) {
        rcond[1] = zero;
        *info = 1;
        return;
    }
    lwamax = (i32)dwork[iwrk] + iwrk;

    if (nd1 > 0) {
        SLC_DLACPY("Full", &np1, &m2, &dwork[iq], &np1, &d[m1 * ldd], &ldd);
        SLC_DLACPY("Full", &np1, &nd1, &dwork[iq + np1 * m2], &np1, &dwork[iq], &np1);
        SLC_DLACPY("Full", &np1, &m2, &d[m1 * ldd], &ldd, &dwork[iq + np1 * nd1], &np1);
    }

    for (i32 j = 0; j < m2 - 1; j++) {
        i32 count = j + 1;
        SLC_DSWAP(&count, &tu[j + 1], &ldtu, &tu[(j + 1) * ldtu], &int1);
    }

    for (i32 j = 0; j < m2; j++) {
        f64 scale = one / dwork[j];
        SLC_DSCAL(&m2, &scale, &tu[j * ldtu], &int1);
    }

    SLC_DGEMM("T", "N", &np1, &n, &np1, &one, &dwork[iq], &np1, c, &ldc,
              &zero, &dwork[iwrk], &np1);
    SLC_DLACPY("Full", &np1, &n, &dwork[iwrk], &np1, c, &ldc);
    i32 lwa_c1 = iwrk + np1 * n;
    lwamax = lwamax > lwa_c1 ? lwamax : lwa_c1;

    iq = np2;
    iwrk = iq + m1 * m1;
    lwork = ldwork - iwrk;

    SLC_DGESVD("A", "A", &np2, &m1, &d[np1], &ldd, dwork, ty, &ldty,
               &dwork[iq], &m1, &dwork[iwrk], &lwork, &info2);

    if (info2 != 0) {
        *info = 3;
        return;
    }

    if (dwork[0] <= zero) {
        rcond[1] = zero;
    } else {
        rcond[1] = dwork[np2 - 1] / dwork[0];
    }
    if (rcond[1] <= toll) {
        *info = 2;
        return;
    }
    i32 lwa_svd2 = (i32)dwork[iwrk] + iwrk;
    lwamax = lwamax > lwa_svd2 ? lwamax : lwa_svd2;

    if (nd2 > 0) {
        SLC_DLACPY("Full", &np2, &m1, &dwork[iq], &m1, &d[np1], &ldd);
        SLC_DLACPY("Full", &nd2, &m1, &dwork[iq + np2], &m1, &dwork[iq], &m1);
        SLC_DLACPY("Full", &np2, &m1, &d[np1], &ldd, &dwork[iq + nd2], &m1);
    }

    for (i32 j = 0; j < np2; j++) {
        f64 scale = one / dwork[j];
        SLC_DSCAL(&np2, &scale, &ty[j * ldty], &int1);
    }

    for (i32 j = 0; j < np2 - 1; j++) {
        i32 count = j + 1;
        SLC_DSWAP(&count, &ty[j + 1], &ldty, &ty[(j + 1) * ldty], &int1);
    }

    SLC_DGEMM("N", "T", &n, &m1, &m1, &one, b, &ldb, &dwork[iq], &m1,
              &zero, &dwork[iwrk], &n);
    SLC_DLACPY("Full", &n, &m1, &dwork[iwrk], &n, b, &ldb);
    i32 lwa_b1 = iwrk + n * m1;
    lwamax = lwamax > lwa_b1 ? lwamax : lwa_b1;

    SLC_DGEMM("N", "N", &n, &m2, &m2, &one, &b[m1 * ldb], &ldb, tu, &ldtu,
              &zero, dwork, &n);
    SLC_DLACPY("Full", &n, &m2, dwork, &n, &b[m1 * ldb], &ldb);

    SLC_DGEMM("N", "N", &np2, &n, &np2, &one, ty, &ldty, &c[np1], &ldc,
              &zero, dwork, &np2);
    SLC_DLACPY("Full", &np2, &n, dwork, &np2, &c[np1], &ldc);

    SLC_DGEMM("N", "N", &np2, &m2, &np2, &one, ty, &ldty, &d[np1 + m1 * ldd], &ldd,
              &zero, dwork, &np2);
    SLC_DGEMM("N", "N", &np2, &m2, &m2, &one, dwork, &np2, tu, &ldtu,
              &zero, &d[np1 + m1 * ldd], &ldd);

    i32 tmp1 = n > m2 ? n : m2;
    tmp1 = tmp1 > np2 ? tmp1 : np2;
    i32 tmp2 = np2 * m2;
    i32 lwa_final = tmp1 > tmp2 ? tmp1 : tmp2;
    lwamax = lwamax > lwa_final ? lwamax : lwa_final;

    dwork[0] = (f64)lwamax;
}
