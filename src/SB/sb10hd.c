/**
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * SB10HD - H2 optimal n-state controller for continuous-time system
 *
 * Computes the matrices of the H2 optimal n-state controller K = (AK, BK, CK, DK)
 * for a given plant P = (A, B, C, D) with NCON control inputs and NMEAS measurements.
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <stdlib.h>
#include <math.h>

void sb10hd(
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
    f64* ak,
    const i32 ldak,
    f64* bk,
    const i32 ldbk,
    f64* ck,
    const i32 ldck,
    f64* dk,
    const i32 lddk,
    f64* rcond,
    const f64 tol,
    i32* iwork,
    f64* dwork,
    const i32 ldwork,
    i32* bwork,
    i32* info)
{
    const f64 zero = 0.0;
    const f64 one = 1.0;

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
    } else {
        i32 t1 = np1 * n > 3 * m2 + np1 ? np1 * n : 3 * m2 + np1;
        t1 = t1 > 5 * m2 ? t1 : 5 * m2;
        i32 part1 = m2 + np1 * np1 + t1;

        i32 t2 = m1 * n > 3 * np2 + m1 ? m1 * n : 3 * np2 + m1;
        t2 = t2 > 5 * np2 ? t2 : 5 * np2;
        i32 part2 = np2 + m1 * m1 + t2;

        i32 maxtemp = part1 > part2 ? part1 : part2;
        maxtemp = maxtemp > n * m2 ? maxtemp : n * m2;
        maxtemp = maxtemp > np2 * n ? maxtemp : np2 * n;
        maxtemp = maxtemp > np2 * m2 ? maxtemp : np2 * m2;
        maxtemp = maxtemp > 1 ? maxtemp : 1;

        i32 riccati = n * (14 * n + 12 + m2 + np2) + 5;

        i32 minwrk = n * m + np * (n + m) + m2 * m2 + np2 * np2 +
                     (maxtemp > riccati ? maxtemp : riccati);

        if (ldwork < minwrk) {
            *info = -26;
        }
    }

    if (*info != 0) {
        return;
    }

    if (n == 0 || m == 0 || np == 0 || m1 == 0 || m2 == 0 ||
        np1 == 0 || np2 == 0) {
        rcond[0] = one;
        rcond[1] = one;
        rcond[2] = one;
        rcond[3] = one;
        dwork[0] = one;
        return;
    }

    f64 toll = tol;
    if (toll <= zero) {
        toll = sqrt(SLC_DLAMCH("Epsilon"));
    }

    i32 iwc = n * m;
    i32 iwd = iwc + np * n;
    i32 iwtu = iwd + np * m;
    i32 iwty = iwtu + m2 * m2;
    i32 iwrk = iwty + np2 * np2;

    SLC_DLACPY("Full", &n, &m, b, &ldb, dwork, &n);
    SLC_DLACPY("Full", &np, &n, c, &ldc, &dwork[iwc], &np);
    SLC_DLACPY("Full", &np, &m, d, &ldd, &dwork[iwd], &np);

    i32 info2 = 0;
    i32 lwork_ud = ldwork - iwrk;
    sb10ud(n, m, np, ncon, nmeas,
           dwork, n, &dwork[iwc], np, &dwork[iwd], np,
           &dwork[iwtu], m2, &dwork[iwty], np2,
           rcond, toll, &dwork[iwrk], lwork_ud, &info2);

    if (info2 > 0) {
        *info = info2;
        return;
    }

    i32 lwamax = (i32)dwork[iwrk] + iwrk;

    i32 iwy = iwrk;
    i32 iwf = iwy + n * n;
    i32 iwh = iwf + m2 * n;
    iwrk = iwh + n * np2;

    i32 lwork_vd = ldwork - iwrk;
    sb10vd(n, m, np, ncon, nmeas,
           a, lda, dwork, n, &dwork[iwc], np,
           &dwork[iwf], m2, &dwork[iwh], n,
           ak, ldak, &dwork[iwy], n, &rcond[2],
           iwork, &dwork[iwrk], lwork_vd, bwork, &info2);

    if (info2 > 0) {
        *info = info2 + 3;
        return;
    }

    i32 lwa_vd = (i32)dwork[iwrk] + iwrk;
    lwamax = lwamax > lwa_vd ? lwamax : lwa_vd;

    sb10wd(n, m, np, ncon, nmeas,
           a, lda, dwork, n, &dwork[iwc], np, &dwork[iwd], np,
           &dwork[iwf], m2, &dwork[iwh], n,
           &dwork[iwtu], m2, &dwork[iwty], np2,
           ak, ldak, bk, ldbk, ck, ldck, dk, lddk, &info2);

    dwork[0] = (f64)lwamax;
}
