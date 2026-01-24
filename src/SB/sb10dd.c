/**
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * SB10DD - H-infinity (sub)optimal controller for discrete-time system
 *
 * Computes the matrices of an H-infinity (sub)optimal n-state controller
 * for a discrete-time system using the method from Green & Limebeer.
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <stdlib.h>
#include <math.h>
#include <ctype.h>
#include <stdbool.h>

void sb10dd(
    const i32 n,
    const i32 m,
    const i32 np,
    const i32 ncon,
    const i32 nmeas,
    const f64 gamma,
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
    f64* x,
    const i32 ldx,
    f64* z,
    const i32 ldz,
    f64* rcond,
    const f64 tol,
    i32* iwork,
    f64* dwork,
    const i32 ldwork,
    i32* info)
{
    const f64 zero = 0.0;
    const f64 one = 1.0;
    const f64 thousn = 1000.0;
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
    } else if (gamma <= zero) {
        *info = -6;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -8;
    } else if (ldb < (n > 1 ? n : 1)) {
        *info = -10;
    } else if (ldc < (np > 1 ? np : 1)) {
        *info = -12;
    } else if (ldd < (np > 1 ? np : 1)) {
        *info = -14;
    } else if (ldak < (n > 1 ? n : 1)) {
        *info = -16;
    } else if (ldbk < (n > 1 ? n : 1)) {
        *info = -18;
    } else if (ldck < (m2 > 1 ? m2 : 1)) {
        *info = -20;
    } else if (lddk < (m2 > 1 ? m2 : 1)) {
        *info = -22;
    } else if (ldx < (n > 1 ? n : 1)) {
        *info = -24;
    } else if (ldz < (n > 1 ? n : 1)) {
        *info = -26;
    } else {
        i32 iwb_ws = (n + np1 + 1) * (n + m2) +
                     ((3 * (n + m2) + n + np1) > (5 * (n + m2)) ?
                      (3 * (n + m2) + n + np1) : (5 * (n + m2)));
        i32 iwc_ws = (n + np2) * (n + m1 + 1) +
                     ((3 * (n + np2) + n + m1) > (5 * (n + np2)) ?
                      (3 * (n + np2) + n + m1) : (5 * (n + np2)));
        i32 iwd_ws = 13 * n * n + 2 * m * m + n * (8 * m + np2) + m1 * (m2 + np2) + 6 * n +
                     ((14 * n + 23) > 16 * n ?
                      ((14 * n + 23) > (2 * n + m) ?
                       ((14 * n + 23) > 3 * m ? (14 * n + 23) : 3 * m) :
                       ((2 * n + m) > 3 * m ? (2 * n + m) : 3 * m)) :
                      (16 * n > (2 * n + m) ?
                       (16 * n > 3 * m ? 16 * n : 3 * m) :
                       ((2 * n + m) > 3 * m ? (2 * n + m) : 3 * m)));
        i32 iwg_ws = 13 * n * n + m * m + (8 * n + m + m2 + 2 * np2) * (m2 + np2) + 6 * n +
                     n * (m + np2) +
                     ((14 * n + 23) > 16 * n ?
                      ((14 * n + 23) > (2 * n + m2 + np2) ?
                       ((14 * n + 23) > 3 * (m2 + np2) ? (14 * n + 23) : 3 * (m2 + np2)) :
                       ((2 * n + m2 + np2) > 3 * (m2 + np2) ? (2 * n + m2 + np2) : 3 * (m2 + np2))) :
                      (16 * n > (2 * n + m2 + np2) ?
                       (16 * n > 3 * (m2 + np2) ? 16 * n : 3 * (m2 + np2)) :
                       ((2 * n + m2 + np2) > 3 * (m2 + np2) ? (2 * n + m2 + np2) : 3 * (m2 + np2))));

        i32 minwrk = iwb_ws;
        if (iwc_ws > minwrk) minwrk = iwc_ws;
        if (iwd_ws > minwrk) minwrk = iwd_ws;
        if (iwg_ws > minwrk) minwrk = iwg_ws;

        if (ldwork < minwrk) {
            *info = -31;
        }
    }

    if (*info != 0) {
        return;
    }

    if (n == 0 || m == 0 || np == 0 || m1 == 0 || m2 == 0 || np1 == 0 || np2 == 0) {
        rcond[0] = one;
        rcond[1] = one;
        rcond[2] = one;
        rcond[3] = one;
        rcond[4] = one;
        rcond[5] = one;
        rcond[6] = one;
        rcond[7] = one;
        dwork[0] = one;
        return;
    }

    f64 toll = tol;
    if (toll <= zero) {
        toll = thousn * SLC_DLAMCH("Epsilon");
    }

    i32 lwamax = 0;
    i32 info2;
    f64 anorm;
    i32 j;

    i32 iws = (n + np1) * (n + m2);
    i32 iwrk = iws + (n + m2);

    SLC_DLACPY("Full", &n, &n, a, &lda, dwork, &(i32){n + np1});
    SLC_DLACPY("Full", &np1, &n, c, &ldc, dwork + n, &(i32){n + np1});
    SLC_DLACPY("Full", &n, &m2, b + m1 * ldb, &ldb, dwork + (n + np1) * n, &(i32){n + np1});
    SLC_DLACPY("Full", &np1, &m2, d + m1 * ldd, &ldd, dwork + (n + np1) * n + n, &(i32){n + np1});

    i32 nm2 = n + m2;
    i32 nnp1 = n + np1;
    i32 ldwork_left = ldwork - iwrk;
    SLC_DGESVD("N", "N", &nnp1, &nm2, dwork, &nnp1, dwork + iws,
               dwork, &nnp1, dwork, &nm2, dwork + iwrk, &ldwork_left, &info2);
    if (info2 > 0) {
        *info = 9;
        return;
    }
    if (dwork[iws + n + m2 - 1] / dwork[iws] <= toll) {
        *info = 1;
        return;
    }
    lwamax = (i32)dwork[iwrk] + iwrk;

    iws = (n + np2) * (n + m1);
    iwrk = iws + (n + np2);

    SLC_DLACPY("Full", &n, &n, a, &lda, dwork, &(i32){n + np2});
    SLC_DLACPY("Full", &np2, &n, c + np1, &ldc, dwork + n, &(i32){n + np2});
    SLC_DLACPY("Full", &n, &m1, b, &ldb, dwork + (n + np2) * n, &(i32){n + np2});
    SLC_DLACPY("Full", &np2, &m1, d + np1, &ldd, dwork + (n + np2) * n + n, &(i32){n + np2});

    i32 nm1 = n + m1;
    i32 nnp2 = n + np2;
    ldwork_left = ldwork - iwrk;
    SLC_DGESVD("N", "N", &nnp2, &nm1, dwork, &nnp2, dwork + iws,
               dwork, &nnp2, dwork, &nm1, dwork + iwrk, &ldwork_left, &info2);
    if (info2 > 0) {
        *info = 9;
        return;
    }
    if (dwork[iws + n + np2 - 1] / dwork[iws] <= toll) {
        *info = 2;
        return;
    }
    if ((i32)dwork[iwrk] + iwrk > lwamax) lwamax = (i32)dwork[iwrk] + iwrk;

    iws = np1 * m2;
    iwrk = iws + m2;

    SLC_DLACPY("Full", &np1, &m2, d + m1 * ldd, &ldd, dwork, &np1);
    ldwork_left = ldwork - iwrk;
    SLC_DGESVD("N", "N", &np1, &m2, dwork, &np1, dwork + iws,
               dwork, &np1, dwork, &m2, dwork + iwrk, &ldwork_left, &info2);
    if (info2 > 0) {
        *info = 9;
        return;
    }
    if (dwork[iws + m2 - 1] / dwork[iws] <= toll) {
        *info = 3;
        return;
    }
    if ((i32)dwork[iwrk] + iwrk > lwamax) lwamax = (i32)dwork[iwrk] + iwrk;

    iws = np2 * m1;
    iwrk = iws + np2;

    SLC_DLACPY("Full", &np2, &m1, d + np1, &ldd, dwork, &np2);
    ldwork_left = ldwork - iwrk;
    SLC_DGESVD("N", "N", &np2, &m1, dwork, &np2, dwork + iws,
               dwork, &np2, dwork, &m1, dwork + iwrk, &ldwork_left, &info2);
    if (info2 > 0) {
        *info = 9;
        return;
    }
    if (dwork[iws + np2 - 1] / dwork[iws] <= toll) {
        *info = 4;
        return;
    }
    if ((i32)dwork[iwrk] + iwrk > lwamax) lwamax = (i32)dwork[iwrk] + iwrk;

    i32 iwv = 0;
    i32 iwb_off = iwv + m * m;
    i32 iwc_off = iwb_off + n * m1;
    i32 iwd_off = iwc_off + (m2 + np2) * n;
    i32 iwq = iwd_off + (m2 + np2) * m1;
    i32 iwl = iwq + n * n;
    i32 iwr = iwl + n * m;
    i32 iwi = iwr + 2 * n;
    i32 iwh = iwi + 2 * n;
    iws = iwh + 2 * n;
    i32 iwt = iws + (2 * n + m) * (2 * n + m);
    i32 iwu = iwt + (2 * n + m) * 2 * n;
    iwrk = iwu + 4 * n * n;
    i32 ir2 = iwv + m1;
    i32 ir3 = ir2 + m * m1;

    f64 gamma2 = gamma * gamma;
    SLC_DSYRK("Lower", "Transpose", &m, &np1, &one, d, &ldd, &zero, dwork, &m);
    for (j = 0; j < m * m1; j += m + 1) {
        dwork[j] -= gamma2;
    }

    SLC_DSYRK("Lower", "Transpose", &n, &np1, &one, c, &ldc, &zero, dwork + iwq, &n);

    SLC_DGEMM("Transpose", "NoTranspose", &n, &m, &np1, &one, c, &ldc,
              d, &ldd, &zero, dwork + iwl, &n);

    f64 rcond2;
    i32 n2pm = 2 * n + m;
    ldwork_left = ldwork - iwrk;
    sb02od("D", "B", "N", "L", "N", "S", n, m, np, a, lda, b, ldb,
           dwork + iwq, n, dwork, m, dwork + iwl, n,
           &rcond2, x, ldx, dwork + iwr, dwork + iwi, dwork + iwh,
           dwork + iws, n2pm, dwork + iwt, n2pm, dwork + iwu, 2 * n,
           toll, iwork, dwork + iwrk, ldwork_left, &info2);
    if (info2 > 0) {
        *info = 6;
        return;
    }
    if ((i32)dwork[iwrk] + iwrk > lwamax) lwamax = (i32)dwork[iwrk] + iwrk;

    iws = iwr;
    iwh = iws + m * m;
    iwt = iwh + n * m;
    iwu = iwt + n * n;
    i32 iwg = iwu + n * n;
    iwrk = iwg + n * n;

    SLC_DLACPY("Lower", &m, &m, dwork, &m, dwork + iws, &m);
    ldwork_left = ldwork - iwrk;
    SLC_DSYTRF("Lower", &m, dwork + iws, &m, iwork, dwork + iwrk, &ldwork_left, &info2);
    if (info2 > 0) {
        *info = 5;
        return;
    }
    if ((i32)dwork[iwrk] + iwrk > lwamax) lwamax = (i32)dwork[iwrk] + iwrk;

    ma02ad("Full", n, m, b, ldb, dwork + iwh, m);
    SLC_DSYTRS("Lower", &m, &n, dwork + iws, &m, iwork, dwork + iwh, &m, &info2);
    slicot_mb01rx('L', 'L', 'N', n, m, zero, one, dwork + iwg, n, b, ldb, dwork + iwh, m);

    f64 sepd, ferr;
    ldwork_left = ldwork - iwrk;
    sb02sd("C", "N", "N", "L", "O", n, a, lda, dwork + iwt, n,
           dwork + iwu, n, dwork + iwg, n, dwork + iwq, n, x, ldx,
           &sepd, &rcond[6], &ferr, iwork, dwork + iwrk, ldwork_left, &info2);
    if (info2 > 0) rcond[6] = zero;
    if ((i32)dwork[iwrk] + iwrk > lwamax) lwamax = (i32)dwork[iwrk] + iwrk;

    iwrk = iwr;

    mb01ru("Lower", "Transpose", m, n, one, one, dwork, m,
           b, ldb, x, ldx, dwork + iwrk, m * n, &info2);

    anorm = SLC_DLANSY("1", "Lower", &m2, dwork + ir3, &m, dwork + iwrk);
    SLC_DPOTRF("Lower", &m2, dwork + ir3, &m, &info2);
    if (info2 > 0) {
        *info = 5;
        return;
    }
    SLC_DPOCON("Lower", &m2, dwork + ir3, &m, &anorm, &rcond[0], dwork + iwrk, iwork, &info2);

    if (rcond[0] < toll) {
        *info = 5;
        return;
    }

    SLC_DTRCON("1", "Lower", "NonUnit", &m2, dwork + ir3, &m, &rcond[4], dwork + iwrk, iwork, &info2);

    if (rcond[4] < toll) {
        *info = 5;
        return;
    }

    SLC_DTRSM("Left", "Lower", "NoTranspose", "NonUnit", &m2, &m1,
              &one, dwork + ir3, &m, dwork + ir2, &m);

    f64 mone = -1.0;
    SLC_DSYRK("Lower", "Transpose", &m1, &m2, &one, dwork + ir2, &m, &mone, dwork, &m);

    anorm = SLC_DLANSY("1", "Lower", &m1, dwork, &m, dwork + iwrk);
    SLC_DPOTRF("Lower", &m1, dwork, &m, &info2);
    if (info2 > 0) {
        *info = 5;
        return;
    }
    SLC_DPOCON("Lower", &m1, dwork, &m, &anorm, &rcond[1], dwork + iwrk, iwork, &info2);

    if (rcond[1] < toll) {
        *info = 5;
        return;
    }

    SLC_DTRCON("1", "Lower", "NonUnit", &m1, dwork, &m, &rcond[2], dwork + iwrk, iwork, &info2);

    if (rcond[2] < toll) {
        *info = 5;
        return;
    }

    SLC_DGEMM("NoTranspose", "NoTranspose", &n, &n, &n, &one, x, &ldx,
              a, &lda, &zero, dwork + iwq, &n);

    ma02ad("Full", n, m, dwork + iwl, n, dwork + iwrk, m);
    SLC_DLACPY("Full", &m, &n, dwork + iwrk, &m, dwork + iwl, &m);
    SLC_DGEMM("Transpose", "NoTranspose", &m, &n, &n, &one, b, &ldb,
              dwork + iwq, &n, &one, dwork + iwl, &m);

    SLC_DTRSM("Left", "Lower", "NoTranspose", "NonUnit", &m2, &n, &one,
              dwork + ir3, &m, dwork + iwl + m1, &m);

    SLC_DGEMM("Transpose", "NoTranspose", &m1, &n, &m2, &mone,
              dwork + ir2, &m, dwork + iwl + m1, &m, &one, dwork + iwl, &m);

    SLC_DTRSM("Left", "Lower", "NoTranspose", "NonUnit", &m1, &n, &one,
              dwork, &m, dwork + iwl, &m);

    SLC_DLACPY("Full", &n, &m1, b, &ldb, dwork + iwb_off, &n);
    SLC_DTRSM("Right", "Lower", "Transpose", "NonUnit", &n, &m1, &one,
              dwork, &m, dwork + iwb_off, &n);

    SLC_DLACPY("Full", &n, &n, a, &lda, ak, &ldak);
    SLC_DGEMM("NoTranspose", "NoTranspose", &n, &n, &m1, &one,
              dwork + iwb_off, &n, dwork + iwl, &m, &one, ak, &ldak);

    i32 nm1_size = n * m1;
    SLC_DSCAL(&nm1_size, &gamma, dwork + iwb_off, &int1);

    i32 m2np2 = m2 + np2;
    SLC_DLACPY("Full", &m2, &m1, dwork + ir2, &m, dwork + iwd_off, &m2np2);
    SLC_DLACPY("Full", &np2, &m1, d + np1, &ldd, dwork + iwd_off + m2, &m2np2);
    SLC_DTRSM("Right", "Lower", "Transpose", "NonUnit", &m2np2, &m1, &one,
              dwork, &m, dwork + iwd_off, &m2np2);

    SLC_DLACPY("Full", &m2, &n, dwork + iwl + m1, &m, dwork + iwc_off, &m2np2);
    SLC_DLACPY("Full", &np2, &n, c + np1, &ldc, dwork + iwc_off + m2, &m2np2);
    SLC_DGEMM("NoTranspose", "NoTranspose", &m2np2, &n, &m1, &one,
              dwork + iwd_off, &m2np2, dwork + iwl, &m, &one, dwork + iwc_off, &m2np2);

    i32 m2np2_m1 = m2np2 * m1;
    SLC_DSCAL(&m2np2_m1, &gamma, dwork + iwd_off, &int1);

    i32 iww = iwd_off + m2np2 * m1;
    iwq = iww + m2np2 * m2np2;
    iwl = iwq + n * n;
    iwr = iwl + n * m2np2;
    iwi = iwr + 2 * n;
    iwh = iwi + 2 * n;
    iws = iwh + 2 * n;
    i32 n2pmp = 2 * n + m2np2;
    iwt = iws + n2pmp * n2pmp;
    iwu = iwt + n2pmp * 2 * n;
    iwg = iwu + 4 * n * n;
    iwrk = iwg + m2np2 * n;
    i32 is2 = iww + m2np2 * m2;
    i32 is3 = is2 + m2;

    SLC_DSYRK("Upper", "NoTranspose", &m2np2, &m1, &one, dwork + iwd_off,
              &m2np2, &zero, dwork + iww, &m2np2);
    for (j = iww; j < iww + m2np2 * m2; j += m2np2 + 1) {
        dwork[j] -= gamma2;
    }

    SLC_DSYRK("Upper", "NoTranspose", &n, &m1, &one, dwork + iwb_off, &n,
              &zero, dwork + iwq, &n);

    SLC_DGEMM("NoTranspose", "Transpose", &n, &m2np2, &m1, &one,
              dwork + iwb_off, &n, dwork + iwd_off, &m2np2, &zero, dwork + iwl, &n);

    for (j = 1; j < n; j++) {
        SLC_DSWAP(&j, ak + j, &ldak, ak + j * ldak, &int1);
    }

    ma02ad("Full", m2np2, n, dwork + iwc_off, m2np2, dwork + iwg, n);

    ldwork_left = ldwork - iwrk;
    sb02od("D", "B", "N", "U", "N", "S", n, m2np2, np, ak, ldak, dwork + iwg, n,
           dwork + iwq, n, dwork + iww, m2np2, dwork + iwl, n,
           &rcond2, z, ldz, dwork + iwr, dwork + iwi, dwork + iwh,
           dwork + iws, n2pmp, dwork + iwt, n2pmp, dwork + iwu, 2 * n,
           toll, iwork, dwork + iwrk, ldwork_left, &info2);
    if (info2 > 0) {
        *info = 7;
        return;
    }
    if ((i32)dwork[iwrk] + iwrk > lwamax) lwamax = (i32)dwork[iwrk] + iwrk;

    iws = iwr;
    iwh = iws + m2np2 * m2np2;
    iwt = iwh + n * m2np2;
    iwu = iwt + n * n;
    iwg = iwu + n * n;
    iwrk = iwg + n * n;

    SLC_DLACPY("Upper", &m2np2, &m2np2, dwork + iww, &m2np2, dwork + iws, &m2np2);
    ldwork_left = ldwork - iwrk;
    SLC_DSYTRF("Upper", &m2np2, dwork + iws, &m2np2, iwork, dwork + iwrk, &ldwork_left, &info2);
    if (info2 > 0) {
        *info = 5;
        return;
    }
    if ((i32)dwork[iwrk] + iwrk > lwamax) lwamax = (i32)dwork[iwrk] + iwrk;

    SLC_DLACPY("Full", &m2np2, &n, dwork + iwc_off, &m2np2, dwork + iwh, &m2np2);
    SLC_DSYTRS("Upper", &m2np2, &n, dwork + iws, &m2np2, iwork, dwork + iwh, &m2np2, &info2);
    slicot_mb01rx('L', 'U', 'T', n, m2np2, zero, one, dwork + iwg, n, dwork + iwc_off, m2np2, dwork + iwh, m2np2);

    ldwork_left = ldwork - iwrk;
    sb02sd("C", "N", "N", "U", "O", n, ak, ldak, dwork + iwt, n,
           dwork + iwu, n, dwork + iwg, n, dwork + iwq, n, z, ldz,
           &sepd, &rcond[7], &ferr, iwork, dwork + iwrk, ldwork_left, &info2);
    if (info2 > 0) rcond[7] = zero;
    if ((i32)dwork[iwrk] + iwrk > lwamax) lwamax = (i32)dwork[iwrk] + iwrk;

    iwrk = iwr;

    mb01ru("Upper", "NoTranspose", m2np2, n, one, one, dwork + iww, m2np2,
           dwork + iwc_off, m2np2, z, ldz, dwork + iwrk, m2np2 * n, &info2);

    anorm = SLC_DLANSY("1", "Upper", &np2, dwork + is3, &m2np2, dwork + iwrk);
    SLC_DPOTRF("Upper", &np2, dwork + is3, &m2np2, &info2);
    if (info2 > 0) {
        *info = 5;
        return;
    }
    SLC_DPOCON("Upper", &np2, dwork + is3, &m2np2, &anorm, &rcond[3], dwork + iwrk, iwork, &info2);

    if (rcond[3] < toll) {
        *info = 5;
        return;
    }

    SLC_DTRSM("Right", "Upper", "NoTranspose", "NonUnit", &m2, &np2,
              &one, dwork + is3, &m2np2, dwork + is2, &m2np2);

    SLC_DSYRK("Upper", "NoTranspose", &m2, &np2, &one, dwork + is2, &m2np2,
              &mone, dwork + iww, &m2np2);
    SLC_DPOTRF("Upper", &m2, dwork + iww, &m2np2, &info2);
    if (info2 > 0) {
        *info = 5;
        return;
    }

    for (j = 1; j < n; j++) {
        SLC_DSWAP(&j, ak + j, &ldak, ak + j * ldak, &int1);
    }

    SLC_DGEMM("NoTranspose", "NoTranspose", &n, &n, &n, &one, ak, &ldak,
              z, &ldz, &zero, dwork + iwrk, &n);

    SLC_DLACPY("Full", &n, &np2, dwork + iwl + n * m2, &n, bk, &ldbk);
    SLC_DGEMM("NoTranspose", "Transpose", &n, &np2, &n, &one,
              dwork + iwrk, &n, dwork + iwc_off + m2, &m2np2, &one, bk, &ldbk);

    SLC_DTRSM("Right", "Upper", "Transpose", "NonUnit", &m2, &np2,
              &one, dwork + is3, &m2np2, dwork + is2, &m2np2);

    SLC_DLACPY("Full", &m2, &np2, dwork + is2, &m2np2, dk, &lddk);
    SLC_DTRSM("Left", "Lower", "Transpose", "NonUnit", &m2, &np2,
              &mone, dwork + ir3, &m, dk, &lddk);

    SLC_DLACPY("Full", &m2, &n, dwork + iwc_off, &m2np2, ck, &ldck);
    SLC_DGEMM("NoTranspose", "NoTranspose", &m2, &n, &np2, &mone,
              dwork + is2, &m2np2, dwork + iwc_off + m2, &m2np2, &one, ck, &ldck);
    SLC_DTRSM("Left", "Lower", "Transpose", "NonUnit", &m2, &n,
              &mone, dwork + ir3, &m, ck, &ldck);

    SLC_DTRSM("Right", "Upper", "NoTranspose", "NonUnit", &n, &np2,
              &one, dwork + is3, &m2np2, bk, &ldbk);
    SLC_DTRSM("Right", "Upper", "Transpose", "NonUnit", &n, &np2,
              &one, dwork + is3, &m2np2, bk, &ldbk);

    SLC_DGEMM("NoTranspose", "NoTranspose", &n, &n, &m2, &one,
              b + m1 * ldb, &ldb, ck, &ldck, &one, ak, &ldak);
    SLC_DGEMM("NoTranspose", "NoTranspose", &n, &n, &np2, &mone, bk,
              &ldbk, dwork + iwc_off + m2, &m2np2, &one, ak, &ldak);

    SLC_DGEMM("NoTranspose", "NoTranspose", &n, &np2, &m2, &one,
              b + m1 * ldb, &ldb, dk, &lddk, &one, bk, &ldbk);

    iwrk = m2 * m2;
    SLC_DLASET("Full", &m2, &m2, &zero, &one, dwork, &m2);
    SLC_DGEMM("NoTranspose", "NoTranspose", &m2, &m2, &np2, &one, dk,
              &lddk, d + np1 + m1 * ldd, &ldd, &one, dwork, &m2);
    anorm = SLC_DLANGE("1", &m2, &m2, dwork, &m2, dwork + iwrk);
    SLC_DGETRF(&m2, &m2, dwork, &m2, iwork, &info2);
    if (info2 > 0) {
        *info = 8;
        return;
    }
    SLC_DGECON("1", &m2, dwork, &m2, &anorm, &rcond[5], dwork + iwrk, iwork + m2, &info2);

    if (rcond[5] < toll) {
        *info = 8;
        return;
    }

    SLC_DGETRS("NoTranspose", &m2, &n, dwork, &m2, iwork, ck, &ldck, &info2);

    SLC_DGETRS("NoTranspose", &m2, &np2, dwork, &m2, iwork, dk, &lddk, &info2);

    SLC_DGEMM("NoTranspose", "NoTranspose", &n, &m2, &np2, &one, bk,
              &ldbk, d + np1 + m1 * ldd, &ldd, &zero, dwork, &n);
    SLC_DGEMM("NoTranspose", "NoTranspose", &n, &n, &m2, &mone, dwork,
              &n, ck, &ldck, &one, ak, &ldak);

    SLC_DGEMM("NoTranspose", "NoTranspose", &n, &np2, &m2, &mone, dwork,
              &n, dk, &lddk, &one, bk, &ldbk);

    dwork[0] = (f64)lwamax;
}
