/**
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * SB10AD - H-infinity optimal controller synthesis
 *
 * Computes an H-infinity optimal n-state controller K = (AK,BK,CK,DK)
 * using modified Glover's and Doyle's 1988 formulas, and the closed-loop
 * system G = (AC,BC,CC,DC) for the plant P = (A,B,C,D).
 *
 * JOB modes:
 *   1: Bisection for gamma reduction
 *   2: Scan from gamma to 0
 *   3: Bisection then scanning
 *   4: Suboptimal controller only
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <stdlib.h>
#include <math.h>

static int select_fn(const f64* wr, const f64* wi) {
    (void)wr;
    (void)wi;
    return 0;
}

void sb10ad(
    const i32 job,
    const i32 n,
    const i32 m,
    const i32 np,
    const i32 ncon,
    const i32 nmeas,
    f64* gamma,
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
    f64* ac,
    const i32 ldac,
    f64* bc,
    const i32 ldbc,
    f64* cc,
    const i32 ldcc,
    f64* dc,
    const i32 lddc,
    f64* rcond,
    const f64 gtol,
    const f64 actol,
    i32* iwork,
    const i32 liwork,
    f64* dwork,
    const i32 ldwork,
    i32* bwork,
    const i32 lbwork,
    i32* info)
{
    const f64 zero = 0.0;
    const f64 one = 1.0;
    const f64 two = 2.0;
    const f64 p1 = 0.1;
    const f64 thous = 1000.0;
    const f64 negone = -1.0;

    i32 m1 = m - ncon;
    i32 m2 = ncon;
    i32 np1 = np - nmeas;
    i32 np2 = nmeas;
    i32 np11 = np1 - m2;
    i32 m11 = m1 - np2;

    *info = 0;

    if (job < 1 || job > 4) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (m < 0) {
        *info = -3;
    } else if (np < 0) {
        *info = -4;
    } else if (ncon < 0 || m1 < 0 || m2 > np1) {
        *info = -5;
    } else if (nmeas < 0 || np1 < 0 || np2 > m1) {
        *info = -6;
    } else if (*gamma < zero) {
        *info = -7;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -9;
    } else if (ldb < (n > 1 ? n : 1)) {
        *info = -11;
    } else if (ldc < (np > 1 ? np : 1)) {
        *info = -13;
    } else if (ldd < (np > 1 ? np : 1)) {
        *info = -15;
    } else if (ldak < (n > 1 ? n : 1)) {
        *info = -17;
    } else if (ldbk < (n > 1 ? n : 1)) {
        *info = -19;
    } else if (ldck < (m2 > 1 ? m2 : 1)) {
        *info = -21;
    } else if (lddk < (m2 > 1 ? m2 : 1)) {
        *info = -23;
    } else if (ldac < (2*n > 1 ? 2*n : 1)) {
        *info = -25;
    } else if (ldbc < (2*n > 1 ? 2*n : 1)) {
        *info = -27;
    } else if (ldcc < (np1 > 1 ? np1 : 1)) {
        *info = -29;
    } else if (lddc < (np1 > 1 ? np1 : 1)) {
        *info = -31;
    } else {
        i32 nn = n * n;
        i32 lw1 = n*m + np*n + np*m + m2*m2 + np2*np2;

        i32 lw2a = (n + np1 + 1)*(n + m2) + (3*(n + m2) + n + np1 > 5*(n + m2) ? 3*(n + m2) + n + np1 : 5*(n + m2));
        i32 lw2b = (n + np2)*(n + m1 + 1) + (3*(n + np2) + n + m1 > 5*(n + np2) ? 3*(n + np2) + n + m1 : 5*(n + np2));
        i32 lw2c_max1 = np1*(n > m1 ? n : m1);
        i32 lw2c_max2 = 3*m2 + np1;
        i32 lw2c_max3 = 5*m2;
        i32 lw2c = m2 + np1*np1 + (lw2c_max1 > lw2c_max2 ? (lw2c_max1 > lw2c_max3 ? lw2c_max1 : lw2c_max3) : (lw2c_max2 > lw2c_max3 ? lw2c_max2 : lw2c_max3));
        i32 lw2d_max1 = (n > np1 ? n : np1)*m1;
        i32 lw2d_max2 = 3*np2 + m1;
        i32 lw2d_max3 = 5*np2;
        i32 lw2d = np2 + m1*m1 + (lw2d_max1 > lw2d_max2 ? (lw2d_max1 > lw2d_max3 ? lw2d_max1 : lw2d_max3) : (lw2d_max2 > lw2d_max3 ? lw2d_max2 : lw2d_max3));
        i32 lw2 = lw2a > lw2b ? (lw2a > lw2c ? (lw2a > lw2d ? lw2a : lw2d) : (lw2c > lw2d ? lw2c : lw2d))
                              : (lw2b > lw2c ? (lw2b > lw2d ? lw2b : lw2d) : (lw2c > lw2d ? lw2c : lw2d));

        i32 min_np11_m1 = np11 < m1 ? np11 : m1;
        i32 max_np11_m1 = np11 > m1 ? np11 : m1;
        i32 min_np1_m11 = np1 < m11 ? np1 : m11;
        i32 max_np1_m11 = np1 > m11 ? np1 : m11;
        i32 lw3a = np11*m1 + (4*min_np11_m1 + max_np11_m1 > 6*min_np11_m1 ? 4*min_np11_m1 + max_np11_m1 : 6*min_np11_m1);
        i32 lw3b = np1*m11 + (4*min_np1_m11 + max_np1_m11 > 6*min_np1_m11 ? 4*min_np1_m11 + max_np1_m11 : 6*min_np1_m11);
        i32 lw3 = lw3a > lw3b ? lw3a : lw3b;

        i32 lw4 = 2*m*m + np*np + 2*m*n + m*np + 2*n*np;
        i32 lw5 = 2*nn + m*n + n*np;

        i32 lw6a_inner = n*m > 10*nn + 12*n + 5 ? n*m : 10*nn + 12*n + 5;
        i32 lw6a = m*m + (2*m1 > 3*nn + lw6a_inner ? 2*m1 : 3*nn + lw6a_inner);
        i32 lw6b_inner = n*np > 10*nn + 12*n + 5 ? n*np : 10*nn + 12*n + 5;
        i32 lw6b = np*np + (2*np1 > 3*nn + lw6b_inner ? 2*np1 : 3*nn + lw6b_inner);
        i32 lw6 = lw6a > lw6b ? lw6a : lw6b;

        i32 lw7a = np11*np11 + (2*np11 > (np11 + m11)*np2 ? 2*np11 : (np11 + m11)*np2);
        i32 lw7b = m11*m11 + (2*m11 > m11*m2 ? 2*m11 : m11*m2);
        i32 lw7c = 3*n;
        i32 lw7d_inner2 = m2*m2 + 3*m2 > np2*(2*np2 + m2 + (np2 > n ? np2 : n)) ? m2*m2 + 3*m2 : np2*(2*np2 + m2 + (np2 > n ? np2 : n));
        i32 lw7d_inner = 2*n*m2 > m2*np2 + lw7d_inner2 ? 2*n*m2 : m2*np2 + lw7d_inner2;
        i32 lw7d = n*(2*np2 + m2) + lw7d_inner;
        i32 lw7_max1 = lw7a > lw7b ? lw7a : lw7b;
        i32 lw7_max2 = lw7c > lw7d ? lw7c : lw7d;
        i32 lw7 = m2*np2 + np2*np2 + m2*m2 + (lw7_max1 > lw7_max2 ? lw7_max1 : lw7_max2);

        i32 lw56_max = lw6 > lw7 ? lw6 : lw7;
        i32 lw_other = lw2 > lw3 ? (lw2 > lw4 ? (lw2 > lw5 + lw56_max ? lw2 : lw5 + lw56_max) : (lw4 > lw5 + lw56_max ? lw4 : lw5 + lw56_max))
                                 : (lw3 > lw4 ? (lw3 > lw5 + lw56_max ? lw3 : lw5 + lw56_max) : (lw4 > lw5 + lw56_max ? lw4 : lw5 + lw56_max));
        i32 minwrk = lw1 + (1 > lw_other ? 1 : lw_other);

        i32 max_dims = n;
        if (m1 > max_dims) max_dims = m1;
        if (np1 > max_dims) max_dims = np1;
        if (m2 > max_dims) max_dims = m2;
        if (np2 > max_dims) max_dims = np2;
        i32 min_liwork = 2*max_dims > nn ? 2*max_dims : nn;

        if (ldwork < minwrk) {
            *info = -38;
        } else if (liwork < min_liwork) {
            *info = -36;
        } else if (lbwork < 2*n) {
            *info = -40;
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
        dwork[0] = one;
        return;
    }

    i32 mode = job;
    if (mode > 2) mode = 1;

    f64 gtoll = gtol;
    if (gtoll <= zero) {
        gtoll = sqrt(SLC_DLAMCH("Epsilon"));
    }

    i32 iwc = n*m;
    i32 iwd = iwc + np*n;
    i32 iwtu = iwd + np*m;
    i32 iwty = iwtu + m2*m2;
    i32 iwrk = iwty + np2*np2;

    i32 ldb_w = n > 1 ? n : 1;
    i32 ldc_w = np > 1 ? np : 1;
    i32 ldd_w = np > 1 ? np : 1;

    SLC_DLACPY("Full", &n, &m, b, &ldb, dwork, &ldb_w);
    SLC_DLACPY("Full", &np, &n, c, &ldc, &dwork[iwc], &ldc_w);
    SLC_DLACPY("Full", &np, &m, d, &ldd, &dwork[iwd], &ldd_w);

    f64 tol2 = negone;
    i32 ldtu = m2 > 1 ? m2 : 1;
    i32 ldty = np2 > 1 ? np2 : 1;
    i32 info2 = 0;
    i32 wrk_avail = ldwork - iwrk;

    sb10pd(n, m, np, ncon, nmeas, a, lda, dwork, ldb_w, &dwork[iwc], ldc_w,
           &dwork[iwd], ldd_w, &dwork[iwtu], ldtu, &dwork[iwty], ldty,
           rcond, tol2, &dwork[iwrk], wrk_avail, &info2);

    i32 lwamax = (i32)dwork[iwrk] + iwrk;

    if (info2 != 0) {
        *info = info2;
        return;
    }

    i32 iwd1 = iwrk;
    i32 iws1 = iwd1 + np11*m1;

    info2 = 0;
    i32 info3 = 0;

    if (np11 > 0 && m1 > 0) {
        i32 min_np11_m1 = np11 < m1 ? np11 : m1;
        iwrk = iws1 + min_np11_m1;
        i32 ldnp11 = np11 > 1 ? np11 : 1;
        SLC_DLACPY("Full", &np11, &m1, &dwork[iwd], &ldd_w, &dwork[iwd1], &ldnp11);
        i32 int1 = 1;
        wrk_avail = ldwork - iwrk;
        SLC_DGESVD("N", "N", &np11, &m1, &dwork[iwd1], &ldnp11,
                   &dwork[iws1], &dwork[iws1], &int1, &dwork[iws1], &int1,
                   &dwork[iwrk], &wrk_avail, &info2);
        i32 wused = (i32)dwork[iwrk] + iwrk;
        if (wused > lwamax) lwamax = wused;
    } else {
        dwork[iws1] = zero;
    }

    i32 iws2 = iwd1 + np1*m11;
    if (np1 > 0 && m11 > 0) {
        i32 min_np1_m11 = np1 < m11 ? np1 : m11;
        iwrk = iws2 + min_np1_m11;
        i32 ldnp1 = np1 > 1 ? np1 : 1;
        SLC_DLACPY("Full", &np1, &m11, &dwork[iwd], &ldd_w, &dwork[iwd1], &ldnp1);
        i32 int1 = 1;
        wrk_avail = ldwork - iwrk;
        SLC_DGESVD("N", "N", &np1, &m11, &dwork[iwd1], &ldnp1,
                   &dwork[iws2], &dwork[iws2], &int1, &dwork[iws2], &int1,
                   &dwork[iwrk], &wrk_avail, &info3);
        i32 wused = (i32)dwork[iwrk] + iwrk;
        if (wused > lwamax) lwamax = wused;
    } else {
        dwork[iws2] = zero;
    }

    f64 gamamn = dwork[iws1] > dwork[iws2] ? dwork[iws1] : dwork[iws2];

    if (info2 > 0 || info3 > 0) {
        *info = 10;
        return;
    } else if (*gamma <= gamamn) {
        *info = 6;
        return;
    }

    i32 nn = n * n;
    i32 n2 = 2 * n;
    i32 iwx = iwd1;
    i32 iwy = iwx + nn;
    i32 iwf = iwy + nn;
    i32 iwh = iwf + m*n;
    iwrk = iwh + n*np;
    i32 iwac = iwd1;
    i32 iwwr = iwac + 4*nn;
    i32 iwwi = iwwr + n2;
    i32 iwre = iwwi + n2;

    f64 stepg = *gamma - gamamn;
    f64 gamabs = *gamma;
    f64 gamamx = *gamma;
    i32 inf = 0;

    i32 ldf = m > 1 ? m : 1;
    i32 ldh = n > 1 ? n : 1;
    i32 ldx = n > 1 ? n : 1;
    i32 ldy = n > 1 ? n : 1;

    do {
        stepg = stepg / two;

        wrk_avail = ldwork - iwrk;
        sb10qd(n, m, np, ncon, nmeas, *gamma, a, lda, dwork, ldb_w,
               &dwork[iwc], ldc_w, &dwork[iwd], ldd_w, &dwork[iwf], ldf,
               &dwork[iwh], ldh, &dwork[iwx], ldx, &dwork[iwy], ldy,
               &rcond[2], iwork, &dwork[iwrk], wrk_avail, bwork, &info2);

        f64 mineac = -thous;

        if (info2 != 0) {
            goto label30;
        }

        wrk_avail = ldwork - iwrk;
        sb10rd(n, m, np, ncon, nmeas, *gamma, a, lda, dwork, ldb_w,
               &dwork[iwc], ldc_w, &dwork[iwd], ldd_w, &dwork[iwf], ldf,
               &dwork[iwh], ldh, &dwork[iwtu], ldtu, &dwork[iwty], ldty,
               &dwork[iwx], ldx, &dwork[iwy], ldy, ak, ldak, bk, ldbk,
               ck, ldck, dk, lddk, iwork, &dwork[iwrk], wrk_avail, &info2);

        if (info2 != 0) {
            goto label30;
        }

        wrk_avail = ldwork - iwd1;
        sb10ld(n, m, np, ncon, nmeas, a, lda, b, ldb, c, ldc, d, ldd,
               ak, ldak, bk, ldbk, ck, ldck, dk, lddk, ac, ldac, bc, ldbc,
               cc, ldcc, dc, lddc, iwork, &dwork[iwd1], wrk_avail, &info2);

        if (info2 != 0) {
            goto label30;
        }

        {
            i32 wused = (i32)dwork[iwd1] + iwd1;
            if (wused > lwamax) lwamax = wused;
        }

        {
            i32 ldac_copy = n2 > 1 ? n2 : 1;
            SLC_DLACPY("Full", &n2, &n2, ac, &ldac, &dwork[iwac], &ldac_copy);

            i32 sdim = 0;
            i32 int1 = 1;
            wrk_avail = ldwork - iwre;
            SLC_DGEES("N", "N", select_fn, &n2, &dwork[iwac], &ldac_copy, &sdim,
                      &dwork[iwwr], &dwork[iwwi], &dwork[iwre], &int1,
                      &dwork[iwre], &wrk_avail, bwork, &info2);

            i32 wused = (i32)dwork[iwre] + iwre;
            if (wused > lwamax) lwamax = wused;
        }

        mineac = -thous;
        for (i32 i = 0; i < n2; i++) {
            if (dwork[iwwr + i] > mineac) {
                mineac = dwork[iwwr + i];
            }
        }

    label30:
        if (mode == 1) {
            if (info2 == 0 && mineac < actol) {
                gamabs = *gamma;
                *gamma = *gamma - stepg;
                inf = 1;
            } else {
                f64 gnew = *gamma + stepg;
                *gamma = gnew < gamamx ? gnew : gamamx;
            }
        } else if (mode == 2) {
            if (info2 == 0 && mineac < actol) {
                gamabs = *gamma;
                inf = 1;
            }
            f64 step_decr = p1 > gtoll ? p1 : gtoll;
            *gamma = *gamma - step_decr;
        }

        if (mode == 1 && job == 3 && two*stepg < gtoll) {
            mode = 2;
            *gamma = gamabs;
        }

    } while (job != 4 && ((mode == 1 && two*stepg >= gtoll) || (mode == 2 && *gamma > zero)));

    if (inf == 0) {
        *info = 12;
        return;
    }

    *gamma = gamabs;

    wrk_avail = ldwork - iwrk;
    sb10qd(n, m, np, ncon, nmeas, *gamma, a, lda, dwork, ldb_w,
           &dwork[iwc], ldc_w, &dwork[iwd], ldd_w, &dwork[iwf], ldf,
           &dwork[iwh], ldh, &dwork[iwx], ldx, &dwork[iwy], ldy,
           &rcond[2], iwork, &dwork[iwrk], wrk_avail, bwork, &info2);

    {
        i32 wused = (i32)dwork[iwrk] + iwrk;
        if (wused > lwamax) lwamax = wused;
    }

    if (info2 > 0) {
        *info = info2 + 5;
        return;
    }

    wrk_avail = ldwork - iwrk;
    sb10rd(n, m, np, ncon, nmeas, *gamma, a, lda, dwork, ldb_w,
           &dwork[iwc], ldc_w, &dwork[iwd], ldd_w, &dwork[iwf], ldf,
           &dwork[iwh], ldh, &dwork[iwtu], ldtu, &dwork[iwty], ldty,
           &dwork[iwx], ldx, &dwork[iwy], ldy, ak, ldak, bk, ldbk,
           ck, ldck, dk, lddk, iwork, &dwork[iwrk], wrk_avail, &info2);

    {
        i32 wused = (i32)dwork[iwrk] + iwrk;
        if (wused > lwamax) lwamax = wused;
    }

    if (info2 == 1) {
        *info = 6;
        return;
    } else if (info2 == 2) {
        *info = 9;
        return;
    }

    sb10ld(n, m, np, ncon, nmeas, a, lda, b, ldb, c, ldc, d, ldd,
           ak, ldak, bk, ldbk, ck, ldck, dk, lddk, ac, ldac, bc, ldbc,
           cc, ldcc, dc, lddc, iwork, dwork, ldwork, &info2);

    if (info2 > 0) {
        *info = 11;
        return;
    }

    dwork[0] = (f64)lwamax;
}
