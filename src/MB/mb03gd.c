/*
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * MB03GD - Exchange eigenvalues of 2x2 or 4x4 skew-Hamiltonian/Hamiltonian pencil
 *          (factored version)
 */

#include "slicot/mb03.h"
#include "slicot/mb01.h"
#include "slicot/mb04.h"
#include "slicot_blas.h"
#include <math.h>

void mb03gd(i32 n, const f64 *b, i32 ldb, const f64 *d, i32 ldd,
            const f64 *macpar, f64 *q, i32 ldq, f64 *u, i32 ldu,
            f64 *dwork, i32 ldwork, i32 *info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const f64 TWO = 2.0;

    i32 int1 = 1;
    i32 int2 = 2;
    i32 int4 = 4;

    *info = 0;

    if (n == 4) {
        f64 eps = macpar[0];
        f64 sfmin = macpar[1];

        f64 co, si, r, f, g, t, s;
        f64 co2, si2;
        f64 smin1, smax1, sr1, cr1, sl1, cl1;
        f64 smin2, smax2, sr2, cr2, sl2, cl2;
        f64 r1, f1, g1, t1;

        SLC_DLARTG(&b[0 + 0*ldb], &b[1 + 0*ldb], &co, &si, &r);
        f = co * b[0 + 1*ldb] + si * b[1 + 1*ldb];
        g = co * b[1 + 1*ldb] - si * b[0 + 1*ldb];
        SLC_DLASV2(&r, &f, &g, &smin1, &smax1, &sr1, &cr1, &sl1, &cl1);

        if (fabs(smin1) < fmax(sfmin, eps * fabs(smax1))) {
            *info = 1;
            return;
        }

        SLC_DLARTG(&b[2 + 2*ldb], &b[3 + 2*ldb], &co2, &si2, &r);
        f = co2 * b[2 + 3*ldb] + si2 * b[3 + 3*ldb];
        g = co2 * b[3 + 3*ldb] - si2 * b[2 + 3*ldb];
        SLC_DLASV2(&r, &f, &g, &smin2, &smax2, &sr2, &cr2, &sl2, &cl2);

        if (fabs(smin2) < fmax(sfmin, eps * fabs(smax2))) {
            *info = 1;
            return;
        }

        r  = (cr1 * d[0 + 0*ldd] + sr1 * d[0 + 1*ldd]) / smax1;
        f  = (cr1 * d[1 + 0*ldd] + sr1 * d[1 + 1*ldd]) / smax1;
        t  = (cr1 * d[0 + 1*ldd] - sr1 * d[0 + 0*ldd]) / smin1;
        g  = (cr1 * d[1 + 1*ldd] - sr1 * d[1 + 0*ldd]) / smin1;

        r1 = cl1 * r - sl1 * t;
        f1 = cl1 * f - sl1 * g;
        t1 = cl1 * t + sl1 * r;
        g1 = cl1 * g + sl1 * f;

        u[0 + 2*ldu] = co * r1 - si * t1;
        u[1 + 2*ldu] = co * t1 + si * r1;
        u[0 + 3*ldu] = co * f1 - si * g1;
        u[1 + 3*ldu] = co * g1 + si * f1;

        q[0 + 0*ldq] = d[0 + 2*ldd];
        q[0 + 1*ldq] = d[0 + 3*ldd];
        q[1 + 1*ldq] = d[1 + 3*ldd];

        f64 alpha = ONE;
        f64 beta = -ONE;
        SLC_DSYR2K("U", "T", &int2, &int2, &alpha, &u[0 + 2*ldu], &ldu,
                   &b[0 + 2*ldb], &ldb, &beta, q, &ldq);

        r  =  cr2 / smax2;
        t  =  sr2 / smax2;
        f  = -sr2 / smin2;
        g  =  cr2 / smin2;

        r1 = cl2 * r - sl2 * f;
        t1 = cl2 * t - sl2 * g;
        f1 = cl2 * f + sl2 * r;
        g1 = cl2 * g + sl2 * t;

        u[2 + 2*ldu] = co2 * r1 - si2 * f1;
        u[3 + 2*ldu] = co2 * t1 - si2 * g1;
        u[2 + 3*ldu] = co2 * f1 + si2 * r1;
        u[3 + 3*ldu] = co2 * g1 + si2 * t1;

        f64 neg_one = -ONE;
        SLC_DGEMM("N", "N", &int2, &int2, &int2, &neg_one,
                  &u[0 + 2*ldu], &ldu, &u[2 + 2*ldu], &ldu, &ZERO, u, &ldu);

        mb01ru("U", "T", int2, int2, ZERO, ONE, &u[2 + 0*ldu], ldu,
               &u[2 + 2*ldu], ldu, q, ldq, dwork, 4, info);
        u[3 + 0*ldu] = u[2 + 1*ldu];

        s = -(u[0 + 0*ldu] + u[1 + 1*ldu]);

        t = u[0 + 0*ldu] * u[1 + 1*ldu] - u[1 + 0*ldu] * u[0 + 1*ldu];

        SLC_DLACPY("F", &int4, &int2, u, &ldu, q, &ldq);
        q[0 + 2*ldq] = u[0 + 0*ldu] - s;
        q[1 + 2*ldq] = u[1 + 0*ldu];
        q[2 + 2*ldq] = u[0 + 0*ldu];
        q[3 + 2*ldq] = u[1 + 0*ldu];
        q[0 + 3*ldq] = u[0 + 1*ldu];
        q[1 + 3*ldq] = u[1 + 1*ldu] - s;
        q[2 + 3*ldq] = u[0 + 1*ldu];
        q[3 + 3*ldq] = u[1 + 1*ldu];

        SLC_DGEMM("N", "N", &int4, &int2, &int2, &ONE,
                  q, &ldq, &q[0 + 2*ldq], &ldq, &ZERO, u, &ldu);
        SLC_DGEMM("T", "N", &int2, &int2, &int2, &neg_one,
                  &q[2 + 2*ldq], &ldq, &q[2 + 0*ldq], &ldq, &ONE, &u[2 + 0*ldu], &ldu);
        u[0 + 0*ldu] += t;
        u[1 + 1*ldu] += t;

        i32 ics = 0;
        i32 itau = ics + 4;
        i32 iwrk2 = itau + 2;

        mb04su(int2, int2, &u[0 + 0*ldu], ldu, &u[2 + 0*ldu], ldu,
               &dwork[ics], &dwork[itau], &dwork[iwrk2], ldwork - iwrk2, info);
        mb04wu(false, false, int2, int2, int2,
               &u[0 + 0*ldu], ldu, &u[2 + 0*ldu], ldu,
               &dwork[ics], &dwork[itau], &dwork[iwrk2], ldwork - iwrk2, info);

        f64 temp1 = u[0 + 0*ldu];
        f64 temp2 = u[1 + 0*ldu];
        f64 temp3 = u[0 + 1*ldu];
        f64 temp4 = u[1 + 1*ldu];

        u[0 + 2*ldu] = temp1;
        u[1 + 2*ldu] = temp2;
        u[0 + 3*ldu] = temp3;
        u[1 + 3*ldu] = temp4;

        u[0 + 0*ldu] = -u[2 + 0*ldu];
        u[1 + 0*ldu] = -u[3 + 0*ldu];
        u[0 + 1*ldu] = -u[2 + 1*ldu];
        u[1 + 1*ldu] = -u[3 + 1*ldu];

        u[2 + 0*ldu] = -u[0 + 2*ldu];
        u[3 + 0*ldu] = -u[1 + 2*ldu];
        u[2 + 1*ldu] = -u[0 + 3*ldu];
        u[3 + 1*ldu] = -u[1 + 3*ldu];

        u[2 + 2*ldu] = u[0 + 0*ldu];
        u[3 + 2*ldu] = u[1 + 0*ldu];
        u[2 + 3*ldu] = u[0 + 1*ldu];
        u[3 + 3*ldu] = u[1 + 1*ldu];

        SLC_DGEMM("T", "N", &int4, &int2, &int2, &ONE, u, &ldu,
                  b, &ldb, &ZERO, q, &ldq);
        SLC_DGEMM("T", "N", &int4, &int2, &int4, &ONE, u, &ldu,
                  &b[0 + 2*ldb], &ldb, &ZERO, &q[0 + 2*ldq], &ldq);

        i32 itau_rq = 0;
        i32 iwrk1 = itau_rq + n;
        SLC_DGERQ2(&int4, &int4, q, &ldq, &dwork[itau_rq], &dwork[iwrk1], info);

        i32 ir = iwrk1;
        iwrk2 = ir + 4;
        dwork[ir + 0] = q[2 + 2*ldq];
        dwork[ir + 1] = q[2 + 3*ldq];
        dwork[ir + 2] = ZERO;
        dwork[ir + 3] = q[3 + 3*ldq];

        SLC_DORGR2(&int4, &int4, &int4, q, &ldq, &dwork[itau_rq], &dwork[iwrk2], info);

        for (i32 i = 1; i < n; i++) {
            i32 len = n - i;
            SLC_DSWAP(&len, &q[i + (i-1)*ldq], &int1, &q[(i-1) + i*ldq], &ldq);
        }

        SLC_DGEQR2(&int2, &int2, &dwork[ir], &int2, &dwork[itau_rq], &dwork[iwrk2], info);
        SLC_DORM2R("R", "N", &int4, &int2, &int2, &dwork[ir], &int2,
                   &dwork[itau_rq], &q[0 + 2*ldq], &ldq, &dwork[iwrk2], info);

    } else {
        f64 g2 = TWO * b[0 + 0*ldb] * b[1 + 1*ldb] * d[0 + 0*ldd];
        f64 arg1 = b[0 + 0*ldb] * b[1 + 1*ldb] * d[0 + 1*ldd];
        f64 co, si, r;
        SLC_DLARTG(&arg1, &g2, &co, &si, &r);

        q[0 + 0*ldq] =  co;
        q[1 + 0*ldq] = -si;
        q[0 + 1*ldq] =  si;
        q[1 + 1*ldq] =  co;

        arg1 = b[0 + 0*ldb] * q[0 + 0*ldq] + b[0 + 1*ldb] * q[1 + 0*ldq];
        f64 arg2 = b[1 + 1*ldb] * q[1 + 0*ldq];
        SLC_DLARTG(&arg1, &arg2, &co, &si, &r);

        u[0 + 0*ldu] =  co;
        u[1 + 0*ldu] =  si;
        u[0 + 1*ldu] = -si;
        u[1 + 1*ldu] =  co;
    }
}
