/**
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * SB03MU - Solve small discrete-time Sylvester equation
 *
 * Solves: ISGN*op(TL)*X*op(TR) - X = SCALE*B
 * where TL is N1-by-N1, TR is N2-by-N2, B is N1-by-N2,
 * N1,N2 in {0,1,2}, ISGN = 1 or -1, op(T) = T or T'.
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>

void sb03mu(
    const bool ltranl,
    const bool ltranr,
    const i32 isgn,
    const i32 n1,
    const i32 n2,
    const f64* tl,
    const i32 ldtl,
    const f64* tr,
    const i32 ldtr,
    const f64* b,
    const i32 ldb,
    f64* scale,
    f64* x,
    const i32 ldx,
    f64* xnorm,
    i32* info
)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const f64 TWO = 2.0;
    const f64 HALF = 0.5;
    const f64 EIGHT = 8.0;

    static const i32 locu12[4] = {2, 3, 0, 1};
    static const i32 locl21[4] = {1, 0, 3, 2};
    static const i32 locu22[4] = {3, 2, 1, 0};
    static const bool xswpiv[4] = {false, false, true, true};
    static const bool bswpiv[4] = {false, true, false, true};

    f64 btmp[4], t16[16], tmp[4], x2[2];
    i32 jpiv[4];

    *info = 0;
    *scale = ONE;

    if (n1 == 0 || n2 == 0) {
        *xnorm = ZERO;
        return;
    }

    f64 eps = SLC_DLAMCH("P");
    f64 smlnum = SLC_DLAMCH("S") / eps;
    f64 sgn = (f64)isgn;

    i32 k = n1 + n1 + n2 - 2;

    if (k == 1) {
        f64 tau1 = sgn * tl[0] * tr[0] - ONE;
        f64 bet = fabs(tau1);
        if (bet <= smlnum) {
            tau1 = smlnum;
            bet = smlnum;
            *info = 1;
        }

        f64 gam = fabs(b[0]);
        if (smlnum * gam > bet) {
            *scale = ONE / gam;
        }

        x[0] = (b[0] * (*scale)) / tau1;
        *xnorm = fabs(x[0]);
        return;
    }

    f64 smin;
    if (k == 2) {
        f64 max_tr = fmax(fmax(fabs(tr[0]), fabs(tr[0 + ldtr])),
                         fmax(fabs(tr[1]), fabs(tr[1 + ldtr])));
        smin = fmax(max_tr * fabs(tl[0]) * eps, smlnum);

        tmp[0] = sgn * tl[0] * tr[0] - ONE;
        tmp[3] = sgn * tl[0] * tr[1 + ldtr] - ONE;
        if (ltranr) {
            tmp[1] = sgn * tl[0] * tr[1];
            tmp[2] = sgn * tl[0] * tr[0 + ldtr];
        } else {
            tmp[1] = sgn * tl[0] * tr[0 + ldtr];
            tmp[2] = sgn * tl[0] * tr[1];
        }
        btmp[0] = b[0];
        btmp[1] = b[0 + ldb];
    } else if (k == 3) {
        f64 max_tl = fmax(fmax(fabs(tl[0]), fabs(tl[0 + ldtl])),
                         fmax(fabs(tl[1]), fabs(tl[1 + ldtl])));
        smin = fmax(max_tl * fabs(tr[0]) * eps, smlnum);

        tmp[0] = sgn * tl[0] * tr[0] - ONE;
        tmp[3] = sgn * tl[1 + ldtl] * tr[0] - ONE;
        if (ltranl) {
            tmp[1] = sgn * tl[0 + ldtl] * tr[0];
            tmp[2] = sgn * tl[1] * tr[0];
        } else {
            tmp[1] = sgn * tl[1] * tr[0];
            tmp[2] = sgn * tl[0 + ldtl] * tr[0];
        }
        btmp[0] = b[0];
        btmp[1] = b[1];
    }

    if (k == 2 || k == 3) {
        i32 ipiv = 0;
        f64 maxval = fabs(tmp[0]);
        for (i32 i = 1; i < 4; i++) {
            if (fabs(tmp[i]) > maxval) {
                maxval = fabs(tmp[i]);
                ipiv = i;
            }
        }

        f64 u11 = tmp[ipiv];
        if (fabs(u11) <= smin) {
            *info = 1;
            u11 = smin;
        }
        f64 u12 = tmp[locu12[ipiv]];
        f64 l21 = tmp[locl21[ipiv]] / u11;
        f64 u22 = tmp[locu22[ipiv]] - u12 * l21;
        bool xswap = xswpiv[ipiv];
        bool bswap = bswpiv[ipiv];

        if (fabs(u22) <= smin) {
            *info = 1;
            u22 = smin;
        }

        if (bswap) {
            f64 temp = btmp[1];
            btmp[1] = btmp[0] - l21 * temp;
            btmp[0] = temp;
        } else {
            btmp[1] = btmp[1] - l21 * btmp[0];
        }

        if ((TWO * smlnum) * fabs(btmp[1]) > fabs(u22) ||
            (TWO * smlnum) * fabs(btmp[0]) > fabs(u11)) {
            *scale = HALF / fmax(fabs(btmp[0]), fabs(btmp[1]));
            btmp[0] = btmp[0] * (*scale);
            btmp[1] = btmp[1] * (*scale);
        }

        x2[1] = btmp[1] / u22;
        x2[0] = btmp[0] / u11 - (u12 / u11) * x2[1];

        if (xswap) {
            f64 temp = x2[1];
            x2[1] = x2[0];
            x2[0] = temp;
        }

        x[0] = x2[0];
        if (n1 == 1) {
            x[0 + ldx] = x2[1];
            *xnorm = fabs(x2[0]) + fabs(x2[1]);
        } else {
            x[1] = x2[1];
            *xnorm = fmax(fabs(x2[0]), fabs(x2[1]));
        }
        return;
    }

    f64 max_tr = fmax(fmax(fabs(tr[0]), fabs(tr[0 + ldtr])),
                     fmax(fabs(tr[1]), fabs(tr[1 + ldtr])));
    f64 max_tl = fmax(fmax(fabs(tl[0]), fabs(tl[0 + ldtl])),
                     fmax(fabs(tl[1]), fabs(tl[1 + ldtl])));
    smin = fmax(eps * max_tl * max_tr, smlnum);

    t16[0 + 0*4] = sgn * tl[0] * tr[0] - ONE;
    t16[1 + 1*4] = sgn * tl[1 + ldtl] * tr[0] - ONE;
    t16[2 + 2*4] = sgn * tl[0] * tr[1 + ldtr] - ONE;
    t16[3 + 3*4] = sgn * tl[1 + ldtl] * tr[1 + ldtr] - ONE;

    if (ltranl) {
        t16[0 + 1*4] = sgn * tl[1] * tr[0];
        t16[1 + 0*4] = sgn * tl[0 + ldtl] * tr[0];
        t16[2 + 3*4] = sgn * tl[1] * tr[1 + ldtr];
        t16[3 + 2*4] = sgn * tl[0 + ldtl] * tr[1 + ldtr];
    } else {
        t16[0 + 1*4] = sgn * tl[0 + ldtl] * tr[0];
        t16[1 + 0*4] = sgn * tl[1] * tr[0];
        t16[2 + 3*4] = sgn * tl[0 + ldtl] * tr[1 + ldtr];
        t16[3 + 2*4] = sgn * tl[1] * tr[1 + ldtr];
    }

    if (ltranr) {
        t16[0 + 2*4] = sgn * tl[0] * tr[0 + ldtr];
        t16[1 + 3*4] = sgn * tl[1 + ldtl] * tr[0 + ldtr];
        t16[2 + 0*4] = sgn * tl[0] * tr[1];
        t16[3 + 1*4] = sgn * tl[1 + ldtl] * tr[1];
    } else {
        t16[0 + 2*4] = sgn * tl[0] * tr[1];
        t16[1 + 3*4] = sgn * tl[1 + ldtl] * tr[1];
        t16[2 + 0*4] = sgn * tl[0] * tr[0 + ldtr];
        t16[3 + 1*4] = sgn * tl[1 + ldtl] * tr[0 + ldtr];
    }

    if (ltranl && ltranr) {
        t16[0 + 3*4] = sgn * tl[1] * tr[0 + ldtr];
        t16[1 + 2*4] = sgn * tl[0 + ldtl] * tr[0 + ldtr];
        t16[2 + 1*4] = sgn * tl[1] * tr[1];
        t16[3 + 0*4] = sgn * tl[0 + ldtl] * tr[1];
    } else if (ltranl && !ltranr) {
        t16[0 + 3*4] = sgn * tl[1] * tr[1];
        t16[1 + 2*4] = sgn * tl[0 + ldtl] * tr[1];
        t16[2 + 1*4] = sgn * tl[1] * tr[0 + ldtr];
        t16[3 + 0*4] = sgn * tl[0 + ldtl] * tr[0 + ldtr];
    } else if (!ltranl && ltranr) {
        t16[0 + 3*4] = sgn * tl[0 + ldtl] * tr[0 + ldtr];
        t16[1 + 2*4] = sgn * tl[1] * tr[0 + ldtr];
        t16[2 + 1*4] = sgn * tl[0 + ldtl] * tr[1];
        t16[3 + 0*4] = sgn * tl[1] * tr[1];
    } else {
        t16[0 + 3*4] = sgn * tl[0 + ldtl] * tr[1];
        t16[1 + 2*4] = sgn * tl[1] * tr[1];
        t16[2 + 1*4] = sgn * tl[0 + ldtl] * tr[0 + ldtr];
        t16[3 + 0*4] = sgn * tl[1] * tr[0 + ldtr];
    }

    btmp[0] = b[0];
    btmp[1] = b[1];
    btmp[2] = b[0 + ldb];
    btmp[3] = b[1 + ldb];

    for (i32 i = 0; i < 3; i++) {
        f64 xmax = ZERO;
        i32 ipsv = i, jpsv = i;

        for (i32 ip = i; ip < 4; ip++) {
            for (i32 jp = i; jp < 4; jp++) {
                if (fabs(t16[ip + jp*4]) >= xmax) {
                    xmax = fabs(t16[ip + jp*4]);
                    ipsv = ip;
                    jpsv = jp;
                }
            }
        }

        if (ipsv != i) {
            i32 four = 4;
            SLC_DSWAP(&four, &t16[ipsv + 0*4], &four, &t16[i + 0*4], &four);
            f64 temp = btmp[i];
            btmp[i] = btmp[ipsv];
            btmp[ipsv] = temp;
        }
        if (jpsv != i) {
            i32 four = 4;
            i32 one = 1;
            SLC_DSWAP(&four, &t16[0 + jpsv*4], &one, &t16[0 + i*4], &one);
        }
        jpiv[i] = jpsv;

        if (fabs(t16[i + i*4]) < smin) {
            *info = 1;
            t16[i + i*4] = smin;
        }

        for (i32 j = i + 1; j < 4; j++) {
            t16[j + i*4] = t16[j + i*4] / t16[i + i*4];
            btmp[j] = btmp[j] - t16[j + i*4] * btmp[i];
            for (i32 kk = i + 1; kk < 4; kk++) {
                t16[j + kk*4] = t16[j + kk*4] - t16[j + i*4] * t16[i + kk*4];
            }
        }
    }

    if (fabs(t16[3 + 3*4]) < smin) {
        t16[3 + 3*4] = smin;
    }

    if ((EIGHT * smlnum) * fabs(btmp[0]) > fabs(t16[0 + 0*4]) ||
        (EIGHT * smlnum) * fabs(btmp[1]) > fabs(t16[1 + 1*4]) ||
        (EIGHT * smlnum) * fabs(btmp[2]) > fabs(t16[2 + 2*4]) ||
        (EIGHT * smlnum) * fabs(btmp[3]) > fabs(t16[3 + 3*4])) {
        f64 maxb = fmax(fmax(fabs(btmp[0]), fabs(btmp[1])),
                        fmax(fabs(btmp[2]), fabs(btmp[3])));
        *scale = (ONE / EIGHT) / maxb;
        btmp[0] = btmp[0] * (*scale);
        btmp[1] = btmp[1] * (*scale);
        btmp[2] = btmp[2] * (*scale);
        btmp[3] = btmp[3] * (*scale);
    }

    for (i32 i = 0; i < 4; i++) {
        i32 kk = 3 - i;
        f64 temp = ONE / t16[kk + kk*4];
        tmp[kk] = btmp[kk] * temp;
        for (i32 j = kk + 1; j < 4; j++) {
            tmp[kk] = tmp[kk] - (temp * t16[kk + j*4]) * tmp[j];
        }
    }

    for (i32 i = 0; i < 3; i++) {
        if (jpiv[2-i] != 2-i) {
            f64 temp = tmp[2-i];
            tmp[2-i] = tmp[jpiv[2-i]];
            tmp[jpiv[2-i]] = temp;
        }
    }

    x[0]         = tmp[0];
    x[1]         = tmp[1];
    x[0 + ldx]   = tmp[2];
    x[1 + ldx]   = tmp[3];

    *xnorm = fmax(fabs(tmp[0]) + fabs(tmp[2]),
                  fabs(tmp[1]) + fabs(tmp[3]));
}
