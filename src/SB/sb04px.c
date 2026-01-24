/**
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * SB04PX - Solve 2x2 Sylvester equation
 *
 * Solves for N1-by-N2 matrix X (1 <= N1,N2 <= 2) in:
 *   op(TL)*X*op(TR) + ISGN*X = SCALE*B
 *
 * where op(T) = T or T', ISGN = 1 or -1.
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <float.h>

void sb04px(
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
    i32* info)
{
    const f64 zero = 0.0;
    const f64 one = 1.0;
    const f64 two = 2.0;
    const f64 half = 0.5;
    const f64 eight = 8.0;

    static const i32 locu12[4] = {2, 3, 0, 1};
    static const i32 locl21[4] = {1, 0, 3, 2};
    static const i32 locu22[4] = {3, 2, 1, 0};
    static const bool xswpiv[4] = {false, false, true, true};
    static const bool bswpiv[4] = {false, true, false, true};

    f64 btmp[4], tmp[4], x2[2];
    f64 t16[16];
    i32 jpiv[4];

    i32 i, j, k, ip, jp, ipiv, ipsv, jpsv;
    f64 bet, eps, gam, l21, sgn, smin, smlnum, tau1;
    f64 temp, u11, u12, u22, xmax;
    bool bswap, xswap;

    *info = 0;
    *scale = one;

    if (n1 == 0 || n2 == 0) {
        *xnorm = zero;
        return;
    }

    eps = DBL_EPSILON;
    smlnum = DBL_MIN / eps;
    sgn = (f64)isgn;

    k = n1 + n1 + n2 - 2;

    switch (k) {
        case 1:
            /* 1-by-1: TL11*X*TR11 + ISGN*X = B11 */
            tau1 = tl[0] * tr[0] + sgn;
            bet = fabs(tau1);
            if (bet <= smlnum) {
                tau1 = smlnum;
                bet = smlnum;
                *info = 1;
            }

            gam = fabs(b[0]);
            if (smlnum * gam > bet) {
                *scale = one / gam;
            }

            x[0] = (b[0] * (*scale)) / tau1;
            *xnorm = fabs(x[0]);
            return;

        case 2:
            /* 1-by-2: TL11*[X11 X12]*op[TR] + ISGN*[X11 X12] = [B11 B12] */
            {
                f64 abs_tr00 = fabs(tr[0]);
                f64 abs_tr01 = fabs(tr[0 + 1*ldtr]);
                f64 abs_tr10 = fabs(tr[1]);
                f64 abs_tr11 = fabs(tr[1 + 1*ldtr]);
                f64 max_tr = abs_tr00;
                if (abs_tr01 > max_tr) max_tr = abs_tr01;
                if (abs_tr10 > max_tr) max_tr = abs_tr10;
                if (abs_tr11 > max_tr) max_tr = abs_tr11;

                smin = max_tr * fabs(tl[0]) * eps;
                if (smin < smlnum) smin = smlnum;

                tmp[0] = tl[0] * tr[0] + sgn;
                tmp[3] = tl[0] * tr[1 + 1*ldtr] + sgn;
                if (ltranr) {
                    tmp[1] = tl[0] * tr[1];
                    tmp[2] = tl[0] * tr[0 + 1*ldtr];
                } else {
                    tmp[1] = tl[0] * tr[0 + 1*ldtr];
                    tmp[2] = tl[0] * tr[1];
                }
                btmp[0] = b[0];
                btmp[1] = b[0 + 1*ldb];
            }
            goto solve_2x2;

        case 3:
            /* 2-by-1: op[TL]*[X11; X21]*TR11 + ISGN*[X11; X21] = [B11; B21] */
            {
                f64 abs_tl00 = fabs(tl[0]);
                f64 abs_tl01 = fabs(tl[0 + 1*ldtl]);
                f64 abs_tl10 = fabs(tl[1]);
                f64 abs_tl11 = fabs(tl[1 + 1*ldtl]);
                f64 max_tl = abs_tl00;
                if (abs_tl01 > max_tl) max_tl = abs_tl01;
                if (abs_tl10 > max_tl) max_tl = abs_tl10;
                if (abs_tl11 > max_tl) max_tl = abs_tl11;

                smin = max_tl * fabs(tr[0]) * eps;
                if (smin < smlnum) smin = smlnum;

                tmp[0] = tl[0] * tr[0] + sgn;
                tmp[3] = tl[1 + 1*ldtl] * tr[0] + sgn;
                if (ltranl) {
                    tmp[1] = tl[0 + 1*ldtl] * tr[0];
                    tmp[2] = tl[1] * tr[0];
                } else {
                    tmp[1] = tl[1] * tr[0];
                    tmp[2] = tl[0 + 1*ldtl] * tr[0];
                }
                btmp[0] = b[0];
                btmp[1] = b[1];
            }
            goto solve_2x2;

        case 4:
            /* 2-by-2 case */
            goto solve_4x4;
    }

    return;

solve_2x2:
    /* Solve 2-by-2 system using complete pivoting */
    {
        f64 abs_tmp[4];
        abs_tmp[0] = fabs(tmp[0]);
        abs_tmp[1] = fabs(tmp[1]);
        abs_tmp[2] = fabs(tmp[2]);
        abs_tmp[3] = fabs(tmp[3]);

        ipiv = 0;
        f64 maxval = abs_tmp[0];
        for (i = 1; i < 4; i++) {
            if (abs_tmp[i] > maxval) {
                maxval = abs_tmp[i];
                ipiv = i;
            }
        }

        u11 = tmp[ipiv];
        if (fabs(u11) <= smin) {
            *info = 1;
            u11 = smin;
        }
        u12 = tmp[locu12[ipiv]];
        l21 = tmp[locl21[ipiv]] / u11;
        u22 = tmp[locu22[ipiv]] - u12 * l21;
        xswap = xswpiv[ipiv];
        bswap = bswpiv[ipiv];
        if (fabs(u22) <= smin) {
            *info = 1;
            u22 = smin;
        }

        if (bswap) {
            temp = btmp[1];
            btmp[1] = btmp[0] - l21 * temp;
            btmp[0] = temp;
        } else {
            btmp[1] = btmp[1] - l21 * btmp[0];
        }

        if ((two * smlnum) * fabs(btmp[1]) > fabs(u22) ||
            (two * smlnum) * fabs(btmp[0]) > fabs(u11)) {
            f64 maxb = fabs(btmp[0]);
            if (fabs(btmp[1]) > maxb) maxb = fabs(btmp[1]);
            *scale = half / maxb;
            btmp[0] = btmp[0] * (*scale);
            btmp[1] = btmp[1] * (*scale);
        }

        x2[1] = btmp[1] / u22;
        x2[0] = btmp[0] / u11 - (u12 / u11) * x2[1];

        if (xswap) {
            temp = x2[1];
            x2[1] = x2[0];
            x2[0] = temp;
        }

        x[0] = x2[0];
        if (n1 == 1) {
            x[0 + 1*ldx] = x2[1];
            *xnorm = fabs(x2[0]) + fabs(x2[1]);
        } else {
            x[1] = x2[1];
            f64 ax0 = fabs(x2[0]);
            f64 ax1 = fabs(x2[1]);
            *xnorm = (ax0 > ax1) ? ax0 : ax1;
        }
    }
    return;

solve_4x4:
    /* 2-by-2: solve equivalent 4-by-4 system */
    {
        f64 abs_tr00 = fabs(tr[0]);
        f64 abs_tr01 = fabs(tr[0 + 1*ldtr]);
        f64 abs_tr10 = fabs(tr[1]);
        f64 abs_tr11 = fabs(tr[1 + 1*ldtr]);
        f64 max_tr = abs_tr00;
        if (abs_tr01 > max_tr) max_tr = abs_tr01;
        if (abs_tr10 > max_tr) max_tr = abs_tr10;
        if (abs_tr11 > max_tr) max_tr = abs_tr11;

        f64 abs_tl00 = fabs(tl[0]);
        f64 abs_tl01 = fabs(tl[0 + 1*ldtl]);
        f64 abs_tl10 = fabs(tl[1]);
        f64 abs_tl11 = fabs(tl[1 + 1*ldtl]);
        f64 max_tl = abs_tl00;
        if (abs_tl01 > max_tl) max_tl = abs_tl01;
        if (abs_tl10 > max_tl) max_tl = abs_tl10;
        if (abs_tl11 > max_tl) max_tl = abs_tl11;

        smin = max_tr * max_tl;
        smin = eps * smin;
        if (smin < smlnum) smin = smlnum;

        #define T16(i, j) t16[(i) + (j)*4]

        T16(0, 0) = tl[0] * tr[0] + sgn;
        T16(1, 1) = tl[1 + 1*ldtl] * tr[0] + sgn;
        T16(2, 2) = tl[0] * tr[1 + 1*ldtr] + sgn;
        T16(3, 3) = tl[1 + 1*ldtl] * tr[1 + 1*ldtr] + sgn;

        if (ltranl) {
            T16(0, 1) = tl[1] * tr[0];
            T16(1, 0) = tl[0 + 1*ldtl] * tr[0];
            T16(2, 3) = tl[1] * tr[1 + 1*ldtr];
            T16(3, 2) = tl[0 + 1*ldtl] * tr[1 + 1*ldtr];
        } else {
            T16(0, 1) = tl[0 + 1*ldtl] * tr[0];
            T16(1, 0) = tl[1] * tr[0];
            T16(2, 3) = tl[0 + 1*ldtl] * tr[1 + 1*ldtr];
            T16(3, 2) = tl[1] * tr[1 + 1*ldtr];
        }

        if (ltranr) {
            T16(0, 2) = tl[0] * tr[0 + 1*ldtr];
            T16(1, 3) = tl[1 + 1*ldtl] * tr[0 + 1*ldtr];
            T16(2, 0) = tl[0] * tr[1];
            T16(3, 1) = tl[1 + 1*ldtl] * tr[1];
        } else {
            T16(0, 2) = tl[0] * tr[1];
            T16(1, 3) = tl[1 + 1*ldtl] * tr[1];
            T16(2, 0) = tl[0] * tr[0 + 1*ldtr];
            T16(3, 1) = tl[1 + 1*ldtl] * tr[0 + 1*ldtr];
        }

        if (ltranl && ltranr) {
            T16(0, 3) = tl[1] * tr[0 + 1*ldtr];
            T16(1, 2) = tl[0 + 1*ldtl] * tr[0 + 1*ldtr];
            T16(2, 1) = tl[1] * tr[1];
            T16(3, 0) = tl[0 + 1*ldtl] * tr[1];
        } else if (ltranl && !ltranr) {
            T16(0, 3) = tl[1] * tr[1];
            T16(1, 2) = tl[0 + 1*ldtl] * tr[1];
            T16(2, 1) = tl[1] * tr[0 + 1*ldtr];
            T16(3, 0) = tl[0 + 1*ldtl] * tr[0 + 1*ldtr];
        } else if (!ltranl && ltranr) {
            T16(0, 3) = tl[0 + 1*ldtl] * tr[0 + 1*ldtr];
            T16(1, 2) = tl[1] * tr[0 + 1*ldtr];
            T16(2, 1) = tl[0 + 1*ldtl] * tr[1];
            T16(3, 0) = tl[1] * tr[1];
        } else {
            T16(0, 3) = tl[0 + 1*ldtl] * tr[1];
            T16(1, 2) = tl[1] * tr[1];
            T16(2, 1) = tl[0 + 1*ldtl] * tr[0 + 1*ldtr];
            T16(3, 0) = tl[1] * tr[0 + 1*ldtr];
        }

        btmp[0] = b[0];
        btmp[1] = b[1];
        btmp[2] = b[0 + 1*ldb];
        btmp[3] = b[1 + 1*ldb];

        /* Gaussian elimination with complete pivoting */
        for (i = 0; i < 3; i++) {
            xmax = zero;

            for (ip = i; ip < 4; ip++) {
                for (jp = i; jp < 4; jp++) {
                    if (fabs(T16(ip, jp)) >= xmax) {
                        xmax = fabs(T16(ip, jp));
                        ipsv = ip;
                        jpsv = jp;
                    }
                }
            }

            /* Row swap */
            if (ipsv != i) {
                i32 incx = 4;
                i32 n4 = 4;
                SLC_DSWAP(&n4, &t16[ipsv], &incx, &t16[i], &incx);
                temp = btmp[i];
                btmp[i] = btmp[ipsv];
                btmp[ipsv] = temp;
            }

            /* Column swap */
            if (jpsv != i) {
                i32 incy = 1;
                i32 n4 = 4;
                SLC_DSWAP(&n4, &t16[jpsv*4], &incy, &t16[i*4], &incy);
            }
            jpiv[i] = jpsv;

            if (fabs(T16(i, i)) < smin) {
                *info = 1;
                T16(i, i) = smin;
            }

            /* Eliminate below diagonal */
            for (j = i + 1; j < 4; j++) {
                T16(j, i) = T16(j, i) / T16(i, i);
                btmp[j] = btmp[j] - T16(j, i) * btmp[i];

                for (k = i + 1; k < 4; k++) {
                    T16(j, k) = T16(j, k) - T16(j, i) * T16(i, k);
                }
            }
        }

        if (fabs(T16(3, 3)) < smin) {
            T16(3, 3) = smin;
        }

        /* Check for overflow */
        if ((eight * smlnum) * fabs(btmp[0]) > fabs(T16(0, 0)) ||
            (eight * smlnum) * fabs(btmp[1]) > fabs(T16(1, 1)) ||
            (eight * smlnum) * fabs(btmp[2]) > fabs(T16(2, 2)) ||
            (eight * smlnum) * fabs(btmp[3]) > fabs(T16(3, 3))) {
            f64 maxb = fabs(btmp[0]);
            if (fabs(btmp[1]) > maxb) maxb = fabs(btmp[1]);
            if (fabs(btmp[2]) > maxb) maxb = fabs(btmp[2]);
            if (fabs(btmp[3]) > maxb) maxb = fabs(btmp[3]);
            *scale = (one / eight) / maxb;
            btmp[0] = btmp[0] * (*scale);
            btmp[1] = btmp[1] * (*scale);
            btmp[2] = btmp[2] * (*scale);
            btmp[3] = btmp[3] * (*scale);
        }

        /* Back substitution */
        for (i = 0; i < 4; i++) {
            k = 3 - i;
            temp = one / T16(k, k);
            tmp[k] = btmp[k] * temp;

            for (j = k + 1; j < 4; j++) {
                tmp[k] = tmp[k] - (temp * T16(k, j)) * tmp[j];
            }
        }

        /* Undo column pivoting */
        for (i = 0; i < 3; i++) {
            if (jpiv[2 - i] != 2 - i) {
                temp = tmp[2 - i];
                tmp[2 - i] = tmp[jpiv[2 - i]];
                tmp[jpiv[2 - i]] = temp;
            }
        }

        x[0] = tmp[0];
        x[1] = tmp[1];
        x[0 + 1*ldx] = tmp[2];
        x[1 + 1*ldx] = tmp[3];

        f64 col0_sum = fabs(tmp[0]) + fabs(tmp[2]);
        f64 col1_sum = fabs(tmp[1]) + fabs(tmp[3]);
        *xnorm = (col0_sum > col1_sum) ? col0_sum : col1_sum;

        #undef T16
    }
    return;
}
