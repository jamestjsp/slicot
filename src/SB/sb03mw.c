/**
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * SB03MW - Solve 2x2 continuous-time Lyapunov equation
 *
 * Solves for the 2-by-2 symmetric matrix X in:
 *     op(T)'*X + X*op(T) = SCALE*B
 *
 * where T is 2-by-2, B is symmetric 2-by-2, and op(T) = T or T'.
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>

void sb03mw(bool ltran, bool lupper, const f64* t, i32 ldt,
            const f64* b, i32 ldb, f64* scale, f64* x, i32 ldx,
            f64* xnorm, i32* info)
{
    const f64 zero = 0.0;
    const f64 one = 1.0;
    const f64 two = 2.0;
    const f64 four = 4.0;

    i32 i, ip, ipsv, j, jp, jpsv, k;
    f64 eps, smin, smlnum, temp, xmax;

    i32 jpiv[3];
    f64 btmp[3], t9[9], tmp[3];

    *info = 0;

    eps = SLC_DLAMCH("P");
    smlnum = SLC_DLAMCH("S") / eps;

    f64 max_abs = fmax(fabs(t[0]), fabs(t[0 + 1*ldt]));
    max_abs = fmax(max_abs, fabs(t[1]));
    max_abs = fmax(max_abs, fabs(t[1 + 1*ldt]));
    smin = fmax(max_abs * eps, smlnum);

    t9[0 + 2*3] = zero;
    t9[2 + 0*3] = zero;
    t9[0 + 0*3] = t[0];
    t9[1 + 1*3] = t[0] + t[1 + 1*ldt];
    t9[2 + 2*3] = t[1 + 1*ldt];

    if (ltran) {
        t9[0 + 1*3] = t[0 + 1*ldt];
        t9[1 + 0*3] = t[1];
        t9[1 + 2*3] = t[0 + 1*ldt];
        t9[2 + 1*3] = t[1];
    } else {
        t9[0 + 1*3] = t[1];
        t9[1 + 0*3] = t[0 + 1*ldt];
        t9[1 + 2*3] = t[1];
        t9[2 + 1*3] = t[0 + 1*ldt];
    }

    btmp[0] = b[0] / two;
    if (lupper) {
        btmp[1] = b[0 + 1*ldb];
    } else {
        btmp[1] = b[1];
    }
    btmp[2] = b[1 + 1*ldb] / two;

    for (i = 0; i < 2; i++) {
        xmax = zero;

        for (ip = i; ip < 3; ip++) {
            for (jp = i; jp < 3; jp++) {
                if (fabs(t9[ip + jp*3]) >= xmax) {
                    xmax = fabs(t9[ip + jp*3]);
                    ipsv = ip;
                    jpsv = jp;
                }
            }
        }

        if (ipsv != i) {
            i32 three = 3;
            i32 inc3 = 3;
            SLC_DSWAP(&three, &t9[ipsv], &inc3, &t9[i], &inc3);
            temp = btmp[i];
            btmp[i] = btmp[ipsv];
            btmp[ipsv] = temp;
        }

        if (jpsv != i) {
            i32 three = 3;
            i32 inc1 = 1;
            SLC_DSWAP(&three, &t9[jpsv*3], &inc1, &t9[i*3], &inc1);
        }
        jpiv[i] = jpsv;

        if (fabs(t9[i + i*3]) < smin) {
            *info = 1;
            t9[i + i*3] = smin;
        }

        for (j = i + 1; j < 3; j++) {
            t9[j + i*3] = t9[j + i*3] / t9[i + i*3];
            btmp[j] = btmp[j] - t9[j + i*3] * btmp[i];

            for (k = i + 1; k < 3; k++) {
                t9[j + k*3] = t9[j + k*3] - t9[j + i*3] * t9[i + k*3];
            }
        }
    }

    if (fabs(t9[2 + 2*3]) < smin) {
        *info = 1;
        t9[2 + 2*3] = smin;
    }

    *scale = one;
    if ((four * smlnum) * fabs(btmp[0]) > fabs(t9[0 + 0*3]) ||
        (four * smlnum) * fabs(btmp[1]) > fabs(t9[1 + 1*3]) ||
        (four * smlnum) * fabs(btmp[2]) > fabs(t9[2 + 2*3])) {
        f64 maxbtmp = fmax(fabs(btmp[0]), fabs(btmp[1]));
        maxbtmp = fmax(maxbtmp, fabs(btmp[2]));
        *scale = (one / four) / maxbtmp;
        btmp[0] = btmp[0] * (*scale);
        btmp[1] = btmp[1] * (*scale);
        btmp[2] = btmp[2] * (*scale);
    }

    for (i = 0; i < 3; i++) {
        k = 2 - i;
        temp = one / t9[k + k*3];
        tmp[k] = btmp[k] * temp;

        for (j = k + 1; j < 3; j++) {
            tmp[k] = tmp[k] - (temp * t9[k + j*3]) * tmp[j];
        }
    }

    for (i = 0; i < 2; i++) {
        if (jpiv[1 - i] != 1 - i) {
            temp = tmp[1 - i];
            tmp[1 - i] = tmp[jpiv[1 - i]];
            tmp[jpiv[1 - i]] = temp;
        }
    }

    x[0] = tmp[0];
    if (lupper) {
        x[0 + 1*ldx] = tmp[1];
    } else {
        x[1] = tmp[1];
    }
    x[1 + 1*ldx] = tmp[2];

    *xnorm = fmax(fabs(tmp[0]) + fabs(tmp[1]), fabs(tmp[1]) + fabs(tmp[2]));
}
