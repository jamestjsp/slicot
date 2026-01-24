/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <ctype.h>

void fd01ad(const char* jp, i32 l, f64 lambda, f64 xin, f64 yin,
            f64* efor, f64* xf, f64* epsbck, f64* cteta, f64* steta,
            f64* yq, f64* epos, f64* eout, f64* salph,
            i32* iwarn, i32* info)
{
    const f64 zero = 0.0;
    const f64 one = 1.0;

    bool both = (toupper((unsigned char)jp[0]) == 'B');

    *iwarn = 0;
    *info = 0;

    if (!both && toupper((unsigned char)jp[0]) != 'P') {
        *info = -1;
    } else if (l < 1) {
        *info = -2;
    } else if (lambda <= zero || lambda > one) {
        *info = -3;
    }

    if (*info != 0) {
        return;
    }

    f64 eps = SLC_DLAMCH("Epsilon");

    f64 fnode = xin;
    for (i32 i = 0; i < l; i++) {
        f64 xfi = xf[i] * lambda;
        xf[i] = steta[i] * fnode + cteta[i] * xfi;
        fnode = cteta[i] * fnode - steta[i] * xfi;
    }

    *epos = fnode * epsbck[l];

    *efor = (*efor) * lambda;
    f64 temp = SLC_DLAPY2(&fnode, efor);
    if (temp < eps) {
        fnode = zero;
        *iwarn = 1;
    } else {
        fnode = fnode * epsbck[l] / temp;
    }
    *efor = temp;

    for (i32 i = l - 1; i >= 0; i--) {
        if (fabs(xf[i]) < eps) {
            *iwarn = 1;
        }
        f64 ctemp, norm;
        SLC_DLARTG(&temp, &xf[i], &ctemp, &salph[i], &norm);
        epsbck[i + 1] = ctemp * epsbck[i] - salph[i] * fnode;
        fnode = ctemp * fnode + salph[i] * epsbck[i];
        temp = norm;
    }

    epsbck[0] = fnode;

    f64 norm_val = SLC_DNRM2(&l, epsbck, &(i32){1});
    temp = sqrt((one + norm_val) * (one - norm_val));
    epsbck[l] = temp;

    for (i32 i = l - 1; i >= 0; i--) {
        if (fabs(epsbck[i]) < eps) {
            *iwarn = 1;
        }
        f64 norm;
        SLC_DLARTG(&temp, &epsbck[i], &cteta[i], &steta[i], &norm);
        temp = norm;
    }

    if (both) {
        fnode = yin;

        for (i32 i = 0; i < l; i++) {
            f64 yqi = yq[i] * lambda;
            yq[i] = steta[i] * fnode + cteta[i] * yqi;
            fnode = cteta[i] * fnode - steta[i] * yqi;
        }

        *eout = fnode * epsbck[l];
    }
}
