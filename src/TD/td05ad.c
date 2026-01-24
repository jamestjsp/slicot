/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <ctype.h>
#include <math.h>

void td05ad(
    const char* unitf,
    const char* output,
    const i32 np1,
    const i32 mp1,
    const f64 w,
    const f64* a,
    const f64* b,
    f64* valr,
    f64* vali,
    i32* info
)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const f64 EIGHT = 8.0;
    const f64 TWENTY = 20.0;
    const f64 NINETY = 90.0;
    const f64 ONE80 = 180.0;
    const f64 THRE60 = 360.0;

    *info = 0;

    char uf = (char)toupper((unsigned char)unitf[0]);
    char op = (char)toupper((unsigned char)output[0]);
    bool lunitf = (uf == 'H');
    bool loutpu = (op == 'P');

    if (!lunitf && uf != 'R') {
        *info = -1;
    } else if (!loutpu && op != 'C') {
        *info = -2;
    } else if (np1 < 1) {
        *info = -3;
    } else if (mp1 < 1) {
        *info = -4;
    }

    if (*info != 0) {
        return;
    }

    i32 m = mp1 - 1;
    i32 n = np1 - 1;
    f64 wc = w;
    f64 twopi = EIGHT * atan(ONE);
    if (lunitf) wc = wc * twopi;
    f64 w2 = wc * wc;

    i32 nzzero = 0;
    for (i32 i = 0; i < m; i++) {
        if (b[i] != ZERO) break;
        nzzero++;
    }

    i32 npzero = 0;
    for (i32 i = 0; i < n; i++) {
        if (a[i] != ZERO) break;
        npzero++;
    }

    i32 iphase = nzzero - npzero;

    i32 m2 = (m - nzzero) % 2;

    f64 treal = b[mp1 - m2 - 1];

    for (i32 fi = m - 1 - m2; fi >= nzzero + 1; fi -= 2) {
        treal = b[fi - 1] - w2 * treal;
    }

    f64 timag;
    if (m == 0) {
        timag = ZERO;
    } else {
        timag = b[m + m2 - 1];

        for (i32 fi = m + m2 - 2; fi >= nzzero + 2; fi -= 2) {
            timag = b[fi - 1] - w2 * timag;
        }

        timag = timag * wc;
    }

    i32 n2 = (n - npzero) % 2;

    f64 breal = a[np1 - n2 - 1];

    for (i32 fi = n - 1 - n2; fi >= npzero + 1; fi -= 2) {
        breal = a[fi - 1] - w2 * breal;
    }

    f64 bimag;
    if (n == 0) {
        bimag = ZERO;
    } else {
        bimag = a[n + n2 - 1];

        for (i32 fi = n + n2 - 2; fi >= npzero + 2; fi -= 2) {
            bimag = a[fi - 1] - w2 * bimag;
        }

        bimag = bimag * wc;
    }

    f64 max_b = fabs(breal);
    if (fabs(bimag) > max_b) max_b = fabs(bimag);

    if (max_b == ZERO || (w == ZERO && iphase < 0)) {
        *info = 1;
        return;
    }

    f64 denom = breal * breal + bimag * bimag;
    f64 zreal = (treal * breal + timag * bimag) / denom;
    f64 zimag = (timag * breal - treal * bimag) / denom;

    f64 wc_power = pow(fabs(wc), (f64)iphase);
    *valr = zreal * wc_power;
    *vali = zimag * wc_power;

    if (!loutpu) {
        i32 abs_iphase = iphase >= 0 ? iphase : -iphase;
        i32 imod = abs_iphase % 4;

        if ((iphase > 0 && imod > 1) ||
            (iphase < 0 && (imod == 1 || imod == 2))) {
            *valr = -(*valr);
            *vali = -(*vali);
        }

        if (imod % 2 != 0) {
            f64 g = *valr;
            *valr = -(*vali);
            *vali = g;
        }

    } else {
        f64 g = SLC_DLAPY2(valr, vali);

        if (*valr == ZERO) {
            *vali = (*vali >= 0) ? NINETY : -NINETY;
        } else {
            *vali = (atan(*vali / *valr) / twopi) * THRE60;
            if (*vali == ZERO && nzzero == m && npzero == n &&
                b[nzzero] * a[npzero] < ZERO) {
                *vali = ONE80;
            }
        }

        *valr = TWENTY * log10(g);

        if (iphase != 0) {
            *vali = *vali + (f64)(nzzero - npzero) * NINETY;
        }
    }
}
