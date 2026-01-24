/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"

void mb04xy(const char *jobu, const char *jobv, const i32 m, const i32 n,
            f64 *x, const i32 ldx, const f64 *taup, const f64 *tauq,
            f64 *u, const i32 ldu, f64 *v, const i32 ldv,
            const bool *inul, i32 *info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    bool ljobua, ljobus, ljobva, ljobvs, wantu, wantv;
    i32 i, im, ioff, l, ncol, p;
    f64 first;
    f64 dwork[1];
    i32 int1 = 1;

    *info = 0;

    ljobua = (jobu[0] == 'A' || jobu[0] == 'a');
    ljobus = (jobu[0] == 'S' || jobu[0] == 's');
    ljobva = (jobv[0] == 'A' || jobv[0] == 'a');
    ljobvs = (jobv[0] == 'S' || jobv[0] == 's');
    wantu = ljobua || ljobus;
    wantv = ljobva || ljobvs;

    if (!wantu && !(jobu[0] == 'N' || jobu[0] == 'n')) {
        *info = -1;
    } else if (!wantv && !(jobv[0] == 'N' || jobv[0] == 'n')) {
        *info = -2;
    } else if (m < 0) {
        *info = -3;
    } else if (n < 0) {
        *info = -4;
    } else if (ldx < (1 > m ? 1 : m)) {
        *info = -6;
    } else if ((wantu && ldu < (1 > m ? 1 : m)) ||
               (!wantu && ldu < 1)) {
        *info = -10;
    } else if ((wantv && ldv < (1 > n ? 1 : n)) ||
               (!wantv && ldv < 1)) {
        *info = -12;
    }

    if (*info != 0) {
        return;
    }

    p = (m < n) ? m : n;
    if (p == 0) {
        return;
    }

    if (m < n) {
        ioff = 1;
    } else {
        ioff = 0;
    }

    im = ((m - 1) < n) ? (m - 1) : n;
    if (wantu && im > 0) {
        if (ljobua) {
            ncol = m;
        } else {
            ncol = p;
        }

        for (i = 0; i < ncol; i++) {
            if (inul[i]) {
                for (l = im - 1; l >= 0; l--) {
                    if (taup[l] != ZERO) {
                        first = x[(l + ioff) + l * ldx];
                        x[(l + ioff) + l * ldx] = ONE;
                        i32 len = m - l - ioff;
                        SLC_DLARF("Left", &len, &int1, &x[(l + ioff) + l * ldx], &int1,
                                  &taup[l], &u[(l + ioff) + i * ldu], &ldu, dwork);
                        x[(l + ioff) + l * ldx] = first;
                    }
                }
            }
        }
    }

    im = ((n - 1) < m) ? (n - 1) : m;
    if (wantv && im > 0) {
        if (ljobva) {
            ncol = n;
        } else {
            ncol = p;
        }

        for (i = 0; i < ncol; i++) {
            if (inul[i]) {
                for (l = im - 1; l >= 0; l--) {
                    if (tauq[l] != ZERO) {
                        first = x[l + (l + 1 - ioff) * ldx];
                        x[l + (l + 1 - ioff) * ldx] = ONE;
                        i32 len = n - l - 1 + ioff;
                        SLC_DLARF("Left", &len, &int1, &x[l + (l + 1 - ioff) * ldx], &ldx,
                                  &tauq[l], &v[(l + 1 - ioff) + i * ldv], &ldv, dwork);
                        x[l + (l + 1 - ioff) * ldx] = first;
                    }
                }
            }
        }
    }
}
