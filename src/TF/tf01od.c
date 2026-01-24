/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"

void tf01od(const i32 nh1, const i32 nh2, const i32 nr, const i32 nc,
            const f64* h, const i32 ldh, f64* t, const i32 ldt, i32* info)
{
    *info = 0;

    i32 max_nh1 = (nh1 > 1) ? nh1 : 1;
    i32 nh1_nr = nh1 * nr;
    i32 max_nh1_nr = (nh1_nr > 1) ? nh1_nr : 1;

    if (nh1 < 0) {
        *info = -1;
    } else if (nh2 < 0) {
        *info = -2;
    } else if (nr < 0) {
        *info = -3;
    } else if (nc < 0) {
        *info = -4;
    } else if (ldh < max_nh1) {
        *info = -6;
    } else if (ldt < max_nh1_nr) {
        *info = -8;
    }

    if (*info != 0) {
        return;
    }

    i32 max4 = nh1;
    if (nh2 > max4) max4 = nh2;
    if (nr > max4) max4 = nr;
    if (nc > max4) max4 = nc;
    if (max4 == 0 || nr == 0 || nc == 0) {
        return;
    }

    i32 ih = 0;
    i32 nrow = (nr - 1) * nh1;
    if (nrow < 0) nrow = 0;

    for (i32 it = 0; it < nr * nh1; it += nh1) {
        SLC_DLACPY("Full", &nh1, &nh2, &h[ih * ldh], &ldh, &t[it], &ldt);
        ih += nh2;
    }

    for (i32 jt = nh2; jt < nc * nh2; jt += nh2) {
        if (nrow > 0) {
            SLC_DLACPY("Full", &nrow, &nh2, &t[nh1 + (jt - nh2) * ldt], &ldt, &t[jt * ldt], &ldt);
        }
        SLC_DLACPY("Full", &nh1, &nh2, &h[ih * ldh], &ldh, &t[nrow + jt * ldt], &ldt);
        ih += nh2;
    }
}
