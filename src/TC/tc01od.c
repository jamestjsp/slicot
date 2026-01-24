/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <ctype.h>

void tc01od(
    const char leri,
    const i32 m,
    const i32 p,
    const i32 indlim,
    f64* pcoeff,
    const i32 ldpco1,
    const i32 ldpco2,
    f64* qcoeff,
    const i32 ldqco1,
    const i32 ldqco2,
    i32* info
)
{
    i32 j, k, minmp, mplim, porm;
    bool lleri;
    i32 int1 = 1;

    *info = 0;
    lleri = (toupper((unsigned char)leri) == 'L');
    mplim = (m > p) ? m : p;
    minmp = (m < p) ? m : p;

    if (!lleri && toupper((unsigned char)leri) != 'R') {
        *info = -1;
    } else if (m < 0) {
        *info = -2;
    } else if (p < 0) {
        *info = -3;
    } else if (indlim < 1) {
        *info = -4;
    } else if ((lleri && ldpco1 < (1 > p ? 1 : p)) ||
               (!lleri && ldpco1 < (1 > m ? 1 : m))) {
        *info = -6;
    } else if ((lleri && ldpco2 < (1 > p ? 1 : p)) ||
               (!lleri && ldpco2 < (1 > m ? 1 : m))) {
        *info = -7;
    } else if (ldqco1 < (1 > mplim ? 1 : mplim)) {
        *info = -9;
    } else if (ldqco2 < (1 > mplim ? 1 : mplim)) {
        *info = -10;
    }

    if (*info != 0) {
        return;
    }

    if (m == 0 || p == 0) {
        return;
    }

    if (mplim != 1) {
        for (k = 0; k < indlim; k++) {
            i32 slice_offset = k * ldqco1 * ldqco2;

            for (j = 0; j < mplim; j++) {
                if (j < minmp - 1) {
                    i32 n_swap = minmp - j - 1;
                    f64* col_ptr = &qcoeff[slice_offset + (j + 1) + j * ldqco1];
                    f64* row_ptr = &qcoeff[slice_offset + j + (j + 1) * ldqco1];
                    SLC_DSWAP(&n_swap, col_ptr, &int1, row_ptr, &ldqco1);
                } else if (j >= p) {
                    f64* src = &qcoeff[slice_offset + j * ldqco1];
                    f64* dst = &qcoeff[slice_offset + j];
                    SLC_DCOPY(&p, src, &int1, dst, &ldqco1);
                } else if (j >= m) {
                    f64* src = &qcoeff[slice_offset + j];
                    f64* dst = &qcoeff[slice_offset + j * ldqco1];
                    SLC_DCOPY(&m, src, &ldqco1, dst, &int1);
                }
            }
        }

        porm = m;
        if (lleri) porm = p;

        if (porm != 1) {
            for (k = 0; k < indlim; k++) {
                i32 slice_offset = k * ldpco1 * ldpco2;

                for (j = 0; j < porm - 1; j++) {
                    i32 n_swap = porm - j - 1;
                    f64* col_ptr = &pcoeff[slice_offset + (j + 1) + j * ldpco1];
                    f64* row_ptr = &pcoeff[slice_offset + j + (j + 1) * ldpco1];
                    SLC_DSWAP(&n_swap, col_ptr, &int1, row_ptr, &ldpco1);
                }
            }
        }
    }
}
