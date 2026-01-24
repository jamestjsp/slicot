/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <ctype.h>

void tc05ad(
    const char leri,
    const i32 m,
    const i32 p,
    const c128 sval,
    const i32* index,
    f64* pcoeff,
    const i32 ldpco1,
    const i32 ldpco2,
    f64* qcoeff,
    const i32 ldqco1,
    const i32 ldqco2,
    f64* rcond,
    c128* cfreqr,
    const i32 ldcfre,
    i32* iwork,
    f64* dwork,
    c128* zwork,
    i32* info
)
{
    bool lleri;
    i32 i, izwork, ij, info1, j, k, kpcoef, ldzwor;
    i32 maxind, minmp, mplim, mwork, pwork;
    f64 cnorm;
    f64 zero = 0.0, one = 1.0;
    i32 int1 = 1;

    *info = 0;
    lleri = (toupper((unsigned char)leri) == 'L');
    mplim = (m > p) ? m : p;

    if (!lleri && toupper((unsigned char)leri) != 'R') {
        *info = -1;
    } else if (m < 0) {
        *info = -2;
    } else if (p < 0) {
        *info = -3;
    } else if ((lleri && ldpco1 < (1 > p ? 1 : p)) ||
               (!lleri && ldpco1 < (1 > m ? 1 : m))) {
        *info = -7;
    } else if ((lleri && ldpco2 < (1 > p ? 1 : p)) ||
               (!lleri && ldpco2 < (1 > m ? 1 : m))) {
        *info = -8;
    } else if ((lleri && ldqco1 < (1 > p ? 1 : p)) ||
               (!lleri && ldqco1 < (1 > mplim ? 1 : mplim))) {
        *info = -10;
    } else if ((lleri && ldqco2 < (1 > m ? 1 : m)) ||
               (!lleri && ldqco2 < (1 > mplim ? 1 : mplim))) {
        *info = -11;
    } else if ((lleri && ldcfre < (1 > p ? 1 : p)) ||
               (!lleri && ldcfre < (1 > mplim ? 1 : mplim))) {
        *info = -14;
    }

    if (*info != 0) {
        return;
    }

    if (m == 0 || p == 0) {
        *rcond = one;
        return;
    }

    if (lleri) {
        pwork = p;
        mwork = m;
    } else {
        pwork = m;
        mwork = p;
    }

    ldzwor = pwork;
    izwork = ldzwor * ldzwor;
    maxind = 0;

    for (i = 0; i < pwork; i++) {
        if (index[i] > maxind) maxind = index[i];
    }

    kpcoef = maxind + 1;

    if (!lleri && mplim > 1) {
        tc01od(leri, m, p, kpcoef, pcoeff, ldpco1, ldpco2,
               qcoeff, ldqco1, ldqco2, info);
    }

    for (i = 0; i < pwork; i++) {
        ij = i;
        for (j = 0; j < pwork; j++) {
            zwork[ij] = pcoeff[i + j * ldpco1];
            ij += pwork;
        }

        for (k = 1; k < index[i] + 1; k++) {
            ij = i;
            for (j = 0; j < pwork; j++) {
                zwork[ij] = (sval * zwork[ij]) +
                            pcoeff[i + j * ldpco1 + k * ldpco1 * ldpco2];
                ij += pwork;
            }
        }
    }

    cnorm = SLC_ZLANGE("1", &pwork, &pwork, zwork, &ldzwor, dwork);

    SLC_ZGETRF(&pwork, &pwork, zwork, &ldzwor, iwork, info);

    if (*info > 0) {
        *info = 1;
        *rcond = zero;
    } else {
        SLC_ZGECON("1", &pwork, zwork, &ldzwor, &cnorm, rcond,
                   &zwork[izwork], dwork, info);

        if (*rcond <= SLC_DLAMCH("Epsilon")) {
            *info = 1;
        } else {
            for (i = 0; i < pwork; i++) {
                for (j = 0; j < mwork; j++) {
                    cfreqr[i + j * ldcfre] = qcoeff[i + j * ldqco1];
                }

                for (k = 1; k < index[i] + 1; k++) {
                    for (j = 0; j < mwork; j++) {
                        cfreqr[i + j * ldcfre] = (sval * cfreqr[i + j * ldcfre]) +
                            qcoeff[i + j * ldqco1 + k * ldqco1 * ldqco2];
                    }
                }
            }

            SLC_ZGETRS("N", &pwork, &mwork, zwork, &ldzwor,
                       iwork, cfreqr, &ldcfre, info);
        }
    }

    if (!lleri && mplim != 1) {
        tc01od('L', mwork, pwork, kpcoef, pcoeff, ldpco1, ldpco2,
               qcoeff, ldqco1, ldqco2, &info1);

        if (*info == 0) {
            minmp = (m < p) ? m : p;

            for (j = 0; j < mplim; j++) {
                if (j < minmp - 1) {
                    i32 n_swap = minmp - j - 1;
                    SLC_ZSWAP(&n_swap, &cfreqr[(j + 1) + j * ldcfre], &int1,
                              &cfreqr[j + (j + 1) * ldcfre], &ldcfre);
                } else if (j >= p) {
                    SLC_ZCOPY(&p, &cfreqr[j * ldcfre], &int1,
                              &cfreqr[j], &ldcfre);
                } else if (j >= m) {
                    SLC_ZCOPY(&m, &cfreqr[j], &ldcfre,
                              &cfreqr[j * ldcfre], &int1);
                }
            }
        }
    }
}
