/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <ctype.h>

void tc04ad(
    const char leri,
    const i32 m,
    const i32 p,
    const i32* index,
    f64* pcoeff,
    const i32 ldpco1,
    const i32 ldpco2,
    f64* qcoeff,
    const i32 ldqco1,
    const i32 ldqco2,
    i32* n,
    f64* rcond,
    f64* a,
    const i32 lda,
    f64* b,
    const i32 ldb,
    f64* c,
    const i32 ldc,
    f64* d,
    const i32 ldd,
    i32* iwork,
    f64* dwork,
    const i32 ldwork,
    i32* info
)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    bool lleri;
    i32 i, ia, ibias, j, ja, jc, jw, jwork, ldw, k;
    i32 kpcoef, kstop, maxind, mindex, mwork, pwork;
    i32 wrkopt;
    f64 dwnorm;

    i32 int1 = 1;
    char leri_upper = (char)toupper((unsigned char)leri);

    *info = 0;
    lleri = (leri_upper == 'L');
    mindex = (m > p) ? m : p;

    if (!lleri && leri_upper != 'R') {
        *info = -1;
    } else if (m < 0) {
        *info = -2;
    } else if (p < 0) {
        *info = -3;
    } else if ((lleri && ldpco1 < (1 > p ? 1 : p)) ||
               (!lleri && ldpco1 < (1 > m ? 1 : m))) {
        *info = -6;
    } else if ((lleri && ldpco2 < (1 > p ? 1 : p)) ||
               (!lleri && ldpco2 < (1 > m ? 1 : m))) {
        *info = -7;
    } else if ((lleri && ldqco1 < (1 > p ? 1 : p)) ||
               (!lleri && ldqco1 < (1 > mindex ? 1 : mindex))) {
        *info = -9;
    } else if ((lleri && ldqco2 < (1 > m ? 1 : m)) ||
               (!lleri && ldqco2 < (1 > mindex ? 1 : mindex))) {
        *info = -10;
    }

    *n = 0;
    if (*info == 0) {
        if (lleri) {
            pwork = p;
            mwork = m;
        } else {
            pwork = m;
            mwork = p;
        }

        maxind = 0;
        for (i = 0; i < pwork; i++) {
            *n += index[i];
            if (index[i] > maxind) maxind = index[i];
        }
        kpcoef = maxind + 1;
    }

    i32 max1n = (1 > *n) ? 1 : *n;
    if (lda < max1n) {
        *info = -14;
    } else if (ldb < max1n) {
        *info = -16;
    } else if (ldc < (1 > mindex ? 1 : mindex)) {
        *info = -18;
    } else if (ldd < (1 > mindex ? 1 : mindex)) {
        *info = -20;
    } else if (ldwork < (1 > mindex * (mindex + 4) ? 1 : mindex * (mindex + 4))) {
        *info = -23;
    }

    if (*info != 0) {
        return;
    }

    if (m == 0 || p == 0) {
        *n = 0;
        *rcond = ONE;
        dwork[0] = ONE;
        return;
    }

    if (!lleri) {
        i32 tc01od_info;
        tc01od('R', m, p, kpcoef, pcoeff, ldpco1, ldpco2,
               qcoeff, ldqco1, ldqco2, &tc01od_info);
    }

    ldw = (1 > pwork) ? 1 : pwork;

    SLC_DLACPY("F", &pwork, &pwork, pcoeff, &ldpco1, dwork, &ldw);

    dwnorm = SLC_DLANGE("1", &pwork, &pwork, dwork, &ldw, dwork);

    SLC_DGETRF(&pwork, &pwork, dwork, &ldw, iwork, info);

    jwork = ldw * pwork;

    SLC_DGECON("1", &pwork, dwork, &ldw, &dwnorm, rcond,
               &dwork[jwork], &iwork[pwork], info);

    wrkopt = (1 > pwork * (pwork + 4)) ? 1 : pwork * (pwork + 4);

    f64 eps = SLC_DLAMCH("E");
    if (*rcond <= eps) {
        *info = 1;
        return;
    }

    SLC_DLASET("F", n, n, &ZERO, &ZERO, a, &lda);

    dwork[jwork] = ONE;
    if (*n > 1) {
        i32 nm1 = *n - 1;
        i32 inc0 = 0;
        i32 lda_p1 = lda + 1;
        SLC_DCOPY(&nm1, &dwork[jwork], &inc0, &a[1], &lda_p1);
    }

    ibias = 2;

    for (i = 0; i < pwork; i++) {
        kstop = index[i] + 1;
        if (kstop != 1) {
            ibias = ibias + index[i];

            for (k = 1; k < kstop; k++) {
                ia = ibias - k - 2;

                for (j = 0; j < pwork; j++) {
                    i32 slice = k * ldpco1 * ldpco2;
                    dwork[jwork + j] = -pcoeff[slice + i + j * ldpco1];
                }

                SLC_DGETRS("T", &pwork, &int1, dwork, &ldw,
                           iwork, &dwork[jwork], &ldw, info);

                ja = 0;
                for (j = 0; j < pwork; j++) {
                    if (index[j] != 0) {
                        ja = ja + index[j];
                        a[ia + (ja - 1) * lda] = dwork[jwork + j];
                    }
                }

                i32 slice_q = k * ldqco1 * ldqco2;
                SLC_DCOPY(&mwork, &qcoeff[slice_q + i], &ldqco1, &b[ia], &ldb);

                i32 slice_p = k * ldpco1 * ldpco2;
                SLC_DCOPY(&pwork, &pcoeff[slice_p + i], &ldpco1, &c[ia * ldc], &int1);
            }
        }
    }

    SLC_DLACPY("F", &pwork, &mwork, qcoeff, &ldqco1, d, &ldd);

    SLC_DGETRS("N", &pwork, &mwork, dwork, &ldw, iwork, d, &ldd, info);

    f64 neg_one = -ONE;
    SLC_DGEMM("T", "N", n, &mwork, &pwork, &neg_one,
              c, &ldc, d, &ldd, &ONE, b, &ldb);

    SLC_DLASET("F", &pwork, n, &ZERO, &ZERO, c, &ldc);

    i32 dwork_size = ldwork - jwork;
    SLC_DGETRI(&pwork, dwork, &ldw, iwork, &dwork[jwork], &dwork_size, info);

    i32 opt = (i32)dwork[jwork] + jwork;
    wrkopt = (wrkopt > opt) ? wrkopt : opt;

    jc = 0;
    jw = 0;

    for (j = 0; j < pwork; j++) {
        if (index[j] != 0) {
            jc = jc + index[j];
            SLC_DCOPY(&pwork, &dwork[jw], &int1, &c[(jc - 1) * ldc], &int1);
        }
        jw = jw + ldw;
    }

    if (!lleri) {
        i32 tc01od_info;
        tc01od('L', mwork, pwork, kpcoef, pcoeff, ldpco1, ldpco2,
               qcoeff, ldqco1, ldqco2, &tc01od_info);

        i32 ab07md_info = ab07md('D', *n, mwork, pwork, a, lda, b, ldb, c, ldc, d, ldd);
        (void)ab07md_info;
    }

    dwork[0] = (f64)wrkopt;
}
