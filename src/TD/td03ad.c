/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <stdbool.h>

void td03ad(
    const char* rowcol,
    const char* leri,
    const char* equil,
    const i32 m,
    const i32 p,
    const i32* indexd,
    const f64* dcoeff,
    const i32 lddcoe,
    f64* ucoeff,
    const i32 lduco1,
    const i32 lduco2,
    i32* nr,
    f64* a,
    const i32 lda,
    f64* b,
    const i32 ldb,
    f64* c,
    const i32 ldc,
    f64* d,
    const i32 ldd,
    i32* indexp,
    f64* pcoeff,
    const i32 ldpco1,
    const i32 ldpco2,
    f64* qcoeff,
    const i32 ldqco1,
    const i32 ldqco2,
    f64* vcoeff,
    const i32 ldvco1,
    const i32 ldvco2,
    const f64 tol,
    i32* iwork,
    f64* dwork,
    const i32 ldwork,
    i32* info
)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    bool lrowco = (rowcol[0] == 'R' || rowcol[0] == 'r');
    bool lleri  = (leri[0] == 'L' || leri[0] == 'l');

    i32 pwork, mwork, maxmp, mplim;
    i32 i, idual, itemp, j, jstop, k, kdcoef, kpcoef, n;
    i32 int1 = 1;

    *info = 0;

    maxmp = (m > p) ? m : p;
    mplim = (maxmp > 1) ? maxmp : 1;

    if (lrowco) {
        pwork = p;
        mwork = m;
    } else {
        pwork = m;
        mwork = p;
    }

    bool rowcol_valid = (rowcol[0] == 'R' || rowcol[0] == 'r' ||
                         rowcol[0] == 'C' || rowcol[0] == 'c');
    bool leri_valid = (leri[0] == 'L' || leri[0] == 'l' ||
                       leri[0] == 'R' || leri[0] == 'r');
    bool equil_valid = (equil[0] == 'S' || equil[0] == 's' ||
                        equil[0] == 'N' || equil[0] == 'n');

    if (!rowcol_valid) {
        *info = -1;
    } else if (!leri_valid) {
        *info = -2;
    } else if (!equil_valid) {
        *info = -3;
    } else if (m < 0) {
        *info = -4;
    } else if (p < 0) {
        *info = -5;
    } else if (lddcoe < ((pwork > 1) ? pwork : 1)) {
        *info = -8;
    } else if (lduco1 < ((pwork > 1) ? pwork : 1) ||
               (!lrowco && lduco1 < mplim)) {
        *info = -10;
    } else if (lduco2 < ((mwork > 1) ? mwork : 1) ||
               (!lrowco && lduco2 < mplim)) {
        *info = -11;
    }

    n = 0;
    if (*info == 0) {
        kdcoef = 0;

        for (i = 0; i < pwork; i++) {
            if (indexd[i] > kdcoef) kdcoef = indexd[i];
            n += indexd[i];
        }
        kdcoef++;

        i32 n1 = (n > 1) ? n : 1;
        if (lda < n1) {
            *info = -14;
        } else if (ldb < n1) {
            *info = -16;
        } else if (ldc < mplim) {
            *info = -18;
        } else if (ldd < mplim) {
            *info = -20;
        } else if (ldpco1 < pwork) {
            *info = -23;
        } else if (ldpco2 < pwork) {
            *info = -24;
        } else if (ldqco1 < ((pwork > 1) ? pwork : 1) ||
                   (!lleri && ldqco1 < mplim)) {
            *info = -26;
        } else if (ldqco2 < ((mwork > 1) ? mwork : 1) ||
                   (!lleri && ldqco2 < mplim)) {
            *info = -27;
        } else if (ldvco1 < ((pwork > 1) ? pwork : 1)) {
            *info = -29;
        } else if (ldvco2 < n1) {
            *info = -30;
        } else {
            i32 ldwork_min = n + ((n > 3*maxmp) ? n : 3*maxmp);
            i32 pm2 = pwork * (pwork + 2);
            if (pm2 > ldwork_min) ldwork_min = pm2;
            if (ldwork_min < 1) ldwork_min = 1;

            if (ldwork < ldwork_min) {
                *info = -34;
            }
        }
    }

    if (*info != 0) {
        return;
    }

    i32 max_nmp = n;
    if (m > max_nmp) max_nmp = m;
    if (p > max_nmp) max_nmp = p;
    if (max_nmp == 0) {
        *nr = 0;
        dwork[0] = ONE;
        return;
    }

    idual = 0;
    if (!lrowco) idual = 1;
    if (!lleri) idual = idual + 1;

    if (!lrowco) {
        if (p < m) {
            for (k = 0; k < kdcoef; k++) {
                SLC_DLASET("F", &(i32){m - p}, &mplim, &ZERO, &ZERO,
                           &ucoeff[(k * lduco2 + 0) * lduco1 + p], &lduco1);
            }
        } else if (p > m) {
            for (k = 0; k < kdcoef; k++) {
                SLC_DLASET("F", &mplim, &(i32){p - m}, &ZERO, &ZERO,
                           &ucoeff[(k * lduco2 + m) * lduco1], &lduco1);
            }
        }

        if (mplim != 1) {
            jstop = mplim - 1;
            for (k = 0; k < kdcoef; k++) {
                for (j = 0; j < jstop; j++) {
                    i32 len = mplim - j - 1;
                    SLC_DSWAP(&len,
                              &ucoeff[(k * lduco2 + j) * lduco1 + j + 1], &int1,
                              &ucoeff[(k * lduco2 + j + 1) * lduco1 + j], &lduco1);
                }
            }
        }
    }

    td03ay(mwork, pwork, indexd, dcoeff, lddcoe, ucoeff, lduco1, lduco2,
           n, a, lda, b, ldb, c, ldc, d, ldd, info);

    if (*info > 0) {
        if (!lrowco && mplim != 1) {
            for (k = 0; k < kdcoef; k++) {
                for (j = 0; j < jstop; j++) {
                    i32 len = mplim - j - 1;
                    SLC_DSWAP(&len,
                              &ucoeff[(k * lduco2 + j) * lduco1 + j + 1], &int1,
                              &ucoeff[(k * lduco2 + j + 1) * lduco1 + j], &lduco1);
                }
            }
        }
        return;
    }

    if (idual == 1) {
        ab07md('D', n, mwork, pwork, a, lda, b, ldb, c, ldc, d, ldd);
        itemp = pwork;
        pwork = mwork;
        mwork = itemp;
    }

    tb03ad("L", equil, n, mwork, pwork, a, lda, b, ldb, c, ldc, d, ldd,
           nr, indexp, pcoeff, ldpco1, ldpco2, qcoeff, ldqco1, ldqco2,
           vcoeff, ldvco1, ldvco2, tol, iwork, dwork, ldwork, info);

    if (*info > 0) {
        *info = pwork + *info;
        if (!lrowco && mplim != 1) {
            for (k = 0; k < kdcoef; k++) {
                for (j = 0; j < jstop; j++) {
                    i32 len = mplim - j - 1;
                    SLC_DSWAP(&len,
                              &ucoeff[(k * lduco2 + j) * lduco1 + j + 1], &int1,
                              &ucoeff[(k * lduco2 + j + 1) * lduco1 + j], &lduco1);
                }
            }
        }
        return;
    }

    if (!lleri) {
        k = iwork[0] - 1;
        if (n >= 2) {
            k = k + iwork[1];
        }
        tb01xd("D", *nr, mwork, pwork, k, *nr - 1, a, lda, b, ldb, c, ldc,
               d, ldd, info);

        kpcoef = 0;
        for (i = 0; i < pwork; i++) {
            if (indexp[i] > kpcoef) kpcoef = indexp[i];
        }
        kpcoef++;

        tc01od('L', mwork, pwork, kpcoef, pcoeff, ldpco1, ldpco2,
               qcoeff, ldqco1, ldqco2, info);
    }

    if (!lrowco && mplim != 1) {
        for (k = 0; k < kdcoef; k++) {
            for (j = 0; j < jstop; j++) {
                i32 len = mplim - j - 1;
                SLC_DSWAP(&len,
                          &ucoeff[(k * lduco2 + j) * lduco1 + j + 1], &int1,
                          &ucoeff[(k * lduco2 + j + 1) * lduco1 + j], &lduco1);
            }
        }
    }
}
