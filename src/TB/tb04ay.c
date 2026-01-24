/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>

static i32 imax(i32 a, i32 b) {
    return a > b ? a : b;
}

void tb04ay(i32 n, i32 mwork, i32 pwork,
            f64* a, i32 lda, f64* b, i32 ldb,
            f64* c, i32 ldc, f64* d, i32 ldd,
            i32* ncont, i32* indexd, f64* dcoeff, i32 lddcoe,
            f64* ucoeff, i32 lduco1, i32 lduco2,
            f64* at, i32 n1, f64* tau, f64 tol1, f64 tol2,
            i32* iwork, f64* dwork, i32 ldwork, i32* info) {

    const f64 ONE = 1.0;
    const i32 int1 = 1;

    i32 i, ib, ibi, ic, indcon, is, iv, ivmin1, iwplus, iz, j, jwork, k, l;
    i32 lwork, maxm, nminl, nplus, wrkopt;
    f64 temp;

    tb01ud("N", n, mwork, pwork, a, lda, b, ldb, c, ldc,
           ncont, &indcon, iwork, at, 1, tau, tol2,
           &iwork[n], dwork, ldwork, info);
    wrkopt = (i32)dwork[0];

    is = 0;
    ic = is + *ncont;
    iz = ic;
    ib = ic + *ncont;
    lwork = ib + mwork * (*ncont > 0 ? *ncont : 1);
    maxm = imax(1, mwork);

    for (i = 0; i < pwork; i++) {
        i32 ncont_val = *ncont;
        SLC_DCOPY(&ncont_val, &c[i], &ldc, &dwork[ic], &int1);

        for (j = 0; j < *ncont; j++) {
            i32 ncont_val2 = *ncont;
            SLC_DCOPY(&ncont_val2, &a[j], &lda, &at[j * n1], &int1);
            SLC_DCOPY(&mwork, &b[j], &ldb, &dwork[ib + j * maxm], &int1);
        }

        tb01zd("N", *ncont, mwork, at, n1, &dwork[ic],
               &dwork[ib], maxm, &nminl, &dwork[iz], 1, tau, tol1,
               &dwork[lwork], ldwork - lwork, info);
        wrkopt = imax(wrkopt, (i32)dwork[lwork] + lwork);

        indexd[i] = nminl;
        dcoeff[i] = ONE;
        SLC_DCOPY(&mwork, &d[i], &ldd, &ucoeff[i], &lduco1);

        if (nminl == 1) {
            temp = -at[0];
            dcoeff[i + lddcoe] = temp;
            SLC_DCOPY(&mwork, &d[i], &ldd, &ucoeff[i + lduco1 * lduco2], &lduco1);
            SLC_DSCAL(&mwork, &temp, &ucoeff[i + lduco1 * lduco2], &lduco1);
            f64 dwork_ic = dwork[ic];
            SLC_DAXPY(&mwork, &dwork_ic, &dwork[ib], &int1, &ucoeff[i + lduco1 * lduco2], &lduco1);
        } else if (nminl > 1) {
            i32 nminl_m1 = nminl - 1;
            i32 n1p1 = n1 + 1;
            SLC_DCOPY(&nminl_m1, &at[1], &n1p1, &dwork[ic + 1], &int1);
            nplus = nminl + 1;

            for (l = is; l < is + nminl; l++) {
                dwork[l] = ONE;
            }

            for (jwork = nminl - 1; jwork >= 0; jwork--) {
                for (j = jwork; j < nminl; j++) {
                    at[jwork + j * n1] = dwork[is + j] * at[jwork + j * n1];
                }
                i32 len = nminl - jwork;
                SLC_DSCAL(&len, &dwork[ic + jwork], &dwork[is + jwork], &int1);
            }

            for (iv = 2; iv <= nminl; iv++) {
                jwork = nplus - iv;
                iwplus = jwork + 1;
                ivmin1 = iv - 1;

                for (k = 1; k <= ivmin1; k++) {
                    at[(iv - 1) + (k - 1) * n1] = -at[(iwplus - 1) + (jwork + k - 1) * n1];
                }

                if (iv != 2) {
                    i32 iv_m2 = iv - 2;
                    i32 iv_m1_idx = iv - 2;
                    i32 iv_idx = iv - 1;
                    SLC_DAXPY(&iv_m2, &ONE, &at[iv_m1_idx], &n1, &at[iv_idx], &n1);

                    for (k = 2; k <= ivmin1; k++) {
                        i32 neg_n1p1 = -(n1 + 1);
                        i32 k_m1 = k - 1;
                        temp = -SLC_DDOT(&k_m1, &at[(iwplus - 1) + jwork * n1], &n1,
                                         &at[iv - k], &neg_n1p1);
                        at[(iv - 1) + (k - 1) * n1] += temp;
                    }
                }
            }

            for (k = 2; k <= nplus; k++) {
                dcoeff[i + (k - 1) * lddcoe] = -at[(k - 2) * n1];
            }

            i32 nminl_m1_2 = nminl - 1;
            SLC_DAXPY(&nminl_m1_2, &ONE, &at[nminl - 1], &n1, &dcoeff[i + lddcoe], &lddcoe);

            for (k = 3; k <= nplus; k++) {
                i32 k_m2 = k - 2;
                i32 neg_n1p1 = -(n1 + 1);
                temp = -SLC_DDOT(&k_m2, at, &n1, &at[nminl - k + 2], &neg_n1p1);
                dcoeff[i + (k - 1) * lddcoe] += temp;
            }

            ibi = ib;
            for (l = 1; l <= nminl; l++) {
                SLC_DSCAL(&mwork, &dwork[is + l - 1], &dwork[ibi], &int1);
                ibi += maxm;
            }

            ibi = ib;
            for (k = 2; k <= nplus; k++) {
                SLC_DCOPY(&mwork, &dwork[ibi], &int1, &ucoeff[i + (k - 1) * lduco1 * lduco2], &lduco1);
                f64 dcoeff_ik = dcoeff[i + (k - 1) * lddcoe];
                SLC_DAXPY(&mwork, &dcoeff_ik, &d[i], &ldd, &ucoeff[i + (k - 1) * lduco1 * lduco2], &lduco1);
                ibi += maxm;
            }

            for (k = 3; k <= nplus; k++) {
                for (j = 1; j <= mwork; j++) {
                    i32 k_m2 = k - 2;
                    i32 neg_n1p1 = -(n1 + 1);
                    temp = SLC_DDOT(&k_m2, &at[nminl - k + 2], &neg_n1p1,
                                    &dwork[ib + j - 1], &maxm);
                    ucoeff[i + (j - 1) * lduco1 + (k - 1) * lduco1 * lduco2] += temp;
                }
            }
        }
    }

    dwork[0] = (f64)wrkopt;
}
