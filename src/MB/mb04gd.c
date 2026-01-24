/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Translated from SLICOT Fortran77 to C11
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <stdlib.h>

void mb04gd(i32 m, i32 n, f64 *a, i32 lda, i32 *jpvt, f64 *tau,
            f64 *dwork, i32 *info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    i32 i, itemp, j, k, ma, mki, nfree, nki, pvt;
    f64 aii, temp, temp2, tolz;
    i32 int1 = 1;

    *info = 0;
    if (m < 0) {
        *info = -1;
        return;
    }
    if (n < 0) {
        *info = -2;
        return;
    }
    i32 max_m_1 = (m > 1) ? m : 1;
    if (lda < max_m_1) {
        *info = -4;
        return;
    }

    k = (m < n) ? m : n;
    if (k == 0) {
        return;
    }

    itemp = m;
    for (i = m - 1; i >= 0; i--) {
        if (jpvt[i] != 0) {
            if (i != itemp - 1) {
                SLC_DSWAP(&n, &a[i], &lda, &a[itemp - 1], &lda);
                jpvt[i] = jpvt[itemp - 1];
                jpvt[itemp - 1] = i + 1;
            } else {
                jpvt[i] = i + 1;
            }
            itemp = itemp - 1;
        } else {
            jpvt[i] = i + 1;
        }
    }
    nfree = m - itemp;
    tolz = sqrt(SLC_DLAMCH("E"));

    if (nfree > 0) {
        ma = (nfree < n) ? nfree : n;
        i32 ma_info = 0;

        SLC_DGERQ2(&ma, &n, &a[(m - ma) * 1 + 0 * lda], &lda,
                   &tau[k - ma], dwork, &ma_info);

        i32 rows_to_apply = m - ma;
        if (rows_to_apply > 0) {
            SLC_DORMR2("R", "T", &rows_to_apply, &n, &ma,
                       &a[(m - ma)], &lda, &tau[k - ma],
                       a, &lda, dwork, &ma_info);
        }
    }

    if (nfree < k) {
        i32 cols_to_norm = n - nfree;
        for (i = 0; i < itemp; i++) {
            dwork[i] = SLC_DNRM2(&cols_to_norm, &a[i], &lda);
            dwork[m + i] = dwork[i];
        }

        for (i = k - nfree - 1; i >= 0; i--) {
            mki = m - k + i;
            nki = n - k + i;

            i32 mki_plus1 = mki + 1;
            pvt = SLC_IDAMAX(&mki_plus1, dwork, &int1) - 1;

            if (pvt != mki) {
                SLC_DSWAP(&n, &a[pvt], &lda, &a[mki], &lda);
                itemp = jpvt[pvt];
                jpvt[pvt] = jpvt[mki];
                jpvt[mki] = itemp;
                dwork[pvt] = dwork[mki];
                dwork[m + pvt] = dwork[m + mki];
            }

            i32 nki_plus1 = nki + 1;
            SLC_DLARFG(&nki_plus1, &a[mki + nki * lda], &a[mki], &lda, &tau[i]);

            aii = a[mki + nki * lda];
            a[mki + nki * lda] = ONE;

            i32 rows_to_apply = mki;
            if (rows_to_apply > 0) {
                SLC_DLARF("R", &rows_to_apply, &nki_plus1, &a[mki], &lda,
                          &tau[i], a, &lda, &dwork[2 * m]);
            }
            a[mki + nki * lda] = aii;

            for (j = 0; j < mki; j++) {
                if (dwork[j] != ZERO) {
                    temp = fabs(a[j + nki * lda]) / dwork[j];
                    temp = (ONE + temp) * (ONE - temp);
                    temp = (temp > ZERO) ? temp : ZERO;
                    temp2 = temp * (dwork[j] / dwork[m + j]) * (dwork[j] / dwork[m + j]);
                    if (temp2 <= tolz) {
                        i32 nki_val = nki;
                        dwork[j] = SLC_DNRM2(&nki_val, &a[j], &lda);
                        dwork[m + j] = dwork[j];
                    } else {
                        dwork[j] = dwork[j] * sqrt(temp);
                    }
                }
            }
        }
    }
}
