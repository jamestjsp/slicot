/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1998-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>

i32 slicot_mb3pyz(i32 m, i32 n, c128* a, i32 lda, f64 rcond, f64 svlmax,
                  i32* rank, f64* sval, i32* jpvt, c128* tau, f64* dwork,
                  c128* zwork) {
    const i32 IMAX = 1, IMIN = 2;
    const f64 ZERO = 0.0, ONE = 1.0;
    const c128 CONE = 1.0 + 0.0*I;

    i32 info = 0;

    if (m < 0) {
        info = -1;
    } else if (n < 0) {
        info = -2;
    } else if (lda < (m > 1 ? m : 1)) {
        info = -4;
    } else if (rcond < ZERO || rcond > ONE) {
        info = -5;
    } else if (svlmax < ZERO) {
        info = -6;
    }

    if (info != 0) {
        i32 xinfo = -info;
        SLC_XERBLA("MB3PYZ", &xinfo);
        return info;
    }

    i32 k = m < n ? m : n;
    if (k == 0) {
        *rank = 0;
        sval[0] = ZERO;
        sval[1] = ZERO;
        sval[2] = ZERO;
        return 0;
    }

    f64 tolz = sqrt(SLC_DLAMCH("Epsilon"));
    i32 ismin = 0;
    i32 ismax = m;
    i32 jwork = 2 * m;

    i32 one = 1;
    for (i32 i = 0; i < m; i++) {
        dwork[i] = SLC_DZNRM2(&n, &a[i], &lda);
        dwork[m + i] = dwork[i];
        jpvt[i] = i + 1;
    }

    *rank = 0;
    f64 smax = ZERO, smin = ZERO, smaxpr = ZERO, sminpr = ZERO;
    c128 c1 = CONE, c2 = CONE, s1, s2;
    c128 aii = 0.0 + 0.0*I;

    while (*rank < k) {
        i32 mki = m - *rank;
        i32 nki = n - *rank;

        i32 pvt = SLC_IDAMAX(&mki, dwork, &one) - 1;

        if (pvt != mki - 1) {
            SLC_ZSWAP(&n, &a[pvt], &lda, &a[mki - 1], &lda);
            i32 itemp = jpvt[pvt];
            jpvt[pvt] = jpvt[mki - 1];
            jpvt[mki - 1] = itemp;
            dwork[pvt] = dwork[mki - 1];
            dwork[m + pvt] = dwork[m + mki - 1];
        }

        if (nki > 1) {
            SLC_ZLACGV(&nki, &a[mki - 1], &lda);
            aii = a[mki - 1 + (nki - 1) * lda];
            SLC_ZLARFG(&nki, &a[mki - 1 + (nki - 1) * lda], &a[mki - 1], &lda, &tau[k - *rank - 1]);
        }

        if (*rank == 0) {
            smax = cabs(a[m - 1 + (n - 1) * lda]);
            if (smax == ZERO) {
                sval[0] = ZERO;
                sval[1] = ZERO;
                sval[2] = ZERO;
                return 0;
            }
            smin = smax;
            smaxpr = smax;
            sminpr = smin;
            c1 = CONE;
            c2 = CONE;
        } else {
            for (i32 ii = 0; ii < *rank; ii++) {
                zwork[jwork + ii] = a[mki - 1 + (nki + ii) * lda];
            }
            SLC_ZLAIC1(&IMIN, rank, &zwork[ismin], &smin,
                      &zwork[jwork], &a[mki - 1 + (nki - 1) * lda], &sminpr, &s1, &c1);
            SLC_ZLAIC1(&IMAX, rank, &zwork[ismax], &smax,
                      &zwork[jwork], &a[mki - 1 + (nki - 1) * lda], &smaxpr, &s2, &c2);
        }

        if (svlmax * rcond <= smaxpr) {
            if (svlmax * rcond <= sminpr) {
                if (smaxpr * rcond <= sminpr) {
                    if (mki > 1) {
                        aii = a[mki - 1 + (nki - 1) * lda];
                        a[mki - 1 + (nki - 1) * lda] = CONE;
                        i32 mki_m1 = mki - 1;
                        SLC_ZLARF("Right", &mki_m1, &nki, &a[mki - 1], &lda,
                                 &tau[k - *rank - 1], a, &lda, &zwork[jwork]);
                        a[mki - 1 + (nki - 1) * lda] = aii;

                        for (i32 j = 0; j < mki - 1; j++) {
                            if (dwork[j] != ZERO) {
                                f64 temp = cabs(a[j + (nki - 1) * lda]) / dwork[j];
                                temp = (ONE + temp) * (ONE - temp);
                                if (temp < ZERO) temp = ZERO;
                                f64 temp2 = temp * (dwork[j] / dwork[m + j]) * (dwork[j] / dwork[m + j]);
                                if (temp2 <= tolz) {
                                    i32 nki_m1 = nki - 1;
                                    dwork[j] = SLC_DZNRM2(&nki_m1, &a[j], &lda);
                                    dwork[m + j] = dwork[j];
                                } else {
                                    dwork[j] = dwork[j] * sqrt(temp);
                                }
                            }
                        }
                    }

                    for (i32 ii = 0; ii < *rank; ii++) {
                        zwork[ismin + ii] = s1 * zwork[ismin + ii];
                        zwork[ismax + ii] = s2 * zwork[ismax + ii];
                    }

                    zwork[ismin + *rank] = c1;
                    zwork[ismax + *rank] = c2;
                    smin = sminpr;
                    smax = smaxpr;
                    (*rank)++;
                    i32 nki_m1 = nki - 1;
                    if (nki_m1 > 0) {
                        SLC_ZLACGV(&nki_m1, &a[mki - 1], &lda);
                    }
                    continue;
                }
            }
        }
        break;
    }

    i32 i = k - *rank - 1;
    i32 mki = m - *rank;
    i32 nki = n - *rank;
    if (*rank < k && nki > 1) {
        i32 nki_m1 = nki - 1;
        SLC_ZLACGV(&nki_m1, &a[mki - 1], &lda);
        c128 scale = -a[mki - 1 + (nki - 1) * lda] * tau[i];
        SLC_ZSCAL(&nki_m1, &scale, &a[mki - 1], &lda);
        a[mki - 1 + (nki - 1) * lda] = aii;
    }

    sval[0] = smax;
    sval[1] = smin;
    sval[2] = sminpr;

    return 0;
}
