/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>

static inline i32 max_i32(i32 a, i32 b) {
    return a > b ? a : b;
}

static inline i32 min_i32(i32 a, i32 b) {
    return a < b ? a : b;
}

static inline f64 max_d(f64 a, f64 b) {
    return a > b ? a : b;
}

void mb03py(i32 m, i32 n, f64* a, i32 lda, f64 rcond,
            f64 svlmax, i32* rank, f64* sval, i32* jpvt,
            f64* tau, f64* dwork, i32* info) {

    const i32 IMAX = 1, IMIN = 2;
    const f64 ZERO = 0.0, ONE = 1.0;

    i32 i, ismax, ismin, itemp, j, jwork, k, mki, nki, pvt;
    f64 aii, c1, c2, s1, s2, smax, smaxpr, smin, sminpr;
    f64 temp, temp2, tolz;

    *info = 0;
    if (m < 0) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (lda < max_i32(1, m)) {
        *info = -4;
    } else if (rcond < ZERO || rcond > ONE) {
        *info = -5;
    } else if (svlmax < ZERO) {
        *info = -6;
    }

    if (*info != 0) {
        return;
    }

    k = min_i32(m, n);
    if (k == 0) {
        *rank = 0;
        sval[0] = ZERO;
        sval[1] = ZERO;
        sval[2] = ZERO;
        return;
    }

    tolz = sqrt(SLC_DLAMCH("Epsilon"));
    ismin = m - 1;
    ismax = ismin + m;
    jwork = ismax + 1;

    for (i = 0; i < m; i++) {
        sl_int n_int = n;
        dwork[i] = SLC_DNRM2(&n_int, &a[i], &lda);
        dwork[m + i] = dwork[i];
        jpvt[i] = i + 1;
    }

    *rank = 0;
    aii = ZERO;

    while (*rank < k) {
        i = k - *rank;
        mki = m - *rank;
        nki = n - *rank;

        sl_int mki_int = mki;
        sl_int inc = 1;
        pvt = SLC_IDAMAX(&mki_int, dwork, &inc);
        pvt--;

        if (pvt != mki - 1) {
            sl_int n_int = n;
            SLC_DSWAP(&n_int, &a[pvt], &lda, &a[mki - 1], &lda);
            itemp = jpvt[pvt];
            jpvt[pvt] = jpvt[mki - 1];
            jpvt[mki - 1] = itemp;
            dwork[pvt] = dwork[mki - 1];
            dwork[m + pvt] = dwork[m + mki - 1];
        }

        if (nki > 1) {
            aii = a[(nki - 1) * lda + mki - 1];
            sl_int nki_int = nki;
            SLC_DLARFG(&nki_int, &a[(nki - 1) * lda + mki - 1],
                       &a[mki - 1], &lda, &tau[i - 1]);
        }

        if (*rank == 0) {
            smax = fabs(a[(n - 1) * lda + m - 1]);
            if (smax <= rcond) {
                sval[0] = ZERO;
                sval[1] = ZERO;
                sval[2] = ZERO;
            }
            smin = smax;
            smaxpr = smax;
            sminpr = smin;
            c1 = ONE;
            c2 = ONE;
        } else {
            sl_int rank_int = *rank;
            SLC_DCOPY(&rank_int, &a[nki * lda + mki - 1], &lda, &dwork[jwork], &inc);
            SLC_DLAIC1(&IMIN, &rank_int, &dwork[ismin], &smin,
                       &dwork[jwork], &a[(nki - 1) * lda + mki - 1], &sminpr, &s1, &c1);
            SLC_DLAIC1(&IMAX, &rank_int, &dwork[ismax], &smax,
                       &dwork[jwork], &a[(nki - 1) * lda + mki - 1], &smaxpr, &s2, &c2);
        }

        if (svlmax * rcond <= smaxpr) {
            if (svlmax * rcond <= sminpr) {
                if (smaxpr * rcond < sminpr) {
                    if (mki > 1) {
                        aii = a[(nki - 1) * lda + mki - 1];
                        a[(nki - 1) * lda + mki - 1] = ONE;
                        sl_int mki_minus_1 = mki - 1;
                        sl_int nki_int = nki;
                        SLC_DLARF("Right", &mki_minus_1, &nki_int, &a[mki - 1], &lda,
                                  &tau[i - 1], a, &lda, &dwork[jwork]);
                        a[(nki - 1) * lda + mki - 1] = aii;

                        for (j = 0; j < mki - 1; j++) {
                            if (dwork[j] != ZERO) {
                                temp = fabs(a[(nki - 1) * lda + j]) / dwork[j];
                                temp = max_d((ONE + temp) * (ONE - temp), ZERO);
                                temp2 = temp * pow(dwork[j] / dwork[m + j], 2);
                                if (temp2 <= tolz) {
                                    if (nki > 1) {
                                        sl_int nki_minus_1 = nki - 1;
                                        dwork[j] = SLC_DNRM2(&nki_minus_1, &a[j], &lda);
                                        dwork[m + j] = dwork[j];
                                    } else {
                                        dwork[j] = ZERO;
                                        dwork[m + j] = ZERO;
                                    }
                                } else {
                                    dwork[j] = dwork[j] * sqrt(temp);
                                }
                            }
                        }
                    }

                    for (i32 ii = 0; ii < *rank; ii++) {
                        dwork[ismin + ii] = s1 * dwork[ismin + ii];
                        dwork[ismax + ii] = s2 * dwork[ismax + ii];
                    }

                    if (*rank > 0) {
                        ismin--;
                        ismax--;
                    }
                    dwork[ismin] = c1;
                    dwork[ismax] = c2;
                    smin = sminpr;
                    smax = smaxpr;
                    (*rank)++;
                    continue;
                }
            }
        }
        break;
    }

    if (*rank < k && nki > 1) {
        i32 i_cur = k - *rank;
        i32 mki_cur = m - *rank;
        i32 nki_cur = n - *rank;
        sl_int nki_minus_1 = nki_cur - 1;
        f64 scale = -a[(nki_cur - 1) * lda + mki_cur - 1] * tau[i_cur - 1];
        SLC_DSCAL(&nki_minus_1, &scale, &a[mki_cur - 1], &lda);
        a[(nki_cur - 1) * lda + mki_cur - 1] = aii;
    }

    sval[0] = smax;
    sval[1] = smin;
    sval[2] = sminpr;
}
