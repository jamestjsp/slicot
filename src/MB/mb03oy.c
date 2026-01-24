/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <float.h>

static inline i32 max_i32(i32 a, i32 b) {
    return a > b ? a : b;
}

static inline i32 min_i32(i32 a, i32 b) {
    return a < b ? a : b;
}

static inline f64 max_d(f64 a, f64 b) {
    return a > b ? a : b;
}

void mb03oy(i32 m, i32 n, f64* a, i32 lda, f64 rcond,
            f64 svlmax, i32* rank, f64* sval, i32* jpvt,
            f64* tau, f64* dwork, i32* info) {

    const i32 IMAX = 1, IMIN = 2;
    const f64 ZERO = 0.0, ONE = 1.0;

    i32 i, ismax, ismin, itemp, j, mn, pvt;
    f64 aii, aii_saved, c1, c2, s1, s2, smax, smaxpr, smin, sminpr;
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

    mn = min_i32(m, n);
    if (mn == 0) {
        *rank = 0;
        sval[0] = ZERO;
        sval[1] = ZERO;
        sval[2] = ZERO;
        return;
    }

    tolz = sqrt(SLC_DLAMCH("Epsilon"));
    ismin = 0;
    ismax = ismin + n;

    for (i = 0; i < n; i++) {
        sl_int m_int = m;
        sl_int inc = 1;
        dwork[i] = SLC_DNRM2(&m_int, &a[i * lda], &inc);
        dwork[n + i] = dwork[i];
        jpvt[i] = i + 1;
    }

    *rank = 0;
    aii_saved = ZERO;

    while (*rank < mn) {
        i = *rank;

        sl_int n_minus_i = n - i;
        sl_int inc = 1;
        pvt = i + SLC_IDAMAX(&n_minus_i, &dwork[i], &inc) - 1;

        if (pvt != i) {
            sl_int m_int = m;
            sl_int inc = 1;
            SLC_DSWAP(&m_int, &a[pvt * lda], &inc, &a[i * lda], &inc);
            itemp = jpvt[pvt];
            jpvt[pvt] = jpvt[i];
            jpvt[i] = itemp;
            dwork[pvt] = dwork[i];
            dwork[n + pvt] = dwork[n + i];
        }

        if (i < m - 1) {
            aii = a[i * lda + i];
            aii_saved = aii;
            sl_int m_minus_i = m - i;
            sl_int inc = 1;
            SLC_DLARFG(&m_minus_i, &a[i * lda + i], &a[i * lda + i + 1], &inc, &tau[i]);
        } else {
            tau[m - 1] = ZERO;
        }

        if (*rank == 0) {
            smax = fabs(a[0]);
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
            SLC_DLAIC1(&IMIN, &rank_int, &dwork[ismin], &smin, &a[i * lda],
                    &a[i * lda + i], &sminpr, &s1, &c1);
            SLC_DLAIC1(&IMAX, &rank_int, &dwork[ismax], &smax, &a[i * lda],
                    &a[i * lda + i], &smaxpr, &s2, &c2);
        }

        if (svlmax * rcond <= smaxpr) {
            if (svlmax * rcond <= sminpr) {
                if (smaxpr * rcond < sminpr) {
                    if (i < n - 1) {
                        aii = a[i * lda + i];
                        a[i * lda + i] = ONE;
                        sl_int m_minus_i = m - i;
                        sl_int n_minus_i_minus_1 = n - i - 1;
                        sl_int inc = 1;
                        SLC_DLARF("Left", &m_minus_i, &n_minus_i_minus_1, &a[i * lda + i], &inc,
                               &tau[i], &a[(i + 1) * lda + i], &lda, &dwork[2 * n]);
                        a[i * lda + i] = aii;
                    }

                    for (j = i + 1; j < n; j++) {
                        if (dwork[j] != ZERO) {
                            temp = fabs(a[j * lda + i]) / dwork[j];
                            temp = max_d((ONE + temp) * (ONE - temp), ZERO);
                            temp2 = temp * pow(dwork[j] / dwork[n + j], 2);
                            if (temp2 <= tolz) {
                                if (m - i - 1 > 0) {
                                    sl_int m_minus_i_minus_1 = m - i - 1;
                                    sl_int inc = 1;
                                    dwork[j] = SLC_DNRM2(&m_minus_i_minus_1, &a[j * lda + i + 1], &inc);
                                    dwork[n + j] = dwork[j];
                                } else {
                                    dwork[j] = ZERO;
                                    dwork[n + j] = ZERO;
                                }
                            } else {
                                dwork[j] = dwork[j] * sqrt(temp);
                            }
                        }
                    }

                    for (i32 ii = 0; ii < *rank; ii++) {
                        dwork[ismin + ii] = s1 * dwork[ismin + ii];
                        dwork[ismax + ii] = s2 * dwork[ismax + ii];
                    }

                    dwork[ismin + *rank] = c1;
                    dwork[ismax + *rank] = c2;
                    smin = sminpr;
                    smax = smaxpr;
                    (*rank)++;
                    continue;
                }
            }
        }
        break;
    }

    if (*rank < n) {
        if (i < m - 1) {
            sl_int m_minus_i_minus_1 = m - i - 1;
            sl_int inc = 1;
            f64 scale = -a[i * lda + i] * tau[i];
            SLC_DSCAL(&m_minus_i_minus_1, &scale, &a[i * lda + i + 1], &inc);
            a[i * lda + i] = aii_saved;
        }
    }

    if (*rank == 0) {
        smin = ZERO;
        sminpr = ZERO;
    }

    sval[0] = smax;
    sval[1] = smin;
    sval[2] = sminpr;
}
