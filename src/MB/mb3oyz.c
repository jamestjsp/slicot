/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1998-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>

i32 slicot_mb3oyz(i32 m, i32 n, c128* a, i32 lda, f64 rcond, f64 svlmax,
                  i32* rank, f64* sval, i32* jpvt, c128* tau, f64* dwork,
                  c128* zwork) {
    const i32 IMAX = 1, IMIN = 2;
    const f64 ZERO = 0.0, ONE = 1.0;
    const c128 CZERO = 0.0 + 0.0*I;
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
        SLC_XERBLA("MB3OYZ", &xinfo);
        return info;
    }

    i32 mn = m < n ? m : n;
    if (mn == 0) {
        *rank = 0;
        sval[0] = ZERO;
        sval[1] = ZERO;
        sval[2] = ZERO;
        return 0;
    }

    f64 tolz = sqrt(SLC_DLAMCH("Epsilon"));
    i32 ismin = 0;
    i32 ismax = n;

    i32 one = 1;
    for (i32 i = 0; i < n; i++) {
        dwork[i] = SLC_DZNRM2(&m, &a[i * lda], &one);
        dwork[n + i] = dwork[i];
        jpvt[i] = i + 1;
    }

    *rank = 0;
    f64 smax = ZERO, smin = ZERO, smaxpr = ZERO, sminpr = ZERO;
    c128 c1 = CONE, c2 = CONE, s1, s2;
    c128 aii = CZERO;

    while (*rank < mn) {
        i32 i = *rank;

        i32 pvt = i + SLC_IDAMAX(&(i32){n - i}, &dwork[i], &one) - 1;

        if (pvt != i) {
            SLC_ZSWAP(&m, &a[pvt * lda], &one, &a[i * lda], &one);
            i32 itemp = jpvt[pvt];
            jpvt[pvt] = jpvt[i];
            jpvt[i] = itemp;
            dwork[pvt] = dwork[i];
            dwork[n + pvt] = dwork[n + i];
        }

        if (i < m - 1) {
            aii = a[i + i * lda];
            i32 len = m - i;
            SLC_ZLARFG(&len, &a[i + i * lda], &a[i + 1 + i * lda], &one, &tau[i]);
        } else {
            tau[m - 1] = CZERO;
        }

        if (*rank == 0) {
            smax = cabs(a[0]);
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
            SLC_ZLAIC1(&IMIN, rank, &zwork[ismin], &smin, &a[i * lda],
                      &a[i + i * lda], &sminpr, &s1, &c1);
            SLC_ZLAIC1(&IMAX, rank, &zwork[ismax], &smax, &a[i * lda],
                      &a[i + i * lda], &smaxpr, &s2, &c2);
        }

        if (svlmax * rcond <= smaxpr) {
            if (svlmax * rcond <= sminpr) {
                if (smaxpr * rcond <= sminpr) {
                    if (i < n - 1) {
                        aii = a[i + i * lda];
                        a[i + i * lda] = CONE;
                        i32 len_m = m - i;
                        i32 len_n = n - i - 1;
                        c128 tau_conj = conj(tau[i]);
                        SLC_ZLARF("Left", &len_m, &len_n, &a[i + i * lda], &one,
                                 &tau_conj, &a[i + (i + 1) * lda], &lda, &zwork[2 * n]);
                        a[i + i * lda] = aii;
                    }

                    for (i32 j = i + 1; j < n; j++) {
                        if (dwork[j] != ZERO) {
                            f64 temp = cabs(a[i + j * lda]) / dwork[j];
                            temp = (ONE + temp) * (ONE - temp);
                            if (temp < ZERO) temp = ZERO;
                            f64 temp2 = temp * (dwork[j] / dwork[n + j]) * (dwork[j] / dwork[n + j]);
                            if (temp2 <= tolz) {
                                if (m - i - 1 > 0) {
                                    i32 len = m - i - 1;
                                    dwork[j] = SLC_DZNRM2(&len, &a[i + 1 + j * lda], &one);
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
                        zwork[ismin + ii] = s1 * zwork[ismin + ii];
                        zwork[ismax + ii] = s2 * zwork[ismax + ii];
                    }

                    zwork[ismin + *rank] = c1;
                    zwork[ismax + *rank] = c2;
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
        i32 i = *rank;
        if (i < m - 1) {
            i32 len = m - i - 1;
            c128 scale = -a[i + i * lda] * tau[i];
            SLC_ZSCAL(&len, &scale, &a[i + 1 + i * lda], &one);
            a[i + i * lda] = aii;
        }
    }

    if (*rank == 0) {
        smin = ZERO;
        sminpr = ZERO;
    }

    sval[0] = smax;
    sval[1] = smin;
    sval[2] = sminpr;

    return 0;
}
