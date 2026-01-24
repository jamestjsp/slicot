/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <complex.h>
#include <math.h>

void mb03rz(const char* jobx, const char* sort, const i32 n, const f64 pmax,
            c128* a, const i32 lda, c128* x, const i32 ldx, i32* nblcks,
            i32* blsize, c128* w, const f64 tol, i32* info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const c128 CZERO = 0.0 + 0.0*I;
    const c128 CONE = 1.0 + 0.0*I;

    bool ljobx, lsorn, lsors, lsort;
    char jobv;
    i32 da11, da22, i, ierr, j, k, l, l11, l22, l22m1;
    f64 bignum, c, d, edif, safemn, thresh;
    c128 av, sc;

    *info = 0;
    ljobx = (jobx[0] == 'U' || jobx[0] == 'u');
    lsorn = (sort[0] == 'N' || sort[0] == 'n');
    lsors = (sort[0] == 'S' || sort[0] == 's');
    lsort = (sort[0] == 'B' || sort[0] == 'b') || lsors;

    if (!ljobx && !(jobx[0] == 'N' || jobx[0] == 'n')) {
        *info = -1;
    } else if (!lsorn && !lsort && !(sort[0] == 'C' || sort[0] == 'c')) {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    } else if (pmax < ONE) {
        *info = -4;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -6;
    } else if ((ldx < 1) || (ljobx && ldx < n)) {
        *info = -8;
    }

    if (*info != 0) {
        return;
    }

    *nblcks = 0;
    if (n == 0) {
        return;
    }

    safemn = SLC_DLAMCH("Safe minimum");
    bignum = ONE / safemn;
    SLC_DLABAD(&safemn, &bignum);
    safemn = safemn / SLC_DLAMCH("Precision");
    jobv = jobx[0];
    if (ljobx) {
        jobv = 'V';
    }

    i32 int1 = 1;
    i32 lda_plus_1 = lda + 1;
    SLC_ZCOPY(&n, a, &lda_plus_1, w, &int1);

    if (lsort) {
        thresh = fabs(tol);
        if (thresh == ZERO) {
            thresh = sqrt(sqrt(SLC_DLAMCH("Epsilon")));
        }

        if (tol <= ZERO) {
            l = SLC_IZAMAX(&n, w, &int1);
            thresh = thresh * cabs(w[l - 1]);
        }
    }

    l11 = 0;

    while (l11 < n) {
        (*nblcks)++;
        da11 = 1;

        if (lsort) {
            l22 = l11 + da11;
            k = l22;

            while (k < n) {
                edif = cabs(w[l11] - w[k]);
                if (edif <= thresh) {
                    if (k > l22) {
                        i32 kp1 = k + 1;
                        i32 l22p1 = l22 + 1;
                        char jobv_str[2] = {jobv, '\0'};
                        SLC_ZTREXC(jobv_str, &n, a, &lda, x, &ldx, &kp1, &l22p1, &ierr);
                        i32 count = k - l22 + 1;
                        SLC_ZCOPY(&count, &a[l22 + l22 * lda], &lda_plus_1, &w[l22], &int1);
                    }

                    da11++;
                    l22 = l11 + da11;
                }
                k++;
            }
        }

        l22 = l11 + da11;
        l22m1 = l22 - 1;

        while (l22 < n) {
            da22 = n - l22;

            ma02az("T", "F", da11, da22, &a[l11 + l22 * lda], lda,
                   &a[l22 + l11 * lda], lda);

            mb03rw(da11, da22, pmax, &a[l11 + l11 * lda], lda,
                   &a[l22 + l22 * lda], lda, &a[l11 + l22 * lda], lda, &ierr);

            if (ierr == 1) {
                ma02az("T", "F", da22, da11, &a[l22 + l11 * lda], lda,
                       &a[l11 + l22 * lda], lda);
                SLC_ZLASET("F", &da22, &da11, &CZERO, &CZERO, &a[l22 + l11 * lda], &lda);

                if (lsorn || lsors) {
                    av = CZERO;
                    for (i = l11; i <= l22m1; i++) {
                        av = av + w[i];
                    }
                    av = av / (c128)da11;

                    d = cabs(av - w[l22]);
                    k = l22;
                    l = l22 + 1;

                    while (l < n) {
                        c = cabs(av - w[l]);
                        if (c < d) {
                            d = c;
                            k = l;
                        }
                        l++;
                    }
                } else {
                    d = bignum;
                    l = l22;
                    k = l22;

                    while (l < n) {
                        i = l11;
                        while (i <= l22m1) {
                            c = cabs(w[i] - w[l]);
                            if (c < d) {
                                d = c;
                                k = l;
                            }
                            i++;
                        }
                        l++;
                    }
                }

                if (k > l22) {
                    i32 kp1 = k + 1;
                    i32 l22p1 = l22 + 1;
                    char jobv_str[2] = {jobv, '\0'};
                    SLC_ZTREXC(jobv_str, &n, a, &lda, x, &ldx, &kp1, &l22p1, &ierr);
                    i32 count = k - l22 + 1;
                    SLC_ZCOPY(&count, &a[l22 + l22 * lda], &lda_plus_1, &w[l22], &int1);
                }

                da11++;
                l22 = l11 + da11;
                l22m1 = l22 - 1;
                continue;
            }
            break;
        }

        if (ljobx) {
            if (l22 < n) {
                SLC_ZGEMM("N", "N", &n, &da22, &da11, &CONE, &x[l11 * ldx], &ldx,
                          &a[l11 + l22 * lda], &lda, &CONE, &x[l22 * ldx], &ldx);
            }

            for (j = l11; j <= l22m1; j++) {
                c = SLC_DZNRM2(&n, &x[j * ldx], &int1);
                sc = c + 0.0*I;
                if (c > safemn) {
                    SLC_ZSCAL(&da11, &sc, &a[j + l11 * lda], &lda);
                    sc = CONE / sc;
                    SLC_ZSCAL(&n, &sc, &x[j * ldx], &int1);
                    SLC_ZSCAL(&da11, &sc, &a[l11 + j * lda], &int1);
                }
            }
        }

        if (l22 < n) {
            SLC_ZLASET("F", &da11, &da22, &CZERO, &CZERO, &a[l11 + l22 * lda], &lda);
            SLC_ZLASET("F", &da22, &da11, &CZERO, &CZERO, &a[l22 + l11 * lda], &lda);
        }

        blsize[*nblcks - 1] = da11;
        l11 = l22;
    }
}
