/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <ctype.h>
#include <math.h>

void mb03ag(const char *shft, i32 k, i32 n, const i32 *amap, const i32 *s,
            i32 sinv, const f64 *a, i32 lda1, i32 lda2,
            f64 *c1, f64 *s1, f64 *c2, f64 *s2,
            i32 *iwork, f64 *dwork) {
    const f64 ONE = 1.0;
    const f64 TWO = 2.0;
    const f64 ZERO = 0.0;

    char shft_upper = (char)toupper((unsigned char)shft[0]);
    bool sgle = (shft_upper == 'S');

    i32 nn = n * n;
    i32 ir = nn;
    i32 ii = ir + n;
    i32 ldas = lda1 * lda2;

    i32 one_i = 1;
    i32 zero_i = 0;
    i32 im, is_loc, ai, ic, l, j;
    f64 z_arr[1] = {ZERO};
    f64 sm, pr, mc, mn, mx, e1, e2, p1, p2, p3;

    ai = amap[k - 1] - 1;
    ic = 0;

    if (k > 1) {
        if (s[ai] == sinv) {
            is_loc = 1;
            for (l = 0; l < n; l++) {
                i32 lp1 = l + 1;
                SLC_DCOPY(&lp1, &a[ai * ldas + l * lda1], &one_i, &dwork[ic], &one_i);
                i32 nlm1 = n - l - 1;
                if (nlm1 > 0) {
                    SLC_DCOPY(&nlm1, z_arr, &zero_i, &dwork[ic + lp1], &one_i);
                }
                ic += n;
            }
        } else {
            is_loc = 0;
            SLC_DLASET("Full", &n, &n, &ZERO, &ONE, dwork, &n);
        }

        for (j = k - is_loc - 1; j >= 1; j--) {
            ai = amap[j] - 1;
            if (s[ai] == sinv) {
                ic = 0;
                for (l = 0; l < n; l++) {
                    i32 lp1 = l + 1;
                    SLC_DTRMV("Upper", "NoTran", "NonUnit", &lp1, &a[ai * ldas], &lda1,
                              &dwork[ic], &one_i);
                    ic += n;
                }
            } else {
                ic = ir;
                for (l = 0; l < n; l++) {
                    i32 lp1 = l + 1;
                    SLC_DCOPY(&lp1, &a[ai * ldas + l * lda1], &one_i, &dwork[ic], &one_i);
                    i32 nlm1 = n - l - 1;
                    if (nlm1 > 0) {
                        SLC_DCOPY(&nlm1, z_arr, &zero_i, &dwork[ic + lp1], &one_i);
                    }
                    ic += n;
                }

                SLC_DGETC2(&n, &dwork[ir], &n, iwork, &iwork[n], &im);

                for (ic = 0; ic < nn; ic += n) {
                    SLC_DGESC2(&n, &dwork[ir], &n, &dwork[ic], iwork, &iwork[n], &sm);
                }
            }
        }

        ai = amap[0] - 1;
        ic = 0;

        for (j = 0; j < n - 1; j++) {
            i32 jp1 = j + 1;
            SLC_DCOPY(&jp1, &dwork[ic], &one_i, &dwork[ir], &one_i);
            SLC_DTRMV("Upper", "NoTran", "NoDiag", &jp1, &a[ai * ldas], &lda1,
                      &dwork[ic], &one_i);

            for (l = 0; l < j + 1; l++) {
                dwork[ic + l + 1] += a[ai * ldas + (l + 1) + l * lda1] * dwork[ir + l];
            }
            ic += n;
        }

        SLC_DCOPY(&n, &dwork[ic], &one_i, &dwork[ir], &one_i);
        SLC_DTRMV("Upper", "NoTran", "NoDiag", &n, &a[ai * ldas], &lda1,
                  &dwork[ic], &one_i);

        for (l = 0; l < n; l++) {
            i32 next_row_idx = l + 1;
            if (next_row_idx < n) {
                dwork[ic + next_row_idx] += a[ai * ldas + next_row_idx + l * lda1] * dwork[ir + l];
            }
        }
    } else {
        ai = amap[0] - 1;

        for (l = 0; l < n - 1; l++) {
            i32 lp2 = l + 2;
            SLC_DCOPY(&lp2, &a[ai * ldas + l * lda1], &one_i, &dwork[ic], &one_i);
            i32 nlm2 = n - l - 2;
            if (nlm2 > 0) {
                SLC_DCOPY(&nlm2, z_arr, &zero_i, &dwork[ic + lp2], &one_i);
            }
            ic += n;
        }

        SLC_DCOPY(&n, &a[ai * ldas + (n - 1) * lda1], &one_i, &dwork[ic], &one_i);
    }

    if (sgle) {
        f64 arg1 = dwork[0] - dwork[nn - 1];
        SLC_DLARTG(&arg1, &dwork[1], c1, s1, &e1);
        *c2 = ONE;
        *s2 = ZERO;
    } else {
        e1 = dwork[0];
        e2 = dwork[1];
        p1 = dwork[n];
        p2 = dwork[n + 1];
        p3 = dwork[n + 2];

        f64 z_dummy[1];
        SLC_DLAHQR(&(i32){0}, &(i32){0}, &n, &(i32){1}, &n, dwork, &n,
                   &dwork[ir], &dwork[ii], &(i32){1}, &(i32){1}, z_dummy,
                   &(i32){1}, &im);

        i32 i_max = SLC_IDAMAX(&n, &dwork[ii], &one_i);
        if (dwork[ii + i_max - 1] == ZERO) {
            im = ir + SLC_IDAMAX(&n, &dwork[ir], &one_i) - 1;
            mx = fabs(dwork[im]);
            mn = mx;

            for (i32 i = ir; i < ir + n; i++) {
                mc = fabs(dwork[i]);
                if (mc < mn) {
                    mn = mc;
                    im = i;
                }
            }

            pr = dwork[im];
            mn = mx;
            dwork[im] = mx;
            is_loc = im;
            mx = pr;

            for (i32 i = ir; i < ir + n; i++) {
                mc = fabs(dwork[i]);
                if (mc < mn) {
                    mn = mc;
                    im = i;
                }
            }

            sm = pr + dwork[im];
            pr = pr * dwork[im];
            dwork[is_loc] = mx;
        } else {
            i32 i = ii;
            bool fc = false;

            while (i < ii + n) {
                if (dwork[i] != ZERO) {
                    mc = SLC_DLAPY2(&dwork[i - n], &dwork[i]);
                    if (!fc) {
                        fc = true;
                        im = i;
                        mn = mc;
                    } else if (mc < mn) {
                        mn = mc;
                        im = i;
                    }
                    i += 2;
                } else {
                    i += 1;
                }
            }

            sm = TWO * dwork[im - n];
            pr = mn * mn;
        }

        p1 = p1 + ((e1 - sm) * e1 + pr) / e2;
        p2 = p2 + e1 - sm;

        SLC_DLARTG(&p2, &p3, c2, s2, &e1);
        SLC_DLARTG(&p1, &e1, c1, s1, &e2);
    }
}
