/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <ctype.h>
#include <stdbool.h>

void mb02dd(const char* job, const char* typet, i32 k, i32 m, i32 n,
            f64* ta, i32 ldta, f64* t, i32 ldt, f64* g, i32 ldg,
            f64* r, i32 ldr, f64* l, i32 ldl, f64* cs, i32 lcs,
            f64* dwork, i32 ldwork, i32* info)
{
    const f64 zero = 0.0;
    const f64 one = 1.0;

    char job_u = (char)toupper((unsigned char)job[0]);
    char typet_u = (char)toupper((unsigned char)typet[0]);

    bool compl = (job_u == 'A');
    bool compg = (job_u == 'R') || compl;
    bool isrow = (typet_u == 'R');

    *info = 0;

    if (!compg && job_u != 'O') {
        *info = -1;
    } else if (!isrow && typet_u != 'C') {
        *info = -2;
    } else if (k < 0) {
        *info = -3;
    } else if (m < 0) {
        *info = -4;
    } else if (n < 0) {
        *info = -5;
    } else if (ldta < 1 || (isrow && ldta < k) || (!isrow && ldta < m * k)) {
        *info = -7;
    } else if (ldt < 1 || (isrow && ldt < k) || (!isrow && ldt < n * k)) {
        *info = -9;
    } else if ((compg && ((isrow && ldg < 2 * k) || (!isrow && ldg < (n + m) * k))) || ldg < 1) {
        *info = -11;
    } else if ((isrow && ldr < (n + m) * k) || (!isrow && ldr < m * k) || ldr < 1) {
        *info = -13;
    } else if ((compl && ((isrow && ldl < m * k) || (!isrow && ldl < (n + m) * k))) || ldl < 1) {
        *info = -15;
    } else if (lcs < 3 * (n + m - 1) * k) {
        *info = -17;
    } else if (ldwork < ((n + m > 1) ? (n + m - 1) * k : 1)) {
        dwork[0] = (f64)((n + m > 1) ? (n + m - 1) * k : 1);
        *info = -19;
    }

    if (*info != 0) {
        return;
    }

    i32 mink = k;
    if (n < mink) mink = n;
    if (m < mink) mink = m;
    if (mink == 0) {
        dwork[0] = one;
        return;
    }

    i32 maxwrk = 1;
    i32 ierr = 0;

    if (isrow) {
        i32 mk = m * k;
        SLC_DTRSM("L", "U", "T", "N", &k, &mk, &one, t, &ldt, ta, &ldta);

        if (compg) {
            i32 nk = n * k;
            SLC_DLASET("A", &k, &mk, &zero, &zero, &g[0 + nk * ldg], &ldg);
            if (m >= n - 1 && n > 1) {
                i32 nm1k = (n - 1) * k;
                SLC_DLACPY("A", &k, &nm1k, &g[k + k * ldg], &ldg, &g[k + (m + 1) * k * ldg], &ldg);
            } else {
                for (i32 ii = n * k - 1; ii >= k; ii--) {
                    SLC_DCOPY(&k, &g[k + ii * ldg], &(i32){1}, &g[k + (m * k + ii) * ldg], &(i32){1});
                }
            }
            SLC_DLASET("A", &k, &mk, &zero, &zero, &g[k + k * ldg], &ldg);
        }

        SLC_DLACPY("A", &k, &mk, ta, &ldta, r, &ldr);

        for (i32 i = 2; i <= n; i++) {
            i32 startr = (i - 1) * k;
            i32 startt = 3 * (i - 2) * k;
            i32 mm1k = (m - 1) * k;
            SLC_DLACPY("A", &k, &mm1k, &r[startr - k + 0 * ldr], &ldr, &r[startr + k * ldr], &ldr);

            mb02cy("R", "N", k, k, mk, k, &r[startr + 0 * ldr], ldr, ta, ldta,
                   &t[0 + startr * ldt], ldt, &cs[startt], 3 * k, dwork, ldwork, &ierr);
            maxwrk = (i32)dwork[0] > maxwrk ? (i32)dwork[0] : maxwrk;
        }

        for (i32 i = 1; i <= m; i++) {
            i32 startt = 3 * (n + i - 2) * k;
            i32 starti = (m - i + 1) * k;
            i32 startr = (n + i - 1) * k;

            if (i == 1) {
                i32 mm1k = (m - 1) * k;
                SLC_DLACPY("A", &k, &mm1k, &r[startr - k + 0 * ldr], &ldr, &r[startr + k * ldr], &ldr);
            } else {
                i32 mip1k = (m - i + 1) * k;
                i32 im2k = (i - 2) * k;
                SLC_DLACPY("U", &k, &mip1k, &r[startr - k + im2k * ldr], &ldr,
                           &r[startr + (i - 1) * k * ldr], &ldr);
            }

            i32 im1k = (i - 1) * k;
            mb02cx("R", k, k, k, &r[startr + im1k * ldr], ldr, &ta[0 + im1k * ldta], ldta,
                   &cs[startt], 3 * k, dwork, ldwork, &ierr);

            if (ierr != 0) {
                *info = 1;
                return;
            }

            maxwrk = (i32)dwork[0] > maxwrk ? (i32)dwork[0] : maxwrk;

            if (m > i) {
                i32 mik = (m - i) * k;
                i32 ik = i * k;
                mb02cy("R", "N", k, k, mik, k, &r[startr + ik * ldr], ldr, &ta[0 + ik * ldta], ldta,
                       &ta[0 + im1k * ldta], ldta, &cs[startt], 3 * k, dwork, ldwork, &ierr);
                maxwrk = (i32)dwork[0] > maxwrk ? (i32)dwork[0] : maxwrk;
            }

            if (compg) {
                mb02cy("R", "T", k, k, k, k, &g[k + 0 * ldg], ldg, &g[0 + startr * ldg], ldg,
                       &ta[0 + im1k * ldta], ldta, &cs[startt], 3 * k, dwork, ldwork, &ierr);
                maxwrk = (i32)dwork[0] > maxwrk ? (i32)dwork[0] : maxwrk;

                i32 npi_m1 = (n + i - 1) * k;
                mb02cy("R", "N", k, k, npi_m1, k, &g[k + starti * ldg], ldg, g, ldg,
                       &ta[0 + im1k * ldta], ldta, &cs[startt], 3 * k, dwork, ldwork, &ierr);
                maxwrk = (i32)dwork[0] > maxwrk ? (i32)dwork[0] : maxwrk;

                if (compl) {
                    i32 npim1k = (n + i - 1) * k;
                    SLC_DLACPY("A", &k, &npim1k, &g[k + starti * ldg], &ldg, &l[im1k + 0 * ldl], &ldl);
                    SLC_DLACPY("L", &k, &k, &g[k + 0 * ldg], &ldg, &l[im1k + startr * ldl], &ldl);
                }
            }
        }
    } else {
        i32 mk = m * k;
        SLC_DTRSM("R", "L", "T", "N", &mk, &k, &one, t, &ldt, ta, &ldta);

        if (compg) {
            i32 nk = n * k;
            SLC_DLASET("A", &mk, &k, &zero, &zero, &g[nk + 0 * ldg], &ldg);
            if (m >= n - 1 && n > 1) {
                i32 nm1k = (n - 1) * k;
                SLC_DLACPY("A", &nm1k, &k, &g[k + k * ldg], &ldg, &g[(m + 1) * k + k * ldg], &ldg);
            } else {
                for (i32 i = 0; i < k; i++) {
                    for (i32 j = n * k - 1; j >= k; j--) {
                        g[j + m * k + (k + i) * ldg] = g[j + (k + i) * ldg];
                    }
                }
            }
            SLC_DLASET("A", &mk, &k, &zero, &zero, &g[k + k * ldg], &ldg);
        }

        SLC_DLACPY("A", &mk, &k, ta, &ldta, r, &ldr);

        for (i32 i = 2; i <= n; i++) {
            i32 startr = (i - 1) * k;
            i32 startt = 3 * (i - 2) * k;
            i32 mm1k = (m - 1) * k;
            SLC_DLACPY("A", &mm1k, &k, &r[0 + (startr - k) * ldr], &ldr, &r[k + startr * ldr], &ldr);

            mb02cy("C", "N", k, k, mk, k, &r[0 + startr * ldr], ldr, ta, ldta,
                   &t[startr + 0 * ldt], ldt, &cs[startt], 3 * k, dwork, ldwork, &ierr);
            maxwrk = (i32)dwork[0] > maxwrk ? (i32)dwork[0] : maxwrk;
        }

        for (i32 i = 1; i <= m; i++) {
            i32 startt = 3 * (n + i - 2) * k;
            i32 starti = (m - i + 1) * k;
            i32 startr = (n + i - 1) * k;

            if (i == 1) {
                i32 mm1k = (m - 1) * k;
                SLC_DLACPY("A", &mm1k, &k, &r[0 + (startr - k) * ldr], &ldr, &r[k + startr * ldr], &ldr);
            } else {
                i32 mip1k = (m - i + 1) * k;
                i32 im2k = (i - 2) * k;
                SLC_DLACPY("L", &mip1k, &k, &r[im2k + (startr - k) * ldr], &ldr,
                           &r[(i - 1) * k + startr * ldr], &ldr);
            }

            i32 im1k = (i - 1) * k;
            mb02cx("C", k, k, k, &r[im1k + startr * ldr], ldr, &ta[im1k + 0 * ldta], ldta,
                   &cs[startt], 3 * k, dwork, ldwork, &ierr);

            if (ierr != 0) {
                *info = 1;
                return;
            }

            maxwrk = (i32)dwork[0] > maxwrk ? (i32)dwork[0] : maxwrk;

            if (m > i) {
                i32 mik = (m - i) * k;
                i32 ik = i * k;
                mb02cy("C", "N", k, k, mik, k, &r[ik + startr * ldr], ldr, &ta[ik + 0 * ldta], ldta,
                       &ta[im1k + 0 * ldta], ldta, &cs[startt], 3 * k, dwork, ldwork, &ierr);
                maxwrk = (i32)dwork[0] > maxwrk ? (i32)dwork[0] : maxwrk;
            }

            if (compg) {
                mb02cy("C", "T", k, k, k, k, &g[0 + k * ldg], ldg, &g[startr + 0 * ldg], ldg,
                       &ta[im1k + 0 * ldta], ldta, &cs[startt], 3 * k, dwork, ldwork, &ierr);
                maxwrk = (i32)dwork[0] > maxwrk ? (i32)dwork[0] : maxwrk;

                i32 npi_m1 = (n + i - 1) * k;
                mb02cy("C", "N", k, k, npi_m1, k, &g[starti + k * ldg], ldg, g, ldg,
                       &ta[im1k + 0 * ldta], ldta, &cs[startt], 3 * k, dwork, ldwork, &ierr);
                maxwrk = (i32)dwork[0] > maxwrk ? (i32)dwork[0] : maxwrk;

                if (compl) {
                    i32 npim1k = (n + i - 1) * k;
                    SLC_DLACPY("A", &npim1k, &k, &g[starti + k * ldg], &ldg, &l[0 + im1k * ldl], &ldl);
                    SLC_DLACPY("U", &k, &k, &g[0 + k * ldg], &ldg, &l[startr + im1k * ldl], &ldl);
                }
            }
        }
    }

    dwork[0] = (f64)maxwrk;
}
