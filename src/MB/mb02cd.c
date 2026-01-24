/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <ctype.h>
#include <stdbool.h>

void mb02cd(const char* job, const char* typet, i32 k, i32 n,
            f64* t, i32 ldt, f64* g, i32 ldg, f64* r, i32 ldr,
            f64* l, i32 ldl, f64* cs, i32 lcs, f64* dwork, i32 ldwork,
            i32* info)
{
    const f64 zero = 0.0;
    const f64 one = 1.0;

    char job_u = (char)toupper((unsigned char)job[0]);
    char typet_u = (char)toupper((unsigned char)typet[0]);

    bool compl = (job_u == 'L') || (job_u == 'A');
    bool compg = (job_u == 'G') || (job_u == 'R') || compl;
    bool compr = (job_u == 'R') || (job_u == 'A') || (job_u == 'O');
    bool isrow = (typet_u == 'R');

    *info = 0;

    if (!compg && !compr) {
        *info = -1;
    } else if (!isrow && typet_u != 'C') {
        *info = -2;
    } else if (k < 0) {
        *info = -3;
    } else if (n < 0) {
        *info = -4;
    } else if (ldt < 1 || (isrow && ldt < k) || (!isrow && ldt < n * k)) {
        *info = -6;
    } else if (ldg < 1 || (compg && ((isrow && ldg < 2 * k) || (!isrow && ldg < n * k)))) {
        *info = -8;
    } else if (ldr < 1 || (compr && ldr < n * k)) {
        *info = -10;
    } else if (ldl < 1 || (compl && ldl < n * k)) {
        *info = -12;
    } else if (lcs < 3 * (n - 1) * k) {
        *info = -14;
    } else if (ldwork < (n > 1 ? (n - 1) * k : 1)) {
        dwork[0] = (f64)((n > 1) ? (n - 1) * k : 1);
        *info = -16;
    }

    if (*info != 0) {
        return;
    }

    i32 mink = (k < n) ? k : n;
    if (mink == 0) {
        dwork[0] = one;
        return;
    }

    i32 maxwrk = 1;
    i32 ierr = 0;

    if (isrow) {
        SLC_DPOTRF("U", &k, t, &ldt, &ierr);
        if (ierr != 0) {
            *info = 1;
            return;
        }

        if (n > 1) {
            i32 nm1k = (n - 1) * k;
            SLC_DTRSM("L", "U", "T", "N", &k, &nm1k, &one, t, &ldt, &t[0 + k * ldt], &ldt);
        }

        if (compg) {
            i32 twok = 2 * k;
            i32 nk = n * k;
            SLC_DLASET("A", &twok, &nk, &zero, &zero, g, &ldg);
            i32 ldgp1 = ldg + 1;
            for (i32 ii = 0; ii < k; ii++) {
                g[k + ii * ldgp1] = one;
            }
            SLC_DTRSM("L", "U", "T", "N", &k, &k, &one, t, &ldt, &g[k + 0 * ldg], &ldg);
            if (n > 1) {
                i32 nm1k = (n - 1) * k;
                SLC_DLACPY("U", &k, &nm1k, t, &ldt, &g[k + k * ldg], &ldg);
            }
            SLC_DLACPY("L", &k, &k, &g[k + 0 * ldg], &ldg, g, &ldg);
        }

        if (compl) {
            SLC_DLACPY("L", &k, &k, &g[k + 0 * ldg], &ldg, l, &ldl);
        }

        if (compr) {
            i32 nk = n * k;
            SLC_DLACPY("U", &k, &nk, t, &ldt, r, &ldr);
        }

        if (compg) {
            for (i32 i = 1; i < n; i++) {
                i32 startr = i * k;
                i32 starti = (n - i) * k;
                i32 startt = 3 * (i - 1) * k;

                mb02cx("R", k, k, k, &g[k + k * ldg], ldg, &t[0 + startr * ldt], ldt,
                       &cs[startt], 3 * k, dwork, ldwork, &ierr);

                if (ierr != 0) {
                    *info = 1;
                    return;
                }

                maxwrk = (i32)dwork[0] > maxwrk ? (i32)dwork[0] : maxwrk;

                if (n > i + 1) {
                    i32 nmik = (n - i - 1) * k;
                    mb02cy("R", "N", k, k, nmik, k, &g[k + 2 * k * ldg], ldg,
                           &t[0 + (startr + k) * ldt], ldt, &t[0 + startr * ldt], ldt,
                           &cs[startt], 3 * k, dwork, ldwork, &ierr);
                    maxwrk = (i32)dwork[0] > maxwrk ? (i32)dwork[0] : maxwrk;
                }

                if (compr) {
                    i32 nmip1k = (n - i) * k;
                    SLC_DLACPY("U", &k, &nmip1k, &g[k + k * ldg], &ldg, &r[startr + startr * ldr], &ldr);
                }

                SLC_DLASET("A", &k, &k, &zero, &zero, &g[k + starti * ldg], &ldg);
                mb02cy("R", "T", k, k, k, k, &g[k + 0 * ldg], ldg, &g[0 + startr * ldg], ldg,
                       &t[0 + startr * ldt], ldt, &cs[startt], 3 * k, dwork, ldwork, &ierr);
                maxwrk = (i32)dwork[0] > maxwrk ? (i32)dwork[0] : maxwrk;

                i32 ik = i * k;
                mb02cy("R", "N", k, k, ik, k, &g[k + starti * ldg], ldg, g, ldg,
                       &t[0 + startr * ldt], ldt, &cs[startt], 3 * k, dwork, ldwork, &ierr);
                maxwrk = (i32)dwork[0] > maxwrk ? (i32)dwork[0] : maxwrk;

                if (compl) {
                    i32 im1k = (i) * k;
                    SLC_DLACPY("A", &k, &im1k, &g[k + starti * ldg], &ldg, &l[startr + 0 * ldl], &ldl);
                    SLC_DLACPY("L", &k, &k, &g[k + 0 * ldg], &ldg, &l[startr + im1k * ldl], &ldl);
                }
            }
        } else {
            if (n > 1) {
                i32 nm1k = (n - 1) * k;
                SLC_DLACPY("U", &k, &nm1k, t, &ldt, &r[k + k * ldr], &ldr);
            }

            for (i32 i = 1; i < n; i++) {
                i32 startr = i * k;
                i32 startt = 3 * (i - 1) * k;

                mb02cx("R", k, k, k, &r[startr + startr * ldr], ldr, &t[0 + startr * ldt], ldt,
                       &cs[startt], 3 * k, dwork, ldwork, &ierr);

                if (ierr != 0) {
                    *info = 1;
                    return;
                }

                maxwrk = (i32)dwork[0] > maxwrk ? (i32)dwork[0] : maxwrk;

                if (n > i + 1) {
                    i32 nmik = (n - i - 1) * k;
                    mb02cy("R", "N", k, k, nmik, k, &r[startr + (startr + k) * ldr], ldr,
                           &t[0 + (startr + k) * ldt], ldt, &t[0 + startr * ldt], ldt,
                           &cs[startt], 3 * k, dwork, ldwork, &ierr);
                    maxwrk = (i32)dwork[0] > maxwrk ? (i32)dwork[0] : maxwrk;

                    i32 nmik2 = (n - i - 1) * k;
                    SLC_DLACPY("U", &k, &nmik2, &r[startr + startr * ldr], &ldr,
                               &r[startr + k + (startr + k) * ldr], &ldr);
                }
            }
        }
    } else {
        SLC_DPOTRF("L", &k, t, &ldt, &ierr);
        if (ierr != 0) {
            *info = 1;
            return;
        }

        if (n > 1) {
            i32 nm1k = (n - 1) * k;
            SLC_DTRSM("R", "L", "T", "N", &nm1k, &k, &one, t, &ldt, &t[k + 0 * ldt], &ldt);
        }

        if (compg) {
            i32 nk = n * k;
            i32 twok = 2 * k;
            SLC_DLASET("A", &nk, &twok, &zero, &zero, g, &ldg);
            i32 ldgp1 = ldg + 1;
            for (i32 ii = 0; ii < k; ii++) {
                g[ii * ldgp1 + k * ldg] = one;
            }
            SLC_DTRSM("R", "L", "T", "N", &k, &k, &one, t, &ldt, &g[0 + k * ldg], &ldg);
            if (n > 1) {
                i32 nm1k = (n - 1) * k;
                SLC_DLACPY("L", &nm1k, &k, t, &ldt, &g[k + k * ldg], &ldg);
            }
            SLC_DLACPY("U", &k, &k, &g[0 + k * ldg], &ldg, g, &ldg);
        }

        if (compl) {
            SLC_DLACPY("U", &k, &k, &g[0 + k * ldg], &ldg, l, &ldl);
        }

        if (compr) {
            i32 nk = n * k;
            SLC_DLACPY("L", &nk, &k, t, &ldt, r, &ldr);
        }

        if (compg) {
            for (i32 i = 1; i < n; i++) {
                i32 startr = i * k;
                i32 starti = (n - i) * k;
                i32 startt = 3 * (i - 1) * k;

                mb02cx("C", k, k, k, &g[k + k * ldg], ldg, &t[startr + 0 * ldt], ldt,
                       &cs[startt], 3 * k, dwork, ldwork, &ierr);

                if (ierr != 0) {
                    *info = 1;
                    return;
                }

                maxwrk = (i32)dwork[0] > maxwrk ? (i32)dwork[0] : maxwrk;

                if (n > i + 1) {
                    i32 nmik = (n - i - 1) * k;
                    mb02cy("C", "N", k, k, nmik, k, &g[2 * k + k * ldg], ldg,
                           &t[startr + k + 0 * ldt], ldt, &t[startr + 0 * ldt], ldt,
                           &cs[startt], 3 * k, dwork, ldwork, &ierr);
                    maxwrk = (i32)dwork[0] > maxwrk ? (i32)dwork[0] : maxwrk;
                }

                if (compr) {
                    i32 nmip1k = (n - i) * k;
                    SLC_DLACPY("L", &nmip1k, &k, &g[k + k * ldg], &ldg, &r[startr + startr * ldr], &ldr);
                }

                SLC_DLASET("A", &k, &k, &zero, &zero, &g[starti + k * ldg], &ldg);
                mb02cy("C", "T", k, k, k, k, &g[0 + k * ldg], ldg, &g[startr + 0 * ldg], ldg,
                       &t[startr + 0 * ldt], ldt, &cs[startt], 3 * k, dwork, ldwork, &ierr);
                maxwrk = (i32)dwork[0] > maxwrk ? (i32)dwork[0] : maxwrk;

                i32 ik = i * k;
                mb02cy("C", "N", k, k, ik, k, &g[starti + k * ldg], ldg, g, ldg,
                       &t[startr + 0 * ldt], ldt, &cs[startt], 3 * k, dwork, ldwork, &ierr);
                maxwrk = (i32)dwork[0] > maxwrk ? (i32)dwork[0] : maxwrk;

                if (compl) {
                    i32 im1k = (i) * k;
                    SLC_DLACPY("A", &im1k, &k, &g[starti + k * ldg], &ldg, &l[0 + startr * ldl], &ldl);
                    SLC_DLACPY("U", &k, &k, &g[0 + k * ldg], &ldg, &l[im1k + startr * ldl], &ldl);
                }
            }
        } else {
            if (n > 1) {
                i32 nm1k = (n - 1) * k;
                SLC_DLACPY("L", &nm1k, &k, t, &ldt, &r[k + k * ldr], &ldr);
            }

            for (i32 i = 1; i < n; i++) {
                i32 startr = i * k;
                i32 startt = 3 * (i - 1) * k;

                mb02cx("C", k, k, k, &r[startr + startr * ldr], ldr, &t[startr + 0 * ldt], ldt,
                       &cs[startt], 3 * k, dwork, ldwork, &ierr);

                if (ierr != 0) {
                    *info = 1;
                    return;
                }

                maxwrk = (i32)dwork[0] > maxwrk ? (i32)dwork[0] : maxwrk;

                if (n > i + 1) {
                    i32 nmik = (n - i - 1) * k;
                    mb02cy("C", "N", k, k, nmik, k, &r[startr + k + startr * ldr], ldr,
                           &t[startr + k + 0 * ldt], ldt, &t[startr + 0 * ldt], ldt,
                           &cs[startt], 3 * k, dwork, ldwork, &ierr);
                    maxwrk = (i32)dwork[0] > maxwrk ? (i32)dwork[0] : maxwrk;

                    i32 nmik2 = (n - i - 1) * k;
                    SLC_DLACPY("L", &nmik2, &k, &r[startr + startr * ldr], &ldr,
                               &r[startr + k + (startr + k) * ldr], &ldr);
                }
            }
        }
    }

    dwork[0] = (f64)maxwrk;
}
