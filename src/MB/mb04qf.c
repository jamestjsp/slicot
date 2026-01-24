/*
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <stdlib.h>

void mb04qf(
    const char *direct, const char *storev, const char *storew,
    i32 n, i32 k,
    f64 *v, i32 ldv, f64 *w, i32 ldw,
    const f64 *cs, const f64 *tau,
    f64 *rs, i32 ldrs, f64 *t, i32 ldt,
    f64 *dwork)
{
    i32 i, j;
    i32 int1 = 1;
    f64 dbl0 = 0.0, dbl1 = 1.0;
    bool lcolv, lcolw;
    f64 taui, vii, wii, cm1;
    i32 k2, pr1, pr2, pr3, ps1, ps2, ps3;
    i32 pt11, pt12, pt13, pt21, pt22, pt23, pt31, pt32, pt33;
    i32 ni1, im1, im2;

    if (n == 0) {
        return;
    }

    lcolv = (storev[0] == 'C' || storev[0] == 'c');
    lcolw = (storew[0] == 'C' || storew[0] == 'c');

    k2 = k + k;
    pr1 = 0;
    pr2 = pr1 + k;
    pr3 = pr2 + k;
    ps1 = pr3 + k;
    ps2 = ps1 + k;
    ps3 = ps2 + k;

    pt11 = 0;
    pt12 = pt11 + k;
    pt13 = pt12 + k;
    pt21 = pt13 + k;
    pt22 = pt21 + k;
    pt23 = pt22 + k;
    pt31 = pt23 + k;
    pt32 = pt31 + k;
    pt33 = pt32 + k;

    for (i = 0; i < k; i++) {
        taui = tau[i];
        vii = v[i + i * ldv];
        v[i + i * ldv] = dbl1;
        wii = w[i + i * ldw];
        w[i + i * ldw] = dbl1;

        ni1 = n - i;
        im1 = i;
        im2 = (i > 1) ? i - 1 : 0;

        if (wii == 0.0) {
            for (j = 0; j <= i; j++) {
                t[j + (pt11 + i) * ldt] = dbl0;
            }
            for (j = 0; j < i; j++) {
                t[j + (pt21 + i) * ldt] = dbl0;
            }
            for (j = 0; j < i; j++) {
                t[j + (pt31 + i) * ldt] = dbl0;
            }
            for (j = 0; j < i; j++) {
                rs[j + (ps1 + i) * ldrs] = dbl0;
            }
        } else {
            f64 neg_wii = -wii;

            if (lcolv && lcolw) {
                SLC_DGEMV("T", &ni1, &im1, &neg_wii, &w[i], &ldw,
                          &w[i + i * ldw], &int1, &dbl0, dwork, &int1);
                SLC_DGEMV("T", &ni1, &im1, &neg_wii, &v[i], &ldv,
                          &w[i + i * ldw], &int1, &dbl0, &dwork[k], &int1);
            } else if (lcolv) {
                SLC_DGEMV("N", &im1, &ni1, &neg_wii, &w[i * ldw], &ldw,
                          &w[i + i * ldw], &ldw, &dbl0, dwork, &int1);
                SLC_DGEMV("T", &ni1, &im1, &neg_wii, &v[i], &ldv,
                          &w[i + i * ldw], &ldw, &dbl0, &dwork[k], &int1);
            } else if (lcolw) {
                SLC_DGEMV("T", &ni1, &im1, &neg_wii, &w[i], &ldw,
                          &w[i + i * ldw], &int1, &dbl0, dwork, &int1);
                SLC_DGEMV("N", &im1, &ni1, &neg_wii, &v[i * ldv], &ldv,
                          &w[i + i * ldw], &int1, &dbl0, &dwork[k], &int1);
            } else {
                SLC_DGEMV("N", &im1, &ni1, &neg_wii, &w[i * ldw], &ldw,
                          &w[i + i * ldw], &ldw, &dbl0, dwork, &int1);
                SLC_DGEMV("N", &im1, &ni1, &neg_wii, &v[i * ldv], &ldv,
                          &w[i + i * ldw], &ldw, &dbl0, &dwork[k], &int1);
            }

            SLC_DCOPY(&im1, dwork, &int1, &t[(pt11 + i) * ldt], &int1);
            SLC_DTRMV("U", "N", "N", &im1, &t[(pt11) * ldt], &ldt, &t[(pt11 + i) * ldt], &int1);
            SLC_DCOPY(&im1, &dwork[k], &int1, &t[(pt13 + i) * ldt], &int1);
            SLC_DTRMV("U", "N", "N", &im1, &t[(pt13) * ldt], &ldt, &t[(pt13 + i) * ldt], &int1);
            SLC_DAXPY(&im1, &dbl1, &t[(pt13 + i) * ldt], &int1, &t[(pt11 + i) * ldt], &int1);
            t[i + (pt11 + i) * ldt] = -wii;

            if (i > 0) {
                SLC_DCOPY(&im2, &dwork[1], &int1, &t[(pt21 + i) * ldt], &int1);
                SLC_DTRMV("U", "N", "N", &im2, &t[(pt21 + 1) * ldt], &ldt, &t[(pt21 + i) * ldt], &int1);
                t[i - 1 + (pt21 + i) * ldt] = dbl0;
                SLC_DCOPY(&im1, &dwork[k], &int1, &t[(pt23 + i) * ldt], &int1);
                SLC_DTRMV("U", "N", "N", &im1, &t[(pt23) * ldt], &ldt, &t[(pt23 + i) * ldt], &int1);
                SLC_DAXPY(&im1, &dbl1, &t[(pt23 + i) * ldt], &int1, &t[(pt21 + i) * ldt], &int1);

                SLC_DCOPY(&im2, &dwork[1], &int1, &t[(pt31 + i) * ldt], &int1);
                SLC_DTRMV("U", "N", "N", &im2, &t[(pt31 + 1) * ldt], &ldt, &t[(pt31 + i) * ldt], &int1);
                t[i - 1 + (pt31 + i) * ldt] = dbl0;
                SLC_DCOPY(&im1, &dwork[k], &int1, &t[(pt33 + i) * ldt], &int1);
                SLC_DTRMV("U", "N", "N", &im1, &t[(pt33) * ldt], &ldt, &t[(pt33 + i) * ldt], &int1);
                SLC_DAXPY(&im1, &dbl1, &t[(pt33 + i) * ldt], &int1, &t[(pt31 + i) * ldt], &int1);

                SLC_DCOPY(&im2, &dwork[1], &int1, &rs[(ps1 + i) * ldrs], &int1);
                SLC_DTRMV("U", "N", "N", &im2, &rs[(ps1 + 1) * ldrs], &ldrs, &rs[(ps1 + i) * ldrs], &int1);
                rs[i - 1 + (ps1 + i) * ldrs] = dbl0;
                SLC_DCOPY(&im1, &dwork[k], &int1, &rs[(ps3 + i) * ldrs], &int1);
                SLC_DTRMV("U", "N", "N", &im1, &rs[(ps3) * ldrs], &ldrs, &rs[(ps3 + i) * ldrs], &int1);
                SLC_DAXPY(&im1, &dbl1, &rs[(ps3 + i) * ldrs], &int1, &rs[(ps1 + i) * ldrs], &int1);
            }
        }

        cm1 = cs[2*i] - dbl1;
        if (lcolw) {
            SLC_DCOPY(&(i32){i + 1}, &w[i], &ldw, dwork, &int1);
        } else {
            SLC_DCOPY(&(i32){i + 1}, &w[i * ldw], &int1, dwork, &int1);
        }
        if (lcolv) {
            SLC_DCOPY(&im1, &v[i], &ldv, &dwork[k], &int1);
        } else {
            SLC_DCOPY(&im1, &v[i * ldv], &int1, &dwork[k], &int1);
        }

        i32 ip1 = i + 1;
        SLC_DCOPY(&ip1, dwork, &int1, &rs[(pr1 + i) * ldrs], &int1);
        SLC_DTRMV("U", "N", "N", &ip1, &t[(pt11) * ldt], &ldt, &rs[(pr1 + i) * ldrs], &int1);
        SLC_DCOPY(&im1, &dwork[k], &int1, &t[(pt13 + i) * ldt], &int1);
        SLC_DTRMV("U", "N", "N", &im1, &t[(pt13) * ldt], &ldt, &t[(pt13 + i) * ldt], &int1);
        SLC_DAXPY(&im1, &dbl1, &t[(pt13 + i) * ldt], &int1, &rs[(pr1 + i) * ldrs], &int1);

        SLC_DCOPY(&im1, &dwork[1], &int1, &rs[(pr2 + i) * ldrs], &int1);
        SLC_DTRMV("U", "N", "N", &im1, &t[(pt21 + 1) * ldt], &ldt, &rs[(pr2 + i) * ldrs], &int1);
        SLC_DCOPY(&im1, &dwork[k], &int1, &t[(pt23 + i) * ldt], &int1);
        SLC_DTRMV("U", "N", "N", &im1, &t[(pt23) * ldt], &ldt, &t[(pt23 + i) * ldt], &int1);
        SLC_DAXPY(&im1, &dbl1, &t[(pt23 + i) * ldt], &int1, &rs[(pr2 + i) * ldrs], &int1);

        SLC_DCOPY(&im1, &dwork[1], &int1, &rs[(pr3 + i) * ldrs], &int1);
        SLC_DTRMV("U", "N", "N", &im1, &t[(pt31 + 1) * ldt], &ldt, &rs[(pr3 + i) * ldrs], &int1);
        SLC_DCOPY(&im1, &dwork[k], &int1, &t[(pt33 + i) * ldt], &int1);
        SLC_DTRMV("U", "N", "N", &im1, &t[(pt33) * ldt], &ldt, &t[(pt33 + i) * ldt], &int1);
        SLC_DAXPY(&im1, &dbl1, &t[(pt33 + i) * ldt], &int1, &rs[(pr3 + i) * ldrs], &int1);

        SLC_DCOPY(&im1, &dwork[1], &int1, &rs[(ps2 + i) * ldrs], &int1);
        SLC_DTRMV("U", "N", "N", &im1, &rs[(ps1 + 1) * ldrs], &ldrs, &rs[(ps2 + i) * ldrs], &int1);
        SLC_DTRMV("U", "N", "N", &im1, &rs[(ps3) * ldrs], &ldrs, &dwork[k], &int1);
        SLC_DAXPY(&im1, &dbl1, &dwork[k], &int1, &rs[(ps2 + i) * ldrs], &int1);
        rs[i + (ps2 + i) * ldrs] = -cs[2*i + 1];

        SLC_DCOPY(&im1, &rs[(ps2 + i) * ldrs], &int1, &t[(pt12 + i) * ldt], &int1);
        SLC_DSCAL(&im1, &cm1, &rs[(ps2 + i) * ldrs], &int1);
        f64 s_val = cs[2*i + 1];
        SLC_DSCAL(&im1, &s_val, &t[(pt12 + i) * ldt], &int1);
        SLC_DCOPY(&im1, &t[(pt12 + i) * ldt], &int1, &t[(pt22 + i) * ldt], &int1);
        SLC_DTRMV("U", "N", "N", &im1, &rs[(pr1) * ldrs], &ldrs, &t[(pt12 + i) * ldt], &int1);
        t[i + (pt12 + i) * ldt] = dbl0;
        SLC_DAXPY(&ip1, &cm1, &rs[(pr1 + i) * ldrs], &int1, &t[(pt12 + i) * ldt], &int1);

        if (i > 0) {
            SLC_DCOPY(&im2, &t[1 + (pt22 + i) * ldt], &int1, &t[(pt32 + i) * ldt], &int1);
        }
        SLC_DTRMV("U", "N", "U", &im1, &rs[(pr2) * ldrs], &ldrs, &t[(pt22 + i) * ldt], &int1);
        SLC_DAXPY(&im1, &cm1, &rs[(pr2 + i) * ldrs], &int1, &t[(pt22 + i) * ldt], &int1);
        t[i + (pt22 + i) * ldt] = cm1;

        if (i > 0) {
            SLC_DTRMV("U", "N", "N", &im2, &rs[(pr3 + 1) * ldrs], &ldrs, &t[(pt32 + i) * ldt], &int1);
            t[i - 1 + (pt32 + i) * ldt] = dbl0;
            SLC_DAXPY(&im1, &cm1, &rs[(pr3 + i) * ldrs], &int1, &t[(pt32 + i) * ldt], &int1);
        }

        if (taui == 0.0) {
            for (j = 0; j <= i; j++) {
                t[j + (pt13 + i) * ldt] = dbl0;
            }
            for (j = 0; j <= i; j++) {
                t[j + (pt23 + i) * ldt] = dbl0;
            }
            for (j = 0; j <= i; j++) {
                t[j + (pt33 + i) * ldt] = dbl0;
            }
            for (j = 0; j <= i; j++) {
                rs[j + (ps3 + i) * ldrs] = dbl0;
            }
        } else {
            f64 neg_taui = -taui;

            if (lcolv && lcolw) {
                SLC_DGEMV("T", &ni1, &ip1, &neg_taui, &w[i], &ldw,
                          &v[i + i * ldv], &int1, &dbl0, dwork, &int1);
                SLC_DGEMV("T", &ni1, &im1, &neg_taui, &v[i], &ldv,
                          &v[i + i * ldv], &int1, &dbl0, &dwork[k2], &int1);
            } else if (lcolv) {
                SLC_DGEMV("N", &ip1, &ni1, &neg_taui, &w[i * ldw], &ldw,
                          &v[i + i * ldv], &int1, &dbl0, dwork, &int1);
                SLC_DGEMV("T", &ni1, &im1, &neg_taui, &v[i], &ldv,
                          &v[i + i * ldv], &int1, &dbl0, &dwork[k2], &int1);
            } else if (lcolw) {
                SLC_DGEMV("T", &ni1, &ip1, &neg_taui, &w[i], &ldw,
                          &v[i + i * ldv], &ldv, &dbl0, dwork, &int1);
                SLC_DGEMV("N", &im1, &ni1, &neg_taui, &v[i * ldv], &ldv,
                          &v[i + i * ldv], &ldv, &dbl0, &dwork[k2], &int1);
            } else {
                SLC_DGEMV("N", &ip1, &ni1, &neg_taui, &w[i * ldw], &ldw,
                          &v[i + i * ldv], &ldv, &dbl0, dwork, &int1);
                SLC_DGEMV("N", &im1, &ni1, &neg_taui, &v[i * ldv], &ldv,
                          &v[i + i * ldv], &ldv, &dbl0, &dwork[k2], &int1);
            }

            SLC_DCOPY(&im1, &dwork[k2], &int1, &t[(pt13 + i) * ldt], &int1);
            SLC_DTRMV("U", "N", "N", &im1, &t[(pt13) * ldt], &ldt, &t[(pt13 + i) * ldt], &int1);
            t[i + (pt13 + i) * ldt] = dbl0;
            SLC_DCOPY(&ip1, dwork, &int1, &dwork[k], &int1);
            SLC_DTRMV("U", "N", "N", &ip1, &t[(pt11) * ldt], &ldt, &dwork[k], &int1);
            SLC_DAXPY(&ip1, &dbl1, &dwork[k], &int1, &t[(pt13 + i) * ldt], &int1);
            SLC_DAXPY(&ip1, &neg_taui, &t[(pt12 + i) * ldt], &int1, &t[(pt13 + i) * ldt], &int1);

            SLC_DCOPY(&im1, &dwork[k2], &int1, &t[(pt23 + i) * ldt], &int1);
            SLC_DTRMV("U", "N", "N", &im1, &t[(pt23) * ldt], &ldt, &t[(pt23 + i) * ldt], &int1);
            t[i + (pt23 + i) * ldt] = dbl0;
            SLC_DCOPY(&im1, &dwork[1], &int1, &dwork[k], &int1);
            SLC_DTRMV("U", "N", "N", &im1, &t[(pt21 + 1) * ldt], &ldt, &dwork[k], &int1);
            SLC_DAXPY(&im1, &dbl1, &dwork[k], &int1, &t[(pt23 + i) * ldt], &int1);
            SLC_DAXPY(&ip1, &neg_taui, &t[(pt22 + i) * ldt], &int1, &t[(pt23 + i) * ldt], &int1);

            SLC_DCOPY(&im1, &dwork[k2], &int1, &t[(pt33 + i) * ldt], &int1);
            SLC_DTRMV("U", "N", "N", &im1, &t[(pt33) * ldt], &ldt, &t[(pt33 + i) * ldt], &int1);
            SLC_DCOPY(&im1, &dwork[1], &int1, &dwork[k], &int1);
            SLC_DTRMV("U", "N", "N", &im1, &t[(pt31 + 1) * ldt], &ldt, &dwork[k], &int1);
            SLC_DAXPY(&im1, &dbl1, &dwork[k], &int1, &t[(pt33 + i) * ldt], &int1);
            SLC_DAXPY(&im1, &neg_taui, &t[(pt32 + i) * ldt], &int1, &t[(pt33 + i) * ldt], &int1);
            t[i + (pt33 + i) * ldt] = -taui;

            SLC_DCOPY(&im1, &dwork[k2], &int1, &rs[(ps3 + i) * ldrs], &int1);
            SLC_DTRMV("U", "N", "N", &im1, &rs[(ps3) * ldrs], &ldrs, &rs[(ps3 + i) * ldrs], &int1);
            SLC_DTRMV("U", "N", "N", &im1, &rs[(ps1 + 1) * ldrs], &ldrs, &dwork[1], &int1);
            SLC_DAXPY(&im1, &dbl1, &dwork[1], &int1, &rs[(ps3 + i) * ldrs], &int1);
            rs[i + (ps3 + i) * ldrs] = dbl0;
            SLC_DAXPY(&ip1, &neg_taui, &rs[(ps2 + i) * ldrs], &int1, &rs[(ps3 + i) * ldrs], &int1);
        }

        w[i + i * ldw] = wii;
        v[i + i * ldv] = vii;
    }
}
