/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <ctype.h>
#include <stdbool.h>

void mb02kd(const char *ldblk, const char *trans, i32 k, i32 l, i32 m, i32 n,
            i32 r, f64 alpha, f64 beta, f64 *tc, i32 ldtc, f64 *tr, i32 ldtr,
            f64 *b, i32 ldb, f64 *c, i32 ldc, f64 *dwork, i32 ldwork, i32 *info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const f64 TWO = 2.0;
    const f64 THREE = 3.0;
    const f64 FOUR = 4.0;
    const f64 THOM50 = 950.0;
    i32 int1 = 1;

    char ldblk_c = (char)toupper((unsigned char)ldblk[0]);
    char trans_c = (char)toupper((unsigned char)trans[0]);

    bool fullc = (ldblk_c == 'C');
    bool ltran = (trans_c == 'T') || (trans_c == 'C');
    bool lmult = (alpha != ZERO);
    i32 mk = m * k;
    i32 nl = n * l;
    bool lquery = (ldwork == -1);

    *info = 0;

    if (!fullc && ldblk_c != 'R') {
        *info = -1;
    } else if (!ltran && trans_c != 'N') {
        *info = -2;
    } else if (k < 0) {
        *info = -3;
    } else if (l < 0) {
        *info = -4;
    } else if (m < 0) {
        *info = -5;
    } else if (n < 0) {
        *info = -6;
    } else if (r < 0) {
        *info = -7;
    } else if (lmult && fullc && ldtc < (mk > 1 ? mk : 1)) {
        *info = -11;
    } else if (lmult && !fullc && ldtc < ((m - 1) * k > 1 ? (m - 1) * k : 1)) {
        *info = -11;
    } else if (lmult && ldtr < (k > 1 ? k : 1)) {
        *info = -13;
    } else if (lmult && !ltran && ldb < (nl > 1 ? nl : 1)) {
        *info = -15;
    } else if (lmult && ltran && ldb < (mk > 1 ? mk : 1)) {
        *info = -15;
    } else if (!ltran && ldc < (mk > 1 ? mk : 1)) {
        *info = -17;
    } else if (ltran && ldc < (nl > 1 ? nl : 1)) {
        *info = -17;
    } else if (ldwork < 1 && !lquery) {
        dwork[0] = ONE;
        *info = -19;
    }

    if (*info != 0) {
        return;
    }

    i32 len = 1;
    i32 p = 0;
    while (len < m + n - 1) {
        len = len * 2;
        p = p + 1;
    }

    f64 coef = THREE * (f64)(m * n) * (f64)(k * l) * (f64)r /
               (f64)(len * (k * l + l * r + k * r));

    i32 shft = fullc ? 0 : 1;
    i32 p1_val;
    if (fullc) {
        p1_val = mk * l;
    } else {
        p1_val = (m - 1) * k * l;
    }

    i32 meth;
    i32 wrkopt;
    if (k * l == 1 && (m > 1 && n > 1)) {
        wrkopt = len * (2 + r) - p;
        meth = 3;
    } else if ((len < m * n) && (coef >= THOM50)) {
        wrkopt = len * (k * l + k * r + l * r + 1) - p;
        meth = 3;
    } else {
        meth = 2;
        wrkopt = p1_val;
    }
    if (wrkopt < 1) wrkopt = 1;

    if (lquery) {
        dwork[0] = (f64)wrkopt;
        return;
    }

    if (beta == ZERO) {
        if (ltran) {
            i32 nl_r = nl, r_r = r;
            SLC_DLASET("All", &nl_r, &r_r, &ZERO, &ZERO, c, &ldc);
        } else {
            i32 mk_r = mk, r_r = r;
            SLC_DLASET("All", &mk_r, &r_r, &ZERO, &ZERO, c, &ldc);
        }
    } else if (beta != ONE) {
        if (ltran) {
            for (i32 i = 0; i < r; i++) {
                SLC_DSCAL(&nl, &beta, &c[i * ldc], &int1);
            }
        } else {
            for (i32 i = 0; i < r; i++) {
                SLC_DSCAL(&mk, &beta, &c[i * ldc], &int1);
            }
        }
    }

    if (!lmult || (mk == 0) || (nl == 0) || (r == 0)) {
        dwork[0] = ONE;
        return;
    }

    if (ldwork < wrkopt) meth = meth - 1;
    if (ldwork < p1_val) meth = 1;

    if (meth == 1 && !ltran) {
        i32 pc = 0;
        for (i32 i = 0; i < m; i++) {
            i32 pt = (i - shft) * k;
            i32 pb = 0;
            for (i32 j = shft; j < i + 1; j++) {
                i32 k_r = k, r_r = r, l_r = l;
                SLC_DGEMM("N", "N", &k_r, &r_r, &l_r, &alpha, &tc[pt], &ldtc,
                          &b[pb], &ldb, &ONE, &c[pc], &ldc);
                pt = pt - k;
                pb = pb + l;
            }
            if (n > i + 1 - shft) {
                i32 k_r = k, r_r = r;
                i32 ncols = (n - i - 1 + shft) * l;
                SLC_DGEMM("N", "N", &k_r, &r_r, &ncols, &alpha, tr, &ldtr,
                          &b[pb], &ldb, &ONE, &c[pc], &ldc);
            }
            pc = pc + k;
        }
    } else if (meth == 1 && ltran) {
        i32 pb = 0;
        for (i32 i = 0; i < m; i++) {
            i32 pt = (i - shft) * k;
            i32 pc = 0;
            for (i32 j = shft; j < i + 1; j++) {
                i32 l_r = l, r_r = r, k_r = k;
                SLC_DGEMM("T", "N", &l_r, &r_r, &k_r, &alpha, &tc[pt], &ldtc,
                          &b[pb], &ldb, &ONE, &c[pc], &ldc);
                pt = pt - k;
                pc = pc + l;
            }
            if (n > i + 1 - shft) {
                i32 k_r = k, r_r = r;
                i32 ncols = (n - i - 1 + shft) * l;
                SLC_DGEMM("T", "N", &ncols, &r_r, &k_r, &alpha, tr, &ldtr,
                          &b[pb], &ldb, &ONE, &c[pc], &ldc);
            }
            pb = pb + k;
        }
    } else if (meth == 2 && !ltran) {
        i32 pt = (m - 1 - shft) * k;
        for (i32 i = 0; i < (m - shft) * k * l; i += k * l) {
            i32 k_r = k, l_r = l;
            SLC_DLACPY("All", &k_r, &l_r, &tc[pt], &ldtc, &dwork[i], &k_r);
            pt = pt - k;
        }

        pt = (m - 1) * k * l;
        i32 pc = 0;
        for (i32 i = 0; i < m; i++) {
            i32 minval = (i + 1 - shft < n) ? (i + 1 - shft) : n;
            i32 cols = minval * l;
            i32 k_r = k, r_r = r;
            SLC_DGEMM("N", "N", &k_r, &r_r, &cols, &alpha, &dwork[pt], &k_r,
                      b, &ldb, &ONE, &c[pc], &ldc);
            if (n > i + 1 - shft) {
                i32 ncols = (n - i - 1 + shft) * l;
                i32 boff = (i + 1 - shft) * l;
                SLC_DGEMM("N", "N", &k_r, &r_r, &ncols, &alpha, tr, &ldtr,
                          &b[boff], &ldb, &ONE, &c[pc], &ldc);
            }
            pc = pc + k;
            pt = pt - k * l;
        }
    } else if (meth == 2 && ltran) {
        i32 pt = (m - 1 - shft) * k;
        for (i32 i = 0; i < (m - shft) * k * l; i += k * l) {
            i32 k_r = k, l_r = l;
            SLC_DLACPY("All", &k_r, &l_r, &tc[pt], &ldtc, &dwork[i], &k_r);
            pt = pt - k;
        }

        pt = (m - 1) * k * l;
        i32 pb = 0;
        for (i32 i = 0; i < m; i++) {
            i32 minval = (i + 1 - shft < n) ? (i + 1 - shft) : n;
            i32 cols = minval * l;
            i32 k_r = k, r_r = r;
            SLC_DGEMM("T", "N", &cols, &r_r, &k_r, &alpha, &dwork[pt], &k_r,
                      &b[pb], &ldb, &ONE, c, &ldc);
            if (n > i + 1 - shft) {
                i32 ncols = (n - i - 1 + shft) * l;
                i32 coff = (i + 1 - shft) * l;
                SLC_DGEMM("T", "N", &ncols, &r_r, &k_r, &alpha, tr, &ldtr,
                          &b[pb], &ldb, &ONE, &c[coff], &ldc);
            }
            pb = pb + k;
            pt = pt - k * l;
        }
    } else if (meth == 3) {
        i32 dimb, dimc;
        if (ltran) {
            dimb = k;
            dimc = l;
        } else {
            dimb = l;
            dimc = k;
        }
        i32 pb_off = len * k * l;
        i32 pc_off = len * (k * l + dimb * r);

        if (ltran) {
            if (fullc) {
                i32 k_r = k, l_r = l;
                i32 lenk = len * k;
                SLC_DLACPY("All", &k_r, &l_r, tc, &ldtc, dwork, &lenk);
            }
            for (i32 i = 0; i < n - 1 + shft; i++) {
                i32 k_r = k, l_r = l;
                i32 lenk = len * k;
                i32 dest = (i + 1 - shft) * k;
                i32 src = i * l;
                SLC_DLACPY("All", &k_r, &l_r, &tr[src * ldtr], &ldtr,
                           &dwork[dest], &lenk);
            }

            i32 pdw = n * k;
            i32 r1 = (len - m - n + 1) * k;
            i32 lenk = len * k;
            SLC_DLASET("All", &r1, &l, &ZERO, &ZERO, &dwork[pdw], &lenk);
            pdw = pdw + r1;

            for (i32 idx = (m - 1 - shft) * k; idx >= k - shft * k; idx -= k) {
                i32 k_r = k, l_r = l;
                i32 lenk = len * k;
                SLC_DLACPY("All", &k_r, &l_r, &tc[idx], &ldtc, &dwork[pdw], &lenk);
                pdw = pdw + k;
            }

            pdw = pb_off;
            i32 lenk2 = len * k;
            SLC_DLACPY("All", &mk, &r, b, &ldb, &dwork[pdw], &lenk2);
            pdw = pdw + mk;
            i32 rem = (len - m) * k;
            SLC_DLASET("All", &rem, &r, &ZERO, &ZERO, &dwork[pdw], &lenk2);
        } else {
            if (!fullc) {
                i32 k_r = k, l_r = l;
                i32 lenk = len * k;
                SLC_DLACPY("All", &k_r, &l_r, tr, &ldtr, dwork, &lenk);
            }
            i32 mk_sh = (m - shft) * k;
            i32 lenk = len * k;
            SLC_DLACPY("All", &mk_sh, &l, tc, &ldtc, &dwork[shft * k], &lenk);

            i32 pdw = mk;
            i32 r1 = (len - m - n + 1) * k;
            SLC_DLASET("All", &r1, &l, &ZERO, &ZERO, &dwork[pdw], &lenk);
            pdw = pdw + r1;

            for (i32 idx = (n - 2 + shft) * l; idx >= shft * l; idx -= l) {
                i32 k_r = k, l_r = l;
                i32 lenk2 = len * k;
                SLC_DLACPY("All", &k_r, &l_r, &tr[idx * ldtr], &ldtr, &dwork[pdw], &lenk2);
                pdw = pdw + k;
            }

            pdw = pb_off;
            i32 lenl = len * l;
            SLC_DLACPY("All", &nl, &r, b, &ldb, &dwork[pdw], &lenl);
            pdw = pdw + nl;
            i32 rem = (len - n) * l;
            SLC_DLASET("All", &rem, &r, &ZERO, &ZERO, &dwork[pdw], &lenl);
        }

        if (k * l == 1) {
            const char *wght = "N";
            i32 ierr;
            dg01od("O", wght, len, dwork, &dwork[pc_off], &ierr);

            for (i32 i = pb_off; i < pb_off + len * r; i += len) {
                dg01od("O", wght, len, &dwork[i], &dwork[pc_off], &ierr);
                f64 scal = alpha / (f64)len;
                dwork[i] = scal * dwork[i] * dwork[0];
                dwork[i + 1] = scal * dwork[i + 1] * dwork[1];
                scal = scal / TWO;

                i32 ln = 1;
                for (i32 ll = 0; ll < p - 1; ll++) {
                    ln = 2 * ln;
                    i32 r1 = 2 * ln;

                    for (i32 pp1 = ln; pp1 < ln + ln / 2; pp1++) {
                        f64 t1 = dwork[pp1] + dwork[r1 - 1];
                        f64 t2 = dwork[pp1] - dwork[r1 - 1];
                        f64 th = t2 * dwork[i + pp1];
                        dwork[i + pp1] = scal * (t1 * dwork[i + pp1] + t2 * dwork[i + r1 - 1]);
                        dwork[i + r1 - 1] = scal * (t1 * dwork[i + r1 - 1] - th);
                        r1 = r1 - 1;
                    }
                }

                dg01od("I", wght, len, &dwork[i], &dwork[pc_off], &ierr);
            }

            pc_off = pb_off;
        } else {
            i32 pdw = pc_off;
            i32 r1 = 0;
            i32 ln = 1;
            f64 th = FOUR * atan(ONE) / (f64)len;

            for (i32 ll = 0; ll < p - 2; ll++) {
                ln = 2 * ln;
                th = TWO * th;
                f64 cf = cos(th);
                f64 sf = sin(th);
                dwork[pdw + r1] = cf;
                dwork[pdw + r1 + 1] = sf;
                r1 = r1 + 2;

                for (i32 i = 0; i < ln - 2; i += 2) {
                    dwork[pdw + r1] = cf * dwork[pdw + i] - sf * dwork[pdw + i + 1];
                    dwork[pdw + r1 + 1] = sf * dwork[pdw + i] + cf * dwork[pdw + i + 1];
                    r1 = r1 + 2;
                }
            }

            i32 pp1 = 2;
            i32 q1 = r1 - 2;
            for (i32 ll = p - 3; ll >= 0; ll--) {
                for (i32 i = pp1; i <= q1; i += 4) {
                    dwork[pdw + r1] = dwork[pdw + i];
                    dwork[pdw + r1 + 1] = dwork[pdw + i + 1];
                    r1 = r1 + 2;
                }
                pp1 = q1 + 4;
                q1 = r1 - 2;
            }

            i32 j = 0;
            i32 kk = k;

            while (j < pc_off) {
                ln = len;
                i32 wpos = pdw;

                for (i32 pp = p - 2; pp >= 0; pp--) {
                    ln = ln / 2;
                    i32 p2 = 0;
                    i32 q2 = ln * kk;
                    i32 r2 = (ln / 2) * kk;
                    i32 s2 = r2 + q2;

                    for (i32 i = 0; i < len / (2 * ln); i++) {
                        for (i32 ir = 0; ir < kk; ir++) {
                            f64 t1 = dwork[q2 + ir + j];
                            dwork[q2 + ir + j] = dwork[p2 + ir + j] - t1;
                            dwork[p2 + ir + j] = dwork[p2 + ir + j] + t1;
                            t1 = dwork[s2 + ir + j];
                            dwork[s2 + ir + j] = dwork[r2 + ir + j] - t1;
                            dwork[r2 + ir + j] = dwork[r2 + ir + j] + t1;
                        }

                        i32 pp1_inner = p2 + kk;
                        i32 q1_inner = pp1_inner + ln * kk;
                        i32 r1_inner = q1_inner - 2 * kk;
                        i32 s1 = r1_inner + ln * kk;

                        for (i32 jj = wpos; jj < wpos + ln - 2; jj += 2) {
                            f64 cf = dwork[jj];
                            f64 sf = dwork[jj + 1];

                            for (i32 ir = 0; ir < kk; ir++) {
                                f64 t1 = dwork[pp1_inner + ir + j] - dwork[q1_inner + ir + j];
                                f64 t2 = dwork[r1_inner + ir + j] - dwork[s1 + ir + j];
                                dwork[pp1_inner + ir + j] = dwork[pp1_inner + ir + j] + dwork[q1_inner + ir + j];
                                dwork[r1_inner + ir + j] = dwork[r1_inner + ir + j] + dwork[s1 + ir + j];
                                dwork[q1_inner + ir + j] = cf * t1 + sf * t2;
                                dwork[s1 + ir + j] = -cf * t2 + sf * t1;
                            }

                            pp1_inner = pp1_inner + kk;
                            q1_inner = q1_inner + kk;
                            r1_inner = r1_inner - kk;
                            s1 = s1 - kk;
                        }

                        p2 = p2 + 2 * kk * ln;
                        q2 = q2 + 2 * kk * ln;
                        r2 = r2 + 2 * kk * ln;
                        s2 = s2 + 2 * kk * ln;
                    }

                    wpos = wpos + ln - 2;
                }

                for (i32 icp = kk; icp < len * kk; icp += 2 * kk) {
                    i32 icq = icp - kk;
                    for (i32 ir = 0; ir < kk; ir++) {
                        f64 t1 = dwork[icp + ir + j];
                        dwork[icp + ir + j] = dwork[icq + ir + j] - t1;
                        dwork[icq + ir + j] = dwork[icq + ir + j] + t1;
                    }
                }

                j = j + len * kk;
                if (j == l * len * k) {
                    kk = dimb;
                }
            }

            i32 lenpmp = len - p;
            SLC_DCOPY(&lenpmp, &dwork[pdw], &int1, &dwork[pdw + r * len * dimc], &int1);
            pdw = pdw + r * len * dimc;
            f64 scal = alpha / (f64)len;
            pp1 = 0;
            r1 = len * k * l;
            i32 s1 = r1 + len * dimb * r;
            i32 kk_h, ll_h;
            if (ltran) {
                kk_h = l;
                ll_h = k;
            } else {
                kk_h = k;
                ll_h = l;
            }
            i32 lenk = len * k;
            i32 lend = len * dimb;
            i32 lenc = len * dimc;
            SLC_DGEMM(trans, "N", &kk_h, &r, &ll_h, &scal, &dwork[pp1], &lenk,
                      &dwork[r1], &lend, &ZERO, &dwork[s1], &lenc);
            pp1 = pp1 + k;
            r1 = r1 + dimb;
            s1 = s1 + dimc;
            SLC_DGEMM(trans, "N", &kk_h, &r, &ll_h, &scal, &dwork[pp1], &lenk,
                      &dwork[r1], &lend, &ZERO, &dwork[s1], &lenc);
            scal = scal / TWO;
            ln = 1;

            for (i32 pp = 0; pp < p - 1; pp++) {
                ln = 2 * ln;
                i32 p2 = (2 * ln - 1) * k;
                r1 = pb_off + ln * dimb;
                i32 r2 = pb_off + (2 * ln - 1) * dimb;
                s1 = pc_off + ln * dimc;
                i32 s2 = pc_off + (2 * ln - 1) * dimc;

                for (pp1 = ln * k; pp1 < (ln + ln / 2) * k; pp1 += k) {
                    for (j = 0; j < len * k * l; j += len * k) {
                        for (i32 i = pp1; i < pp1 + k; i++) {
                            f64 t1 = dwork[p2];
                            dwork[p2] = dwork[j + i] - t1;
                            dwork[j + i] = dwork[j + i] + t1;
                            p2 = p2 + 1;
                        }
                        p2 = p2 + (len - 1) * k;
                    }
                    p2 = p2 - len * k * l;

                    i32 lenk_l = len * k;
                    i32 lenbd = len * dimb;
                    i32 lencd = len * dimc;
                    SLC_DGEMM(trans, "N", &kk_h, &r, &ll_h, &scal, &dwork[pp1], &lenk_l,
                              &dwork[r1], &lenbd, &ZERO, &dwork[s1], &lencd);
                    SLC_DGEMM(trans, "N", &kk_h, &r, &ll_h, &scal, &dwork[p2], &lenk_l,
                              &dwork[r2], &lenbd, &ONE, &dwork[s1], &lencd);
                    SLC_DGEMM(trans, "N", &kk_h, &r, &ll_h, &scal, &dwork[pp1], &lenk_l,
                              &dwork[r2], &lenbd, &ZERO, &dwork[s2], &lencd);
                    f64 nscal = -scal;
                    SLC_DGEMM(trans, "N", &kk_h, &r, &ll_h, &nscal, &dwork[p2], &lenk_l,
                              &dwork[r1], &lenbd, &ONE, &dwork[s2], &lencd);

                    p2 = p2 - k;
                    r1 = r1 + dimb;
                    r2 = r2 - dimb;
                    s1 = s1 + dimc;
                    s2 = s2 - dimc;
                }
            }

            for (j = pc_off; j < pc_off + len * dimc * r; j += len * dimc) {
                for (i32 icp = dimc; icp < len * dimc; icp += 2 * dimc) {
                    i32 icq = icp - dimc;
                    for (i32 ir = 0; ir < dimc; ir++) {
                        f64 t1 = dwork[icp + ir + j];
                        dwork[icp + ir + j] = dwork[icq + ir + j] - t1;
                        dwork[icq + ir + j] = dwork[icq + ir + j] + t1;
                    }
                }

                ln = 1;
                i32 wpos = pdw + len - 2 * p;

                for (i32 pp = 0; pp < p - 1; pp++) {
                    ln = 2 * ln;
                    i32 p2 = 0;
                    i32 q2 = ln * dimc;
                    i32 r2 = (ln / 2) * dimc;
                    i32 s2 = r2 + q2;

                    for (i32 i = 0; i < len / (2 * ln); i++) {
                        for (i32 ir = 0; ir < dimc; ir++) {
                            f64 t1 = dwork[q2 + ir + j];
                            dwork[q2 + ir + j] = dwork[p2 + ir + j] - t1;
                            dwork[p2 + ir + j] = dwork[p2 + ir + j] + t1;
                            t1 = dwork[s2 + ir + j];
                            dwork[s2 + ir + j] = dwork[r2 + ir + j] - t1;
                            dwork[r2 + ir + j] = dwork[r2 + ir + j] + t1;
                        }

                        pp1 = p2 + dimc;
                        q1 = pp1 + ln * dimc;
                        r1 = q1 - 2 * dimc;
                        s1 = r1 + ln * dimc;

                        for (i32 jj = wpos; jj < wpos + ln - 2; jj += 2) {
                            f64 cf = dwork[jj];
                            f64 sf = dwork[jj + 1];

                            for (i32 ir = 0; ir < dimc; ir++) {
                                f64 t1 = cf * dwork[q1 + ir + j] + sf * dwork[s1 + ir + j];
                                f64 t2 = -cf * dwork[s1 + ir + j] + sf * dwork[q1 + ir + j];
                                dwork[q1 + ir + j] = dwork[pp1 + ir + j] - t1;
                                dwork[pp1 + ir + j] = dwork[pp1 + ir + j] + t1;
                                dwork[s1 + ir + j] = dwork[r1 + ir + j] - t2;
                                dwork[r1 + ir + j] = dwork[r1 + ir + j] + t2;
                            }

                            pp1 = pp1 + dimc;
                            q1 = q1 + dimc;
                            r1 = r1 - dimc;
                            s1 = s1 - dimc;
                        }

                        p2 = p2 + 2 * dimc * ln;
                        q2 = q2 + 2 * dimc * ln;
                        r2 = r2 + 2 * dimc * ln;
                        s2 = s2 + 2 * dimc * ln;
                    }

                    wpos = wpos - 2 * ln + 2;
                }
            }
        }

        i32 i_dim;
        if (ltran) {
            i_dim = nl;
        } else {
            i_dim = mk;
        }

        for (i32 jj = 0; jj < r; jj++) {
            SLC_DAXPY(&i_dim, &ONE, &dwork[pc_off + jj * len * dimc], &int1, &c[jj * ldc], &int1);
        }
    }

    dwork[0] = (f64)wrkopt;
}
