/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <ctype.h>
#include <stdbool.h>

void mb02cv(const char* typeg, const char* strucg, i32 k, i32 n, i32 p, i32 q,
            i32 nb, i32 rnk, f64* a1, i32 lda1, f64* a2, i32 lda2,
            f64* b, i32 ldb, f64* f1, i32 ldf1, f64* f2, i32 ldf2,
            f64* g, i32 ldg, const f64* cs, f64* dwork, i32 ldwork, i32* info)
{
    const f64 zero = 0.0;
    const f64 one = 1.0;

    char typeg_u = (char)toupper((unsigned char)typeg[0]);
    char strucg_u = (char)toupper((unsigned char)strucg[0]);

    bool lrdef = (typeg_u == 'D');
    bool lcol = (typeg_u == 'C');
    bool ltri = (strucg_u == 'T');

    i32 col2 = p - k;
    if (col2 < 0) col2 = 0;

    i32 wrkmin;
    if (lrdef) {
        wrkmin = (n > 1) ? n : 1;
    } else {
        if (nb >= 1) {
            wrkmin = (n + k) * nb;
            if (wrkmin < 1) wrkmin = 1;
        } else {
            wrkmin = (n > 1) ? n : 1;
        }
    }

    *info = 0;
    if (!lcol && !lrdef && typeg_u != 'R') {
        *info = -1;
    } else if (!ltri && strucg_u != 'N') {
        *info = -2;
    } else if (k < 0) {
        *info = -3;
    } else if (n < 0) {
        *info = -4;
    } else if (p < k) {
        *info = -5;
    } else if (q < 0 || (lrdef && q < k)) {
        *info = -6;
    } else if (lrdef && (rnk < 0 || rnk > k)) {
        *info = -8;
    } else if (lda1 < 1 || (lrdef && lda1 < k)) {
        *info = -10;
    } else if ((p == k && lda2 < 1) ||
               (p > k && (lrdef || lcol) && lda2 < ((k > 1) ? k : 1)) ||
               (p > k && !(lrdef || lcol) && lda2 < (p - k))) {
        *info = -12;
    } else if ((q == 0 && ldb < 1) ||
               (q > 0 && (lrdef || lcol) && ldb < ((k > 1) ? k : 1)) ||
               (q > 0 && !(lrdef || lcol) && ldb < q)) {
        *info = -14;
    } else if ((lrdef || lcol) && ldf1 < ((n > 1) ? n : 1)) {
        *info = -16;
    } else if (!(lrdef || lcol) && ldf1 < ((k > 1) ? k : 1)) {
        *info = -16;
    } else if ((p == k && ldf2 < 1) ||
               (p > k && (lrdef || lcol) && ldf2 < ((n > 1) ? n : 1)) ||
               (p > k && !(lrdef || lcol) && ldf2 < (p - k))) {
        *info = -18;
    } else if ((q == 0 && ldg < 1) ||
               (q > 0 && (lrdef || lcol) && ldg < ((n > 1) ? n : 1)) ||
               (q > 0 && !(lrdef || lcol) && ldg < q)) {
        *info = -20;
    } else if (ldwork < wrkmin) {
        dwork[0] = (f64)wrkmin;
        *info = -23;
    }

    if (*info != 0) {
        return;
    }

    i32 min_kn = (k < n) ? k : n;
    if (min_kn == 0 || (!lrdef && q == 0 && p == k)) {
        return;
    }

    i32 int1 = 1;
    i32 pst2;
    i32 len;
    i32 pos;
    f64 c, s, tau, alpha, beta, temp;

    if (lrdef) {
        if (col2 == 0) {
            pst2 = 2 * k;
        } else {
            pst2 = 4 * k;
        }

        for (i32 i = 0; i < rnk; i++) {
            if (col2 > 1) {
                tau = a2[i + 0 * lda2];
                a2[i + 0 * lda2] = one;
                SLC_DLARF("Right", &n, &col2, &a2[i], &lda2, &tau, f2, &ldf2, dwork);
                a2[i + 0 * lda2] = tau;
            }

            if (k > i + 1) {
                alpha = a1[i + i * lda1];
                a1[i + i * lda1] = one;
                i32 len_f1 = k - i;
                f64 tau_cs = cs[pst2 + i];
                SLC_DLARF("Right", &n, &len_f1, &a1[i + i * lda1], &lda1, &tau_cs,
                          &f1[0 + i * ldf1], &ldf1, dwork);
                a1[i + i * lda1] = alpha;
            }

            if (col2 > 0) {
                c = cs[2 * k + i * 2];
                s = cs[2 * k + i * 2 + 1];
                SLC_DROT(&n, &f1[0 + i * ldf1], &int1, f2, &int1, &c, &s);
            }

            if (q > 1) {
                tau = b[i + 0 * ldb];
                b[i + 0 * ldb] = one;
                SLC_DLARF("Right", &n, &q, &b[i], &ldb, &tau, g, &ldg, dwork);
                b[i + 0 * ldb] = tau;
            }

            c = cs[i * 2];
            s = cs[i * 2 + 1];
            f64 inv_c = one / c;
            f64 neg_s_c = -s / c;
            f64 neg_s = -s;
            SLC_DSCAL(&n, &inv_c, &f1[0 + i * ldf1], &int1);
            SLC_DAXPY(&n, &neg_s_c, &g[0], &int1, &f1[0 + i * ldf1], &int1);
            SLC_DSCAL(&n, &c, &g[0], &int1);
            SLC_DAXPY(&n, &neg_s, &f1[0 + i * ldf1], &int1, &g[0], &int1);
        }

        len = q;
        pos = 0;

        for (i32 j = rnk; j < k; j++) {
            if (col2 > 1) {
                tau = a2[j + 0 * lda2];
                a2[j + 0 * lda2] = one;
                SLC_DLARF("Right", &n, &col2, &a2[j], &lda2, &tau, f2, &ldf2, dwork);
                a2[j + 0 * lda2] = tau;
            }
            if (k > j + 1) {
                alpha = a1[j + j * lda1];
                a1[j + j * lda1] = one;
                i32 len_f1 = k - j;
                f64 tau_cs = cs[pst2 + j];
                SLC_DLARF("Right", &n, &len_f1, &a1[j + j * lda1], &lda1, &tau_cs,
                          &f1[0 + j * ldf1], &ldf1, dwork);
                a1[j + j * lda1] = alpha;
            }
            if (col2 > 0) {
                c = cs[2 * k + j * 2];
                s = cs[2 * k + j * 2 + 1];
                SLC_DROT(&n, &f1[0 + j * ldf1], &int1, f2, &int1, &c, &s);
            }
            if (len > 1) {
                beta = b[j + pos * ldb];
                b[j + pos * ldb] = one;
                f64 tau_j = cs[j * 2];
                SLC_DLARF("Right", &n, &len, &b[j + pos * ldb], &ldb, &tau_j,
                          &g[0 + pos * ldg], &ldg, dwork);
                b[j + pos * ldb] = beta;
            }
            len--;
            pos++;
        }

    } else if (lcol) {
        if (ltri) {
            len = (n > k) ? (n - k) : 0;
        } else {
            len = n;
        }

        if (col2 > 0) {
            i32 nbl = (col2 < nb) ? col2 : nb;
            i32 i = 0;
            if (nbl > 0) {
                for (i = 0; i <= k - nbl; i += nbl) {
                    i32 ib = k - i;
                    if (ib > nbl) ib = nbl;
                    i32 ldt = n + k;
                    SLC_DLARFT("Forward", "Rowwise", &col2, &ib, &a2[i], &lda2,
                               (f64*)&cs[4 * k + i], dwork, &ldt);
                    SLC_DLARFB("Right", "No Transpose", "Forward", "Rowwise",
                               &len, &col2, &ib, &a2[i], &lda2,
                               dwork, &ldt, f2, &ldf2, &dwork[ib], &ldt);

                    for (i32 j = i; j < i + ib; j++) {
                        tau = a2[j + 0 * lda2];
                        a2[j + 0 * lda2] = one;
                        i32 min_col2_j = (col2 < j - i + 1) ? col2 : (j - i + 1);
                        SLC_DLARF("Right", &len, &min_col2_j, &a2[j], &lda2, &tau,
                                  f2, &ldf2, dwork);
                        a2[j + 0 * lda2] = tau;

                        c = cs[2 * k + j * 2];
                        s = cs[2 * k + j * 2 + 1];
                        SLC_DROT(&len, &f1[0 + j * ldf1], &int1, f2, &int1, &c, &s);

                        if (ltri) {
                            len++;
                            temp = f1[len - 1 + j * ldf1];
                            f1[len - 1 + j * ldf1] = c * temp;
                            f2[len - 1 + 0 * ldf2] = -s * temp;

                            for (i32 jj = 1; jj < col2; jj++) {
                                f2[len - 1 + jj * ldf2] = zero;
                            }
                        }
                    }
                }
            } else {
                i = 0;
            }

            for (i32 j = i; j < k; j++) {
                if (col2 > 1) {
                    tau = a2[j + 0 * lda2];
                    a2[j + 0 * lda2] = one;
                    SLC_DLARF("Right", &len, &col2, &a2[j], &lda2, &tau,
                              f2, &ldf2, dwork);
                    a2[j + 0 * lda2] = tau;
                }

                c = cs[2 * k + j * 2];
                s = cs[2 * k + j * 2 + 1];
                SLC_DROT(&len, &f1[0 + j * ldf1], &int1, f2, &int1, &c, &s);

                if (ltri) {
                    len++;
                    temp = f1[len - 1 + j * ldf1];
                    f1[len - 1 + j * ldf1] = c * temp;
                    f2[len - 1 + 0 * ldf2] = -s * temp;

                    for (i32 jj = 1; jj < col2; jj++) {
                        f2[len - 1 + jj * ldf2] = zero;
                    }
                }
            }

            pst2 = 5 * k;
        } else {
            pst2 = 2 * k;
        }

        if (ltri) {
            len = n - k;
        } else {
            len = n;
        }

        i32 nbl = (q < nb) ? q : nb;
        i32 i = 0;
        if (nbl > 0) {
            for (i = 0; i <= k - nbl; i += nbl) {
                i32 ib = k - i;
                if (ib > nbl) ib = nbl;
                i32 ldt = n + k;
                SLC_DLARFT("Forward", "Rowwise", &q, &ib, &b[i], &ldb,
                           (f64*)&cs[pst2 + i], dwork, &ldt);
                SLC_DLARFB("Right", "NonTranspose", "Forward", "Rowwise",
                           &len, &q, &ib, &b[i], &ldb,
                           dwork, &ldt, g, &ldg, &dwork[ib], &ldt);

                for (i32 j = i; j < i + ib; j++) {
                    tau = b[j + 0 * ldb];
                    b[j + 0 * ldb] = one;
                    i32 len_b = j - i + 1;
                    SLC_DLARF("Right", &len, &len_b, &b[j], &ldb, &tau,
                              g, &ldg, dwork);
                    b[j + 0 * ldb] = tau;

                    c = cs[j * 2];
                    s = cs[j * 2 + 1];
                    f64 inv_c = one / c;
                    f64 neg_s_c = -s / c;
                    f64 neg_s = -s;
                    SLC_DSCAL(&len, &inv_c, &f1[0 + j * ldf1], &int1);
                    SLC_DAXPY(&len, &neg_s_c, g, &int1, &f1[0 + j * ldf1], &int1);
                    SLC_DSCAL(&len, &c, g, &int1);
                    SLC_DAXPY(&len, &neg_s, &f1[0 + j * ldf1], &int1, g, &int1);

                    if (ltri) {
                        len++;
                        g[len - 1 + 0 * ldg] = neg_s_c * f1[len - 1 + j * ldf1];
                        f1[len - 1 + j * ldf1] = f1[len - 1 + j * ldf1] / c;

                        for (i32 jj = 1; jj < q; jj++) {
                            g[len - 1 + jj * ldg] = zero;
                        }
                    }
                }
            }
        } else {
            i = 0;
        }

        for (i32 j = i; j < k; j++) {
            if (q > 1) {
                tau = b[j + 0 * ldb];
                b[j + 0 * ldb] = one;
                SLC_DLARF("Right", &len, &q, &b[j], &ldb, &tau,
                          g, &ldg, dwork);
                b[j + 0 * ldb] = tau;
            }
            if (q > 0) {
                c = cs[j * 2];
                s = cs[j * 2 + 1];
                f64 inv_c = one / c;
                f64 neg_s_c = -s / c;
                f64 neg_s = -s;
                SLC_DSCAL(&len, &inv_c, &f1[0 + j * ldf1], &int1);
                SLC_DAXPY(&len, &neg_s_c, g, &int1, &f1[0 + j * ldf1], &int1);
                SLC_DSCAL(&len, &c, g, &int1);
                SLC_DAXPY(&len, &neg_s, &f1[0 + j * ldf1], &int1, g, &int1);

                if (ltri) {
                    len++;
                    g[len - 1 + 0 * ldg] = -s / c * f1[len - 1 + j * ldf1];
                    f1[len - 1 + j * ldf1] = f1[len - 1 + j * ldf1] / c;

                    for (i32 jj = 1; jj < q; jj++) {
                        g[len - 1 + jj * ldg] = zero;
                    }
                }
            }
        }

    } else {
        if (ltri) {
            len = (n > k) ? (n - k) : 0;
        } else {
            len = n;
        }

        if (col2 > 0) {
            i32 nbl = (nb < col2) ? nb : col2;
            i32 i = 0;
            if (nbl > 0) {
                for (i = 0; i <= k - nbl; i += nbl) {
                    i32 ib = k - i;
                    if (ib > nbl) ib = nbl;
                    i32 ldt = n + k;
                    SLC_DLARFT("Forward", "Columnwise", &col2, &ib, &a2[0 + i * lda2], &lda2,
                               (f64*)&cs[4 * k + i], dwork, &ldt);
                    SLC_DLARFB("Left", "Transpose", "Forward", "Columnwise",
                               &col2, &len, &ib, &a2[0 + i * lda2], &lda2,
                               dwork, &ldt, f2, &ldf2, &dwork[ib], &ldt);

                    for (i32 j = i; j < i + ib; j++) {
                        tau = a2[0 + j * lda2];
                        a2[0 + j * lda2] = one;
                        i32 min_col2_j = (col2 < j - i + 1) ? col2 : (j - i + 1);
                        SLC_DLARF("Left", &min_col2_j, &len, &a2[0 + j * lda2], &int1, &tau,
                                  f2, &ldf2, dwork);
                        a2[0 + j * lda2] = tau;

                        c = cs[2 * k + j * 2];
                        s = cs[2 * k + j * 2 + 1];
                        SLC_DROT(&len, &f1[j], &ldf1, f2, &ldf2, &c, &s);

                        if (ltri) {
                            len++;
                            temp = f1[j + (len - 1) * ldf1];
                            f1[j + (len - 1) * ldf1] = c * temp;
                            f2[0 + (len - 1) * ldf2] = -s * temp;

                            for (i32 jj = 1; jj < col2; jj++) {
                                f2[jj + (len - 1) * ldf2] = zero;
                            }
                        }
                    }
                }
            } else {
                i = 0;
            }

            for (i32 j = i; j < k; j++) {
                if (col2 > 1) {
                    tau = a2[0 + j * lda2];
                    a2[0 + j * lda2] = one;
                    SLC_DLARF("Left", &col2, &len, &a2[0 + j * lda2], &int1, &tau,
                              f2, &ldf2, dwork);
                    a2[0 + j * lda2] = tau;
                }

                c = cs[2 * k + j * 2];
                s = cs[2 * k + j * 2 + 1];
                SLC_DROT(&len, &f1[j], &ldf1, f2, &ldf2, &c, &s);

                if (ltri) {
                    len++;
                    temp = f1[j + (len - 1) * ldf1];
                    f1[j + (len - 1) * ldf1] = c * temp;
                    f2[0 + (len - 1) * ldf2] = -s * temp;

                    for (i32 jj = 1; jj < col2; jj++) {
                        f2[jj + (len - 1) * ldf2] = zero;
                    }
                }
            }

            pst2 = 5 * k;
        } else {
            pst2 = 2 * k;
        }

        if (ltri) {
            len = n - k;
        } else {
            len = n;
        }

        i32 nbl = (q < nb) ? q : nb;
        i32 i = 0;
        if (nbl > 0) {
            for (i = 0; i <= k - nbl; i += nbl) {
                i32 ib = k - i;
                if (ib > nbl) ib = nbl;
                i32 ldt = n + k;
                SLC_DLARFT("Forward", "Columnwise", &q, &ib, &b[0 + i * ldb], &ldb,
                           (f64*)&cs[pst2 + i], dwork, &ldt);
                SLC_DLARFB("Left", "Transpose", "Forward", "Columnwise",
                           &q, &len, &ib, &b[0 + i * ldb], &ldb,
                           dwork, &ldt, g, &ldg, &dwork[ib], &ldt);

                for (i32 j = i; j < i + ib; j++) {
                    tau = b[0 + j * ldb];
                    b[0 + j * ldb] = one;
                    i32 len_b = j - i + 1;
                    SLC_DLARF("Left", &len_b, &len, &b[0 + j * ldb], &int1, &tau,
                              g, &ldg, dwork);
                    b[0 + j * ldb] = tau;

                    c = cs[j * 2];
                    s = cs[j * 2 + 1];
                    f64 inv_c = one / c;
                    f64 neg_s_c = -s / c;
                    f64 neg_s = -s;
                    SLC_DSCAL(&len, &inv_c, &f1[j], &ldf1);
                    SLC_DAXPY(&len, &neg_s_c, g, &ldg, &f1[j], &ldf1);
                    SLC_DSCAL(&len, &c, g, &ldg);
                    SLC_DAXPY(&len, &neg_s, &f1[j], &ldf1, g, &ldg);

                    if (ltri) {
                        len++;
                        g[0 + (len - 1) * ldg] = neg_s_c * f1[j + (len - 1) * ldf1];
                        f1[j + (len - 1) * ldf1] = f1[j + (len - 1) * ldf1] / c;

                        for (i32 jj = 1; jj < q; jj++) {
                            g[jj + (len - 1) * ldg] = zero;
                        }
                    }
                }
            }
        } else {
            i = 0;
        }

        for (i32 j = i; j < k; j++) {
            if (q > 1) {
                tau = b[0 + j * ldb];
                b[0 + j * ldb] = one;
                SLC_DLARF("Left", &q, &len, &b[0 + j * ldb], &int1, &tau,
                          g, &ldg, dwork);
                b[0 + j * ldb] = tau;
            }
            if (q > 0) {
                c = cs[j * 2];
                s = cs[j * 2 + 1];
                f64 inv_c = one / c;
                f64 neg_s_c = -s / c;
                f64 neg_s = -s;
                SLC_DSCAL(&len, &inv_c, &f1[j], &ldf1);
                SLC_DAXPY(&len, &neg_s_c, g, &ldg, &f1[j], &ldf1);
                SLC_DSCAL(&len, &c, g, &ldg);
                SLC_DAXPY(&len, &neg_s, &f1[j], &ldf1, g, &ldg);

                if (ltri) {
                    len++;
                    g[0 + (len - 1) * ldg] = -s / c * f1[j + (len - 1) * ldf1];
                    f1[j + (len - 1) * ldf1] = f1[j + (len - 1) * ldf1] / c;

                    for (i32 jj = 1; jj < q; jj++) {
                        g[jj + (len - 1) * ldg] = zero;
                    }
                }
            }
        }
    }
}
