/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <ctype.h>
#include <stdbool.h>

void mb02gd(const char* typet, const char* triu, i32 k, i32 n, i32 nl,
            i32 p, i32 s, f64* t, i32 ldt, f64* rb, i32 ldrb,
            f64* dwork, i32 ldwork, i32* info)
{
    const f64 zero = 0.0;
    const f64 one = 1.0;

    char typet_u = (char)toupper((unsigned char)typet[0]);
    char triu_u = (char)toupper((unsigned char)triu[0]);
    bool isrow = (typet_u == 'R');
    bool ltri = (triu_u == 'T');

    i32 lenr = (nl + 1) * k;
    i32 sizr = ltri ? (nl * k + 1) : lenr;

    *info = 0;

    i32 wrkmin = 1 + (lenr + nl) * k;

    if (!isrow && typet_u != 'C') {
        *info = -1;
    } else if (!ltri && triu_u != 'N') {
        *info = -2;
    } else if (k < 0) {
        *info = -3;
    } else if ((ltri && n < 2) || (!ltri && n < 1)) {
        *info = -4;
    } else if (nl >= n || (ltri && nl < 1) || (!ltri && nl < 0)) {
        *info = -5;
    } else if (p < 0 || p > n) {
        *info = -6;
    } else if (s < 0 || s > n - p) {
        *info = -7;
    } else if ((isrow && ldt < (k > 1 ? k : 1)) ||
               (!isrow && ldt < (lenr > 1 ? lenr : 1))) {
        *info = -9;
    } else if ((ltri && ldrb < sizr) ||
               (!ltri && ldrb < (lenr > 1 ? lenr : 1))) {
        *info = -11;
    } else {
        bool lquery = (ldwork == -1);
        i32 pdw = lenr * k + 1;
        i32 kk = pdw + 4 * k;

        f64 dwork_query;
        i32 query_info = 0;
        i32 neg1 = -1;
        if (isrow) {
            SLC_DGEQRF(&k, &lenr, t, &ldt, &dwork_query, &dwork_query, &neg1, &query_info);
        } else {
            SLC_DGELQF(&lenr, &k, t, &ldt, &dwork_query, &dwork_query, &neg1, &query_info);
        }
        i32 wrkopt = kk + (i32)dwork_query;

        if (ldwork < wrkmin && !lquery) {
            dwork[0] = (f64)wrkmin;
            *info = -13;
        }

        if (*info == 0) {
            if (lquery) {
                dwork[0] = (f64)wrkopt;
                return;
            }
        }
    }

    if (*info != 0) {
        return;
    }

    if (s * k == 0) {
        dwork[0] = one;
        return;
    }

    i32 ierr = 0;
    i32 posr;
    i32 pre, stps;
    i32 int1 = 1;
    i32 ipvt[1];
    f64 dum[1];
    i32 rnk;
    i32 pdw_val = lenr * k + 1;
    i32 kk = pdw_val + 4 * k;

    i32 nb;
    if (ldwork > kk) {
        nb = (ldwork - kk) / lenr;
        if (nb > k) nb = k;
    } else {
        nb = 0;
    }
    i32 nbmin;
    if (isrow) {
        nbmin = SLC_ILAENV(&(i32){2}, "DGEQRF", " ", &k, &lenr, &(i32){-1}, &(i32){-1});
    } else {
        nbmin = SLC_ILAENV(&(i32){2}, "DGELQF", " ", &lenr, &k, &(i32){-1}, &(i32){-1});
    }
    if (nbmin < 2) nbmin = 2;
    if (nb < nbmin) nb = 0;

    if (p == 0) {
        if (isrow) {
            SLC_DPOTRF("Upper", &k, t, &ldt, &ierr);
            if (ierr != 0) {
                *info = 1;
                return;
            }
            if (nl > 0) {
                i32 nlk = nl * k;
                SLC_DTRSM("Left", "Upper", "Transpose", "NonUnit", &k, &nlk,
                          &one, t, &ldt, &t[k * ldt], &ldt);
            }

            if (ltri) {
                for (i32 i = 1; i <= lenr - k; i++) {
                    i32 cnt = (i < k) ? i : k;
                    i32 dst_row = sizr - i + 1 - 1;
                    if (dst_row < 0) dst_row = 0;
                    SLC_DCOPY(&cnt, &t[(i - 1) * ldt], &int1,
                              &rb[dst_row + (i - 1) * ldrb], &int1);
                }

                for (i32 i = k; i >= 1; i--) {
                    i32 src_row = k - i;
                    i32 src_col = lenr - i;
                    SLC_DCOPY(&i, &t[src_row + src_col * ldt], &int1,
                              &rb[(lenr - i) * ldrb], &int1);
                }
            } else {
                for (i32 i = 1; i <= lenr; i++) {
                    i32 cnt = (i < k) ? i : k;
                    i32 dst_row = sizr - i + 1 - 1;
                    if (dst_row < 0) dst_row = 0;
                    SLC_DCOPY(&cnt, &t[(i - 1) * ldt], &int1,
                              &rb[dst_row + (i - 1) * ldrb], &int1);
                }
            }

            if (n == 1) {
                dwork[0] = one;
                return;
            }

            i32 nlk = nl * k;
            SLC_DLACPY("All", &k, &nlk, &t[k * ldt], &ldt, &dwork[1], &k);
            SLC_DLASET("All", &k, &k, &zero, &zero, &dwork[nl * k * k + 1], &k);
            posr = k;
        } else {
            SLC_DPOTRF("Lower", &k, t, &ldt, &ierr);
            if (ierr != 0) {
                *info = 1;
                return;
            }
            if (nl > 0) {
                i32 nlk = nl * k;
                SLC_DTRSM("Right", "Lower", "Transpose", "NonUnit", &nlk, &k,
                          &one, t, &ldt, &t[k], &ldt);
            }

            posr = 0;
            if (ltri) {
                for (i32 i = 1; i <= k; i++) {
                    SLC_DCOPY(&sizr, &t[(i - 1) + (i - 1) * ldt], &int1,
                              &rb[posr * ldrb], &int1);
                    posr++;
                }
            } else {
                for (i32 i = 1; i <= k; i++) {
                    i32 cnt = lenr - i + 1;
                    SLC_DCOPY(&cnt, &t[(i - 1) + (i - 1) * ldt], &int1,
                              &rb[posr * ldrb], &int1);
                    if (lenr < n * k && i > 1) {
                        i32 iminus1 = i - 1;
                        SLC_DLASET("All", &iminus1, &int1, &zero, &zero,
                                   &rb[(lenr - i + 1) + posr * ldrb], &ldrb);
                    }
                    posr++;
                }
            }

            if (n == 1) {
                dwork[0] = one;
                return;
            }

            i32 nlk = nl * k;
            SLC_DLACPY("All", &nlk, &k, &t[k], &ldt, &dwork[1], &lenr);
            SLC_DLASET("All", &k, &k, &zero, &zero, &dwork[nl * k + 1], &lenr);
        }
        pre = 1;
        stps = s - 1;
    } else {
        pre = p;
        stps = s;
        posr = 0;
    }

    i32 head = ((pre - 1) * k) % lenr;

    if (isrow) {
        for (i32 i = pre; i <= pre + stps - 1; i++) {
            i32 ldw_cu = ldwork - pdw_val - 4 * k;
            mb02cu("Row", k, k, k, nb, t, ldt, dum, 1,
                   &dwork[head * k + 1], k, &rnk, ipvt, &dwork[pdw_val],
                   zero, &dwork[pdw_val + 4 * k], ldw_cu, &ierr);

            if (ierr != 0) {
                *info = 1;
                return;
            }

            i32 len_val = (n - i) * k - k;
            i32 max_len = lenr - head - k;
            if (len_val > max_len) len_val = max_len;
            if (len_val < 0) len_val = 0;

            i32 len2 = (n - i) * k - len_val - k;
            if (len2 > head) len2 = head;
            if (len2 < 0) len2 = 0;

            const char* struct_str = (len_val == lenr - k) ? triu : "N";

            i32 ldw_cv = ldwork - pdw_val - 4 * k;
            mb02cv("Row", struct_str, k, len_val, k, k, nb, -1, dum, 1,
                   dum, 1, &dwork[head * k + 1], k, &t[k * ldt], ldt,
                   dum, 1, &dwork[(head + k) * k + 1], k, &dwork[pdw_val],
                   &dwork[pdw_val + 4 * k], ldw_cv, &ierr);

            struct_str = ((n - i) * k >= lenr) ? triu : "N";

            mb02cv("Row", struct_str, k, len2, k, k, nb, -1, dum, 1,
                   dum, 1, &dwork[head * k + 1], k, &t[(k + len_val) * ldt], ldt,
                   dum, 1, &dwork[1], k, &dwork[pdw_val],
                   &dwork[pdw_val + 4 * k], ldw_cv, &ierr);

            SLC_DLASET("All", &k, &k, &zero, &zero, &dwork[head * k + 1], &k);

            if (ltri) {
                i32 jmax = len_val + len2 + k;
                if (jmax > lenr - k) jmax = lenr - k;
                for (i32 j = 1; j <= jmax; j++) {
                    i32 cnt = (j < k) ? j : k;
                    i32 dst_row = sizr - j + 1 - 1;
                    if (dst_row < 0) dst_row = 0;
                    SLC_DCOPY(&cnt, &t[(j - 1) * ldt], &int1,
                              &rb[dst_row + (posr + j - 1) * ldrb], &int1);
                }

                if (len_val + len2 + k >= lenr) {
                    for (i32 jj = k; jj >= 1; jj--) {
                        i32 src_row = k - jj;
                        i32 src_col = lenr - jj;
                        SLC_DCOPY(&jj, &t[src_row + src_col * ldt], &int1,
                                  &rb[(posr + lenr - jj) * ldrb], &int1);
                    }
                }
                posr += k;
            } else {
                for (i32 j = 1; j <= len_val + len2 + k; j++) {
                    i32 cnt = (j < k) ? j : k;
                    i32 dst_row = sizr - j + 1 - 1;
                    if (dst_row < 0) dst_row = 0;
                    SLC_DCOPY(&cnt, &t[(j - 1) * ldt], &int1,
                              &rb[dst_row + (posr + j - 1) * ldrb], &int1);
                    if (j > lenr - k) {
                        i32 cnt2 = sizr - j;
                        if (cnt2 > 0) {
                            SLC_DLASET("All", &cnt2, &int1, &zero, &zero,
                                       &rb[(posr + j - 1) * ldrb], &int1);
                        }
                    }
                }
                posr += k;
            }
            head = (head + k) % lenr;
        }
    } else {
        for (i32 i = pre; i <= pre + stps - 1; i++) {
            i32 ldw_cu = ldwork - pdw_val - 4 * k;
            mb02cu("Column", k, k, k, nb, t, ldt, dum, 1,
                   &dwork[head + 1], lenr, &rnk, ipvt, &dwork[pdw_val],
                   zero, &dwork[pdw_val + 4 * k], ldw_cu, &ierr);

            if (ierr != 0) {
                *info = 1;
                return;
            }

            i32 len_val = (n - i) * k - k;
            i32 max_len = lenr - head - k;
            if (len_val > max_len) len_val = max_len;
            if (len_val < 0) len_val = 0;

            i32 len2 = (n - i) * k - len_val - k;
            if (len2 > head) len2 = head;
            if (len2 < 0) len2 = 0;

            const char* struct_str = (len_val == lenr - k) ? triu : "N";

            i32 ldw_cv = ldwork - pdw_val - 4 * k;
            mb02cv("Column", struct_str, k, len_val, k, k, nb, -1, dum, 1,
                   dum, 1, &dwork[head + 1], lenr, &t[k], ldt,
                   dum, 1, &dwork[head + k + 1], lenr, &dwork[pdw_val],
                   &dwork[pdw_val + 4 * k], ldw_cv, &ierr);

            struct_str = ((n - i) * k >= lenr) ? triu : "N";

            mb02cv("Column", struct_str, k, len2, k, k, nb, -1, dum, 1,
                   dum, 1, &dwork[head + 1], lenr, &t[k + len_val], ldt,
                   dum, 1, &dwork[1], lenr, &dwork[pdw_val],
                   &dwork[pdw_val + 4 * k], ldw_cv, &ierr);

            SLC_DLASET("All", &k, &k, &zero, &zero, &dwork[head + 1], &lenr);

            if (ltri) {
                for (i32 j = 1; j <= k; j++) {
                    i32 cnt = sizr;
                    i32 alt_cnt = (n - i) * k - j + 1;
                    if (alt_cnt < cnt) cnt = alt_cnt;
                    SLC_DCOPY(&cnt, &t[(j - 1) + (j - 1) * ldt], &int1,
                              &rb[posr * ldrb], &int1);
                    posr++;
                }
            } else {
                for (i32 j = 1; j <= k; j++) {
                    i32 cnt = sizr - j + 1;
                    i32 alt_cnt = (n - i) * k - j + 1;
                    if (alt_cnt < cnt) cnt = alt_cnt;
                    SLC_DCOPY(&cnt, &t[(j - 1) + (j - 1) * ldt], &int1,
                              &rb[posr * ldrb], &int1);
                    if (lenr < (n - i) * k) {
                        i32 jminus1 = j - 1;
                        i32 start_row = cnt;
                        if (jminus1 > 0 && start_row < sizr) {
                            SLC_DLASET("All", &jminus1, &int1, &zero, &zero,
                                       &rb[start_row + posr * ldrb], &ldrb);
                        }
                    }
                    posr++;
                }
            }
            head = (head + k) % lenr;
        }
    }

    i32 wrkopt = kk + (ldwork > kk ? (ldwork - kk) : 0);
    dwork[0] = (f64)wrkopt;
}
