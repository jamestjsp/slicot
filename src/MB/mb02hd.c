/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <ctype.h>
#include <stdbool.h>

void mb02hd(const char* triu, i32 k, i32 l, i32 m, i32 ml, i32 n, i32 nu,
            i32 p, i32 s, f64* tc, i32 ldtc, f64* tr, i32 ldtr,
            f64* rb, i32 ldrb, f64* dwork, i32 ldwork, i32* info)
{
    const f64 zero = 0.0;
    const f64 one = 1.0;

    char triu_u = (char)toupper((unsigned char)triu[0]);
    bool ltri = (triu_u == 'T');

    *info = 0;

    i32 x = ml + nu + 1;
    if (x > n) x = n;
    i32 lenr = x * l;
    i32 sizr;
    if (ltri) {
        sizr = (ml + nu) * l + 1;
        if (sizr > n * l) sizr = n * l;
    } else {
        sizr = lenr;
    }

    i32 wrkmin;
    if (p == 0) {
        i32 opt1 = lenr * l + (2 * nu + 1) * l * k;
        i32 opt2 = 2 * lenr * (k + l) + (6 + x) * l;
        wrkmin = 1 + ((opt1 > opt2) ? opt1 : opt2);
    } else {
        wrkmin = 1 + 2 * lenr * (k + l) + (6 + x) * l;
    }
    i32 posr = 0;

    i32 mk = m * k;
    i32 nl_val = n * l;
    i32 min_mk_nl = (mk < nl_val) ? mk : nl_val;

    if (!ltri && triu_u != 'N') {
        *info = -1;
    } else if (k < 0) {
        *info = -2;
    } else if (l < 0) {
        *info = -3;
    } else if (m < 1) {
        *info = -4;
    } else if (ml >= m || (ml + 1) * k < l) {
        *info = -5;
    } else if (mk <= nl_val) {
        i32 lim1 = m - (mk - 1) / l - 1;
        i32 lim2 = m - mk / l;
        if (ml < lim1 || (ml < lim2 && (mk % l) < k)) {
            *info = -5;
        }
    } else {
        if (ml * k < n * (l - k)) {
            *info = -5;
        }
    }
    if (*info == 0) {
        if (n < 1) {
            *info = -6;
        } else if (nu >= n || nu < 0 || (ltri && nu < 1 - ml) ||
                   (m + nu) * l < min_mk_nl) {
            *info = -7;
        } else if (p < 0 || (p * l - l) >= min_mk_nl) {
            *info = -8;
        } else if (s < 0 || (p + s - 1) * l >= min_mk_nl) {
            *info = -9;
        } else if (p == 0 && ldtc < ((1 > (ml + 1) * k) ? 1 : (ml + 1) * k)) {
            *info = -11;
        } else if (p == 0 && ldtr < ((1 > k) ? 1 : k)) {
            *info = -13;
        } else if (ldrb < ((sizr > 1) ? sizr : 1)) {
            *info = -15;
        }
    }

    bool lquery = (ldwork == -1);
    i32 wrkopt = wrkmin;
    i32 ierr = 0;

    if (*info == 0) {
        if (p == 0) {
            i32 lenc = (ml + 1) * k;
            i32 lenl_temp = ml + 1 + ((nu < n - m) ? nu : (n - m));
            i32 lenl = (lenl_temp > 0) ? lenl_temp : 0;
            (void)lenl;
            i32 pdw = (lenr + lenc) * l;
            if (lquery) {
                i32 neg1 = -1;
                SLC_DGEQRF(&lenc, &l, dwork, &lenc, dwork, dwork, &neg1, &ierr);
                i32 opt1 = 1 + (i32)dwork[0] + pdw + l;
                if (opt1 > wrkopt) wrkopt = opt1;
                SLC_DORGQR(&lenc, &l, &l, dwork, &lenc, dwork, dwork, &neg1, &ierr);
                i32 opt2 = (i32)dwork[0] + pdw + l;
                if (opt2 > wrkopt) wrkopt = opt2;
            }
        }
        i32 kk = 2 * lenr * (k + l) + 1 + 6 * l;
        i32 neg1 = -1;
        i32 lenr_max = (lenr > 1) ? lenr : 1;
        SLC_DGELQF(&lenr_max, &l, dwork, &lenr_max, dwork, dwork, &neg1, &ierr);
        i32 cand = kk + (i32)dwork[0];
        if (cand > wrkopt) wrkopt = cand;

        if (ldwork < wrkmin && !lquery) {
            dwork[0] = (f64)wrkmin;
            *info = -17;
        }
    }

    if (*info != 0) {
        return;
    }
    if (lquery) {
        dwork[0] = (f64)wrkopt;
        return;
    }

    if (l * k * s == 0) {
        dwork[0] = one;
        return;
    }

    wrkopt = 1;

    i32 int1 = 1;
    i32 ipvt_dummy[1] = {0};
    i32 pdr, pnr, pfr, pdw;
    i32 pre, stps;
    i32 head;

    if (p == 0) {
        i32 lenc = (ml + 1) * k;
        i32 lenl_temp = ml + 1 + ((nu < n - m) ? nu : (n - m));
        i32 lenl = (lenl_temp > 0) ? lenl_temp : 0;
        i32 pdc = lenr * l;
        pdw = (lenr + lenc) * l;

        SLC_DLACPY("All", &lenc, &l, tc, &ldtc, &dwork[pdc], &lenc);
        i32 lenc_work = ldwork - pdw - l;
        SLC_DGEQRF(&lenc, &l, &dwork[pdc], &lenc, &dwork[pdw],
                   &dwork[pdw + l], &lenc_work, &ierr);
        i32 opt = (i32)dwork[pdw + l] + pdw + l;
        if (opt > wrkopt) wrkopt = opt;

        ma02ad("Upper part", l, l, &dwork[pdc], lenc, &dwork[0], lenr);

        lenc_work = ldwork - pdw - l;
        SLC_DORGQR(&lenc, &l, &l, &dwork[pdc], &lenc, &dwork[pdw],
                   &dwork[pdw + l], &lenc_work, &ierr);
        opt = (i32)dwork[pdw + l] + pdw + l;
        if (opt > wrkopt) wrkopt = opt;

        i32 pt = lenc - 2 * k;
        i32 i = pdw;
        for (; i < pdw + ml * k * l; i += k * l) {
            SLC_DLACPY("All", &k, &l, &tc[pt], &ldtc, &dwork[i], &k);
            pt -= k;
        }

        pdw = i;
        pdr = l;
        i32 len = nu * l;
        i32 lenr_minus_l = lenr - l;
        SLC_DLASET("All", &lenr_minus_l, &l, &zero, &zero, &dwork[pdr], &lenr);
        i32 n_minus_1 = n - 1;

        i32 pdc_cur = pdc;
        i32 pdw_cur = pdw;
        for (i32 ii = 0; ii < ml + 1; ii++) {
            i32 cols = (ii < n_minus_1) ? ii : n_minus_1;
            cols = cols * l;
            if (cols > 0) {
                SLC_DGEMM("Transpose", "NonTranspose", &cols, &l, &k, &one,
                          &dwork[pdw_cur], &k, &dwork[pdc_cur], &lenc, &one,
                          &dwork[pdr], &lenr);
            }
            if (len > 0) {
                i32 pdr_off = pdr + ii * l;
                SLC_DGEMM("Transpose", "NonTranspose", &len, &l, &k, &one,
                          tr, &ldtr, &dwork[pdc_cur], &lenc, &one,
                          &dwork[pdr_off], &lenr);
            }
            pdw_cur -= k * l;
            pdc_cur += k;
            if (ii >= n - nu - 1 && len > 0) len -= l;
        }

        if (ltri) {
            for (i32 ii = 0; ii < l; ii++) {
                i32 copy_len = sizr;
                i32 rem = n * l - ii;
                if (copy_len > rem) copy_len = rem;
                SLC_DCOPY(&copy_len, &dwork[ii * lenr + ii], &int1,
                          &rb[posr * ldrb], &int1);
                posr++;
            }
        } else {
            for (i32 ii = 0; ii < l; ii++) {
                i32 copy_len = lenr - ii;
                SLC_DCOPY(&copy_len, &dwork[ii * lenr + ii], &int1,
                          &rb[posr * ldrb], &int1);
                if (lenr < n * l && ii > 0) {
                    i32 zero_len = ii;
                    i32 zero_off = lenr - ii;
                    SLC_DLASET("All", &zero_len, &int1, &zero, &zero,
                               &rb[zero_off + posr * ldrb], &ldrb);
                }
                posr++;
            }
        }

        if (n == 1) {
            dwork[0] = (f64)wrkopt;
            return;
        }

        pdr = lenr * l;
        i32 nu_l = nu * l;
        ma02ad("All", k, nu_l, tr, ldtr, &dwork[pdr], lenr);
        i32 zero_rows = lenr - nu * l;
        SLC_DLASET("All", &zero_rows, &k, &zero, &zero,
                   &dwork[pdr + nu * l], &lenr);

        pnr = pdr + lenr * k;
        i32 copy_len = lenr - l;
        SLC_DLACPY("All", &copy_len, &l, &dwork[l], &lenr, &dwork[pnr], &lenr);
        SLC_DLASET("All", &l, &l, &zero, &zero, &dwork[pnr + lenr - l], &lenr);

        pfr = pnr + lenr * l;

        i32 start_pdw = pfr + ((m - ml - 1) * l) % lenr;
        pt = ml * k;
        i32 min_ml_lenl = ((ml + 1) < lenl) ? (ml + 1) : lenl;
        for (i32 ii = 0; ii < min_ml_lenl; ii++) {
            ma02ad("All", k, l, &tc[pt], ldtc, &dwork[start_pdw], lenr);
            pt -= k;
            start_pdw = pfr + (start_pdw + l - pfr) % lenr;
        }
        pt = 0;
        for (i32 ii = ml + 1; ii < lenl; ii++) {
            ma02ad("All", k, l, &tr[pt * ldtr], ldtr, &dwork[start_pdw], lenr);
            pt += l;
            start_pdw = pfr + (start_pdw + l - pfr) % lenr;
        }
        pre = 1;
        stps = s - 1;
    } else {
        pdr = lenr * l;
        pnr = pdr + lenr * k;
        pfr = pnr + lenr * l;
        pre = p;
        stps = s;
    }

    pdw = pfr + lenr * k;
    head = ((pre - 1) * l) % lenr;

    i32 nb = (ldwork - (pdw + 6 * l)) / lenr;
    if (nb > l) nb = l;
    i32 nbmin = 2;
    i32 neg1 = -1;
    nbmin = SLC_ILAENV(&nbmin, "DGELQF", " ", &lenr, &l, &neg1, &neg1);
    if (nbmin < 2) nbmin = 2;
    if (nb < nbmin) nb = 0;

    for (i32 ii = pre; ii < pre + stps; ii++) {
        i32 col2;
        if (ii < m - ml) {
            col2 = l;
        } else {
            col2 = k + l;
        }

        i32 kk = l;
        i32 rem = mk - ii * l;
        if (kk > rem) kk = rem;

        i32 kk_plus_k = kk + k;
        i32 rnk_dummy;
        mb02cu("Column", kk, kk_plus_k, col2, nb, &dwork[0], lenr,
               &dwork[pdr + head], lenr, &dwork[pnr + head], lenr,
               &rnk_dummy, ipvt_dummy, &dwork[pdw], zero, &dwork[pdw + 6 * l],
               ldwork - pdw - 6 * l, &ierr);
        if (ierr != 0) {
            *info = 1;
            return;
        }

        i32 len = (n - ii) * l - kk;
        i32 max_len = lenr - head - kk;
        if (len > max_len) len = max_len;
        if (len < 0) len = 0;
        i32 len2 = (n - ii) * l - len - kk;
        if (len2 > head) len2 = head;
        if (len2 < 0) len2 = 0;

        const char* struct_str;
        if (len == (lenr - kk)) {
            struct_str = triu;
        } else {
            struct_str = "N";
        }
        mb02cv("Column", struct_str, kk, len, kk_plus_k, col2, nb, -1,
               &dwork[0], lenr, &dwork[pdr + head], lenr,
               &dwork[pnr + head], lenr, &dwork[kk], lenr,
               &dwork[pdr + head + kk], lenr, &dwork[pnr + head + kk],
               lenr, &dwork[pdw], &dwork[pdw + 6 * l],
               ldwork - pdw - 6 * l, &ierr);

        if ((n - ii) * l >= lenr) {
            struct_str = triu;
        } else {
            struct_str = "N";
        }

        mb02cv("Column", struct_str, kk, len2, kk_plus_k, col2, nb, -1,
               &dwork[0], lenr, &dwork[pdr + head], lenr,
               &dwork[pnr + head], lenr, &dwork[kk + len], lenr,
               &dwork[pdr], lenr, &dwork[pnr], lenr,
               &dwork[pdw], &dwork[pdw + 6 * l],
               ldwork - pdw - 6 * l, &ierr);

        i32 zero_cols = k + col2;
        SLC_DLASET("All", &l, &zero_cols, &zero, &zero,
                   &dwork[pdr + head], &lenr);

        if (ltri) {
            for (i32 j = 0; j < kk; j++) {
                i32 copy_len = sizr;
                i32 rem2 = (n - ii) * l - j;
                if (copy_len > rem2) copy_len = rem2;
                SLC_DCOPY(&copy_len, &dwork[j * lenr + j], &int1,
                          &rb[posr * ldrb], &int1);
                posr++;
            }
        } else {
            for (i32 j = 0; j < kk; j++) {
                i32 copy_len = sizr - j;
                i32 rem2 = (n - ii) * l - j;
                if (copy_len > rem2) copy_len = rem2;
                SLC_DCOPY(&copy_len, &dwork[j * lenr + j], &int1,
                          &rb[posr * ldrb], &int1);
                if (lenr < (n - ii) * l && j > 0) {
                    i32 zero_len = j;
                    i32 off = copy_len;
                    i32 rem3 = sizr - j - copy_len;
                    if (zero_len > rem3) zero_len = rem3;
                    if (zero_len > 0) {
                        SLC_DLASET("All", &zero_len, &int1, &zero, &zero,
                                   &rb[off + posr * ldrb], &ldrb);
                    }
                }
                posr++;
            }
        }

        head = (head + l) % lenr;
    }

    dwork[0] = (f64)wrkopt;
}
