/**
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <complex.h>
#include <string.h>

void sb10md(i32 nc, i32 mp, i32 lendat, i32 f, i32* ord, i32 mnb,
            const i32* nblock, const i32* itype, f64 qutol,
            f64* a, i32 lda, f64* b, i32 ldb, f64* c, i32 ldc,
            const f64* d, i32 ldd, const f64* omega,
            i32* totord, f64* ad, i32 ldad, f64* bd, i32 ldbd,
            f64* cd, i32 ldcd, f64* dd, i32 lddd, f64* mju,
            i32* iwork, i32 liwork, f64* dwork, i32 ldwork,
            c128* zwork, i32 lzwork, i32* info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const f64 TWO = 2.0;
    const f64 THREE = 3.0;
    const i32 HNPTS = 2048;
    i32 one_int = 1;

    i32 i, k, w, ii, ic, info2;
    i32 cord, lord;
    i32 iwx, iwgjom, idwrk, ldsize;
    i32 iwb, icwrk, lcsize;
    i32 iwrfrd, iwifrd, iwad, iwbd, iwcd, iwdd;
    i32 dlwmax, clwmax, maxwrk, maxcwr;
    i32 lw1, lw2, lw3, lw4, lwa, lwb, mn;
    f64 rcnd, rcond, toler;
    f64 meqe, maqe, mod1, mod2, rqe, tol;
    c128 freq;

    *info = 0;

    iwx = mp * lendat;
    iwgjom = iwx + 2 * mnb - 1;
    idwrk = iwgjom + mp;
    ldsize = ldwork - idwrk;

    iwb = mp * mp;
    icwrk = iwb + nc * mp;
    lcsize = lzwork - icwrk;

    if (nc < 0) {
        *info = -1;
    } else if (mp < 0) {
        *info = -2;
    } else if (lendat < 2) {
        *info = -3;
    } else if (f < 0) {
        *info = -4;
    } else if (*ord > lendat - 1) {
        *info = -5;
    } else if (mnb < 1 || (mp > 0 && mnb > mp)) {
        *info = -6;
    } else if (lda < (nc > 1 ? nc : 1)) {
        *info = -11;
    } else if (ldb < (nc > 1 ? nc : 1)) {
        *info = -13;
    } else if (ldc < (mp > 1 ? mp : 1)) {
        *info = -15;
    } else if (ldd < (mp > 1 ? mp : 1)) {
        *info = -17;
    } else if (ldad < 1 || (qutol >= ZERO && mp > 0 && ldad < mp * (*ord))) {
        *info = -21;
    } else if (ldbd < 1 || (qutol >= ZERO && mp > 0 && ldbd < mp * (*ord))) {
        *info = -23;
    } else if (ldcd < 1 || (qutol >= ZERO && ldcd < mp + f)) {
        *info = -25;
    } else if (lddd < 1 || (qutol >= ZERO && lddd < mp + f)) {
        *info = -27;
    } else {
        i32 ii_max = (nc > 4 * mnb - 2) ? nc : 4 * mnb - 2;
        ii_max = (ii_max > mp) ? ii_max : mp;
        mn = (2 * lendat < 2 * (*ord) + 1) ? 2 * lendat : 2 * (*ord) + 1;
        lwa = idwrk - 1;
        lwb = lendat * (mp + 2) + (*ord) * ((*ord) + 2) + 1;
        lw1 = 2 * lendat + 4 * HNPTS;
        lw2 = lendat + 6 * HNPTS;
        i32 term1 = mn + 6 * (*ord) + 4;
        i32 term2 = 2 * mn + 1;
        lw3 = 2 * lendat * (2 * (*ord) + 1) +
              ((2 * lendat > 2 * (*ord) + 1) ? 2 * lendat : 2 * (*ord) + 1) +
              ((term1 > term2) ? term1 : term2);
        i32 term3 = (*ord) * (*ord) + 5 * (*ord);
        i32 term4 = 6 * (*ord) + 1 + ((1 < *ord) ? 1 : *ord);
        lw4 = (term3 > term4) ? term3 : term4;

        i32 lwa_term1 = nc + ((nc > mp - 1) ? nc : mp - 1);
        i32 lwa_term2 = 2 * mp * mp * mnb - mp * mp + 9 * mnb * mnb +
                        mp * mnb + 11 * mp + 33 * mnb - 11;
        dlwmax = lwa + ((lwa_term1 > lwa_term2) ? lwa_term1 : lwa_term2);

        i32 clw_term1 = icwrk - 1 + nc * nc + 2 * nc;
        i32 clw_term2 = 6 * mp * mp * mnb + 13 * mp * mp + 6 * mnb + 6 * mp - 3;
        clwmax = (clw_term1 > clw_term2) ? clw_term1 : clw_term2;

        if (qutol >= ZERO) {
            ii_max = (ii_max > 2 * (*ord) + 1) ? ii_max : 2 * (*ord) + 1;
            i32 dlw_term1 = lwb + 2;
            i32 dlw_term2 = lwb + lw1;
            i32 dlw_term3 = lwb + lw2;
            i32 dlw_term4 = lwb + lw3;
            i32 dlw_term5 = lwb + lw4;
            i32 dlw_term6 = lwb + 2 * (*ord);
            i32 dlw_max = dlw_term1;
            if (dlw_term2 > dlw_max) dlw_max = dlw_term2;
            if (dlw_term3 > dlw_max) dlw_max = dlw_term3;
            if (dlw_term4 > dlw_max) dlw_max = dlw_term4;
            if (dlw_term5 > dlw_max) dlw_max = dlw_term5;
            if (dlw_term6 > dlw_max) dlw_max = dlw_term6;
            if (dlw_max > dlwmax) dlwmax = dlw_max;

            i32 clw1 = lendat * (2 * (*ord) + 3);
            i32 clw2 = (*ord) * ((*ord) + 3) + 1;
            i32 clw_max = (clw1 > clw2) ? clw1 : clw2;
            if (clw_max > clwmax) clwmax = clw_max;
        }

        if (liwork < ii_max) {
            *info = -30;
        } else if (ldwork < (3 > dlwmax ? 3 : dlwmax)) {
            *info = -32;
        } else if (lzwork < clwmax) {
            *info = -34;
        }
    }

    if (*info != 0) {
        return;
    }

    if (*ord < 1) *ord = 1;
    *totord = 0;

    if (nc == 0 || mp == 0) {
        dwork[0] = THREE;
        dwork[1] = ZERO;
        dwork[2] = ONE;
        return;
    }

    toler = sqrt(SLC_DLAMCH("Epsilon"));
    rcond = ONE;
    maxcwr = clwmax;

    for (w = 0; w < lendat; w++) {
        freq = ZERO + I * omega[w];
        char baleig = 'C';
        char inita = (w == 0) ? 'G' : 'H';

        i32 tb05_info = slicot_tb05ad(baleig, inita, nc, mp, mp, freq,
                                       a, lda, b, ldb, c, ldc,
                                       &rcnd, zwork, mp,
                                       dwork, dwork, &zwork[iwb], nc,
                                       &dwork[idwrk], ldsize,
                                       &zwork[icwrk], lcsize);

        if (tb05_info > 0) {
            *info = 1;
            return;
        }

        if (rcnd < rcond) rcond = rcnd;
        if (w == 0) maxwrk = (i32)dwork[idwrk] + idwrk - 1;

        ic = 0;
        for (k = 0; k < mp; k++) {
            for (i = 0; i < mp; i++) {
                zwork[ic] = zwork[ic] + d[i + k * ldd];
                ic++;
            }
        }

        i32 ab13_info = ab13md('N', mp, zwork, mp, mnb, nblock, itype,
                               &dwork[iwx], &mju[w], &dwork[w * mp],
                               &dwork[iwgjom], iwork, &dwork[idwrk], ldsize,
                               &zwork[iwb], lzwork - iwb);

        if (ab13_info != 0) {
            *info = ab13_info + 1;
            return;
        }

        if (w == 0) {
            i32 opt_dw = (i32)dwork[idwrk] + idwrk - 1;
            if (opt_dw > maxwrk) maxwrk = opt_dw;
            i32 opt_cw = (i32)creal(zwork[iwb]) + iwb - 1;
            if (opt_cw > maxcwr) maxcwr = opt_cw;
        }

        if (dwork[w * mp + mp - 1] != ZERO) {
            f64 scale = ONE / dwork[w * mp + mp - 1];
            SLC_DSCAL(&mp, &scale, &dwork[w * mp], &one_int);
        }
    }

    if (qutol < ZERO) {
        dwork[0] = (f64)maxwrk;
        dwork[1] = (f64)maxcwr;
        dwork[2] = rcond;
        return;
    }

    iwrfrd = iwx;
    iwifrd = iwrfrd + lendat;
    iwad = iwifrd + lendat;
    iwbd = iwad + (*ord) * (*ord);
    iwcd = iwbd + (*ord);
    iwdd = iwcd + (*ord);
    idwrk = iwdd + 1;
    ldsize = ldwork - idwrk;

    icwrk = (*ord) + 2;
    lcsize = lzwork - icwrk;
    tol = -ONE;

    for (i = 0; i < lendat; i++) {
        dwork[iwifrd + i] = ZERO;
    }

    i32 mp_ord = mp * (*ord);
    i32 mp_f = mp + f;
    SLC_DLASET("Full", &mp_ord, &mp_ord, &ZERO, &ZERO, ad, &ldad);
    SLC_DLASET("Full", &mp_ord, &mp_f, &ZERO, &ZERO, bd, &ldbd);
    SLC_DLASET("Full", &mp_f, &mp_ord, &ZERO, &ZERO, cd, &ldcd);
    SLC_DLASET("Full", &mp_f, &mp_f, &ZERO, &ONE, dd, &lddd);

    for (ii = 0; ii < mp; ii++) {
        SLC_DCOPY(&lendat, &dwork[ii], &mp, &dwork[iwrfrd], &one_int);

        cord = 1;

        do {
            lord = cord;
            i32 sb10yd_n = lord;
            sb10yd(0, 1, lendat, &dwork[iwrfrd], &dwork[iwifrd],
                   omega, &sb10yd_n, &dwork[iwad], *ord, &dwork[iwbd],
                   &dwork[iwcd], &dwork[iwdd], tol,
                   iwork, &dwork[idwrk], ldsize, zwork, lzwork, &info2);
            lord = sb10yd_n;

            if (info2 != 0) {
                *info = 10 + info2;
                return;
            }

            meqe = ZERO;
            maqe = ZERO;

            for (w = 0; w < lendat; w++) {
                freq = ZERO + I * omega[w];
                char baleig = 'C';
                char inita = 'H';

                i32 tb05_info2 = slicot_tb05ad(baleig, inita, lord, 1, 1, freq,
                                               &dwork[iwad], *ord,
                                               &dwork[iwbd], *ord,
                                               &dwork[iwcd], 1,
                                               &rcnd, zwork, 1,
                                               &dwork[idwrk], &dwork[idwrk],
                                               &zwork[1], *ord,
                                               &dwork[idwrk], ldsize,
                                               &zwork[icwrk], lcsize);

                if (tb05_info2 > 0) {
                    *info = 1;
                    return;
                }

                if (rcnd < rcond) rcond = rcnd;
                if (w == 0) {
                    i32 opt_dw = (i32)dwork[idwrk] + idwrk - 1;
                    if (opt_dw > maxwrk) maxwrk = opt_dw;
                }

                zwork[0] = zwork[0] + dwork[iwdd];

                mod1 = fabs(dwork[iwrfrd + w]);
                mod2 = cabs(zwork[0]);
                rqe = fabs((mod1 - mod2) / (mod1 + toler));
                meqe += rqe;
                if (rqe > maqe) maqe = rqe;
            }

            meqe /= lendat;

            if ((meqe + maqe) / TWO <= qutol || cord == *ord) {
                break;
            }

            cord++;
        } while (1);

        *totord += lord;

        i32 totord_minus_lord_plus_1 = *totord - lord;
        SLC_DLACPY("Full", &lord, &lord, &dwork[iwad], ord,
                   &ad[totord_minus_lord_plus_1 + totord_minus_lord_plus_1 * ldad], &ldad);
        SLC_DCOPY(&lord, &dwork[iwbd], &one_int,
                  &bd[totord_minus_lord_plus_1 + ii * ldbd], &one_int);
        SLC_DCOPY(&lord, &dwork[iwcd], &one_int,
                  &cd[ii + totord_minus_lord_plus_1 * ldcd], &ldcd);

        dd[ii + ii * lddd] = dwork[iwdd];
    }

    dwork[0] = (f64)maxwrk;
    dwork[1] = (f64)maxcwr;
    dwork[2] = rcond;
}
