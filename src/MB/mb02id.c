/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <ctype.h>
#include <math.h>
#include <stdbool.h>

void mb02id(const char* job, i32 k, i32 l, i32 m, i32 n,
            i32 rb, i32 rc, f64* tc, i32 ldtc, f64* tr, i32 ldtr,
            f64* b, i32 ldb, f64* c, i32 ldc,
            f64* dwork, i32 ldwork, i32* info)
{
    const f64 zero = 0.0;
    const f64 one = 1.0;
    const f64 mone = -1.0;
    i32 int1 = 1;

    char job_u = (char)toupper((unsigned char)job[0]);
    bool compo = (job_u == 'O') || (job_u == 'A');
    bool compu = (job_u == 'U') || (job_u == 'A');

    i32 mk = m * k;
    i32 nl = n * l;

    i32 x = 2 * n * l * (l + k) + (6 + n) * l;
    i32 tmp = (nl + mk + 1) * l + mk;
    if (tmp > x) x = tmp;

    i32 y = n * mk * l + nl;

    i32 wrkmin;
    i32 minmn = (m < n) ? m : n;
    if (minmn == 1) {
        wrkmin = (mk > 1) ? mk : 1;
        if (compo && rb > wrkmin) wrkmin = rb;
        if (compu && rc > wrkmin) wrkmin = rc;
        wrkmin = y + wrkmin;
        if (wrkmin < 1) wrkmin = 1;
    } else {
        wrkmin = x;
        if (compo) {
            tmp = nl * rb + 1;
            if (tmp > wrkmin) wrkmin = tmp;
        }
        if (compu) {
            tmp = nl * rc + 1;
            if (tmp > wrkmin) wrkmin = tmp;
        }
    }

    i32 wrkopt = 1;
    bool lquery = (ldwork == -1);

    *info = 0;

    if (!compo && !compu) {
        *info = -1;
    } else if (k < 0) {
        *info = -2;
    } else if (l < 0) {
        *info = -3;
    } else if (m < 0) {
        *info = -4;
    } else if (n < 0 || nl > mk) {
        *info = -5;
    } else if (compo && rb < 0) {
        *info = -6;
    } else if (compu && rc < 0) {
        *info = -7;
    } else if (ldtc < (mk > 1 ? mk : 1)) {
        *info = -9;
    } else if (ldtr < (k > 1 ? k : 1)) {
        *info = -11;
    } else if (ldb < 1 || (compo && ldb < mk)) {
        *info = -13;
    } else if (ldc < 1 || (compu && ldc < mk)) {
        *info = -15;
    }

    bool compus = compu;
    if (compo && (nl == 0 || rb == 0)) compo = false;
    if (compu && (nl == 0 || rc == 0)) compu = false;

    if (*info == 0) {
        if (lquery) {
            i32 pdw = k * l * m * n;
            i32 ierr;
            if ((m < n ? m == 1 : n == 1)) {
                if (compo) {
                    SLC_DGELS("N", &mk, &nl, &rb, dwork, &mk, b, &ldb, dwork, &int1, &ierr);
                    i32 opt = (i32)dwork[0] + pdw;
                    if (opt > wrkopt) wrkopt = opt;
                }
                if (compu) {
                    SLC_DGELS("T", &mk, &nl, &rc, dwork, &mk, c, &ldc, dwork, &int1, &ierr);
                    i32 opt = (i32)dwork[0] + pdw;
                    if (opt > wrkopt) wrkopt = opt;
                }
            } else {
                if (compo) {
                    mb02kd("C", "T", k, l, m, n, rb, one, zero, tc, ldtc, tr, ldtr,
                           b, ldb, dwork, nl, dwork, -1, &ierr);
                    i32 opt = (i32)dwork[0] + nl * rb;
                    if (opt > wrkopt) wrkopt = opt;
                }
                pdw = (nl + mk) * l;
                SLC_DGEQRF(&mk, &l, dwork, &mk, dwork, dwork, &int1, &ierr);
                tmp = (i32)dwork[0] + pdw + l;
                if (tmp > wrkopt) wrkopt = tmp;

                SLC_DORGQR(&mk, &l, &l, dwork, &mk, dwork, dwork, &int1, &ierr);
                tmp = (i32)dwork[0] + pdw + l;
                if (tmp > wrkopt) wrkopt = tmp;

                i32 nm1 = n - 1;
                mb02kd("R", "T", k, l, m, nm1, l, one, zero, tc, ldtc, tr, ldtr,
                       dwork, mk, dwork, nl, dwork, -1, &ierr);
                tmp = (i32)dwork[0] + pdw;
                if (tmp > wrkopt) wrkopt = tmp;

                pdw = 2 * nl * (l + k);
                i32 nlmax = (nl > 1) ? nl : 1;
                SLC_DGELQF(&nl, &l, dwork, &nlmax, dwork, dwork, &int1, &ierr);
                tmp = pdw + 6 * l + (i32)dwork[0];
                if (tmp > wrkopt) wrkopt = tmp;

                if (compu) {
                    mb02kd("C", "N", k, l, m, n, rc, one, zero, tc, ldtc, tr, ldtr,
                           c, ldc, dwork, mk, dwork, -1, &ierr);
                    tmp = (i32)dwork[0] + mk * rc;
                    if (tmp > wrkopt) wrkopt = tmp;
                }
            }
        } else if (ldwork < wrkmin) {
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

    if (compus && (nl == 0 || rc == 0)) {
        SLC_DLASET("F", &mk, &rc, &zero, &zero, c, &ldc);
    }

    if (!compo && !compu) {
        dwork[0] = one;
        return;
    }

    if (minmn == 1) {
        i32 pdw = k * l * m * n;
        i32 ierr;
        i32 nm1l = (n - 1) * l;
        i32 wrkrem = ldwork - pdw;

        if (compo) {
            SLC_DLACPY("A", &mk, &l, tc, &ldtc, dwork, &mk);
            if (nm1l > 0) {
                SLC_DLACPY("A", &k, &nm1l, tr, &ldtr, &dwork[k * l], &mk);
            }
            SLC_DGELS("N", &mk, &nl, &rb, dwork, &mk, b, &ldb, &dwork[pdw], &wrkrem, &ierr);
            i32 opt = (i32)dwork[pdw] + pdw;
            if (opt > wrkopt) wrkopt = opt;
        }
        if (compu) {
            SLC_DLACPY("A", &mk, &l, tc, &ldtc, dwork, &mk);
            if (nm1l > 0) {
                SLC_DLACPY("A", &k, &nm1l, tr, &ldtr, &dwork[k * l], &mk);
            }
            SLC_DGELS("T", &mk, &nl, &rc, dwork, &mk, c, &ldc, &dwork[pdw], &wrkrem, &ierr);
            i32 opt = (i32)dwork[pdw] + pdw;
            if (opt > wrkopt) wrkopt = opt;
        }
        dwork[0] = (f64)wrkopt;
        return;
    }

    i32 ierr;
    i32 nm1 = n - 1;
    i32 nm1l = nm1 * l;

    if (compo) {
        i32 wrkrem = ldwork - nl * rb;
        mb02kd("C", "T", k, l, m, n, rb, one, zero, tc, ldtc, tr, ldtr,
               b, ldb, dwork, nl, &dwork[nl * rb], wrkrem, &ierr);
        i32 opt = (i32)dwork[nl * rb] + nl * rb;
        if (opt > wrkopt) wrkopt = opt;
        SLC_DLACPY("A", &nl, &rb, dwork, &nl, b, &ldb);
    }

    i32 pdw = nl * l + 1 - 1;
    i32 mkl = mk * l;
    i32 mkl_plus_l = mkl + l;
    i32 wrkrem = ldwork - pdw - mkl_plus_l;

    SLC_DLACPY("A", &mk, &l, tc, &ldtc, &dwork[pdw], &mk);
    SLC_DGEQRF(&mk, &l, &dwork[pdw], &mk, &dwork[pdw + mkl], &dwork[pdw + mkl_plus_l], &wrkrem, &ierr);

    i32 opt = (i32)dwork[pdw + mkl_plus_l] + pdw + mkl_plus_l;
    if (opt > wrkopt) wrkopt = opt;

    for (i32 i = pdw; i < pdw + mkl; i += mk + 1) {
        if (dwork[i] == zero) {
            *info = 1;
            return;
        }
    }

    ma02ad("U", l, l, &dwork[pdw], mk, dwork, nl);

    SLC_DORGQR(&mk, &l, &l, &dwork[pdw], &mk, &dwork[pdw + mkl], &dwork[pdw + mkl_plus_l], &wrkrem, &ierr);
    opt = (i32)dwork[pdw + mkl_plus_l] + pdw + mkl_plus_l;
    if (opt > wrkopt) wrkopt = opt;

    wrkrem = ldwork - pdw - mkl + 1;
    mb02kd("R", "T", k, l, m, nm1, l, one, zero, tc, ldtc, tr, ldtr,
           &dwork[pdw], mk, &dwork[l], nl, &dwork[pdw + mkl], wrkrem, &ierr);
    opt = (i32)dwork[pdw + mkl] + pdw + mkl;
    if (opt > wrkopt) wrkopt = opt;

    i32 ppr = nl * l;
    i32 pnr = nl * (l + k);

    ma02ad("A", k, nm1l, tr, ldtr, &dwork[ppr + l], nl);

    SLC_DLACPY("A", &nm1l, &l, &dwork[l], &nl, &dwork[pnr + l], &nl);

    i32 pt = (m - 1) * k;
    i32 pdw2 = pnr + nl * l + l;
    i32 imin = (m < nm1) ? m : nm1;
    for (i32 i = 0; i < imin; i++) {
        ma02ad("A", k, l, &tc[pt], ldtc, &dwork[pdw2], nl);
        pt -= k;
        pdw2 += l;
    }

    pt = 0;
    for (i32 i = m; i < nm1; i++) {
        ma02ad("A", k, l, &tr[pt * ldtr], ldtr, &dwork[pdw2], nl);
        pt += l;
        pdw2 += l;
    }

    if (compo) {
        SLC_DTRSM("L", "L", "N", "N", &l, &rb, &one, dwork, &nl, b, &ldb);
        SLC_DGEMM("N", "N", &nm1l, &rb, &l, &one, &dwork[l], &nl, b, &ldb, &mone, &b[l], &ldb);
        SLC_DTRSM("L", "L", "T", "N", &l, &rb, &one, dwork, &nl, b, &ldb);
    }

    if (compu) {
        SLC_DTRSM("L", "L", "N", "N", &l, &rc, &one, dwork, &nl, c, &ldc);
        SLC_DGEMM("N", "N", &nm1l, &rc, &l, &one, &dwork[l], &nl, c, &ldc, &mone, &c[l], &ldc);
        SLC_DTRSM("L", "L", "T", "N", &l, &rc, &one, dwork, &nl, c, &ldc);
    }

    i32 pdi = nm1l;
    SLC_DLACPY("L", &l, &l, dwork, &nl, &dwork[pdi], &nl);
    SLC_DTRTRI("L", "N", &l, &dwork[pdi], &nl, &ierr);

    i32 lm1 = l - 1;
    if (lm1 > 0) {
        i32 nl2m1_l = (2 * n - 1) * l;
        ma02ad("L", lm1, l, &dwork[pdi + 1], nl, &dwork[nl2m1_l], nl);
        SLC_DLASET("L", &lm1, &l, &zero, &zero, &dwork[pdi + 1], &nl);
    }

    SLC_DLACPY("U", &l, &l, &dwork[pdi], &nl, &dwork[pnr], &nl);
    i32 lm1_b = l - 1;
    if (lm1_b > 0) {
        SLC_DLASET("L", &lm1_b, &l, &zero, &zero, &dwork[pnr + 1], &nl);
    }
    SLC_DLASET("A", &l, &k, &zero, &zero, &dwork[ppr], &nl);
    SLC_DLASET("A", &l, &k, &zero, &zero, &dwork[pnr + nl * l], &nl);

    i32 ppi = ppr;
    ppr = ppr + l;
    i32 pni = pnr;
    pnr = pnr + l;

    pdw2 = 2 * nl * (l + k);
    i32 len = nm1l;

    i32 nb = (ldwork - pdw2 - 6 * l) / nl;
    if (nb > l) nb = l;
    i32 nbmin = SLC_ILAENV(&int1, "DGELQF", " ", &nl, &l, &int1, &int1);
    if (nbmin < 2) nbmin = 2;
    if (nb < nbmin) nb = 0;

    i32 rnk;
    i32 ipvt_dummy = 0;

    for (i32 i = l; i < nl; i += l) {
        i32 lk = l + k;
        i32 wrkrem2 = ldwork - pdw2 - 6 * l;
        mb02cu("C", l, lk, lk, nb, dwork, nl, &dwork[ppr], nl, &dwork[pnr], nl,
               &rnk, &ipvt_dummy, &dwork[pdw2], zero, &dwork[pdw2 + 6 * l], wrkrem2, &ierr);
        if (ierr != 0) {
            *info = 1;
            return;
        }

        i32 len_ml = len - l;
        i32 m1 = -1;
        mb02cv("C", "N", l, len_ml, lk, lk, nb, m1, dwork, nl, &dwork[ppr], nl, &dwork[pnr], nl,
               &dwork[l], nl, &dwork[ppr + l], nl, &dwork[pnr + l], nl,
               &dwork[pdw2], &dwork[pdw2 + 6 * l], wrkrem2, &ierr);

        pdi -= l;

        if (compo) {
            SLC_DTRSM("L", "L", "N", "N", &l, &rb, &mone, dwork, &nl, &b[i], &ldb);
            if (len > l) {
                SLC_DGEMM("N", "N", &len_ml, &rb, &l, &one, &dwork[l], &nl, &b[i], &ldb, &one, &b[i + l], &ldb);
            }
        }

        if (compu) {
            SLC_DTRSM("L", "L", "N", "N", &l, &rc, &mone, dwork, &nl, &c[i], &ldc);
            if (len > l) {
                SLC_DGEMM("N", "N", &len_ml, &rc, &l, &one, &dwork[l], &nl, &c[i], &ldc, &one, &c[i + l], &ldc);
            }
        }

        SLC_DLASET("A", &l, &l, &zero, &zero, &dwork[pdi], &nl);

        i32 ipl = i + l;
        mb02cv("C", "T", l, ipl, lk, lk, nb, m1, dwork, nl, &dwork[ppr], nl, &dwork[pnr], nl,
               &dwork[pdi], nl, &dwork[ppi], nl, &dwork[pni], nl,
               &dwork[pdw2], &dwork[pdw2 + 6 * l], wrkrem2, &ierr);

        if (compo) {
            i32 im1 = i;
            SLC_DGEMM("N", "N", &im1, &rb, &l, &one, &dwork[pdi], &nl, &b[i], &ldb, &one, b, &ldb);
            i32 nm1l_idx = nm1l;
            SLC_DTRMM("L", "U", "N", "N", &l, &rb, &one, &dwork[nm1l_idx], &nl, &b[i], &ldb);
        }

        if (compu) {
            i32 im1 = i;
            SLC_DGEMM("N", "N", &im1, &rc, &l, &one, &dwork[pdi], &nl, &c[i], &ldc, &one, c, &ldc);
            i32 nm1l_idx = nm1l;
            SLC_DTRMM("L", "U", "N", "N", &l, &rc, &one, &dwork[nm1l_idx], &nl, &c[i], &ldc);
        }

        len -= l;
        pnr += l;
        ppr += l;
    }

    if (compu) {
        wrkrem = ldwork - mk * rc;
        mb02kd("C", "N", k, l, m, n, rc, one, zero, tc, ldtc, tr, ldtr,
               c, ldc, dwork, mk, &dwork[mk * rc], wrkrem, &ierr);
        opt = (i32)dwork[mk * rc] + mk * rc;
        if (opt > wrkopt) wrkopt = opt;
        SLC_DLACPY("A", &mk, &rc, dwork, &mk, c, &ldc);
    }

    dwork[0] = (f64)wrkopt;
}
