/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot/lapack_aux.h"
#include "slicot_blas.h"
#include <math.h>
#include <stdlib.h>

i32 slicot_zgegv(const char *jobvl, const char *jobvr, i32 n, c128 *a, i32 lda,
                 c128 *b, i32 ldb, c128 *alpha, c128 *beta,
                 c128 *vl, i32 ldvl, c128 *vr, i32 ldvr,
                 c128 *work, i32 lwork, f64 *rwork) {
    const f64 zero = 0.0;
    const f64 one = 1.0;
    const c128 czero = 0.0 + 0.0 * I;
    const c128 cone = 1.0 + 0.0 * I;

    i32 ijobvl, ijobvr;
    bool ilvl, ilvr, ilv, lquery;
    i32 lwkmin, lwkopt, nb, nb1, nb2, nb3, lopt;
    i32 ilo, ihi, iinfo;
    i32 ileft, iright, irwork;
    i32 itau, iwork;
    i32 irows, icols, in;
    f64 eps, safmin, safmax;
    f64 anrm, bnrm, anrm1, anrm2, bnrm1, bnrm2;
    f64 temp, scale;
    f64 absai, absar, absb, salfai, salfar, sbeta;
    bool ilimit;
    char chtemp;
    i32 ldumma = 1;
    i32 i_one = 1;
    i32 i_neg_one = -1;

    if (*jobvl == 'N' || *jobvl == 'n') {
        ijobvl = 1;
        ilvl = false;
    } else if (*jobvl == 'V' || *jobvl == 'v') {
        ijobvl = 2;
        ilvl = true;
    } else {
        ijobvl = -1;
        ilvl = false;
    }

    if (*jobvr == 'N' || *jobvr == 'n') {
        ijobvr = 1;
        ilvr = false;
    } else if (*jobvr == 'V' || *jobvr == 'v') {
        ijobvr = 2;
        ilvr = true;
    } else {
        ijobvr = -1;
        ilvr = false;
    }
    ilv = ilvl || ilvr;

    lwkmin = (n > 0) ? 2 * n : 1;
    lwkopt = lwkmin;
    if (n > 0) {
        work[0] = (f64)lwkopt + 0.0 * I;
    }
    lquery = (lwork == -1);

    i32 info = 0;
    if (ijobvl <= 0) {
        info = -1;
    } else if (ijobvr <= 0) {
        info = -2;
    } else if (n < 0) {
        info = -3;
    } else if (lda < (n > 1 ? n : 1)) {
        info = -5;
    } else if (ldb < (n > 1 ? n : 1)) {
        info = -7;
    } else if (ldvl < 1 || (ilvl && ldvl < n)) {
        info = -11;
    } else if (ldvr < 1 || (ilvr && ldvr < n)) {
        info = -13;
    } else if (lwork < lwkmin && !lquery) {
        info = -15;
    }

    if (info == 0) {
        nb1 = SLC_ILAENV(&i_one, "ZGEQRF", " ", &n, &n, &i_neg_one, &i_neg_one);
        nb2 = SLC_ILAENV(&i_one, "ZUNMQR", " ", &n, &n, &n, &i_neg_one);
        nb3 = SLC_ILAENV(&i_one, "ZUNGQR", " ", &n, &n, &n, &i_neg_one);
        nb = nb1;
        if (nb2 > nb) nb = nb2;
        if (nb3 > nb) nb = nb3;
        lopt = 2 * n > n * (nb + 1) ? 2 * n : n * (nb + 1);
        work[0] = (f64)lopt + 0.0 * I;
    }

    if (info != 0) {
        return info;
    } else if (lquery) {
        return 0;
    }

    if (n == 0) {
        return 0;
    }

    eps = SLC_DLAMCH("E") * SLC_DLAMCH("B");
    safmin = SLC_DLAMCH("S");
    safmin = safmin + safmin;
    safmax = one / safmin;

    anrm = SLC_ZLANGE("M", &n, &n, a, &lda, rwork);
    anrm1 = anrm;
    anrm2 = one;
    if (anrm < one) {
        if (safmax * anrm < one) {
            anrm1 = safmin;
            anrm2 = safmax * anrm;
        }
    }

    if (anrm > zero) {
        SLC_ZLASCL("G", &i_neg_one, &i_neg_one, &anrm, &one, &n, &n, a, &lda, &iinfo);
        if (iinfo != 0) {
            return n + 10;
        }
    }

    bnrm = SLC_ZLANGE("M", &n, &n, b, &ldb, rwork);
    bnrm1 = bnrm;
    bnrm2 = one;
    if (bnrm < one) {
        if (safmax * bnrm < one) {
            bnrm1 = safmin;
            bnrm2 = safmax * bnrm;
        }
    }

    if (bnrm > zero) {
        SLC_ZLASCL("G", &i_neg_one, &i_neg_one, &bnrm, &one, &n, &n, b, &ldb, &iinfo);
        if (iinfo != 0) {
            return n + 10;
        }
    }

    ileft = 0;
    iright = n;
    irwork = iright + n;

    SLC_ZGGBAL("P", &n, a, &lda, b, &ldb, &ilo, &ihi, &rwork[ileft], &rwork[iright], &rwork[irwork], &iinfo);
    if (iinfo != 0) {
        info = n + 1;
        goto cleanup;
    }

    irows = ihi + 1 - ilo;
    if (ilv) {
        icols = n + 1 - ilo;
    } else {
        icols = irows;
    }
    itau = 0;
    iwork = itau + irows;

    i32 lwork_remaining = lwork - iwork;
    i32 ilo_idx = ilo - 1;
    SLC_ZGEQRF(&irows, &icols, &b[ilo_idx + ilo_idx * ldb], &ldb,
               &work[itau], &work[iwork], &lwork_remaining, &iinfo);
    if (iinfo >= 0 && iwork < lwork) {
        i32 newopt = (i32)creal(work[iwork]) + iwork;
        if (newopt > lwkopt) lwkopt = newopt;
    }
    if (iinfo != 0) {
        info = n + 2;
        goto cleanup;
    }

    lwork_remaining = lwork - iwork;
    SLC_ZUNMQR("L", "C", &irows, &icols, &irows, &b[ilo_idx + ilo_idx * ldb], &ldb,
               &work[itau], &a[ilo_idx + ilo_idx * lda], &lda,
               &work[iwork], &lwork_remaining, &iinfo);
    if (iinfo >= 0 && iwork < lwork) {
        i32 newopt = (i32)creal(work[iwork]) + iwork;
        if (newopt > lwkopt) lwkopt = newopt;
    }
    if (iinfo != 0) {
        info = n + 3;
        goto cleanup;
    }

    if (ilvl) {
        SLC_ZLASET("Full", &n, &n, &czero, &cone, vl, &ldvl);
        i32 irows_m1 = irows - 1;
        if (irows_m1 > 0) {
            i32 ilo_p1 = ilo_idx + 1;
            SLC_ZLACPY("L", &irows_m1, &irows_m1, &b[ilo_p1 + ilo_idx * ldb], &ldb,
                       &vl[ilo_p1 + ilo_idx * ldvl], &ldvl);
        }
        lwork_remaining = lwork - iwork;
        SLC_ZUNGQR(&irows, &irows, &irows, &vl[ilo_idx + ilo_idx * ldvl], &ldvl,
                   &work[itau], &work[iwork], &lwork_remaining, &iinfo);
        if (iinfo >= 0 && iwork < lwork) {
            i32 newopt = (i32)creal(work[iwork]) + iwork;
            if (newopt > lwkopt) lwkopt = newopt;
        }
        if (iinfo != 0) {
            info = n + 4;
            goto cleanup;
        }
    }

    if (ilvr) {
        SLC_ZLASET("Full", &n, &n, &czero, &cone, vr, &ldvr);
    }

    if (ilv) {
        SLC_ZGGHRD(jobvl, jobvr, &n, &ilo, &ihi, a, &lda, b, &ldb, vl, &ldvl, vr, &ldvr, &iinfo);
    } else {
        i32 i_one_local = 1;
        SLC_ZGGHRD("N", "N", &irows, &i_one_local, &irows, &a[ilo_idx + ilo_idx * lda], &lda,
                   &b[ilo_idx + ilo_idx * ldb], &ldb, vl, &ldvl, vr, &ldvr, &iinfo);
    }
    if (iinfo != 0) {
        info = n + 5;
        goto cleanup;
    }

    iwork = itau;

    if (ilv) {
        chtemp = 'S';
    } else {
        chtemp = 'E';
    }
    char chtemp_str[2] = {chtemp, '\0'};
    lwork_remaining = lwork - iwork;
    SLC_ZHGEQZ(chtemp_str, jobvl, jobvr, &n, &ilo, &ihi, a, &lda, b, &ldb,
               alpha, beta, vl, &ldvl, vr, &ldvr,
               &work[iwork], &lwork_remaining, &rwork[irwork], &iinfo);
    if (iinfo >= 0 && iwork < lwork) {
        i32 newopt = (i32)creal(work[iwork]) + iwork;
        if (newopt > lwkopt) lwkopt = newopt;
    }
    if (iinfo != 0) {
        if (iinfo > 0 && iinfo <= n) {
            info = iinfo;
        } else if (iinfo > n && iinfo <= 2 * n) {
            info = iinfo - n;
        } else {
            info = n + 6;
        }
        goto cleanup;
    }

    if (ilv) {
        if (ilvl) {
            if (ilvr) {
                chtemp = 'B';
            } else {
                chtemp = 'L';
            }
        } else {
            chtemp = 'R';
        }
        char side_str[2] = {chtemp, '\0'};
        SLC_ZTGEVC(side_str, "B", &ldumma, &n, a, &lda, b, &ldb, vl, &ldvl,
                   vr, &ldvr, &n, &in, &work[iwork], &rwork[irwork], &iinfo);
        if (iinfo != 0) {
            info = n + 7;
            goto cleanup;
        }

        if (ilvl) {
            SLC_ZGGBAK("P", "L", &n, &ilo, &ihi, &rwork[ileft], &rwork[iright],
                       &n, vl, &ldvl, &iinfo);
            if (iinfo != 0) {
                info = n + 8;
                goto cleanup;
            }

            for (i32 jc = 0; jc < n; jc++) {
                temp = zero;
                for (i32 jr = 0; jr < n; jr++) {
                    f64 absval = fabs(creal(vl[jr + jc * ldvl])) + fabs(cimag(vl[jr + jc * ldvl]));
                    if (absval > temp) temp = absval;
                }
                if (temp < safmin) continue;
                temp = one / temp;
                for (i32 jr = 0; jr < n; jr++) {
                    vl[jr + jc * ldvl] = vl[jr + jc * ldvl] * temp;
                }
            }
        }

        if (ilvr) {
            SLC_ZGGBAK("P", "R", &n, &ilo, &ihi, &rwork[ileft], &rwork[iright],
                       &n, vr, &ldvr, &iinfo);
            if (iinfo != 0) {
                info = n + 9;
                goto cleanup;
            }

            for (i32 jc = 0; jc < n; jc++) {
                temp = zero;
                for (i32 jr = 0; jr < n; jr++) {
                    f64 absval = fabs(creal(vr[jr + jc * ldvr])) + fabs(cimag(vr[jr + jc * ldvr]));
                    if (absval > temp) temp = absval;
                }
                if (temp < safmin) continue;
                temp = one / temp;
                for (i32 jr = 0; jr < n; jr++) {
                    vr[jr + jc * ldvr] = vr[jr + jc * ldvr] * temp;
                }
            }
        }
    }

    for (i32 jc = 0; jc < n; jc++) {
        absar = fabs(creal(alpha[jc]));
        absai = fabs(cimag(alpha[jc]));
        absb = fabs(creal(beta[jc]));
        salfar = anrm * creal(alpha[jc]);
        salfai = anrm * cimag(alpha[jc]);
        sbeta = bnrm * creal(beta[jc]);
        ilimit = false;
        scale = one;

        if (fabs(salfai) < safmin && absai >= fmax(safmin, fmax(eps * absar, eps * absb))) {
            ilimit = true;
            scale = (safmin / anrm1) / fmax(safmin, anrm2 * absai);
        }

        if (fabs(salfar) < safmin && absar >= fmax(safmin, fmax(eps * absai, eps * absb))) {
            ilimit = true;
            f64 newscale = (safmin / anrm1) / fmax(safmin, anrm2 * absar);
            if (newscale > scale) scale = newscale;
        }

        if (fabs(sbeta) < safmin && absb >= fmax(safmin, fmax(eps * absar, eps * absai))) {
            ilimit = true;
            f64 newscale = (safmin / bnrm1) / fmax(safmin, bnrm2 * absb);
            if (newscale > scale) scale = newscale;
        }

        if (ilimit) {
            temp = (scale * safmin) * fmax(fabs(salfar), fmax(fabs(salfai), fabs(sbeta)));
            if (temp > one) {
                scale = scale / temp;
            }
            if (scale < one) {
                ilimit = false;
            }
        }

        if (ilimit) {
            salfar = (scale * creal(alpha[jc])) * anrm;
            salfai = (scale * cimag(alpha[jc])) * anrm;
            sbeta = (scale * creal(beta[jc])) * bnrm;
        }
        alpha[jc] = salfar + salfai * I;
        beta[jc] = sbeta + 0.0 * I;
    }

cleanup:
    work[0] = (f64)lwkopt + 0.0 * I;
    return info;
}
