/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot/lapack_aux.h"
#include "slicot_blas.h"
#include <string.h>
#include <math.h>
#include <stdlib.h>

i32 slicot_dgegv(const char *jobvl, const char *jobvr, i32 n, f64 *a, i32 lda,
                 f64 *b, i32 ldb, f64 *alphar, f64 *alphai, f64 *beta,
                 f64 *vl, i32 ldvl, f64 *vr, i32 ldvr,
                 f64 *work, i32 lwork) {
    const f64 zero = 0.0;
    const f64 one = 1.0;

    i32 ijobvl, ijobvr;
    bool ilvl, ilvr, ilv, lquery;
    i32 lwkmin, lwkopt, nb, nb1, nb2, nb3, lopt;
    i32 ilo, ihi, iinfo;
    i32 ileft, iright, itau, iwork;
    i32 irows, icols, in;
    f64 eps, safmin, safmax, onepls;
    f64 anrm, bnrm, anrm1, anrm2, bnrm1, bnrm2;
    f64 temp, scale;
    f64 absar, absai, absb, salfar, salfai, sbeta;
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

    lwkmin = (n > 0) ? 8 * n : 1;
    lwkopt = lwkmin;
    if (n > 0) {
        work[0] = (f64)lwkopt;
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
        info = -12;
    } else if (ldvr < 1 || (ilvr && ldvr < n)) {
        info = -14;
    } else if (lwork < lwkmin && !lquery) {
        info = -16;
    }

    if (info == 0) {
        nb1 = SLC_ILAENV(&i_one, "DGEQRF", " ", &n, &n, &i_neg_one, &i_neg_one);
        nb2 = SLC_ILAENV(&i_one, "DORMQR", " ", &n, &n, &n, &i_neg_one);
        nb3 = SLC_ILAENV(&i_one, "DORGQR", " ", &n, &n, &n, &i_neg_one);
        nb = nb1;
        if (nb2 > nb) nb = nb2;
        if (nb3 > nb) nb = nb3;
        i32 opt1 = 6 * n;
        i32 opt2 = n * (nb + 1);
        lopt = 2 * n + (opt1 > opt2 ? opt1 : opt2);
        work[0] = (f64)lopt;
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
    onepls = one + 4 * eps;

    anrm = SLC_DLANGE("M", &n, &n, a, &lda, work);
    anrm1 = anrm;
    anrm2 = one;
    if (anrm < one) {
        if (safmax * anrm < one) {
            anrm1 = safmin;
            anrm2 = safmax * anrm;
        }
    }

    if (anrm > zero) {
        SLC_DLASCL("G", &i_neg_one, &i_neg_one, &anrm, &one, &n, &n, a, &lda, &iinfo);
        if (iinfo != 0) {
            return n + 10;
        }
    }

    bnrm = SLC_DLANGE("M", &n, &n, b, &ldb, work);
    bnrm1 = bnrm;
    bnrm2 = one;
    if (bnrm < one) {
        if (safmax * bnrm < one) {
            bnrm1 = safmin;
            bnrm2 = safmax * bnrm;
        }
    }

    if (bnrm > zero) {
        SLC_DLASCL("G", &i_neg_one, &i_neg_one, &bnrm, &one, &n, &n, b, &ldb, &iinfo);
        if (iinfo != 0) {
            return n + 10;
        }
    }

    ileft = 0;
    iright = n;
    iwork = iright + n;

    SLC_DGGBAL("P", &n, a, &lda, b, &ldb, &ilo, &ihi, &work[ileft], &work[iright], &work[iwork], &iinfo);
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
    itau = iwork;
    iwork = itau + irows;

    i32 lwork_remaining = lwork - iwork;
    i32 ilo_idx = ilo - 1;
    SLC_DGEQRF(&irows, &icols, &b[ilo_idx + ilo_idx * ldb], &ldb,
               &work[itau], &work[iwork], &lwork_remaining, &iinfo);
    if (iinfo >= 0 && iwork < lwork) {
        i32 newopt = (i32)work[iwork] + iwork;
        if (newopt > lwkopt) lwkopt = newopt;
    }
    if (iinfo != 0) {
        info = n + 2;
        goto cleanup;
    }

    lwork_remaining = lwork - iwork;
    SLC_DORMQR("L", "T", &irows, &icols, &irows, &b[ilo_idx + ilo_idx * ldb], &ldb,
               &work[itau], &a[ilo_idx + ilo_idx * lda], &lda,
               &work[iwork], &lwork_remaining, &iinfo);
    if (iinfo >= 0 && iwork < lwork) {
        i32 newopt = (i32)work[iwork] + iwork;
        if (newopt > lwkopt) lwkopt = newopt;
    }
    if (iinfo != 0) {
        info = n + 3;
        goto cleanup;
    }

    if (ilvl) {
        SLC_DLASET("Full", &n, &n, &zero, &one, vl, &ldvl);
        i32 irows_m1 = irows - 1;
        if (irows_m1 > 0) {
            i32 ilo_p1 = ilo_idx + 1;
            SLC_DLACPY("L", &irows_m1, &irows_m1, &b[ilo_p1 + ilo_idx * ldb], &ldb,
                       &vl[ilo_p1 + ilo_idx * ldvl], &ldvl);
        }
        lwork_remaining = lwork - iwork;
        SLC_DORGQR(&irows, &irows, &irows, &vl[ilo_idx + ilo_idx * ldvl], &ldvl,
                   &work[itau], &work[iwork], &lwork_remaining, &iinfo);
        if (iinfo >= 0 && iwork < lwork) {
            i32 newopt = (i32)work[iwork] + iwork;
            if (newopt > lwkopt) lwkopt = newopt;
        }
        if (iinfo != 0) {
            info = n + 4;
            goto cleanup;
        }
    }

    if (ilvr) {
        SLC_DLASET("Full", &n, &n, &zero, &one, vr, &ldvr);
    }

    if (ilv) {
        SLC_DGGHRD(jobvl, jobvr, &n, &ilo, &ihi, a, &lda, b, &ldb, vl, &ldvl, vr, &ldvr, &iinfo);
    } else {
        i32 i_one_local = 1;
        SLC_DGGHRD("N", "N", &irows, &i_one_local, &irows, &a[ilo_idx + ilo_idx * lda], &lda,
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
    SLC_DHGEQZ(chtemp_str, jobvl, jobvr, &n, &ilo, &ihi, a, &lda, b, &ldb,
               alphar, alphai, beta, vl, &ldvl, vr, &ldvr,
               &work[iwork], &lwork_remaining, &iinfo);
    if (iinfo >= 0 && iwork < lwork) {
        i32 newopt = (i32)work[iwork] + iwork;
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
        SLC_DTGEVC(side_str, "B", &ldumma, &n, a, &lda, b, &ldb, vl, &ldvl,
                   vr, &ldvr, &n, &in, &work[iwork], &iinfo);
        if (iinfo != 0) {
            info = n + 7;
            goto cleanup;
        }

        if (ilvl) {
            SLC_DGGBAK("P", "L", &n, &ilo, &ihi, &work[ileft], &work[iright],
                       &n, vl, &ldvl, &iinfo);
            if (iinfo != 0) {
                info = n + 8;
                goto cleanup;
            }

            for (i32 jc = 0; jc < n; jc++) {
                if (alphai[jc] < zero) continue;

                temp = zero;
                if (alphai[jc] == zero) {
                    for (i32 jr = 0; jr < n; jr++) {
                        f64 absval = fabs(vl[jr + jc * ldvl]);
                        if (absval > temp) temp = absval;
                    }
                } else {
                    for (i32 jr = 0; jr < n; jr++) {
                        f64 absval = fabs(vl[jr + jc * ldvl]) + fabs(vl[jr + (jc + 1) * ldvl]);
                        if (absval > temp) temp = absval;
                    }
                }
                if (temp < safmin) continue;
                temp = one / temp;
                if (alphai[jc] == zero) {
                    for (i32 jr = 0; jr < n; jr++) {
                        vl[jr + jc * ldvl] *= temp;
                    }
                } else {
                    for (i32 jr = 0; jr < n; jr++) {
                        vl[jr + jc * ldvl] *= temp;
                        vl[jr + (jc + 1) * ldvl] *= temp;
                    }
                }
            }
        }

        if (ilvr) {
            SLC_DGGBAK("P", "R", &n, &ilo, &ihi, &work[ileft], &work[iright],
                       &n, vr, &ldvr, &iinfo);
            if (iinfo != 0) {
                info = n + 9;
                goto cleanup;
            }

            for (i32 jc = 0; jc < n; jc++) {
                if (alphai[jc] < zero) continue;

                temp = zero;
                if (alphai[jc] == zero) {
                    for (i32 jr = 0; jr < n; jr++) {
                        f64 absval = fabs(vr[jr + jc * ldvr]);
                        if (absval > temp) temp = absval;
                    }
                } else {
                    for (i32 jr = 0; jr < n; jr++) {
                        f64 absval = fabs(vr[jr + jc * ldvr]) + fabs(vr[jr + (jc + 1) * ldvr]);
                        if (absval > temp) temp = absval;
                    }
                }
                if (temp < safmin) continue;
                temp = one / temp;
                if (alphai[jc] == zero) {
                    for (i32 jr = 0; jr < n; jr++) {
                        vr[jr + jc * ldvr] *= temp;
                    }
                } else {
                    for (i32 jr = 0; jr < n; jr++) {
                        vr[jr + jc * ldvr] *= temp;
                        vr[jr + (jc + 1) * ldvr] *= temp;
                    }
                }
            }
        }
    }

    for (i32 jc = 0; jc < n; jc++) {
        absar = fabs(alphar[jc]);
        absai = fabs(alphai[jc]);
        absb = fabs(beta[jc]);
        salfar = anrm * alphar[jc];
        salfai = anrm * alphai[jc];
        sbeta = bnrm * beta[jc];
        ilimit = false;
        scale = one;

        if (fabs(salfai) < safmin && absai >= fmax(safmin, fmax(eps * absar, eps * absb))) {
            ilimit = true;
            scale = (onepls * safmin / anrm1) / fmax(onepls * safmin, anrm2 * absai);
        } else if (salfai == zero) {
            if (alphai[jc] < zero && jc > 0) {
                alphai[jc - 1] = zero;
            } else if (alphai[jc] > zero && jc < n - 1) {
                alphai[jc + 1] = zero;
            }
        }

        if (fabs(salfar) < safmin && absar >= fmax(safmin, fmax(eps * absai, eps * absb))) {
            ilimit = true;
            f64 newscale = (onepls * safmin / anrm1) / fmax(onepls * safmin, anrm2 * absar);
            if (newscale > scale) scale = newscale;
        }

        if (fabs(sbeta) < safmin && absb >= fmax(safmin, fmax(eps * absar, eps * absai))) {
            ilimit = true;
            f64 newscale = (onepls * safmin / bnrm1) / fmax(onepls * safmin, bnrm2 * absb);
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
            salfar = (scale * alphar[jc]) * anrm;
            salfai = (scale * alphai[jc]) * anrm;
            sbeta = (scale * beta[jc]) * bnrm;
        }
        alphar[jc] = salfar;
        alphai[jc] = salfai;
        beta[jc] = sbeta;
    }

cleanup:
    work[0] = (f64)lwkopt;
    return info;
}
