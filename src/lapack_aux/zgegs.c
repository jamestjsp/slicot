/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot/lapack_aux.h"
#include "slicot_blas.h"
#include <complex.h>
#include <math.h>
#include <stdlib.h>

i32 slicot_zgegs(const char *jobvsl, const char *jobvsr, i32 n, c128 *a, i32 lda,
                 c128 *b, i32 ldb, c128 *alpha, c128 *beta,
                 c128 *vsl, i32 ldvsl, c128 *vsr, i32 ldvsr,
                 c128 *work, i32 lwork, f64 *rwork) {
    const f64 zero = 0.0;
    const f64 one = 1.0;
    const c128 czero = 0.0 + 0.0 * I;
    const c128 cone = 1.0 + 0.0 * I;

    i32 ijobvl, ijobvr;
    bool ilvsl, ilvsr, lquery;
    i32 lwkmin, lwkopt, nb, nb1, nb2, nb3, lopt;
    i32 ilo, ihi, iinfo;
    i32 ileft, iright, irwork, iwork;
    i32 irows, icols;
    i32 itau;
    f64 eps, safmin, smlnum, bignum;
    f64 anrm, bnrm, anrmto, bnrmto;
    bool ilascl, ilbscl;
    i32 i_one = 1;
    i32 i_neg_one = -1;
    i32 info = 0;

    if (*jobvsl == 'N' || *jobvsl == 'n') {
        ijobvl = 1;
        ilvsl = false;
    } else if (*jobvsl == 'V' || *jobvsl == 'v') {
        ijobvl = 2;
        ilvsl = true;
    } else {
        ijobvl = -1;
        ilvsl = false;
    }

    if (*jobvsr == 'N' || *jobvsr == 'n') {
        ijobvr = 1;
        ilvsr = false;
    } else if (*jobvsr == 'V' || *jobvsr == 'v') {
        ijobvr = 2;
        ilvsr = true;
    } else {
        ijobvr = -1;
        ilvsr = false;
    }

    lwkmin = (2 * n > 1) ? 2 * n : 1;
    lwkopt = lwkmin;
    work[0] = (c128)lwkopt;
    lquery = (lwork == -1);

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
    } else if (ldvsl < 1 || (ilvsl && ldvsl < n)) {
        info = -11;
    } else if (ldvsr < 1 || (ilvsr && ldvsr < n)) {
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
        lopt = n * (nb + 1);
        work[0] = (c128)lopt;
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
    smlnum = n * safmin / eps;
    bignum = one / smlnum;

    anrm = SLC_ZLANGE("M", &n, &n, a, &lda, rwork);
    ilascl = false;
    if (anrm > zero && anrm < smlnum) {
        anrmto = smlnum;
        ilascl = true;
    } else if (anrm > bignum) {
        anrmto = bignum;
        ilascl = true;
    }

    if (ilascl) {
        SLC_ZLASCL("G", &i_neg_one, &i_neg_one, &anrm, &anrmto, &n, &n, a, &lda, &iinfo);
        if (iinfo != 0) {
            return n + 9;
        }
    }

    bnrm = SLC_ZLANGE("M", &n, &n, b, &ldb, rwork);
    ilbscl = false;
    if (bnrm > zero && bnrm < smlnum) {
        bnrmto = smlnum;
        ilbscl = true;
    } else if (bnrm > bignum) {
        bnrmto = bignum;
        ilbscl = true;
    }

    if (ilbscl) {
        SLC_ZLASCL("G", &i_neg_one, &i_neg_one, &bnrm, &bnrmto, &n, &n, b, &ldb, &iinfo);
        if (iinfo != 0) {
            return n + 9;
        }
    }

    ileft = 0;
    iright = n;
    irwork = iright + n;
    iwork = 0;

    SLC_ZGGBAL("P", &n, a, &lda, b, &ldb, &ilo, &ihi, &rwork[ileft],
               &rwork[iright], &rwork[irwork], &iinfo);
    if (iinfo != 0) {
        info = n + 1;
        goto cleanup;
    }

    irows = ihi + 1 - ilo;
    icols = n + 1 - ilo;
    itau = iwork;
    iwork = itau + irows;

    i32 lwork_remaining = lwork - iwork;
    if (lwork_remaining < 1) lwork_remaining = 1;
    i32 ilo_idx = ilo - 1;

    SLC_ZGEQRF(&irows, &icols, &b[ilo_idx + ilo_idx * ldb], &ldb,
               &work[itau], &work[iwork], &lwork_remaining, &iinfo);
    if (iinfo >= 0) {
        i32 newopt = (i32)creal(work[iwork]) + iwork;
        if (newopt > lwkopt) lwkopt = newopt;
    }
    if (iinfo != 0) {
        info = n + 2;
        goto cleanup;
    }

    lwork_remaining = lwork - iwork;
    if (lwork_remaining < 1) lwork_remaining = 1;

    SLC_ZUNMQR("L", "C", &irows, &icols, &irows, &b[ilo_idx + ilo_idx * ldb], &ldb,
               &work[itau], &a[ilo_idx + ilo_idx * lda], &lda,
               &work[iwork], &lwork_remaining, &iinfo);
    if (iinfo >= 0) {
        i32 newopt = (i32)creal(work[iwork]) + iwork;
        if (newopt > lwkopt) lwkopt = newopt;
    }
    if (iinfo != 0) {
        info = n + 3;
        goto cleanup;
    }

    if (ilvsl) {
        SLC_ZLASET("Full", &n, &n, &czero, &cone, vsl, &ldvsl);
        i32 irows_m1 = irows - 1;
        if (irows_m1 > 0) {
            i32 ilo_p1 = ilo_idx + 1;
            SLC_ZLACPY("L", &irows_m1, &irows_m1, &b[ilo_p1 + ilo_idx * ldb], &ldb,
                       &vsl[ilo_p1 + ilo_idx * ldvsl], &ldvsl);
        }
        lwork_remaining = lwork - iwork;
        if (lwork_remaining < 1) lwork_remaining = 1;

        SLC_ZUNGQR(&irows, &irows, &irows, &vsl[ilo_idx + ilo_idx * ldvsl], &ldvsl,
                   &work[itau], &work[iwork], &lwork_remaining, &iinfo);
        if (iinfo >= 0) {
            i32 newopt = (i32)creal(work[iwork]) + iwork;
            if (newopt > lwkopt) lwkopt = newopt;
        }
        if (iinfo != 0) {
            info = n + 4;
            goto cleanup;
        }
    }

    if (ilvsr) {
        SLC_ZLASET("Full", &n, &n, &czero, &cone, vsr, &ldvsr);
    }

    SLC_ZGGHRD(jobvsl, jobvsr, &n, &ilo, &ihi, a, &lda, b, &ldb,
               vsl, &ldvsl, vsr, &ldvsr, &iinfo);
    if (iinfo != 0) {
        info = n + 5;
        goto cleanup;
    }

    iwork = itau;
    lwork_remaining = lwork - iwork;
    if (lwork_remaining < 1) lwork_remaining = 1;

    SLC_ZHGEQZ("S", jobvsl, jobvsr, &n, &ilo, &ihi, a, &lda, b, &ldb,
               alpha, beta, vsl, &ldvsl, vsr, &ldvsr,
               &work[iwork], &lwork_remaining, &rwork[irwork], &iinfo);
    if (iinfo >= 0) {
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

    if (ilvsl) {
        SLC_ZGGBAK("P", "L", &n, &ilo, &ihi, &rwork[ileft],
                   &rwork[iright], &n, vsl, &ldvsl, &iinfo);
        if (iinfo != 0) {
            info = n + 7;
            goto cleanup;
        }
    }
    if (ilvsr) {
        SLC_ZGGBAK("P", "R", &n, &ilo, &ihi, &rwork[ileft],
                   &rwork[iright], &n, vsr, &ldvsr, &iinfo);
        if (iinfo != 0) {
            info = n + 8;
            goto cleanup;
        }
    }

    if (ilascl) {
        SLC_ZLASCL("U", &i_neg_one, &i_neg_one, &anrmto, &anrm, &n, &n, a, &lda, &iinfo);
        if (iinfo != 0) {
            info = n + 9;
            goto cleanup;
        }
        SLC_ZLASCL("G", &i_neg_one, &i_neg_one, &anrmto, &anrm, &n, &i_one, alpha, &n, &iinfo);
        if (iinfo != 0) {
            info = n + 9;
            goto cleanup;
        }
    }

    if (ilbscl) {
        SLC_ZLASCL("U", &i_neg_one, &i_neg_one, &bnrmto, &bnrm, &n, &n, b, &ldb, &iinfo);
        if (iinfo != 0) {
            info = n + 9;
            goto cleanup;
        }
        SLC_ZLASCL("G", &i_neg_one, &i_neg_one, &bnrmto, &bnrm, &n, &i_one, beta, &n, &iinfo);
        if (iinfo != 0) {
            info = n + 9;
            goto cleanup;
        }
    }

cleanup:
    work[0] = (c128)lwkopt;
    return info;
}
