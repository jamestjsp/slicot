/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>

void mb05my(
    const char* balanc,
    const i32 n,
    f64* a,
    const i32 lda,
    f64* wr,
    f64* wi,
    f64* r,
    const i32 ldr,
    f64* q,
    const i32 ldq,
    f64* dwork,
    const i32 ldwork,
    i32* info
)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    bool lquery = (ldwork == -1);
    bool scale = (*balanc == 'S' || *balanc == 's');
    bool no_scale = (*balanc == 'N' || *balanc == 'n');

    *info = 0;

    if (!scale && !no_scale) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -4;
    } else if (ldr < (n > 1 ? n : 1)) {
        *info = -8;
    } else if (ldq < (n > 1 ? n : 1)) {
        *info = -10;
    }

    if (*info == 0) {
        i32 minwrk = (4 * n > 1) ? 4 * n : 1;

        if (ldwork < minwrk && !lquery) {
            *info = -12;
        } else if (lquery) {
            i32 maxwrk;
            i32 ilo = 1, ihi = n;
            i32 ierr;
            i32 lwork_query = -1;
            f64 work_opt;

            SLC_DGEHRD(&n, &ilo, &ihi, a, &lda, dwork, &work_opt, &lwork_query, &ierr);
            maxwrk = (i32)work_opt;

            SLC_DORGHR(&n, &ilo, &ihi, q, &ldq, dwork, &work_opt, &lwork_query, &ierr);
            i32 tmp = 2 * n + (i32)work_opt;
            maxwrk = (maxwrk > tmp) ? maxwrk : tmp;

            SLC_DHSEQR("S", "V", &n, &ilo, &ihi, a, &lda, wr, wi, q, &ldq,
                       &work_opt, &lwork_query, &ierr);
            tmp = n + (i32)work_opt;
            if (tmp > maxwrk) maxwrk = tmp;
            if (minwrk > maxwrk) maxwrk = minwrk;

            dwork[0] = (f64)maxwrk;
            return;
        }
    }

    if (*info != 0) {
        return;
    }

    if (n == 0) {
        dwork[0] = ONE;
        return;
    }

    f64 eps = SLC_DLAMCH("P");
    f64 smlnum = SLC_DLAMCH("S");
    f64 bignum = ONE / smlnum;
    SLC_DLABAD(&smlnum, &bignum);
    smlnum = sqrt(smlnum) / eps;
    bignum = ONE / smlnum;

    f64 anrm = SLC_DLANGE("M", &n, &n, a, &lda, dwork);
    bool scalea = false;
    f64 cscale = ONE;

    if (anrm > ZERO && anrm < smlnum) {
        scalea = true;
        cscale = smlnum;
    } else if (anrm > bignum) {
        scalea = true;
        cscale = bignum;
    }

    if (scalea) {
        i32 kl = 0, ku = 0;
        i32 ierr;
        SLC_DLASCL("G", &kl, &ku, &anrm, &cscale, &n, &n, a, &lda, &ierr);
    }

    i32 ibal = 0;
    i32 ilo, ihi;
    i32 ierr;

    SLC_DGEBAL(balanc, &n, a, &lda, &ilo, &ihi, &dwork[ibal], &ierr);

    i32 itau = ibal + n;
    i32 jwork = itau + n;
    i32 lwork = ldwork - jwork;

    SLC_DGEHRD(&n, &ilo, &ihi, a, &lda, &dwork[itau], &dwork[jwork], &lwork, &ierr);
    i32 maxwrk = (i32)dwork[jwork];

    SLC_DLACPY("Lower", &n, &n, a, &lda, q, &ldq);

    SLC_DORGHR(&n, &ilo, &ihi, q, &ldq, &dwork[itau], &dwork[jwork], &lwork, &ierr);
    i32 tmp = 2 * n + (i32)dwork[jwork];
    if (tmp > maxwrk) maxwrk = tmp;

    jwork = itau;
    lwork = ldwork - jwork;

    SLC_DHSEQR("S", "V", &n, &ilo, &ihi, a, &lda, wr, wi, q, &ldq,
               &dwork[jwork], &lwork, info);

    tmp = n + (i32)dwork[jwork];
    if (tmp > maxwrk) maxwrk = tmp;
    i32 minwrk = (4 * n > 1) ? 4 * n : 1;
    if (minwrk > maxwrk) maxwrk = minwrk;

    if (*info > 0) {
        goto label_10;
    }

    i32 side_n = 1;
    bool select_dummy[1] = {false};
    f64 vl_dummy = ZERO;
    i32 mm = n;
    i32 nout;

    SLC_DTREVC("R", "A", (i32*)select_dummy, &n, a, &lda, &vl_dummy, &side_n,
               r, &ldr, &mm, &nout, &dwork[jwork], &ierr);

label_10:
    if (scalea) {
        i32 nminf = n - *info;
        i32 lda_wr = (nminf > 1) ? nminf : 1;
        i32 kl = 0, ku = 0;

        SLC_DLASCL("G", &kl, &ku, &cscale, &anrm, &nminf, &side_n,
                   &wr[*info], &lda_wr, &ierr);
        SLC_DLASCL("G", &kl, &ku, &cscale, &anrm, &nminf, &side_n,
                   &wi[*info], &lda_wr, &ierr);

        if (*info > 0) {
            i32 ilom1 = ilo - 1;
            SLC_DLASCL("G", &kl, &ku, &cscale, &anrm, &ilom1, &side_n,
                       wr, &n, &ierr);
            SLC_DLASCL("G", &kl, &ku, &cscale, &anrm, &ilom1, &side_n,
                       wi, &n, &ierr);
        }
    }

    if (scale) {
        for (i32 k = n - 1; k >= 0; k--) {
            dwork[k + 1] = dwork[k];
        }
    }
    dwork[0] = (f64)maxwrk;
}
