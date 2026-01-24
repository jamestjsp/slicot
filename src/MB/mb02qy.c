#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <stdbool.h>

void mb02qy(i32 m, i32 n, i32 nrhs, i32 rank, f64 *a, i32 lda,
            const i32 *jpvt, f64 *b, i32 ldb, f64 *tau,
            f64 *dwork, i32 ldwork, i32 *info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    i32 mn = (m < n) ? m : n;

    *info = 0;

    if (m < 0) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (nrhs < 0) {
        *info = -3;
    } else if (rank < 0 || rank > mn) {
        *info = -4;
    } else if (lda < 1 || (m > 0 && lda < m)) {
        *info = -6;
    } else if (ldb < 1 || (nrhs > 0 && ldb < m && ldb < n)) {
        *info = -9;
    } else {
        i32 minwrk = 1;
        if (n > minwrk) minwrk = n;
        if (nrhs > minwrk) minwrk = nrhs;

        bool lquery = (ldwork == -1);
        if (lquery) {
            f64 maxwrk_f = (f64)minwrk;
            i32 query_rank = (n > 1) ? n - 1 : 0;

            i32 tmp_info;
            SLC_DTZRZF(&query_rank, &n, a, &lda, tau, dwork, &(i32){-1}, &tmp_info);
            if (dwork[0] > maxwrk_f) maxwrk_f = dwork[0];

            i32 n_minus_rank = n - query_rank;
            SLC_DORMRZ("Left", "Transpose", &n, &nrhs, &query_rank, &n_minus_rank,
                       a, &lda, tau, b, &ldb, dwork, &(i32){-1}, &tmp_info);
            if (dwork[0] > maxwrk_f) maxwrk_f = dwork[0];

            dwork[0] = maxwrk_f;
            return;
        }

        if (ldwork < minwrk) {
            *info = -12;
        }
    }

    if (*info != 0) {
        return;
    }

    if (mn == 0 || nrhs == 0) {
        dwork[0] = ONE;
        return;
    }

    f64 maxwrk = (f64)n;

    if (rank < n) {
        f64 smlnum = SLC_DLAMCH("Safe minimum") / SLC_DLAMCH("Precision");
        f64 bignum = ONE / smlnum;
        SLC_DLABAD(&smlnum, &bignum);

        f64 anrm = SLC_DLANTR("MaxNorm", "Upper", "Non-unit", &rank, &n, a, &lda, dwork);
        i32 iascl = 0;

        if (anrm > ZERO && anrm < smlnum) {
            SLC_DLASCL("Upper", &(i32){0}, &(i32){0}, &anrm, &smlnum, &rank, &n, a, &lda, info);
            iascl = 1;
        } else if (anrm > bignum) {
            SLC_DLASCL("Upper", &(i32){0}, &(i32){0}, &anrm, &bignum, &rank, &n, a, &lda, info);
            iascl = 2;
        } else if (anrm == ZERO) {
            SLC_DLASET("Full", &n, &nrhs, &ZERO, &ZERO, b, &ldb);
            dwork[0] = ONE;
            return;
        }

        f64 bnrm = SLC_DLANGE("MaxNorm", &m, &nrhs, b, &ldb, dwork);
        i32 ibscl = 0;

        if (bnrm > ZERO && bnrm < smlnum) {
            SLC_DLASCL("General", &(i32){0}, &(i32){0}, &bnrm, &smlnum, &m, &nrhs, b, &ldb, info);
            ibscl = 1;
        } else if (bnrm > bignum) {
            SLC_DLASCL("General", &(i32){0}, &(i32){0}, &bnrm, &bignum, &m, &nrhs, b, &ldb, info);
            ibscl = 2;
        }

        SLC_DTZRZF(&rank, &n, a, &lda, tau, dwork, &ldwork, info);
        if (dwork[0] > maxwrk) maxwrk = dwork[0];

        SLC_DTRSM("Left", "Upper", "No transpose", "Non-unit", &rank, &nrhs,
                  &ONE, a, &lda, b, &ldb);

        i32 n_minus_rank = n - rank;
        SLC_DLASET("Full", &n_minus_rank, &nrhs, &ZERO, &ZERO, &b[rank], &ldb);

        SLC_DORMRZ("Left", "Transpose", &n, &nrhs, &rank, &n_minus_rank,
                   a, &lda, tau, b, &ldb, dwork, &ldwork, info);
        if (dwork[0] > maxwrk) maxwrk = dwork[0];

        if (iascl == 1) {
            SLC_DLASCL("General", &(i32){0}, &(i32){0}, &anrm, &smlnum, &n, &nrhs, b, &ldb, info);
            SLC_DLASCL("Upper", &(i32){0}, &(i32){0}, &smlnum, &anrm, &rank, &rank, a, &lda, info);
        } else if (iascl == 2) {
            SLC_DLASCL("General", &(i32){0}, &(i32){0}, &anrm, &bignum, &n, &nrhs, b, &ldb, info);
            SLC_DLASCL("Upper", &(i32){0}, &(i32){0}, &bignum, &anrm, &rank, &rank, a, &lda, info);
        }
        if (ibscl == 1) {
            SLC_DLASCL("General", &(i32){0}, &(i32){0}, &smlnum, &bnrm, &n, &nrhs, b, &ldb, info);
        } else if (ibscl == 2) {
            SLC_DLASCL("General", &(i32){0}, &(i32){0}, &bignum, &bnrm, &n, &nrhs, b, &ldb, info);
        }
    } else {
        SLC_DTRSM("Left", "Upper", "No transpose", "Non-unit", &rank, &nrhs,
                  &ONE, a, &lda, b, &ldb);
    }

    for (i32 j = 0; j < nrhs; j++) {
        for (i32 i = 0; i < n; i++) {
            i32 k = jpvt[i] - 1;
            dwork[k] = b[i + j * ldb];
        }
        SLC_DCOPY(&n, dwork, &(i32){1}, &b[j * ldb], &(i32){1});
    }

    dwork[0] = maxwrk;
}
