#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <string.h>
#include <stdbool.h>

i32 mb03ud(char jobq, char jobp, i32 n, f64 *a, i32 lda, f64 *q, i32 ldq,
           f64 *sv, f64 *dwork, i32 ldwork, i32 *info) {

    const f64 one = 1.0;
    const f64 zero = 0.0;

    *info = 0;
    bool wantq = (jobq == 'V' || jobq == 'v');
    bool wantp = (jobp == 'V' || jobp == 'v');

    if (!wantq && !(jobq == 'N' || jobq == 'n')) {
        *info = -1;
    } else if (!wantp && !(jobp == 'N' || jobp == 'n')) {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    } else if (lda < (n > 0 ? n : 1)) {
        *info = -5;
    } else if ((wantq && ldq < (n > 0 ? n : 1)) || (!wantq && ldq < 1)) {
        *info = -7;
    }

    i32 minwrk = (n > 0 ? 5 * n : 1);
    bool lquery = (ldwork == -1);

    i32 maxwrk = minwrk;
    if (lquery && *info == 0) {
        i32 info_query = 0;
        SLC_DGEBRD(&n, &n, a, &lda, sv, dwork, dwork, dwork, dwork, &ldwork, &info_query);
        maxwrk = (i32)dwork[0];

        if (wantq) {
            SLC_DORGBR("Q", &n, &n, &n, q, &ldq, dwork, dwork, &ldwork, &info_query);
            i32 tmp = (i32)dwork[0];
            maxwrk = (maxwrk > tmp ? maxwrk : tmp);
        }

        if (wantp) {
            SLC_DORGBR("P", &n, &n, &n, a, &lda, dwork, dwork, &ldwork, &info_query);
            i32 tmp = (i32)dwork[0];
            maxwrk = (maxwrk > tmp ? maxwrk : tmp);
        }

        maxwrk = (3 * n + maxwrk > minwrk ? 3 * n + maxwrk : minwrk);
    }

    if (ldwork < minwrk && !lquery) {
        *info = -10;
    }

    if (*info != 0) {
        return -(*info);
    } else if (lquery) {
        dwork[0] = (f64)maxwrk;
        return 0;
    }

    if (n == 0) {
        dwork[0] = one;
        return 0;
    }

    f64 eps = SLC_DLAMCH("P");
    f64 smlnum = sqrt(SLC_DLAMCH("S")) / eps;
    f64 bignum = one / smlnum;

    f64 dum[1];
    f64 anrm = SLC_DLANTR("M", "U", "N", &n, &n, a, &lda, dum);

    i32 iscl = 0;
    if (anrm > zero && anrm < smlnum) {
        iscl = 1;
        i32 ierr = 0;
        SLC_DLASCL("U", &ierr, &ierr, &anrm, &smlnum, &n, &n, a, &lda, &ierr);
    } else if (anrm > bignum) {
        iscl = 1;
        i32 ierr = 0;
        SLC_DLASCL("U", &ierr, &ierr, &anrm, &bignum, &n, &n, a, &lda, &ierr);
    }

    if (n > 1) {
        i32 nm1 = n - 1;
        SLC_DLASET("L", &nm1, &nm1, &zero, &zero, a + 1, &lda);
    }

    i32 ie = 0;
    i32 itauq = ie + n;
    i32 itaup = itauq + n;
    i32 jwork = itaup + n;

    i32 ldwork_remain = ldwork - jwork;
    SLC_DGEBRD(&n, &n, a, &lda, sv, &dwork[ie], &dwork[itauq], &dwork[itaup],
               &dwork[jwork], &ldwork_remain, info);

    if (*info != 0) {
        return 0;
    }

    i32 tmp = (i32)dwork[jwork];
    maxwrk = (maxwrk > tmp + jwork ? maxwrk : tmp + jwork);

    i32 ncolq = 0;
    if (wantq) {
        ncolq = n;
        SLC_DLACPY("L", &n, &n, a, &lda, q, &ldq);
        SLC_DORGBR("Q", &n, &n, &n, q, &ldq, &dwork[itauq], &dwork[jwork],
                   &ldwork_remain, info);
        if (*info != 0) {
            return 0;
        }
        tmp = (i32)dwork[jwork];
        maxwrk = (maxwrk > tmp + jwork ? maxwrk : tmp + jwork);
    }

    i32 ncolp = 0;
    if (wantp) {
        ncolp = n;
        SLC_DORGBR("P", &n, &n, &n, a, &lda, &dwork[itaup], &dwork[jwork],
                   &ldwork_remain, info);
        if (*info != 0) {
            return 0;
        }
        tmp = (i32)dwork[jwork];
        maxwrk = (maxwrk > tmp + jwork ? maxwrk : tmp + jwork);
    }

    jwork = ie + n;
    i32 ncvt = ncolp;
    i32 nru = ncolq;
    i32 ncc = 0;
    i32 ldvt = lda;
    i32 ldu = ldq;
    i32 ldc = 1;

    SLC_DBDSQR("U", &n, &ncvt, &nru, &ncc, sv, &dwork[ie], a, &ldvt,
               q, &ldu, dum, &ldc, &dwork[jwork], info);

    if (*info != 0) {
        for (i32 i = n - 2; i >= 0; i--) {
            dwork[i + 1] = dwork[i + ie];
        }
    }

    if (iscl == 1) {
        i32 ierr = 0;
        i32 one_int = 1;
        if (anrm > bignum) {
            SLC_DLASCL("G", &ierr, &ierr, &bignum, &anrm, &n, &one_int, sv, &n, &ierr);
            if (*info != 0 && anrm > bignum) {
                i32 nm1 = n - 1;
                SLC_DLASCL("G", &ierr, &ierr, &bignum, &anrm, &nm1, &one_int, &dwork[1], &n, &ierr);
            }
        }
        if (anrm < smlnum) {
            SLC_DLASCL("G", &ierr, &ierr, &smlnum, &anrm, &n, &one_int, sv, &n, &ierr);
            if (*info != 0 && anrm < smlnum) {
                i32 nm1 = n - 1;
                SLC_DLASCL("G", &ierr, &ierr, &smlnum, &anrm, &nm1, &one_int, &dwork[1], &n, &ierr);
            }
        }
    }

    dwork[0] = (f64)maxwrk;

    return 0;
}
