// SPDX-License-Identifier: BSD-3-Clause

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>

void mb03xp(const char *job, const char *compq, const char *compz,
            i32 n, i32 ilo, i32 ihi, f64 *a, i32 lda, f64 *b, i32 ldb,
            f64 *q, i32 ldq, f64 *z, i32 ldz,
            f64 *alphar, f64 *alphai, f64 *beta,
            f64 *dwork, i32 ldwork, i32 *info) {

    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    bool wantt = (job[0] == 'S' || job[0] == 's');
    bool initq = (compq[0] == 'I' || compq[0] == 'i');
    bool wantq = initq || (compq[0] == 'V' || compq[0] == 'v');
    bool initz = (compz[0] == 'I' || compz[0] == 'i');
    bool wantz = initz || (compz[0] == 'V' || compz[0] == 'v');

    *info = 0;

    if (!(job[0] == 'E' || job[0] == 'e') && !wantt) {
        *info = -1;
    } else if (!(compq[0] == 'N' || compq[0] == 'n') && !wantq) {
        *info = -2;
    } else if (!(compz[0] == 'N' || compz[0] == 'n') && !wantz) {
        *info = -3;
    } else if (n < 0) {
        *info = -4;
    } else if (ilo < 1 || ilo > (n > 0 ? n + 1 : 1)) {
        *info = -5;
    } else if (ihi < (ilo < n ? ilo : n) || ihi > n) {
        *info = -6;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -8;
    } else if (ldb < (n > 1 ? n : 1)) {
        *info = -10;
    } else if (ldq < 1 || (wantq && ldq < n)) {
        *info = -12;
    } else if (ldz < 1 || (wantz && ldz < n)) {
        *info = -14;
    } else if (ldwork < (n > 1 ? n : 1)) {
        dwork[0] = (f64)(n > 1 ? n : 1);
        *info = -19;
    }

    if (*info != 0) {
        return;
    }

    if (initq) {
        SLC_DLASET("All", &n, &n, &ZERO, &ONE, q, &ldq);
    }
    if (initz) {
        SLC_DLASET("All", &n, &n, &ZERO, &ONE, z, &ldz);
    }

    for (i32 idx = 0; idx < ilo - 1; idx++) {
        if (b[idx + idx * ldb] < ZERO) {
            if (!wantt) {
                b[idx + idx * ldb] = -b[idx + idx * ldb];
                a[idx + idx * lda] = -a[idx + idx * lda];
            }
        }
        alphar[idx] = a[idx + idx * lda];
        alphai[idx] = ZERO;
        beta[idx] = b[idx + idx * ldb];
    }
    for (i32 idx = ihi; idx < n; idx++) {
        if (b[idx + idx * ldb] < ZERO) {
            if (!wantt) {
                b[idx + idx * ldb] = -b[idx + idx * ldb];
                a[idx + idx * lda] = -a[idx + idx * lda];
            }
        }
        alphar[idx] = a[idx + idx * lda];
        alphai[idx] = ZERO;
        beta[idx] = b[idx + idx * ldb];
    }

    if (n == 0 || ilo == ihi + 1) {
        dwork[0] = ONE;
        return;
    }

    for (i32 j = ilo - 1; j < ihi - 2; j++) {
        for (i32 i = j + 2; i < n; i++) {
            a[i + j * lda] = ZERO;
        }
    }
    for (i32 j = ilo - 1; j < ihi - 1; j++) {
        for (i32 i = j + 1; i < n; i++) {
            b[i + j * ldb] = ZERO;
        }
    }

    i32 ierr;
    mb03yd(wantt, wantq, wantz, n, ilo, ihi, ilo, ihi,
           a, lda, b, ldb, q, ldq, z, ldz,
           alphar, alphai, beta, dwork, ldwork, &ierr);
    *info = ierr;
    dwork[0] = (f64)(n > 1 ? n : 1);
}
