// SPDX-License-Identifier: BSD-3-Clause

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>

void mb03yt(f64 *a, i32 lda, f64 *b, i32 ldb,
            f64 *alphar, f64 *alphai, f64 *beta,
            f64 *csl, f64 *snl, f64 *csr, f64 *snr) {
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const i32 TWO = 2;
    const i32 IONE = 1;

    f64 safmin = SLC_DLAMCH("S");
    f64 ulp = SLC_DLAMCH("P");

    f64 anorm = fmax(fabs(a[0]) + fabs(a[1]),
                     fmax(fabs(a[lda]) + fabs(a[1 + lda]), safmin));
    a[0] /= anorm;
    a[lda] /= anorm;
    a[1] /= anorm;
    a[1 + lda] /= anorm;

    f64 bnorm = fmax(fabs(b[0]),
                     fmax(fabs(b[ldb]) + fabs(b[1 + ldb]), safmin));
    b[0] /= bnorm;
    b[ldb] /= bnorm;
    b[1 + ldb] /= bnorm;

    f64 wi = ZERO;
    f64 wr1, wr2, scale1, scale2;
    f64 r, t, h1, h2, h3, rr, qq;

    if (fabs(a[1]) <= ulp) {
        *csl = ONE;
        *snl = ZERO;
        *csr = ONE;
        *snr = ZERO;
        wi = ZERO;
        a[1] = ZERO;
        b[1] = ZERO;
    } else if (fabs(b[0]) <= ulp) {
        SLC_DLARTG(&a[1 + lda], &a[1], csr, snr, &t);
        *snr = -(*snr);
        SLC_DROT(&TWO, &a[0], &IONE, &a[lda], &IONE, csr, snr);
        SLC_DROT(&TWO, &b[0], &ldb, &b[1], &ldb, csr, snr);
        *csl = ONE;
        *snl = ZERO;
        wi = ZERO;
        a[1] = ZERO;
        b[0] = ZERO;
        b[1] = ZERO;
    } else if (fabs(b[1 + ldb]) <= ulp) {
        SLC_DLARTG(&a[0], &a[1], csl, snl, &r);
        *csr = ONE;
        *snr = ZERO;
        wi = ZERO;
        SLC_DROT(&TWO, &a[0], &lda, &a[1], &lda, csl, snl);
        SLC_DROT(&TWO, &b[0], &IONE, &b[ldb], &IONE, csl, snl);
        a[1] = ZERO;
        b[1] = ZERO;
        b[1 + ldb] = ZERO;
    } else {
        r = b[0];
        b[0] = b[1 + ldb];
        b[1 + ldb] = r;
        b[ldb] = -b[ldb];

        SLC_DLAG2(a, &lda, b, &ldb, &safmin, &scale1, &scale2, &wr1, &wr2, &wi);

        if (wi == ZERO) {
            h1 = scale1 * a[0] - wr1 * b[0];
            h2 = scale1 * a[lda] - wr1 * b[ldb];
            h3 = scale1 * a[1 + lda] - wr1 * b[1 + ldb];

            rr = SLC_DLAPY2(&h1, &h2);
            f64 tmp = scale1 * a[1];
            qq = SLC_DLAPY2(&tmp, &h3);

            if (rr > qq) {
                SLC_DLARTG(&h2, &h1, csr, snr, &t);
            } else {
                tmp = scale1 * a[1];
                SLC_DLARTG(&h3, &tmp, csr, snr, &t);
            }

            *snr = -(*snr);
            SLC_DROT(&TWO, &a[0], &IONE, &a[lda], &IONE, csr, snr);
            SLC_DROT(&TWO, &b[0], &IONE, &b[ldb], &IONE, csr, snr);

            h1 = fmax(fabs(a[0]) + fabs(a[lda]), fabs(a[1]) + fabs(a[1 + lda]));
            h2 = fmax(fabs(b[0]) + fabs(b[ldb]), fabs(b[1]) + fabs(b[1 + ldb]));

            if (scale1 * h1 >= fabs(wr1) * h2) {
                SLC_DLARTG(&b[0], &b[1], csl, snl, &r);
            } else {
                SLC_DLARTG(&a[0], &a[1], csl, snl, &r);
            }

            SLC_DROT(&TWO, &a[0], &lda, &a[1], &lda, csl, snl);
            SLC_DROT(&TWO, &b[0], &ldb, &b[1], &ldb, csl, snl);

            a[1] = ZERO;
            b[1] = ZERO;

            r = b[0];
            b[0] = b[1 + ldb];
            b[1 + ldb] = r;
            b[ldb] = -b[ldb];
        } else {
            r = b[0];
            b[0] = b[1 + ldb];
            b[1 + ldb] = r;
            b[ldb] = -b[ldb];

            SLC_DLASV2(&b[0], &b[ldb], &b[1 + ldb], &r, &t, snl, csl, snr, csr);

            SLC_DROT(&TWO, &a[0], &lda, &a[1], &lda, csl, snl);
            SLC_DROT(&TWO, &b[0], &ldb, &b[1], &ldb, csr, snr);
            SLC_DROT(&TWO, &a[0], &IONE, &a[lda], &IONE, csr, snr);
            SLC_DROT(&TWO, &b[0], &IONE, &b[ldb], &IONE, csl, snl);

            b[1] = ZERO;
            b[ldb] = ZERO;
        }
    }

    r = b[0];
    t = b[1 + ldb];
    a[0] *= anorm;
    a[1] *= anorm;
    a[lda] *= anorm;
    a[1 + lda] *= anorm;
    b[0] *= bnorm;
    b[1] *= bnorm;
    b[ldb] *= bnorm;
    b[1 + ldb] *= bnorm;

    if (wi == ZERO) {
        alphar[0] = a[0];
        alphar[1] = a[1 + lda];
        alphai[0] = ZERO;
        alphai[1] = ZERO;
        beta[0] = b[0];
        beta[1] = b[1 + ldb];
    } else {
        wr1 = anorm * wr1;
        wi = anorm * wi;
        if (fabs(wr1) > ONE || wi > ONE) {
            wr1 = wr1 * r;
            wi = wi * r;
            r = ONE;
        }
        if (fabs(wr1) > ONE || fabs(wi) > ONE) {
            wr1 = wr1 * t;
            wi = wi * t;
            t = ONE;
        }
        alphar[0] = (wr1 / scale1) * r * t;
        alphai[0] = fabs((wi / scale1) * r * t);
        alphar[1] = alphar[0];
        alphai[1] = -alphai[0];
        beta[0] = bnorm;
        beta[1] = bnorm;
    }
}
