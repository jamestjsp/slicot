/*
 * MC01XD - Compute roots of cubic polynomial
 *
 * Computes roots of P(t) = ALPHA + BETA*t + GAMMA*t^2 + DELTA*t^3.
 * Uses QZ or QR algorithm depending on coefficient variation.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>

void mc01xd(f64 alpha, f64 beta, f64 gamma, f64 delta,
            f64 *evr, f64 *evi, f64 *evq,
            f64 *dwork, i32 ldwork, i32 *info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const f64 TEN = 10.0;
    i32 i, j;
    i32 m2pos, wrkpos;
    i32 nmin = 42;
    f64 maxc, minc, var;
    i32 int3 = 3;
    i32 int1 = 1;
    i32 lwork_query;

    *info = 0;

    if (ldwork == -1) {
        f64 work_dgeev, work_dggev;
        SLC_DGEEV("N", "N", &int3, dwork, &int3, evr, evi, dwork, &int1,
                  dwork, &int1, dwork, &ldwork, info);
        work_dgeev = dwork[0];
        SLC_DGGEV("N", "N", &int3, dwork, &int3, dwork, &int3, evr, evi, evq,
                  dwork, &int1, dwork, &int1, &dwork[1], &ldwork, info);
        work_dggev = dwork[1];
        i32 opt1 = 9 + (i32)work_dgeev;
        i32 opt2 = 18 + (i32)work_dggev;
        i32 opt = opt1 > opt2 ? opt1 : opt2;
        dwork[0] = (f64)(nmin > opt ? nmin : opt);
        return;
    } else if (ldwork < nmin) {
        *info = -9;
        return;
    }

    for (i = 0; i < 18; i++) {
        dwork[i] = ZERO;
    }

    m2pos = 9;
    wrkpos = 18;

    if (fabs(alpha) > fabs(beta)) {
        i = 0;
        evr[0] = alpha;
    } else {
        i = 1;
        evr[0] = beta;
    }

    if (fabs(gamma) > fabs(delta)) {
        j = 2;
        evr[1] = gamma;
    } else {
        j = 3;
        evr[1] = delta;
    }

    if (fabs(evr[1]) > fabs(evr[0])) {
        i = j;
        maxc = fabs(evr[1]);
    } else {
        maxc = fabs(evr[0]);
    }

    f64 abs_alpha = fabs(alpha);
    f64 abs_beta = fabs(beta);
    f64 abs_gamma = fabs(gamma);
    f64 abs_delta = fabs(delta);
    minc = abs_alpha;
    if (abs_beta < minc) minc = abs_beta;
    if (abs_gamma < minc) minc = abs_gamma;
    if (abs_delta < minc) minc = abs_delta;

    if (minc > ZERO) {
        var = maxc / minc;
    } else {
        var = maxc;
    }

    if (var > TEN) {
        if (i == 0) {
            dwork[0] = -beta / alpha;
            dwork[1] = ONE;
            dwork[3] = -gamma / alpha;
            dwork[5] = ONE;
            dwork[6] = -delta / alpha;
        } else if (i == 1) {
            dwork[0] = -alpha / beta;
            dwork[3] = -gamma / beta;
            dwork[4] = ONE;
            dwork[6] = -delta / beta;
            dwork[8] = ONE;

            dwork[m2pos + 0] = ONE;
            dwork[m2pos + 1] = dwork[0];
            dwork[m2pos + 4] = dwork[3];
            dwork[m2pos + 5] = ONE;
            dwork[m2pos + 7] = dwork[6];
        } else if (i == 2) {
            dwork[1] = -alpha / gamma;
            dwork[3] = ONE;
            dwork[4] = -beta / gamma;
            dwork[7] = -delta / gamma;
            dwork[8] = ONE;

            dwork[m2pos + 0] = ONE;
            dwork[m2pos + 2] = dwork[1];
            dwork[m2pos + 4] = ONE;
            dwork[m2pos + 5] = dwork[4];
            dwork[m2pos + 8] = dwork[7];
        } else {
            dwork[2] = -alpha / delta;
            dwork[3] = ONE;
            dwork[5] = -beta / delta;
            dwork[7] = ONE;
            dwork[8] = -gamma / delta;
        }

        if (i == 0 || i == 3) {
            lwork_query = ldwork - 9;
            SLC_DGEEV("N", "N", &int3, dwork, &int3, evr, evi, &dwork[wrkpos],
                      &int1, &dwork[wrkpos], &int1, &dwork[m2pos], &lwork_query,
                      info);

            if (i == 0) {
                j = 0;
                while (j < 3 - *info) {
                    if (evi[j] == ZERO) {
                        evr[j] = ONE / evr[j];
                        j++;
                    } else if (evi[j] > ZERO) {
                        f64 re_out, im_out;
                        SLC_DLADIV(&ONE, &ZERO, &evr[j], &evi[j], &re_out, &im_out);
                        evr[j] = re_out;
                        evi[j] = -im_out;
                        evr[j + 1] = re_out;
                        evi[j + 1] = im_out;
                        j += 2;
                    } else {
                        j++;
                    }
                }
            }
            evq[0] = ONE;
            evq[1] = ONE;
            evq[2] = ONE;
        } else {
            lwork_query = ldwork - 18;
            SLC_DGGEV("N", "N", &int3, dwork, &int3, &dwork[m2pos], &int3, evr,
                      evi, evq, &dwork[wrkpos], &int1, &dwork[wrkpos], &int1,
                      &dwork[wrkpos], &lwork_query, info);
        }
    } else {
        if (i == 0) {
            dwork[0] = alpha;
            dwork[4] = alpha;
            dwork[8] = alpha;

            dwork[m2pos + 0] = -beta;
            dwork[m2pos + 1] = alpha;
            dwork[m2pos + 3] = -gamma;
            dwork[m2pos + 5] = alpha;
            dwork[m2pos + 6] = -delta;
        } else if (i == 1) {
            dwork[0] = -alpha;
            dwork[3] = -gamma;
            dwork[4] = beta;
            dwork[6] = -delta;
            dwork[8] = beta;

            dwork[m2pos + 0] = beta;
            dwork[m2pos + 1] = -alpha;
            dwork[m2pos + 4] = -gamma;
            dwork[m2pos + 5] = beta;
            dwork[m2pos + 7] = -delta;
        } else if (i == 2) {
            dwork[1] = -alpha;
            dwork[3] = gamma;
            dwork[4] = -beta;
            dwork[7] = -delta;
            dwork[8] = gamma;

            dwork[m2pos + 0] = gamma;
            dwork[m2pos + 2] = -alpha;
            dwork[m2pos + 4] = gamma;
            dwork[m2pos + 5] = -beta;
            dwork[m2pos + 8] = -delta;
        } else {
            dwork[2] = -alpha;
            dwork[3] = delta;
            dwork[5] = -beta;
            dwork[7] = delta;
            dwork[8] = -gamma;

            dwork[m2pos + 0] = delta;
            dwork[m2pos + 4] = delta;
            dwork[m2pos + 8] = delta;
        }

        lwork_query = ldwork - 18;
        SLC_DGGEV("N", "N", &int3, dwork, &int3, &dwork[m2pos], &int3, evr,
                  evi, evq, &dwork[wrkpos], &int1, &dwork[wrkpos], &int1,
                  &dwork[wrkpos], &lwork_query, info);
    }
}
