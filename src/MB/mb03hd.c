/*
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * MB03HD - Exchange eigenvalues of 2x2 or 4x4 skew-Hamiltonian/Hamiltonian pencil
 *
 * Computes an orthogonal matrix Q for a real regular 2-by-2 or 4-by-4
 * skew-Hamiltonian/Hamiltonian pencil in structured Schur form such that
 * J Q' J' (aA - bB) Q is still in structured Schur form but the eigenvalues
 * are exchanged.
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>

void mb03hd(const i32 n, const f64 *a, const i32 lda,
            const f64 *b, const i32 ldb, const f64 *macpar,
            f64 *q, const i32 ldq, f64 *dwork, i32 *info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const f64 TWO = 2.0;

    f64 par[3];
    f64 co, si, d, nrm, s, smin, smln, t;
    i32 itau, iwrk;

    *info = 0;

    if (n == 4) {
        par[0] = macpar[0];
        par[1] = macpar[1];

        dwork[0] = a[0];
        dwork[1] = ZERO;
        dwork[4] = a[lda];
        dwork[5] = a[1 + lda];
        dwork[8] = ZERO;
        dwork[9] = -a[3 * lda];
        dwork[10] = -dwork[0];
        dwork[11] = -dwork[4];
        dwork[12] = -dwork[9];
        dwork[13] = ZERO;
        dwork[14] = ZERO;
        dwork[15] = -dwork[5];
        dwork[16] = b[2 * ldb];
        dwork[17] = b[3 * ldb];
        dwork[20] = b[3 * ldb];
        dwork[21] = b[1 + 3 * ldb];

        smln = TWO * par[1] / par[0];
        f64 abs_dw0 = fabs(dwork[0]);
        f64 abs_dw9 = fabs(dwork[9]);
        f64 sum1 = fabs(dwork[4]) + fabs(dwork[5]);
        f64 sum2 = fabs(dwork[17]) + fmax(fabs(dwork[16]), fabs(dwork[21]));
        f64 max_val = fmax(abs_dw0, fmax(smln, fmax(abs_dw9, fmax(sum1, sum2))));
        smin = sqrt(smln) / max_val;
        par[2] = smin;

        i32 iwarn_local = 0;
        mb02uw(false, 2, 6, par, b, ldb, dwork, 4, &si, &iwarn_local);
        if (iwarn_local != 0) *info = 1;

        iwarn_local = 0;
        mb02uw(true, 2, 2, par, b, ldb, &dwork[10], 4, &d, &iwarn_local);
        if (iwarn_local != 0) *info = 1;

        i32 two = 2;
        i32 four = 4;
        i32 six = 6;
        i32 zero_i = 0;

        if (si < d) {
            SLC_DLASCL("G", &zero_i, &zero_i, &d, &si, &two, &two, &dwork[10], &four, info);
        } else if (si > d) {
            SLC_DLASCL("G", &zero_i, &zero_i, &si, &d, &two, &six, dwork, &four, info);
        }

        f64 neg_one = -ONE;
        SLC_DGEMM("N", "N", &two, &two, &two, &neg_one,
                  &dwork[16], &four, &dwork[10], &four, &ONE, &dwork[8], &four);

        nrm = fmax(fabs(dwork[0]) + fabs(dwork[1]),
                   fmax(fabs(dwork[4]) + fabs(dwork[5]), smln));
        if (nrm > ONE) {
            SLC_DLASCL("G", &zero_i, &zero_i, &nrm, &ONE, &two, &four, dwork, &four, info);
            SLC_DLASCL("G", &zero_i, &zero_i, &nrm, &ONE, &two, &two, &dwork[10], &four, info);
        }

        s = dwork[0] + dwork[5];

        t = dwork[0] * dwork[5] - dwork[1] * dwork[4];

        SLC_DLACPY("F", &four, &two, &dwork[8], &four, q, &ldq);

        f64 neg_s = -s;
        SLC_DGEMM("N", "N", &two, &two, &four, &ONE,
                  dwork, &four, &dwork[8], &four, &neg_s, q, &ldq);
        SLC_DGEMM("N", "N", &two, &two, &two, &ONE,
                  &dwork[10], &four, &dwork[10], &four, &neg_s, &q[2], &ldq);
        q[2] += t;
        q[3 + ldq] += t;

        itau = 0;
        iwrk = 2;

        SLC_DGEQR2(&four, &two, q, &ldq, &dwork[itau], &dwork[iwrk], info);
        SLC_DORG2R(&four, &four, &two, q, &ldq, &dwork[itau], &dwork[iwrk], info);

        dwork[20] = a[0] * q[0] + a[lda] * q[1] + a[3 * lda] * q[3];
        dwork[21] = a[1 + lda] * q[1] - a[3 * lda] * q[2];
        dwork[22] = a[0] * q[2];
        dwork[23] = a[lda] * q[2] + a[1 + lda] * q[3];
        dwork[8] = q[2 + 2 * ldq] * dwork[20] + q[3 + 2 * ldq] * dwork[21]
                 - q[2 * ldq] * dwork[22] - q[1 + 2 * ldq] * dwork[23];
        dwork[9] = q[2 + 3 * ldq] * dwork[20] + q[3 + 3 * ldq] * dwork[21]
                 - q[3 * ldq] * dwork[22] - q[1 + 3 * ldq] * dwork[23];
        SLC_DLARTG(&dwork[8], &dwork[9], &co, &si, &t);
        i32 int1 = 1;
        SLC_DROT(&four, &q[2 * ldq], &int1, &q[3 * ldq], &int1, &co, &si);

    } else {
        f64 arg2 = TWO * b[0];
        SLC_DLARTG(&b[ldb], &arg2, &co, &si, &t);
        q[0] = co;
        q[1] = -si;
        q[ldq] = si;
        q[1 + ldq] = co;
    }
}
