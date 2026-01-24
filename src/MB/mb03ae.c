/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <ctype.h>
#include <math.h>

void mb03ae(const char *shft, i32 k, i32 n, const i32 *amap, const i32 *s,
            i32 sinv, const f64 *a, i32 lda1, i32 lda2, f64 *c1, f64 *s1,
            f64 *c2, f64 *s2) {
    const f64 ONE = 1.0;
    const f64 TWO = 2.0;
    const f64 ZERO = 0.0;

    char shft_upper = (char)toupper((unsigned char)shft[0]);
    bool sgle = (shft_upper == 'S') || (n == 2);

    i32 m = (n < 3) ? n : 3;
    i32 mm = m * m;
    i32 l = n - m;
    f64 t = ONE;

    f64 dwork[9];
    f64 y[9];
    f64 z[4];
    i32 ip[3], jp[3];
    i32 ind;
    f64 scl;
    f64 e1, e2, p1, p2, p3, pr, sm;
    f64 wr[2], wi[2];
    f64 c1_tmp, s1_tmp;

    i32 ldas = lda1 * lda2;
    i32 one_i = 1;

    SLC_DLASET("Full", &m, &m, &ZERO, &ONE, dwork, &m);

    if (sgle) {
        for (i32 j = k - 1; j >= 1; j--) {
            i32 ai = amap[j] - 1;
            const f64 *a_slice = a + ai * ldas;

            if (s[ai] == sinv) {
                t = t * a_slice[0];

                for (i32 ic = 0; ic < mm; ic += m) {
                    SLC_DTRMV("Upper", "NoTran", "NonUnit", &m, &a_slice[l + l * lda1],
                              &lda1, &dwork[ic], &one_i);
                }
            } else {
                t = t / a_slice[0];

                SLC_DLACPY("Upper", &m, &m, &a_slice[l + l * lda1], &lda1, y, &m);
                i32 m_1 = m - 1;
                i32 idx = 1;
                SLC_DLASET("Lower", &m_1, &m_1, &ZERO, &ZERO, &y[idx], &m);
                SLC_DGETC2(&m, y, &m, ip, jp, &ind);

                for (i32 ic = 0; ic < mm; ic += m) {
                    SLC_DGESC2(&m, y, &m, &dwork[ic], ip, jp, &scl);
                }
            }
        }

        i32 ai = amap[0] - 1;
        const f64 *a_slice = a + ai * ldas;
        e1 = a_slice[0] * t;
        e2 = a_slice[1] * t;

        i32 two_i = 2;
        i32 n_idx = n - 1;
        i32 n_1_idx = n - 2;
        p1 = SLC_DDOT(&two_i, &a_slice[n_idx + n_1_idx * lda1], &lda1, &dwork[m], &one_i);

        f64 arg1 = e1 - p1;
        SLC_DLARTG(&arg1, &e2, c1, s1, &e1);
        *c2 = ONE;
        *s2 = ZERO;
    } else {
        z[0] = ONE;
        z[1] = ZERO;
        z[2] = ZERO;
        z[3] = ONE;

        for (i32 j = k - 1; j >= 1; j--) {
            i32 ai = amap[j] - 1;
            const f64 *a_slice = a + ai * ldas;

            if (s[ai] == sinv) {
                f64 a11 = a_slice[0];
                f64 a12 = a_slice[lda1];
                f64 a22 = a_slice[1 + lda1];

                z[0] = a11 * z[0];
                z[2] = a11 * z[2] + a12 * z[3];
                z[3] = a22 * z[3];

                for (i32 ic = 0; ic < mm; ic += m) {
                    SLC_DTRMV("Upper", "NoTran", "NonUnit", &m, &a_slice[l + l * lda1],
                              &lda1, &dwork[ic], &one_i);
                }
            } else {
                y[0] = a_slice[0];
                y[1] = ZERO;
                i32 two_i = 2;
                SLC_DCOPY(&two_i, &a_slice[lda1], &one_i, &y[2], &one_i);
                SLC_DGETC2(&two_i, y, &two_i, ip, jp, &ind);
                SLC_DGESC2(&two_i, y, &two_i, z, ip, jp, &scl);
                SLC_DGESC2(&two_i, y, &two_i, &z[2], ip, jp, &scl);

                SLC_DLACPY("Upper", &m, &m, &a_slice[l + l * lda1], &lda1, y, &m);
                i32 m_1 = m - 1;
                i32 idx = 1;
                SLC_DLASET("Lower", &m_1, &m_1, &ZERO, &ZERO, &y[idx], &m);
                SLC_DGETC2(&m, y, &m, ip, jp, &ind);

                for (i32 ic = 0; ic < mm; ic += m) {
                    SLC_DGESC2(&m, y, &m, &dwork[ic], ip, jp, &scl);
                }
            }
        }

        i32 ai = amap[0] - 1;
        const f64 *a_slice = a + ai * ldas;
        e1 = a_slice[0] * z[0];
        e2 = a_slice[1] * z[0];

        f64 a11 = a_slice[0];
        f64 a12 = a_slice[lda1];
        f64 a22 = a_slice[1 + lda1];
        f64 a21 = a_slice[1];
        f64 a32 = a_slice[2 + lda1];

        p1 = a11 * z[2] + a12 * z[3];
        p2 = a21 * z[2] + a22 * z[3];
        p3 = a32 * z[3];

        i32 l_new = n - 2;
        i32 l_1 = n - 3;
        i32 n_idx = n - 1;

        i32 two_i = 2;
        z[0] = SLC_DDOT(&two_i, &a_slice[l_new + l_1 * lda1], &lda1, &dwork[m], &one_i);
        z[1] = a_slice[n_idx + l_new * lda1] * dwork[m + 1];
        z[2] = SLC_DDOT(&m, &a_slice[l_new + l_1 * lda1], &lda1, &dwork[mm - m], &one_i);
        z[3] = SLC_DDOT(&two_i, &a_slice[n_idx + l_new * lda1], &lda1, &dwork[mm - 2], &one_i);

        SLC_DLANV2(&z[0], &z[2], &z[1], &z[3], &wr[0], &wi[0], &wr[1], &wi[1],
                   &c1_tmp, &s1_tmp);

        if (wi[0] == ZERO) {
            if (fabs(wr[0]) < fabs(wr[1])) {
                t = wr[0];
            } else {
                t = wr[1];
            }
            sm = TWO * t;
            pr = t * t;
        } else {
            sm = TWO * wr[0];
            pr = wr[0] * wr[0] + wi[0] * wi[0];
        }

        p1 = p1 + ((e1 - sm) * e1 + pr) / e2;
        p2 = p2 + e1 - sm;

        SLC_DLARTG(&p2, &p3, c2, s2, &e1);
        SLC_DLARTG(&p1, &e1, c1, s1, &e2);
    }
}
