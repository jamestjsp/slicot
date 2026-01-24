/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <ctype.h>
#include <math.h>

void mb03ah(const char *shft, i32 k, i32 n, const i32 *amap, const i32 *s,
            i32 sinv, const f64 *a, i32 lda1, i32 lda2, f64 *c1, f64 *s1,
            f64 *c2, f64 *s2) {
    const f64 ONE = 1.0;
    const f64 TWO = 2.0;
    const f64 ZERO = 0.0;
    i32 int1 = 1;
    i32 int2 = 2;

    char shft_upper = (char)toupper((unsigned char)shft[0]);
    bool sgle = (shft_upper == 'S') || (n == 2);

    i32 m = (n < 3) ? n : 3;
    i32 mm = m * m;
    i32 ldas = lda1 * lda2;

    f64 dwork[9];
    f64 y[9];
    f64 z[4];  /* 2x2 matrix stored column-major */
    i32 ip[3], jp[3];
    f64 wi[2], wr[2];

    f64 e1, e2, p1, p2, p3, pr, scl, sm, t;
    i32 i, ic, ind, j;

    /* Initialize DWORK to identity matrix M x M */
    SLC_DLASET("Full", &m, &m, &ZERO, &ONE, dwork, &m);

    /* Evaluate needed part of matrix product */
    for (j = k - 2; j >= 0; j--) {
        i = amap[j] - 1;  /* Convert to 0-based */
        const f64 *a_slice = a + i * ldas;

        if (s[i] == sinv) {
            /* Multiply DWORK by A(:,:,i) from the left */
            for (ic = 0; ic < mm; ic += m) {
                SLC_DTRMV("Upper", "NoTran", "NonUnit", &m, a_slice, &lda1,
                          &dwork[ic], &int1);
            }
        } else {
            /* Solve linear system with complete pivoting */
            SLC_DLACPY("Upper", &m, &m, a_slice, &lda1, y, &m);
            if (m > 1) {
                i32 m1 = m - 1;
                SLC_DLASET("Lower", &m1, &m1, &ZERO, &ZERO, &y[1], &m);
            }
            SLC_DGETC2(&m, y, &m, ip, jp, &ind);

            for (ic = 0; ic < mm; ic += m) {
                SLC_DGESC2(&m, y, &m, &dwork[ic], ip, jp, &scl);
            }
        }
    }

    /* Compute elements of first two columns of product */
    i = amap[k - 1] - 1;  /* Hessenberg matrix index (0-based) */
    const f64 *a_hess = a + i * ldas;

    SLC_DCOPY(&int2, a_hess, &int1, y, &int1);
    SLC_DTRMV("Upper", "NoTran", "NonUnit", &int2, dwork, &m, y, &int1);
    e1 = y[0];
    e2 = y[1];

    if (sgle) {
        /* Single shift: compute (N,N) element of product as shift */
        p1 = ONE;

        for (j = 0; j < k; j++) {
            i = amap[j] - 1;
            const f64 *a_j = a + i * ldas;
            f64 ann = a_j[(n - 1) + (n - 1) * lda1];

            if (s[i] == sinv) {
                p1 = p1 * ann;
            } else {
                p1 = p1 / ann;
            }
        }

        f64 temp = e1 - p1;
        SLC_DLARTG(&temp, &e2, c1, s1, &e1);
        *c2 = ONE;
        *s2 = ZERO;
    } else {
        /* Double shift */
        SLC_DCOPY(&m, &a_hess[lda1], &int1, y, &int1);
        SLC_DTRMV("Upper", "NoTran", "NonUnit", &m, dwork, &m, y, &int1);
        p1 = y[0];
        p2 = y[1];
        p3 = y[2];

        /* Compute bottom 2x2 part using complete pivoting */
        SLC_DLASET("Full", &int2, &int2, &ZERO, &ONE, z, &int2);

        i32 nm1 = n - 1;  /* m = n - 1 for this section (Fortran M = N-1) */

        for (j = k - 2; j >= 0; j--) {
            i = amap[j] - 1;
            const f64 *a_j = a + i * ldas;
            f64 amm = a_j[(nm1 - 1) + (nm1 - 1) * lda1];  /* A(M,M) = A(N-1,N-1) */
            f64 amn = a_j[(nm1 - 1) + (n - 1) * lda1];     /* A(M,N) = A(N-1,N) */
            f64 ann = a_j[(n - 1) + (n - 1) * lda1];       /* A(N,N) */

            if (s[i] == sinv) {
                z[0] = amm * z[0];
                z[2] = amm * z[2] + amn * z[3];  /* Z(1,2) */
                z[3] = ann * z[3];               /* Z(2,2) */
            } else {
                /* Y is 2x2 matrix: [[A(M,M), 0], [A(M,N), A(N,N)]]^T column-major:
                   Y = [A(M,M), A(M,N); 0, A(N,N)] but lower is zeroed */
                f64 ytmp[4];
                ytmp[0] = amm;
                ytmp[1] = ZERO;
                SLC_DCOPY(&int2, &a_j[(nm1 - 1) + (n - 1) * lda1], &int1, &ytmp[2], &int1);

                SLC_DGETC2(&int2, ytmp, &int2, ip, jp, &ind);
                SLC_DGESC2(&int2, ytmp, &int2, z, ip, jp, &scl);
                SLC_DGESC2(&int2, ytmp, &int2, &z[2], ip, jp, &scl);
            }
        }

        /* Final product with Hessenberg factor */
        i = amap[k - 1] - 1;
        const f64 *a_k = a + i * ldas;
        f64 amm = a_k[(nm1 - 1) + (nm1 - 1) * lda1];
        f64 amn = a_k[(nm1 - 1) + (n - 1) * lda1];
        f64 anm = a_k[(n - 1) + (nm1 - 1) * lda1];
        f64 ann = a_k[(n - 1) + (n - 1) * lda1];

        t = z[0] * amm + z[2] * anm;
        z[2] = z[0] * amn + z[2] * ann;
        z[0] = t;
        z[1] = z[3] * anm;
        z[3] = z[3] * ann;

        /* Compute eigenvalues of bottom 2x2 part */
        SLC_DLANV2(&z[0], &z[2], &z[1], &z[3], &wr[0], &wi[0], &wr[1], &wi[1], c1, s1);

        if (wi[0] == ZERO) {
            /* Two real eigenvalues: use the one with minimum modulus */
            if (fabs(wr[0]) < fabs(wr[1])) {
                t = wr[0];
            } else {
                t = wr[1];
            }
            sm = TWO * t;
            pr = t * t;
        } else {
            /* Complex conjugate pair */
            sm = TWO * wr[0];
            pr = wr[0] * wr[0] + wi[0] * wi[0];
        }

        /* Compute first column of double shift polynomial */
        p1 = p1 + ((e1 - sm) * e1 + pr) / e2;
        p2 = p2 + e1 - sm;

        /* Compute rotations to annihilate P2 and P3 */
        SLC_DLARTG(&p2, &p3, c2, s2, &e1);
        SLC_DLARTG(&p1, &e1, c1, s1, &e2);
    }
}
