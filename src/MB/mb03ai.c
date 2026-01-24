/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <ctype.h>
#include <math.h>
#include <string.h>

void mb03ai(const char *shft, i32 k, i32 n, const i32 *amap, const i32 *s,
            i32 sinv, const f64 *a, i32 lda1, i32 lda2, f64 *c1, f64 *s1,
            f64 *c2, f64 *s2, f64 *dwork) {
    const f64 ONE = 1.0;
    const f64 TWO = 2.0;
    const f64 ZERO = 0.0;

    char shft_upper = (char)toupper((unsigned char)shft[0]);
    bool sgle = (shft_upper == 'S');

    i32 nn = n * n;
    i32 ir = nn;
    i32 ii = ir + n;
    i32 ldas = lda1 * lda2;

    i32 one_i = 1;
    i32 ind;

    SLC_DLASET("Full", &n, &n, &ZERO, &ONE, dwork, &n);

    for (i32 j = 0; j < k - 1; j++) {
        i32 ai = amap[j] - 1;
        if (ai < 0) continue;
        const f64 *a_slice = a + ai * ldas;

        if (s[ai] == sinv) {
            SLC_DTRMM("Right", "Upper", "NoTran", "NonUnit", &n, &n,
                      &ONE, a_slice, &lda1, dwork, &n);
        } else {
            SLC_DTRSM("Right", "Upper", "NoTran", "NonUnit", &n, &n,
                      &ONE, a_slice, &lda1, dwork, &n);
        }
    }

    i32 ai = amap[k - 1] - 1;
    const f64 *a_slice = a + ai * ldas;
    SLC_DCOPY(&n, &a_slice[(n - 1) * lda1], &one_i, &dwork[ir], &one_i);
    SLC_DTRMV("Upper", "NoTran", "NonUnit", &n, dwork, &n, &dwork[ir], &one_i);

    i32 jj = ir - n;
    for (i32 ll = n - 1; ll >= 1; ll--) {
        i32 lp1 = ll + 1;
        SLC_DCOPY(&lp1, &a_slice[(ll - 1) * lda1], &one_i, &dwork[ii], &one_i);
        SLC_DTRMV("Upper", "NoTran", "NonUnit", &lp1, dwork, &n, &dwork[ii], &one_i);
        SLC_DCOPY(&lp1, &dwork[ii], &one_i, &dwork[jj], &one_i);
        jj -= n;
    }

    jj = 0;
    for (i32 ll = 1; ll <= n; ll++) {
        i32 lp1 = ll + 1;
        memmove(&dwork[jj], &dwork[jj + n], (size_t)lp1 * sizeof(f64));
        jj += n;
    }

    if (sgle) {
        f64 e1, arg1;
        arg1 = dwork[0] - dwork[nn - 1];
        SLC_DLARTG(&arg1, &dwork[1], c1, s1, &e1);
        *c2 = ONE;
        *s2 = ZERO;
    } else {
        f64 e1 = dwork[0];
        f64 e2 = dwork[1];
        f64 p1 = dwork[n];
        f64 p2 = dwork[n + 1];
        f64 p3 = dwork[n + 2];

        f64 z_dummy[1];
        SLC_DLAHQR(&(i32){0}, &(i32){0}, &n, &(i32){1}, &n, dwork, &n,
                   &dwork[ir], &dwork[ii], &(i32){1}, &(i32){1}, z_dummy,
                   &(i32){1}, &ind);

        ind = 0;
        i32 in2 = 0;
        bool isc = false;
        f64 mxc = ZERO;
        f64 mxr = ZERO;
        f64 md;
        i32 in1 = 0;

        for (i32 i = ii; i < ii + n; i++) {
            if (dwork[i] != ZERO) {
                isc = true;
                md = SLC_DLAPY2(&dwork[i - n], &dwork[i]);
                if (md > mxc) {
                    mxc = md;
                    ind = i;
                }
            } else {
                md = fabs(dwork[i - n]);
                in1 = in2;
                if (md > mxr) {
                    mxr = md;
                    in2 = i - n;
                }
            }
        }

        f64 sm, pr;
        if (isc) {
            sm = TWO * dwork[ind - n];
            pr = mxc * mxc;
        } else {
            if (in1 == in2) {
                mxr = ZERO;
                sm = dwork[in2];
                dwork[in2] = ZERO;

                for (i32 i = ir; i < ir + n; i++) {
                    md = fabs(dwork[i]);
                    if (md > mxr) {
                        mxr = md;
                        in1 = i;
                    }
                }

                dwork[in2] = sm;
            }
            sm = dwork[in1] + dwork[in2];
            pr = dwork[in1] * dwork[in2];
        }

        p1 = p1 + ((e1 - sm) * e1 + pr) / e2;
        p2 = p2 + e1 - sm;

        SLC_DLARTG(&p2, &p3, c2, s2, &e1);
        SLC_DLARTG(&p1, &e1, c1, s1, &e2);
    }
}
