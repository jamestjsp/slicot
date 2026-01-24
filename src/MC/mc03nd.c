/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <stdlib.h>
#include <math.h>

void mc03nd(i32 mp, i32 np, i32 dp, const f64 *p, i32 ldp1, i32 ldp2,
            i32 *dk, i32 *gam, f64 *nullsp, i32 ldnull,
            f64 *ker, i32 ldker1, i32 ldker2, f64 tol,
            i32 *iwork, f64 *dwork, i32 ldwork, i32 *info)
{
    i32 m = dp * mp;
    i32 h = m - mp;
    i32 n = h + np;

    *info = 0;

    i32 max_mp_1 = (mp > 1) ? mp : 1;
    i32 max_np_1 = (np > 1) ? np : 1;

    if (mp < 0) {
        *info = -1;
    } else if (np < 0) {
        *info = -2;
    } else if (dp <= 0) {
        *info = -3;
    } else if (ldp1 < max_mp_1) {
        *info = -5;
    } else if (ldp2 < max_np_1) {
        *info = -6;
    } else if (ldnull < max_np_1) {
        *info = -10;
    } else if (ldker1 < max_np_1) {
        *info = -12;
    } else if (ldker2 < max_np_1) {
        *info = -13;
    } else if (ldwork < n * (m * n + 2 * (m + n))) {
        *info = -17;
    }

    if (*info != 0) {
        i32 neg_info = -(*info);
        SLC_XERBLA("MC03ND", &neg_info);
        return;
    }

    if (mp == 0 || np == 0) {
        *dk = -1;
        return;
    }

    i32 jworka = 0;
    i32 jworke = jworka + m * n;
    i32 jworkz = jworke + m * n;
    i32 jworkv = jworkz + n * n;
    i32 jworkq = jworka;

    mc03nx(mp, np, dp, p, ldp1, ldp2, &dwork[jworka], m, &dwork[jworke], m);

    f64 toler1 = SLC_DLANGE("F", &m, &np, &dwork[jworke + h * m], &m, dwork);
    f64 toler2 = SLC_DLANGE("F", &mp, &np, p, &ldp1, dwork);
    f64 toler = (toler1 > toler2) ? toler1 : toler2;
    f64 eps = SLC_DLAMCH("Epsilon");
    f64 sqrt_h = sqrt((f64)h);
    toler = 10.0 * eps * SLC_DLAPY2(&toler, &sqrt_h);
    if (toler <= tol) {
        toler = tol;
    }

    i32 ranke;
    mb04ud("No Q", "Identity Z", m, n, &dwork[jworka], m,
           &dwork[jworke], m, &dwork[jworkq], m, &dwork[jworkz], n,
           &ranke, iwork, toler, &dwork[jworkv], info);

    i32 max_n_mp1 = (n > m + 1) ? n : (m + 1);
    i32 muk = m;
    i32 nuk = muk + max_n_mp1;
    i32 tail = nuk + max_n_mp1;

    i32 nblcks, nblcki;
    i32 mnei[3];
    mb04vd("Separation", "No Q", "Update Z", m, n, ranke,
           &dwork[jworka], m, &dwork[jworke], m,
           &dwork[jworkq], m, &dwork[jworkz], n,
           iwork, &nblcks, &nblcki, &iwork[muk], &iwork[nuk], &iwork[tail],
           mnei, toler, &iwork[tail], info);

    if (*info > 0) {
        *info = *info + nblcks;
        return;
    }

    if (nblcks < 1 || mnei[1] == 0) {
        *dk = -1;
        return;
    }

    *dk = nblcks - 1;
    i32 nra = mnei[0];
    i32 nca = mnei[1];

    mc03ny(nblcks, nra, nca, &dwork[jworka], m, &dwork[jworke], m,
           &iwork[muk], &iwork[nuk], &dwork[jworkv], n, info);

    if (*info > 0) {
        return;
    }

    i32 ncv = iwork[muk] - iwork[nuk];
    gam[0] = ncv;
    iwork[0] = 0;
    iwork[tail] = iwork[muk];

    for (i32 i = 1; i < nblcks; i++) {
        i32 idiff = iwork[muk + i] - iwork[nuk + i];
        gam[i] = idiff;
        iwork[i] = ncv;
        ncv = ncv + (i + 1) * idiff;
        iwork[tail + i] = iwork[tail + i - 1] + iwork[muk + i];
    }

    f64 zero = 0.0;
    f64 one = 1.0;
    SLC_DLASET("Full", &np, &ncv, &zero, &zero, nullsp, &ldnull);

    i32 vc1 = 0;

    for (i32 i = 0; i < nblcks; i++) {
        i32 vr2 = iwork[tail + i];

        for (i32 j = 0; j <= i; j++) {
            i32 gami = gam[i];
            SLC_DGEMM("No transpose", "No transpose", &np, &gami, &vr2,
                      &one, &dwork[jworkz + h], &n,
                      &dwork[jworkv + vc1 * n], &n, &zero, &nullsp[vc1 * ldnull],
                      &ldnull);
            vc1 = vc1 + gami;
            vr2 = vr2 - iwork[muk + i - j];
        }
    }

    i32 sgamk = 0;

    for (i32 k = 0; k < nblcks; k++) {
        SLC_DLASET("Full", &np, &sgamk, &zero, &zero, &ker[k * ldker1 * ldker2],
                   &ldker1);
        i32 ifir = sgamk;

        for (i32 j = k; j < nblcks; j++) {
            i32 gamj = gam[j];
            vc1 = iwork[j] + k * gamj;
            SLC_DLACPY("Full", &np, &gamj, &nullsp[vc1 * ldnull], &ldnull,
                       &ker[ifir * ldker1 + k * ldker1 * ldker2], &ldker1);
            ifir = ifir + gamj;
        }

        sgamk = sgamk + gam[k];
    }
}
