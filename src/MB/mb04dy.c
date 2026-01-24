/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 *
 * MB04DY - Symplectic scaling of a Hamiltonian matrix
 *
 * Purpose:
 *   Performs symplectic scaling on a Hamiltonian matrix:
 *       H = [ A    G  ]
 *           [ Q   -A' ]
 *   where A is N-by-N and G, Q are symmetric N-by-N matrices.
 *
 *   Scaling strategies:
 *   'S': Symplectic scaling using DGEBAL + equilibration
 *   '1'/'O': 1-norm scaling by power of machine base
 *   'N': No scaling
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <ctype.h>
#include <math.h>

void mb04dy(const char *jobscl, i32 n, f64 *a, i32 lda,
            f64 *qg, i32 ldqg, f64 *d, f64 *dwork, i32 *info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    char job_upper = (char)toupper((unsigned char)jobscl[0]);
    bool symp = (job_upper == 'S');
    bool norm = (job_upper == '1') || (job_upper == 'O');
    bool none = (job_upper == 'N');

    *info = 0;

    if (!symp && !norm && !none) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (lda < (1 > n ? 1 : n) && !none) {
        *info = -4;
    } else if (ldqg < (1 > n ? 1 : n) && !none) {
        *info = -6;
    }

    if (*info != 0) {
        return;
    }

    if (n == 0 || none) {
        return;
    }

    f64 base = SLC_DLAMCH("B");
    f64 eps = SLC_DLAMCH("P");
    f64 ufl = SLC_DLAMCH("S");
    f64 ofl = ONE / ufl;
    SLC_DLABAD(&ufl, &ofl);
    f64 sfmax = (eps / base) / ufl;
    f64 sfmin = ONE / sfmax;

    if (norm) {
        f64 anrm = SLC_DLANGE("1", &n, &n, a, &lda, dwork);
        f64 gnrm = SLC_DLANSY("1", "U", &n, &qg[ldqg], &ldqg, dwork);
        f64 qnrm = SLC_DLANSY("1", "L", &n, qg, &ldqg, dwork);

        f64 y = ONE;
        if (anrm > y) y = anrm;
        if (gnrm > y) y = gnrm;
        if (qnrm > y) y = qnrm;

        f64 tau = ONE;
        f64 sqrt_sfmax = sqrt(sfmax);
        while (tau < y && tau < sqrt_sfmax) {
            tau = tau * base;
        }

        if (tau > ONE) {
            f64 tau_div_base = tau / base;
            if (fabs(tau_div_base - y) < fabs(tau - y)) {
                tau = tau_div_base;
            }

            i32 ierr;
            SLC_DLASCL("G", &(i32){0}, &(i32){0}, &tau, &ONE, &n, &n, a, &lda, &ierr);
            SLC_DLASCL("U", &(i32){0}, &(i32){0}, &tau, &ONE, &n, &n, &qg[ldqg], &ldqg, &ierr);
            SLC_DLASCL("U", &(i32){0}, &(i32){0}, &tau, &ONE, &n, &n, &qg[ldqg], &ldqg, &ierr);
        }

        d[0] = tau;
    } else {
        i32 ilo, ihi, ierr;
        SLC_DGEBAL("S", &n, a, &lda, &ilo, &ihi, d, &ierr);

        for (i32 j = 0; j < n; j++) {
            for (i32 i = j; i < n; i++) {
                qg[i + j * ldqg] = qg[i + j * ldqg] * d[j] * d[i];
            }
        }

        for (i32 j = 1; j <= n; j++) {
            for (i32 i = 0; i < j; i++) {
                qg[i + j * ldqg] = qg[i + j * ldqg] / d[j - 1] / d[i];
            }
        }

        f64 gnrm = SLC_DLANSY("1", "U", &n, &qg[ldqg], &ldqg, dwork);
        f64 qnrm = SLC_DLANSY("1", "L", &n, qg, &ldqg, dwork);

        f64 rho;
        if (gnrm == ZERO) {
            if (qnrm == ZERO) {
                rho = ONE;
            } else {
                rho = sfmax;
            }
        } else if (qnrm == ZERO) {
            rho = sfmin;
        } else {
            rho = sqrt(qnrm) / sqrt(gnrm);
        }

        SLC_DLASCL("L", &(i32){0}, &(i32){0}, &rho, &ONE, &n, &n, qg, &ldqg, &ierr);
        SLC_DLASCL("U", &(i32){0}, &(i32){0}, &ONE, &rho, &n, &n, &qg[ldqg], &ldqg, &ierr);

        f64 sqrt_rho = sqrt(rho);
        i32 int1 = 1;
        SLC_DRSCL(&n, &sqrt_rho, d, &int1);
    }
}
