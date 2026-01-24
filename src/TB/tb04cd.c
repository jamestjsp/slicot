/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 2002-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

/*
 * TB04CD - State-space to minimal pole-zero-gain form
 *
 * Purpose:
 *   To compute the transfer function matrix G of a state-space
 *   representation (A,B,C,D) of a linear time-invariant multivariable
 *   system, using the pole-zeros method. The transfer function matrix
 *   is returned in a minimal pole-zero-gain form.
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <stdbool.h>

static inline bool lsame(char ca, char cb) {
    if (ca >= 'a' && ca <= 'z') ca -= 32;
    if (cb >= 'a' && cb <= 'z') cb -= 32;
    return ca == cb;
}

static inline i32 imax(i32 a, i32 b) { return a > b ? a : b; }
static inline i32 imin(i32 a, i32 b) { return a < b ? a : b; }

void tb04cd(const char* jobd, const char* equil, i32 n, i32 m, i32 p, i32 npz,
            f64* a, i32 lda, f64* b, i32 ldb, f64* c, i32 ldc,
            const f64* d, i32 ldd,
            i32* nz, i32 ldnz, i32* np, i32 ldnp,
            f64* zerosr, f64* zerosi, f64* polesr, f64* polesi,
            f64* gains, i32 ldgain, f64 tol,
            i32* iwork, f64* dwork, i32 ldwork, i32* info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const f64 C100 = 100.0;
    const i32 int1 = 1;

    f64 anorm, dij, epsn, maxred, toldef;
    i32 i, ia, iac, ias, ib, ic, icc, ierr, im, ip, ipm1, itau, itau1, iz, j;
    i32 jwk, jwork, jwork1, k, ncont, wrkopt;
    bool dijnz, fndeig, withd;
    f64 z_dummy[1] = {0.0};

    *info = 0;
    withd = lsame(*jobd, 'D');

    if (!withd && !lsame(*jobd, 'Z')) {
        *info = -1;
    } else if (!lsame(*equil, 'S') && !lsame(*equil, 'N')) {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    } else if (m < 0) {
        *info = -4;
    } else if (p < 0) {
        *info = -5;
    } else if (npz < 0) {
        *info = -6;
    } else if (lda < imax(1, n)) {
        *info = -8;
    } else if (ldb < imax(1, n)) {
        *info = -10;
    } else if (ldc < imax(1, p)) {
        *info = -12;
    } else if (ldd < 1 || (withd && ldd < p)) {
        *info = -14;
    } else if (ldnz < imax(1, p)) {
        *info = -16;
    } else if (ldnp < imax(1, p)) {
        *info = -18;
    } else if (ldgain < imax(1, p)) {
        *info = -24;
    } else if (ldwork < imax(1, n * (n + p) + imax(n + imax(n, p), n * (2 * n + 3)))) {
        *info = -28;
    }

    if (*info != 0) {
        return;
    }

    dij = ZERO;
    if (imin(imin(n, p), m) == 0) {
        if (imin(p, m) > 0) {
            for (j = 0; j < m; j++) {
                for (i = 0; i < p; i++) {
                    nz[i + j * ldnz] = 0;
                    np[i + j * ldnp] = 0;
                    if (withd)
                        dij = d[i + j * ldd];
                    gains[i + j * ldgain] = dij;
                }
            }
        }
        dwork[0] = ONE;
        return;
    }

    toldef = tol;
    if (toldef <= ZERO) {
        epsn = (f64)n * SLC_DLAMCH("E");
        anorm = SLC_DLANGE("F", &n, &n, a, &lda, dwork);
    }

    ia = 0;
    ic = ia + n * n;
    itau = ic + p * n;
    jwork = itau + n;
    iac = itau;

    k = 0;

    if (lsame(*equil, 'S')) {
        maxred = C100;
        tb01id("A", n, m, p, &maxred, a, lda, b, ldb, c, ldc, dwork, &ierr);
    }

    for (j = 0; j < m; j++) {
        SLC_DLACPY("F", &n, &n, a, &lda, &dwork[ia], &n);
        SLC_DLACPY("F", &p, &n, c, &ldc, &dwork[ic], &p);

        tb01zd("N", n, p, &dwork[ia], n, &b[0 + j * ldb], &dwork[ic], p,
               &ncont, z_dummy, 1, &dwork[itau], tol,
               &dwork[jwork], ldwork - jwork, &ierr);

        if (j == 0)
            wrkopt = (i32)dwork[jwork] + jwork;

        ib = iac + ncont * ncont;
        icc = ib + ncont;
        itau1 = icc + ncont;
        jwk = itau1 + ncont;
        ias = itau1;
        jwork1 = ias + ncont * ncont;

        for (i = 0; i < p; i++) {
            if (ncont > 0) {
                if (withd)
                    dij = d[i + j * ldd];

                ma02ad("F", ncont, ncont, &dwork[ia], n, &dwork[iac], ncont);
                SLC_DCOPY(&ncont, &b[0 + j * ldb], &int1, &dwork[ib], &int1);
                for (i32 ii = 0; ii < ncont; ii++) {
                    dwork[icc + ii] = dwork[ic + i + ii * p];
                }

                tb01zd("N", ncont, 1, &dwork[iac], ncont, &dwork[icc],
                       &dwork[ib], 1, &ip, z_dummy, 1,
                       &dwork[itau1], tol, &dwork[jwk], ldwork - jwk, &ierr);

                if (i == 0)
                    wrkopt = imax(wrkopt, (i32)dwork[jwk] + jwk);

                if (ip > 0) {
                    SLC_DLACPY("F", &ip, &ip, &dwork[iac], &ncont, &dwork[ias], &ip);

                    i32 ilo = 1, ihi = ip;
                    i32 lwork1 = ldwork - jwork1;
                    SLC_DHSEQR("E", "N", &ip, &ilo, &ihi, &dwork[iac], &ncont,
                               &polesr[k], &polesi[k], z_dummy, &int1,
                               &dwork[jwork1], &lwork1, &ierr);
                    if (ierr != 0) {
                        *info = 2;
                        return;
                    }
                    wrkopt = imax(wrkopt, (i32)dwork[jwork1] + jwork1);

                    ipm1 = ip - 1;
                    dijnz = withd && dij != ZERO;
                    fndeig = dijnz || ipm1 > 0;

                    if (!fndeig) {
                        iz = 0;
                    } else if (dijnz) {
                        iz = ip;
                        SLC_DLACPY("F", &iz, &iz, &dwork[ias], &ip, &dwork[iac], &ncont);
                        f64 alpha = -dwork[icc] / dij;
                        SLC_DAXPY(&iz, &alpha, &dwork[ib], &int1, &dwork[iac], &ncont);
                    } else {
                        if (tol <= ZERO) {
                            f64 bfnorm = SLC_DLANGE("F", &ip, &int1, &dwork[ib], &int1, &dwork[0]);
                            toldef = epsn * (anorm > bfnorm ? anorm : bfnorm);
                        }

                        im = -1;
                        for (i32 ii = 0; ii < ipm1; ii++) {
                            if (fabs(dwork[ib + ii]) > toldef) {
                                im = ii;
                                break;
                            }
                        }

                        if (im < 0) {
                            iz = 0;
                        } else {
                            iz = ip - im - 1;
                            i32 ias_off = ias + (im + 1) * (ip + 1);
                            SLC_DLACPY("F", &iz, &iz, &dwork[ias_off], &ip, &dwork[iac], &ncont);

                            f64 alpha = -dwork[ias + im * (ip + 1) + 1] / dwork[ib + im];
                            SLC_DAXPY(&iz, &alpha, &dwork[ib + im + 1], &int1, &dwork[iac], &ncont);
                        }
                    }

                    if (fndeig && iz > 0) {
                        i32 ilo = 1, ihi = iz;
                        i32 lwork1 = ldwork - jwork1;
                        SLC_DHSEQR("E", "N", &iz, &ilo, &ihi, &dwork[iac], &ncont,
                                   &zerosr[k], &zerosi[k], z_dummy, &int1,
                                   &dwork[jwork1], &lwork1, &ierr);
                        if (ierr != 0) {
                            *info = 1;
                            return;
                        }
                    }

                    if (dijnz) {
                        gains[i + j * ldgain] = dij;
                    } else {
                        tb04bx(ip, iz, &dwork[ias], ip, &dwork[icc],
                               &dwork[ib], dij, &polesr[k], &polesi[k],
                               &zerosr[k], &zerosi[k], &gains[i + j * ldgain], iwork);
                    }
                    nz[i + j * ldnz] = iz;
                    np[i + j * ldnp] = ip;
                } else {
                    nz[i + j * ldnz] = 0;
                    np[i + j * ldnp] = 0;
                    gains[i + j * ldgain] = withd ? dij : ZERO;
                }
            } else {
                nz[i + j * ldnz] = 0;
                np[i + j * ldnp] = 0;
                gains[i + j * ldgain] = withd ? d[i + j * ldd] : ZERO;
            }
            k += npz;
        }
    }

    dwork[0] = (f64)wrkopt;
}
