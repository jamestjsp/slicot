/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 2002-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 *
 * TB04BD - Transfer function matrix via pole-zero method
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <stdbool.h>

static bool lsame(char ca, char cb) {
    if (ca >= 'a' && ca <= 'z') ca -= 32;
    if (cb >= 'a' && cb <= 'z') cb -= 32;
    return ca == cb;
}

static inline i32 imax(i32 a, i32 b) { return a > b ? a : b; }
static inline i32 imin(i32 a, i32 b) { return a < b ? a : b; }

void tb04bd(const char* jobd, const char* order, const char* equil,
            i32 n, i32 m, i32 p, i32 md,
            f64* a, i32 lda, f64* b, i32 ldb, f64* c, i32 ldc,
            const f64* d, i32 ldd,
            i32* ign, i32 ldign, i32* igd, i32 ldigd,
            f64* gn, f64* gd, f64 tol,
            i32* iwork, f64* dwork, i32 ldwork, i32* info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const f64 C100 = 100.0;
    const i32 int1 = 1;
    const i32 int0 = 0;

    bool ascend, withd;
    f64 anorm, dij, epsn, maxred, toldef, x;
    i32 i, ia, iac, ias, ib, ic, icc, ierr, iip, im;
    i32 ip, ipm1, irp, itau, itau1, iz, j, jj, jwork, jwork1, k, l, ncont, wrkopt;
    bool dijnz, fndeig;
    f64 z_arr[1];

    *info = 0;
    withd = lsame(*jobd, 'D');
    ascend = lsame(*order, 'I');

    if (!withd && !lsame(*jobd, 'Z')) {
        *info = -1;
    } else if (!ascend && !lsame(*order, 'D')) {
        *info = -2;
    } else if (!lsame(*equil, 'S') && !lsame(*equil, 'N')) {
        *info = -3;
    } else if (n < 0) {
        *info = -4;
    } else if (m < 0) {
        *info = -5;
    } else if (p < 0) {
        *info = -6;
    } else if (md < 1) {
        *info = -7;
    } else if (lda < imax(1, n)) {
        *info = -9;
    } else if (ldb < imax(1, n)) {
        *info = -11;
    } else if (ldc < imax(1, p)) {
        *info = -13;
    } else if (ldd < 1 || (withd && ldd < p)) {
        *info = -15;
    } else if (ldign < imax(1, p)) {
        *info = -17;
    } else if (ldigd < imax(1, p)) {
        *info = -19;
    } else if (ldwork < imax(1, n * (n + p) + imax(n + imax(n, p), n * (2 * n + 5)))) {
        *info = -25;
    }

    if (*info != 0) {
        return;
    }

    z_arr[0] = ZERO;
    i32 pmmd = p * m * md;
    SLC_DCOPY(&pmmd, z_arr, &int0, gn, &int1);
    SLC_DCOPY(&pmmd, z_arr, &int0, gd, &int1);

    if (imin(imin(n, p), m) == 0) {
        if (imin(p, m) > 0) {
            k = 0;
            for (j = 0; j < m; j++) {
                for (i = 0; i < p; i++) {
                    ign[i + j * ldign] = 0;
                    igd[i + j * ldigd] = 0;
                    if (withd) {
                        gn[k] = d[i + j * ldd];
                    }
                    gd[k] = ONE;
                    k += md;
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
    dij = ZERO;

    if (lsame(*equil, 'S')) {
        maxred = C100;
        tb01id("All", n, m, p, &maxred, a, lda, b, ldb, c, ldc, dwork, &ierr);
    }

    for (j = 0; j < m; j++) {
        SLC_DLACPY("F", &n, &n, a, &lda, &dwork[ia], &n);
        SLC_DLACPY("F", &p, &n, c, &ldc, &dwork[ic], &p);

        i32 ldwork_sub = ldwork - jwork;
        tb01zd("N", n, p, &dwork[ia], n, &b[j * ldb], &dwork[ic], p,
               &ncont, z_arr, 1, &dwork[itau], tol, &dwork[jwork], ldwork_sub, &ierr);
        if (j == 0) {
            wrkopt = (i32)dwork[jwork] + jwork;
        }

        ib = iac + ncont * ncont;
        icc = ib + ncont;
        itau1 = icc + ncont;
        irp = itau1;
        iip = irp + ncont;
        ias = iip + ncont;
        jwork1 = ias + ncont * ncont;

        for (i = 0; i < p; i++) {
            if (withd) {
                dij = d[i + j * ldd];
            }

            if (ncont > 0) {
                ma02ad("F", ncont, ncont, &dwork[ia], n, &dwork[iac], ncont);
                SLC_DCOPY(&ncont, &b[j * ldb], &int1, &dwork[ib], &int1);
                SLC_DCOPY(&ncont, &dwork[ic + i], &p, &dwork[icc], &int1);

                i32 ldwork_sub2 = ldwork - iip;
                tb01zd("N", ncont, 1, &dwork[iac], ncont, &dwork[icc], &dwork[ib], 1,
                       &ip, z_arr, 1, &dwork[itau1], tol, &dwork[iip], ldwork_sub2, &ierr);
                if (i == 0) {
                    wrkopt = imax(wrkopt, (i32)dwork[iip] + iip);
                }

                if (ip > 0) {
                    SLC_DLACPY("F", &ip, &ip, &dwork[iac], &ncont, &dwork[ias], &ip);

                    i32 ilo = 1, ihi = ip;
                    i32 ldwork_sub3 = ldwork - jwork1;
                    SLC_DHSEQR("E", "N", &ip, &ilo, &ihi, &dwork[iac], &ncont,
                               &dwork[irp], &dwork[iip], z_arr, &int1,
                               &dwork[jwork1], &ldwork_sub3, &ierr);
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
                        f64 coef = -dwork[icc] / dij;
                        SLC_DAXPY(&iz, &coef, &dwork[ib], &int1, &dwork[iac], &ncont);
                    } else {
                        if (tol <= ZERO) {
                            toldef = epsn * fmax(anorm, SLC_DLANGE("F", &ip, &int1, &dwork[ib], &int1, dwork));
                        }

                        im = -1;
                        for (i32 mm = 0; mm < ipm1; mm++) {
                            if (fabs(dwork[ib + mm]) > toldef) {
                                im = mm;
                                break;
                            }
                        }

                        if (im < 0) {
                            iz = 0;
                        } else {
                            iz = ip - im - 1;
                            i32 src = ias + (im + 1) * (ip + 1);
                            SLC_DLACPY("F", &iz, &iz, &dwork[src], &ip, &dwork[iac], &ncont);

                            f64 coef = -dwork[ias + (im + 1) * (ip + 1) - ip] / dwork[ib + im];
                            SLC_DAXPY(&iz, &coef, &dwork[ib + im + 1], &int1, &dwork[iac], &ncont);
                        }
                    }

                    if (fndeig && iz > 0) {
                        i32 ilo_z = 1, ihi_z = iz;
                        i32 ldwork_sub4 = ldwork - jwork1;
                        SLC_DHSEQR("E", "N", &iz, &ilo_z, &ihi_z, &dwork[iac], &ncont,
                                   &gn[k], &gd[k], z_arr, &int1,
                                   &dwork[jwork1], &ldwork_sub4, &ierr);
                        if (ierr != 0) {
                            *info = 1;
                            return;
                        }
                    }

                    if (dijnz) {
                        x = dij;
                    } else {
                        tb04bx(ip, iz, &dwork[ias], ip, &dwork[icc], &dwork[ib], dij,
                               &dwork[irp], &dwork[iip], &gn[k], &gd[k], &x, iwork);
                    }

                    if (ascend) {
                        mc01pd(iz, &gn[k], &gd[k], &dwork[ib], &dwork[ias], &ierr);
                    } else {
                        mc01py(iz, &gn[k], &gd[k], &dwork[ib], &dwork[ias], &ierr);
                    }

                    jj = k;
                    for (l = ib; l <= ib + iz; l++) {
                        gn[jj] = dwork[l] * x;
                        jj++;
                    }

                    if (ascend) {
                        mc01pd(ip, &dwork[irp], &dwork[iip], &gd[k], &dwork[ias], &ierr);
                    } else {
                        mc01py(ip, &dwork[irp], &dwork[iip], &gd[k], &dwork[ias], &ierr);
                    }

                    ign[i + j * ldign] = iz;
                    igd[i + j * ldigd] = ip;
                } else {
                    ign[i + j * ldign] = 0;
                    igd[i + j * ldigd] = 0;
                    gn[k] = dij;
                    gd[k] = ONE;
                }
            } else {
                ign[i + j * ldign] = 0;
                igd[i + j * ldigd] = 0;
                gn[k] = dij;
                gd[k] = ONE;
            }

            k += md;
        }
    }
}
