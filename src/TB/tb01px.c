/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
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

static i32 imax(i32 a, i32 b) {
    return a > b ? a : b;
}

static f64 fmax2(f64 a, f64 b) {
    return a > b ? a : b;
}

static f64 fmax4(f64 a, f64 b, f64 c, f64 d) {
    return fmax2(fmax2(a, b), fmax2(c, d));
}

/**
 * @brief Computes minimal/controllable/observable realization
 *
 * TB01PX finds a reduced (controllable, observable, or minimal) state-space
 * representation (Ar,Br,Cr) for any original state-space representation
 * (A,B,C). The matrix Ar is in an upper block Hessenberg staircase form.
 *
 * @param[in] job   'M': minimal, 'C': controllable, 'O': observable
 * @param[in] equil 'S': balance (scale), 'N': no balancing
 * @param[in] n     Order of original system (n >= 0)
 * @param[in] m     Number of inputs (m >= 0)
 * @param[in] p     Number of outputs (p >= 0)
 * @param[in,out] a State matrix (n x n), on exit contains Ar in (nr x nr)
 * @param[in] lda   Leading dimension of a (lda >= max(1,n))
 * @param[in,out] b Input matrix (n x m or n x max(m,p) if job != 'C')
 * @param[in] ldb   Leading dimension of b (ldb >= max(1,n))
 * @param[in,out] c Output matrix (p x n or max(m,p) x n if job != 'C')
 * @param[in] ldc   Leading dimension of c (ldc >= max(1,m,p) if n > 0)
 * @param[out] nr   Order of reduced system
 * @param[out] infred Information on reduction (4 elements)
 * @param[in] tol   Tolerance for rank determination (if <= 0, default used)
 * @param[out] iwork Integer workspace
 * @param[out] dwork Real workspace
 * @param[in] ldwork Length of dwork
 * @param[out] info  0: success, <0: -i means param i invalid
 */
void tb01px(const char* job, const char* equil, i32 n, i32 m, i32 p,
            f64* a, i32 lda, f64* b, i32 ldb, f64* c, i32 ldc,
            i32* nr, i32* infred, f64 tol,
            i32* iwork, f64* dwork, i32 ldwork, i32* info) {

    const f64 ONE = 1.0;
    const f64 HUNDR = 100.0;
    const i32 LDIZ = 1;

    bool lequil, lnjobc, lnjobo, lspace;
    i32 i, ib, icon, indcon, itau, iz, jwork, kl, kwa, kwb, kwc;
    i32 ldwmin, maxmp, ncont, wrkopt = 1;
    f64 anorm, bnorm, cnorm, maxred;

    *info = 0;
    maxmp = imax(m, p);
    ldwmin = n + imax(n, 3 * maxmp);
    lnjobc = !lsame(*job, 'C');
    lnjobo = !lsame(*job, 'O');
    lequil = lsame(*equil, 'S');

    if (lnjobc && lnjobo && !lsame(*job, 'M')) {
        *info = -1;
    } else if (!lequil && !lsame(*equil, 'N')) {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    } else if (m < 0) {
        *info = -4;
    } else if (p < 0) {
        *info = -5;
    } else if (lda < imax(1, n)) {
        *info = -7;
    } else if (ldb < imax(1, n)) {
        *info = -9;
    } else if ((ldc < 1) || (n > 0 && ldc < maxmp)) {
        *info = -11;
    } else if (ldwork < 1 || (n > 0 && ldwork < ldwmin)) {
        *info = -17;
    }

    if (*info != 0) {
        return;
    }

    infred[0] = -1;
    infred[1] = -1;
    infred[3] = 0;

    if (n == 0 || (lnjobc && p == 0) || (lnjobo && m == 0)) {
        *nr = 0;
        infred[2] = 0;
        dwork[0] = ONE;
        return;
    }

    anorm = SLC_DLANGE("1", &n, &n, a, &lda, dwork);
    bnorm = SLC_DLANGE("1", &n, &m, b, &ldb, dwork);
    cnorm = SLC_DLANGE("1", &p, &n, c, &ldc, dwork);

    if (lequil) {
        maxred = fmax4(HUNDR, anorm, bnorm, cnorm);
        tb01id("A", n, m, p, &maxred, a, lda, b, ldb, c, ldc, dwork, info);
    }

    lspace = ldwork >= n * (n + m + p) + ldwmin;
    if (lspace) {
        kwa = 0;
        kwb = kwa + n * n;
        kwc = kwb + n * m;
        iz = kwc + p * n;
    } else {
        iz = 0;
    }

    itau = iz;
    jwork = itau + n;
    kl = imax(0, n - 1);
    ib = 0;

    if (lnjobo) {
        if (lspace) {
            i32 n_loc = n;
            i32 p_loc_max = imax(1, p);
            SLC_DLACPY("F", &n_loc, &n_loc, a, &lda, &dwork[kwa], &n_loc);
            SLC_DLACPY("F", &n_loc, &m, b, &ldb, &dwork[kwb], &n_loc);
            SLC_DLACPY("F", &p, &n_loc, c, &ldc, &dwork[kwc], &p_loc_max);
        }

        i32 ldwork_sub = ldwork - jwork;
        tb01ud("N", n, m, p, a, lda, b, ldb, c, ldc,
               &ncont, &icon, iwork, &dwork[iz], LDIZ, &dwork[itau], tol,
               &iwork[n], &dwork[jwork], ldwork_sub, info);

        wrkopt = (i32)dwork[jwork] + jwork;

        if (ncont < n || !lspace) {
            if (icon > 1) {
                kl = iwork[0] + iwork[1] - 1;
            } else if (icon == 1) {
                kl = iwork[0] - 1;
            } else {
                kl = 0;
            }
            infred[0] = n - ncont;
            infred[3] = icon;
            if (lnjobc) {
                ib = n;
            }
        } else {
            i32 n_loc = n;
            i32 p_loc_max = imax(1, p);
            SLC_DLACPY("F", &n_loc, &n_loc, &dwork[kwa], &n_loc, a, &lda);
            SLC_DLACPY("F", &n_loc, &m, &dwork[kwb], &n_loc, b, &ldb);
            SLC_DLACPY("F", &p, &n_loc, &dwork[kwc], &p_loc_max, c, &ldc);
        }
    } else {
        ncont = n;
    }

    if (lnjobc) {
        if (lspace && ((lnjobo && ncont < n) || !lnjobo)) {
            i32 ncont_loc = ncont;
            i32 p_loc_max = imax(1, p);
            SLC_DLACPY("F", &ncont_loc, &ncont_loc, a, &lda, &dwork[kwa], &n);
            SLC_DLACPY("F", &ncont_loc, &m, b, &ldb, &dwork[kwb], &n);
            SLC_DLACPY("F", &p, &ncont_loc, c, &ldc, &dwork[kwc], &p_loc_max);
        }

        f64 dummy_d = 0.0;
        ab07md('Z', ncont, m, p, a, lda, b, ldb, c, ldc, &dummy_d, 1);

        i32 ldwork_sub = ldwork - jwork;
        tb01ud("N", ncont, p, m, a, lda, b, ldb, c, ldc,
               nr, &indcon, &iwork[ib], &dwork[iz], LDIZ, &dwork[itau], tol,
               &iwork[ib + n], &dwork[jwork], ldwork_sub, info);

        wrkopt = imax(wrkopt, (i32)dwork[jwork] + jwork);

        if (*nr < ncont || !lspace) {
            if (indcon > 1) {
                kl = iwork[ib] + iwork[ib + 1] - 1;
            } else if (indcon == 1) {
                kl = iwork[ib] - 1;
            } else {
                kl = 0;
            }

            i32 nr_m1 = imax(0, *nr - 1);
            f64 dummy_d = 0.0;
            tb01xd("Z", *nr, p, m, kl, nr_m1, a, lda, b, ldb, c, ldc, &dummy_d, 1, info);

            infred[1] = ncont - *nr;
            infred[3] = indcon;

            if (lnjobo) {
                for (i = 0; i < indcon; i++) {
                    iwork[i] = iwork[ib + i];
                }
            }
        } else {
            i32 ncont_loc = ncont;
            i32 p_loc_max = imax(1, p);
            SLC_DLACPY("F", &ncont_loc, &ncont_loc, &dwork[kwa], &n, a, &lda);
            SLC_DLACPY("F", &ncont_loc, &m, &dwork[kwb], &n, b, &ldb);
            SLC_DLACPY("F", &p, &ncont_loc, &dwork[kwc], &p_loc_max, c, &ldc);
        }
    } else {
        *nr = ncont;
    }

    infred[2] = kl;
    dwork[0] = (f64)imax(wrkopt, ldwmin + n * (n + m + p));
}
