/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>

static inline i32 max_i32(i32 a, i32 b) {
    return a > b ? a : b;
}

static inline i32 min_i32(i32 a, i32 b) {
    return a < b ? a : b;
}

static inline i32 max3_i32(i32 a, i32 b, i32 c) {
    return max_i32(a, max_i32(b, c));
}

void ab08nx(i32 n, i32 m, i32 p, i32* ro, i32* sigma, f64 svlmax,
            f64* abcd, i32 ldabcd, i32* ninfz, i32* infz, i32* kronl,
            i32* mu, i32* nu, i32* nkrol, f64 tol, i32* iwork,
            f64* dwork, i32 ldwork, i32* info) {

    const f64 ZERO = 0.0;

    i32 np = n + p;
    i32 mpm = min_i32(p, m);

    *info = 0;
    bool lquery = (ldwork == -1);

    if (n < 0) {
        *info = -1;
    } else if (m < 0) {
        *info = -2;
    } else if (p < 0) {
        *info = -3;
    } else if (*ro != p && *ro != max_i32(p - m, 0)) {
        *info = -4;
    } else if (*sigma != 0 && *sigma != m) {
        *info = -5;
    } else if (svlmax < ZERO) {
        *info = -6;
    } else if (ldabcd < max_i32(1, np)) {
        *info = -8;
    } else if (*ninfz < 0) {
        *info = -9;
    } else {
        i32 jwork = max_i32(1, max_i32(mpm + max_i32(3*m - 1, n),
                           min_i32(p, n) + max3_i32(3*p - 1, np, n + m)));
        if (lquery) {
            i32 wrkopt = jwork;
            sl_int p_int = p, n_int = n, mpm_int = mpm;
            sl_int ldabcd_int = ldabcd;
            sl_int lwork_query = -1;
            sl_int info_query = 0;

            if (m > 0) {
                SLC_DORMQR("Left", "Transpose", &p_int, &n_int, &mpm_int,
                           abcd, &ldabcd_int, dwork, abcd, &ldabcd_int,
                           dwork, &lwork_query, &info_query);
                wrkopt = max_i32(jwork, mpm + (i32)dwork[0]);
            }

            sl_int min_pn = min_i32(p, n);
            sl_int np_int = np;
            SLC_DORMRQ("Right", "Transpose", &np_int, &n_int, &min_pn,
                       abcd, &ldabcd_int, dwork, abcd, &ldabcd_int,
                       dwork, &lwork_query, &info_query);
            wrkopt = max_i32(wrkopt, min_i32(p, n) + (i32)dwork[0]);

            sl_int mn_int = m + n;
            SLC_DORMRQ("Left", "NoTranspose", &n_int, &mn_int, &min_pn,
                       abcd, &ldabcd_int, dwork, abcd, &ldabcd_int,
                       dwork, &lwork_query, &info_query);
            wrkopt = max_i32(wrkopt, min_i32(p, n) + (i32)dwork[0]);

            dwork[0] = (f64)wrkopt;
            return;
        } else if (ldwork < jwork) {
            *info = -18;
        }
    }

    if (*info != 0) {
        return;
    }

    *mu = p;
    *nu = n;
    i32 iz = 0;
    i32 ik = 0;
    i32 mm1 = m;
    i32 itau = 0;
    *nkrol = 0;
    i32 wrkopt = 1;

    f64 sval[3];
    i32 i1, irow, jwork, mnu, rank, ro1, tau, mntau;

    while (*mu != 0) {
        ro1 = *ro;
        mnu = m + *nu;

        if (m > 0) {
            if (*sigma != 0) {
                irow = *nu;
                for (i1 = 0; i1 < *sigma; i1++) {
                    sl_int ro_plus_1 = *ro + 1;
                    sl_int inc1 = 1;
                    f64 t;
                    SLC_DLARFG(&ro_plus_1, &abcd[i1 * ldabcd + irow],
                               &abcd[i1 * ldabcd + irow + 1], &inc1, &t);

                    sl_int mnu_minus_i1 = mnu - i1 - 1;
                    SLC_DLATZM("L", &ro_plus_1, &mnu_minus_i1,
                               &abcd[i1 * ldabcd + irow + 1], &inc1, &t,
                               &abcd[(i1 + 1) * ldabcd + irow],
                               &abcd[(i1 + 1) * ldabcd + irow + 1],
                               &ldabcd, dwork);
                    irow++;
                }

                sl_int ro_sigma_minus_1 = *ro + *sigma - 1;
                sl_int sigma_int = *sigma;
                sl_int ldabcd_int = ldabcd;
                sl_int nu_plus_1 = *nu + 1;
                SLC_DLASET("Lower", &ro_sigma_minus_1, &sigma_int, &ZERO, &ZERO,
                           &abcd[nu_plus_1], &ldabcd_int);
            }

            if (*sigma < m) {
                jwork = itau + min_i32(ro1, m);
                i1 = *sigma;
                irow = *nu + i1;

                sl_int m_minus_sigma = m - *sigma;
                mb03oy(ro1, m_minus_sigma, &abcd[i1 * ldabcd + irow], ldabcd,
                       tol, svlmax, &rank, sval, iwork, &dwork[itau],
                       &dwork[jwork], info);
                wrkopt = max_i32(wrkopt, jwork + 3*m - 1);

                sl_int nu_plus_sigma = *nu + *sigma;
                sl_int forwrd = 1;
                sl_int ldabcd_int = ldabcd;
                SLC_DLAPMT(&forwrd, &nu_plus_sigma, &m_minus_sigma,
                           &abcd[i1 * ldabcd], &ldabcd_int, iwork);

                if (rank > 0) {
                    sl_int ro1_int = ro1;
                    sl_int nu_int = *nu;
                    sl_int rank_int = rank;
                    sl_int ldwork_minus_jwork = ldwork - jwork;
                    sl_int info_dormqr = 0;
                    SLC_DORMQR("Left", "Transpose", &ro1_int, &nu_int, &rank_int,
                               &abcd[i1 * ldabcd + irow], &ldabcd_int,
                               &dwork[itau], &abcd[mm1 * ldabcd + irow],
                               &ldabcd_int, &dwork[jwork], &ldwork_minus_jwork,
                               &info_dormqr);
                    wrkopt = max_i32(wrkopt, (i32)dwork[jwork] + jwork);

                    if (ro1 > 1) {
                        sl_int ro1_minus_1 = ro1 - 1;
                        sl_int min_ro1_rank = min_i32(ro1 - 1, rank);
                        sl_int irow_plus_1 = irow + 1;
                        SLC_DLASET("Lower", &ro1_minus_1, &min_ro1_rank, &ZERO, &ZERO,
                                   &abcd[i1 * ldabcd + irow_plus_1], &ldabcd_int);
                    }
                    ro1 = ro1 - rank;
                }
            }
        }

        tau = ro1;
        *sigma = *mu - tau;

        if (iz > 0) {
            infz[iz - 1] = infz[iz - 1] + *ro - tau;
            *ninfz = *ninfz + iz * (*ro - tau);
        }
        if (ro1 == 0) {
            break;
        }
        iz++;

        if (*nu <= 0) {
            *mu = *sigma;
            *nu = 0;
            *ro = 0;
        } else {
            i1 = *nu + *sigma;
            mntau = min_i32(tau, *nu);
            jwork = itau + mntau;

            mb03py(tau, *nu, &abcd[mm1 * ldabcd + i1], ldabcd, tol, svlmax,
                   &rank, sval, iwork, &dwork[itau], &dwork[jwork], info);
            wrkopt = max_i32(wrkopt, jwork + 3*tau);

            if (rank > 0) {
                irow = i1 + tau - rank;
                sl_int i1_int = i1;
                sl_int nu_int = *nu;
                sl_int rank_int = rank;
                sl_int ldabcd_int = ldabcd;
                sl_int ldwork_minus_jwork = ldwork - jwork;
                sl_int info_dormrq = 0;

                SLC_DORMRQ("Right", "Transpose", &i1_int, &nu_int, &rank_int,
                           &abcd[mm1 * ldabcd + irow], &ldabcd_int,
                           &dwork[mntau - rank], &abcd[mm1 * ldabcd],
                           &ldabcd_int, &dwork[jwork], &ldwork_minus_jwork,
                           &info_dormrq);
                wrkopt = max_i32(wrkopt, (i32)dwork[jwork] + jwork);

                sl_int mnu_int = mnu;
                SLC_DORMRQ("Left", "NoTranspose", &nu_int, &mnu_int, &rank_int,
                           &abcd[mm1 * ldabcd + irow], &ldabcd_int,
                           &dwork[mntau - rank], abcd, &ldabcd_int,
                           &dwork[jwork], &ldwork_minus_jwork, &info_dormrq);
                wrkopt = max_i32(wrkopt, (i32)dwork[jwork] + jwork);

                sl_int nu_minus_rank = *nu - rank;
                SLC_DLASET("Full", &rank_int, &nu_minus_rank, &ZERO, &ZERO,
                           &abcd[mm1 * ldabcd + irow], &ldabcd_int);

                if (rank > 1) {
                    sl_int rank_minus_1 = rank - 1;
                    sl_int irow_plus_1 = irow + 1;
                    i32 col_offset = mm1 + *nu - rank;
                    SLC_DLASET("Lower", &rank_minus_1, &rank_minus_1, &ZERO, &ZERO,
                               &abcd[col_offset * ldabcd + irow_plus_1], &ldabcd_int);
                }
            }
            *ro = rank;
        }

        kronl[ik] = kronl[ik] + tau - *ro;
        *nkrol = *nkrol + kronl[ik];
        ik++;

        *nu = *nu - *ro;
        *mu = *sigma + *ro;

        if (*ro == 0) {
            break;
        }
    }

    dwork[0] = (f64)wrkopt;
}
