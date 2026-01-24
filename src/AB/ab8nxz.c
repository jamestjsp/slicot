/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>

i32 slicot_mb3oyz(i32 m, i32 n, c128* a, i32 lda, f64 rcond, f64 svlmax,
                  i32* rank, f64* sval, i32* jpvt, c128* tau, f64* dwork,
                  c128* zwork);

i32 slicot_mb3pyz(i32 m, i32 n, c128* a, i32 lda, f64 rcond, f64 svlmax,
                  i32* rank, f64* sval, i32* jpvt, c128* tau, f64* dwork,
                  c128* zwork);

i32 slicot_ab8nxz(i32 n, i32 m, i32 p, i32* ro, i32* sigma, f64 svlmax,
                  c128* abcd, i32 ldabcd, i32* ninfz, i32* infz, i32* kronl,
                  i32* mu, i32* nu, i32* nkrol, f64 tol, i32* iwork,
                  f64* dwork, c128* zwork, i32 lzwork) {
    const c128 ZERO = 0.0 + 0.0*I;
    const f64 DZERO = 0.0;

    i32 np = n + p;
    i32 mpm = (p < m) ? p : m;
    i32 info = 0;
    bool lquery = (lzwork == -1);

    if (n < 0) {
        info = -1;
    } else if (m < 0) {
        info = -2;
    } else if (p < 0) {
        info = -3;
    } else if (*ro != p && *ro != ((p - m > 0) ? p - m : 0)) {
        info = -4;
    } else if (*sigma != 0 && *sigma != m) {
        info = -5;
    } else if (svlmax < DZERO) {
        info = -6;
    } else if (ldabcd < (np > 1 ? np : 1)) {
        info = -8;
    } else if (*ninfz < 0) {
        info = -9;
    } else {
        i32 minpn = (p < n) ? p : n;
        i32 jwork1 = mpm + (3 * m - 1 > n ? 3 * m - 1 : n);
        i32 jwork2 = minpn + (3 * p - 1 > np ? 3 * p - 1 : (np > n + m ? np : n + m));
        i32 jwork = (1 > jwork1 ? 1 : jwork1);
        jwork = (jwork > jwork2 ? jwork : jwork2);

        if (lquery) {
            i32 wrkopt = jwork;
            if (m > 0) {
                i32 len_n = n;
                i32 len_mpm = mpm;
                SLC_ZUNMQR("Left", "Conjugate", &p, &len_n, &len_mpm, abcd,
                          &ldabcd, zwork, abcd, &ldabcd, zwork, &(i32){-1}, &info);
                i32 opt1 = mpm + (i32)creal(zwork[0]);
                wrkopt = (wrkopt > opt1 ? wrkopt : opt1);
            }
            i32 len_np = np;
            i32 len_n = n;
            i32 minpn2 = (p < n) ? p : n;
            SLC_ZUNMRQ("Right", "ConjTranspose", &len_np, &len_n, &minpn2,
                      abcd, &ldabcd, zwork, abcd, &ldabcd, zwork, &(i32){-1}, &info);
            i32 opt2 = minpn2 + (i32)creal(zwork[0]);
            wrkopt = (wrkopt > opt2 ? wrkopt : opt2);
            i32 len_nm = n + m;
            SLC_ZUNMRQ("Left", "NoTranspose", &len_n, &len_nm, &minpn2,
                      abcd, &ldabcd, zwork, abcd, &ldabcd, zwork, &(i32){-1}, &info);
            i32 opt3 = minpn2 + (i32)creal(zwork[0]);
            wrkopt = (wrkopt > opt3 ? wrkopt : opt3);
            zwork[0] = (f64)wrkopt + 0.0*I;
            return 0;
        } else if (lzwork < jwork) {
            info = -19;
        }
    }

    if (info != 0) {
        i32 xinfo = -info;
        SLC_XERBLA("AB8NXZ", &xinfo);
        return info;
    }

    *mu = p;
    *nu = n;
    i32 iz = 0;
    i32 ik = 0;
    i32 mm1 = m;
    i32 itau = 0;
    *nkrol = 0;
    i32 wrkopt = 1;

    i32 one = 1;
    i32 ro1;
    i32 tau_idx;
    f64 sval[3];

    while (*mu != 0) {
        ro1 = *ro;
        i32 mnu = m + *nu;

        if (m > 0) {
            if (*sigma != 0) {
                i32 irow = *nu;
                for (i32 i1 = 0; i1 < *sigma; i1++) {
                    c128 tc;
                    i32 len = *ro + 1;
                    SLC_ZLARFG(&len, &abcd[irow + i1 * ldabcd],
                              &abcd[irow + 1 + i1 * ldabcd], &one, &tc);
                    i32 len2 = mnu - i1 - 1;
                    if (len2 > 0) {
                        c128 tc_conj = conj(tc);
                        SLC_ZLATZM("L", &len, &len2, &abcd[irow + 1 + i1 * ldabcd],
                                  &one, &tc_conj, &abcd[irow + (i1 + 1) * ldabcd],
                                  &abcd[irow + 1 + (i1 + 1) * ldabcd], &ldabcd, zwork);
                    }
                    irow++;
                }
                i32 nset = *ro + *sigma - 1;
                i32 mset = (*sigma < nset ? *sigma : nset);
                for (i32 j = 0; j < mset; j++) {
                    for (i32 i = 0; i < nset - j; i++) {
                        abcd[*nu + 1 + i + j * ldabcd] = ZERO;
                    }
                }
            }

            if (*sigma < m) {
                i32 jwork_idx = itau + ((ro1 < m) ? ro1 : m);
                i32 i1 = *sigma;
                i32 irow = *nu + i1;
                i32 rank;
                i32 m_sigma = m - *sigma;
                slicot_mb3oyz(ro1, m_sigma, &abcd[irow + i1 * ldabcd], ldabcd,
                             tol, svlmax, &rank, sval, iwork, &zwork[itau],
                             dwork, &zwork[jwork_idx]);
                wrkopt = (wrkopt > jwork_idx + 3 * m - 2 ? wrkopt : jwork_idx + 3 * m - 2);

                i32 nu_sigma = *nu + *sigma;
                i32 forward = 1;
                SLC_ZLAPMT(&forward, &nu_sigma, &m_sigma, &abcd[i1 * ldabcd],
                          &ldabcd, iwork);

                if (rank > 0) {
                    i32 jwork2 = jwork_idx;
                    i32 lwork = lzwork - jwork2;
                    SLC_ZUNMQR("Left", "Conjugate", &ro1, nu, &rank,
                              &abcd[irow + i1 * ldabcd], &ldabcd, &zwork[itau],
                              &abcd[irow + mm1 * ldabcd], &ldabcd, &zwork[jwork2],
                              &lwork, &info);
                    i32 opt = (i32)creal(zwork[jwork2]) + jwork2;
                    wrkopt = (wrkopt > opt ? wrkopt : opt);

                    if (ro1 > 1) {
                        i32 nset = ro1 - 1;
                        i32 mset = (nset < rank ? nset : rank);
                        for (i32 j = 0; j < mset; j++) {
                            for (i32 i = 0; i < nset - j; i++) {
                                abcd[irow + 1 + i + (i1 + j) * ldabcd] = ZERO;
                            }
                        }
                    }
                    ro1 = ro1 - rank;
                }
            }
        }

        tau_idx = ro1;
        *sigma = *mu - tau_idx;

        if (iz > 0) {
            infz[iz - 1] = infz[iz - 1] + *ro - tau_idx;
            *ninfz = *ninfz + iz * (*ro - tau_idx);
        }

        if (ro1 == 0) break;
        iz++;

        if (*nu <= 0) {
            *mu = *sigma;
            *nu = 0;
            *ro = 0;
        } else {
            i32 i1 = *nu + *sigma;
            i32 mntau = (tau_idx < *nu) ? tau_idx : *nu;
            i32 jwork_idx = itau + mntau;

            i32 rank;
            slicot_mb3pyz(tau_idx, *nu, &abcd[i1 + mm1 * ldabcd], ldabcd,
                         tol, svlmax, &rank, sval, iwork, &zwork[itau],
                         dwork, &zwork[jwork_idx]);
            wrkopt = (wrkopt > jwork_idx + 3 * tau_idx - 1 ? wrkopt : jwork_idx + 3 * tau_idx - 1);

            if (rank > 0) {
                i32 irow = i1 + tau_idx - rank;
                i32 i1_m1 = i1;
                i32 lwork = lzwork - jwork_idx;
                SLC_ZUNMRQ("Right", "ConjTranspose", &i1_m1, nu, &rank,
                          &abcd[irow + mm1 * ldabcd], &ldabcd, &zwork[mntau - rank],
                          &abcd[mm1 * ldabcd], &ldabcd, &zwork[jwork_idx],
                          &lwork, &info);
                i32 opt1 = (i32)creal(zwork[jwork_idx]) + jwork_idx;
                wrkopt = (wrkopt > opt1 ? wrkopt : opt1);

                i32 mnu2 = m + *nu;
                SLC_ZUNMRQ("Left", "NoTranspose", nu, &mnu2, &rank,
                          &abcd[irow + mm1 * ldabcd], &ldabcd, &zwork[mntau - rank],
                          abcd, &ldabcd, &zwork[jwork_idx], &lwork, &info);
                i32 opt2 = (i32)creal(zwork[jwork_idx]) + jwork_idx;
                wrkopt = (wrkopt > opt2 ? wrkopt : opt2);

                i32 nu_rank = *nu - rank;
                for (i32 j = 0; j < nu_rank; j++) {
                    for (i32 i = 0; i < rank; i++) {
                        abcd[irow + i + (mm1 + j) * ldabcd] = ZERO;
                    }
                }
                if (rank > 1) {
                    for (i32 j = 0; j < rank - 1; j++) {
                        for (i32 i = 1; i < rank - j; i++) {
                            abcd[irow + i + (mm1 + nu_rank + j) * ldabcd] = ZERO;
                        }
                    }
                }
            }
            *ro = rank;
        }

        kronl[ik] = kronl[ik] + tau_idx - *ro;
        *nkrol = *nkrol + kronl[ik];
        ik++;

        *nu = *nu - *ro;
        *mu = *sigma + *ro;

        if (*ro == 0) break;
    }

    zwork[0] = (f64)wrkopt + 0.0*I;
    return 0;
}
