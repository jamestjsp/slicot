/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <ctype.h>
#include <stdlib.h>

static inline i32 max_i32(i32 a, i32 b) {
    return a > b ? a : b;
}

static inline i32 min_i32(i32 a, i32 b) {
    return a < b ? a : b;
}

void ab01nd(const char* jobz, i32 n, i32 m, f64* a, i32 lda, f64* b, i32 ldb,
            i32* ncont, i32* indcon, i32* nblk, f64* z, i32 ldz, f64* tau,
            f64 tol, i32* iwork, f64* dwork, i32 ldwork, i32* info) {

    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    char jobz_upper = (char)toupper((unsigned char)jobz[0]);
    bool ljobf = (jobz_upper == 'F');
    bool ljobi = (jobz_upper == 'I');
    bool ljobz = ljobf || ljobi;

    *info = 0;

    if (!ljobz && jobz_upper != 'N') {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (m < 0) {
        *info = -3;
    } else if (lda < max_i32(1, n)) {
        *info = -5;
    } else if (ldb < max_i32(1, n)) {
        *info = -7;
    } else if (ldz < 1 || (ljobz && ldz < n)) {
        *info = -12;
    } else if (ldwork < max_i32(1, max_i32(n, 3 * m))) {
        *info = -17;
    }

    if (*info != 0) {
        return;
    }

    *ncont = 0;
    *indcon = 0;

    if (min_i32(n, m) == 0) {
        if (n > 0) {
            if (ljobi) {
                sl_int n_int = n;
                SLC_DLASET("Full", &n_int, &n_int, &ZERO, &ONE, z, &ldz);
            } else if (ljobf) {
                sl_int n_int = n;
                sl_int one_int = 1;
                SLC_DLASET("Full", &n_int, &n_int, &ZERO, &ZERO, z, &ldz);
                SLC_DLASET("Full", &n_int, &one_int, &ZERO, &ZERO, tau, &n_int);
            }
        }
        dwork[0] = ONE;
        return;
    }

    f64 anorm = SLC_DLANGE("M", &n, &n, a, &lda, dwork);
    f64 bnorm = SLC_DLANGE("M", &n, &m, b, &ldb, dwork);

    if (bnorm == ZERO) {
        if (ljobi) {
            sl_int n_int = n;
            SLC_DLASET("Full", &n_int, &n_int, &ZERO, &ONE, z, &ldz);
        } else if (ljobf) {
            sl_int n_int = n;
            sl_int one_int = 1;
            SLC_DLASET("Full", &n_int, &n_int, &ZERO, &ZERO, z, &ldz);
            SLC_DLASET("Full", &n_int, &one_int, &ZERO, &ZERO, tau, &n_int);
        }
        dwork[0] = ONE;
        return;
    }

    i32 info_tmp = 0;
    mb01pd("S", "G", n, n, 0, 0, anorm, 0, NULL, a, lda, &info_tmp);
    mb01pd("S", "G", n, m, 0, 0, bnorm, 0, NULL, b, ldb, &info_tmp);

    f64 fnrm = SLC_DLANGE("F", &n, &m, b, &ldb, dwork);

    f64 toldef = tol;
    if (toldef <= ZERO) {
        toldef = (f64)(n * n) * SLC_DLAMCH("Precision");
    }

    i32 wrkopt = 1;
    i32 ni = 0;
    i32 itau = 0;
    i32 ncrt = n;
    i32 mcrt = m;
    i32 iqr = 0;

    f64 sval[3];
    i32 rank;

    while (1) {
        mb03oy(ncrt, mcrt, &b[iqr], ldb, toldef, fnrm, &rank,
               sval, iwork, &tau[itau], dwork, &info_tmp);

        if (rank != 0) {
            i32 nj = ni;
            ni = *ncont;
            *ncont = *ncont + rank;
            (*indcon)++;
            nblk[*indcon - 1] = rank;

            sl_int ncrt_int = ncrt;
            sl_int rank_int = rank;
            sl_int n_int = n;
            sl_int ldwork_int = ldwork;

            SLC_DORMQR("Left", "Transpose", &ncrt_int, &ncrt_int, &rank_int,
                       &b[iqr], &ldb, &tau[itau],
                       &a[(ni) + (ni) * lda], &lda, dwork, &ldwork_int, &info_tmp);
            wrkopt = max_i32(wrkopt, (i32)dwork[0]);

            SLC_DORMQR("Right", "No transpose", &n_int, &ncrt_int, &rank_int,
                       &b[iqr], &ldb, &tau[itau],
                       &a[(ni) * lda], &lda, dwork, &ldwork_int, &info_tmp);
            wrkopt = max_i32(wrkopt, (i32)dwork[0]);

            if (ljobz && ncrt > 1) {
                sl_int ncrt_m1 = ncrt - 1;
                sl_int rank_m1 = min_i32(rank, ncrt - 1);
                SLC_DLACPY("L", &ncrt_m1, &rank_m1,
                           &b[iqr + 1], &ldb,
                           &z[(ni + 1) + itau * ldz], &ldz);
            }

            if (rank > 1) {
                sl_int rank_m1 = rank - 1;
                SLC_DLASET("L", &rank_m1, &rank_m1, &ZERO, &ZERO,
                           &b[iqr + 1], &ldb);
            }

            if (*indcon == 1) {
                sl_int rank_int = rank;
                sl_int m_int = m;
                i32 forward = 0;
                SLC_DLAPMT(&forward, &rank_int, &m_int, &b[iqr], &ldb, iwork);
                iqr = rank;
                fnrm = SLC_DLANGE("F", &n, &n, a, &lda, dwork);
            } else {
                for (i32 j = 0; j < mcrt; j++) {
                    i32 col_idx = nj + iwork[j] - 1;
                    sl_int rank_int = rank;
                    sl_int one_int = 1;
                    SLC_DCOPY(&rank_int, &b[iqr + j * ldb], &one_int,
                              &a[(ni) + col_idx * lda], &one_int);
                }
            }

            itau = itau + rank;

            if (rank != ncrt) {
                mcrt = rank;
                ncrt = ncrt - rank;
                sl_int ncrt_int = ncrt;
                sl_int mcrt_int = mcrt;
                SLC_DLACPY("G", &ncrt_int, &mcrt_int,
                           &a[(*ncont) + (ni) * lda], &lda,
                           &b[iqr], &ldb);
                SLC_DLASET("G", &ncrt_int, &mcrt_int, &ZERO, &ZERO,
                           &a[(*ncont) + (ni) * lda], &lda);
                continue;
            }
        }
        break;
    }

    if (ljobi) {
        sl_int n_int = n;
        sl_int itau_max = max_i32(1, itau);
        sl_int ldwork_int = ldwork;
        SLC_DORGQR(&n_int, &n_int, &itau_max, z, &ldz, tau, dwork, &ldwork_int, &info_tmp);
        wrkopt = max_i32(wrkopt, (i32)dwork[0]);
    }

    if (n >= iqr + 1) {
        sl_int rows = n - iqr;
        sl_int m_int = m;
        SLC_DLASET("G", &rows, &m_int, &ZERO, &ZERO, &b[iqr], &ldb);
    }

    if (ljobf) {
        for (i32 j = itau; j < n; j++) {
            tau[j] = ZERO;
        }
    }

    i32 nbl;
    if (*indcon < n) {
        nbl = *indcon + 1;
        nblk[nbl - 1] = n - *ncont;
    } else {
        nbl = 0;
    }

    mb01pd("U", "H", n, n, 0, 0, anorm, nbl, nblk, a, lda, &info_tmp);
    mb01pd("U", "G", nblk[0], m, 0, 0, bnorm, 0, nblk, b, ldb, &info_tmp);

    dwork[0] = (f64)wrkopt;
}
