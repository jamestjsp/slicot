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

static i32 imin(i32 a, i32 b) {
    return a < b ? a : b;
}

void tb01ud(const char* jobz, i32 n, i32 m, i32 p,
            f64* a, i32 lda, f64* b, i32 ldb, f64* c, i32 ldc,
            i32* ncont, i32* indcon, i32* nblk,
            f64* z, i32 ldz, f64* tau, f64 tol,
            i32* iwork, f64* dwork, i32 ldwork, i32* info) {

    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    bool ljobf, ljobi, ljobz;
    i32 iqr, itau, j, mcrt, nbl, ncrt, ni, nj, rank, wrkopt;
    f64 anorm, bnorm, fnrm, toldef;
    f64 sval[3];

    const i32 one_i = 1;

    *info = 0;
    ljobf = lsame(*jobz, 'F');
    ljobi = lsame(*jobz, 'I');
    ljobz = ljobf || ljobi;

    if (!ljobz && !lsame(*jobz, 'N')) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (m < 0) {
        *info = -3;
    } else if (p < 0) {
        *info = -4;
    } else if (lda < imax(1, n)) {
        *info = -6;
    } else if (ldb < imax(1, n)) {
        *info = -8;
    } else if (ldc < imax(1, p)) {
        *info = -10;
    } else if ((!ljobz && ldz < 1) || (ljobz && ldz < imax(1, n))) {
        *info = -15;
    } else if (ldwork < imax(1, imax(n, imax(3 * m, p)))) {
        *info = -20;
    }

    if (*info != 0) {
        return;
    }

    *ncont = 0;
    *indcon = 0;

    anorm = SLC_DLANGE("M", &n, &n, a, &lda, dwork);
    bnorm = SLC_DLANGE("M", &n, &m, b, &ldb, dwork);

    if (imin(n, m) == 0 || bnorm == ZERO) {
        if (n > 0) {
            if (ljobi) {
                SLC_DLASET("Full", &n, &n, &ZERO, &ONE, z, &ldz);
            } else if (ljobf) {
                SLC_DLASET("Full", &n, &n, &ZERO, &ZERO, z, &ldz);
                SLC_DLASET("Full", &n, &one_i, &ZERO, &ZERO, tau, &n);
            }
        }
        dwork[0] = ONE;
        return;
    }

    mb01pd("S", "G", n, n, 0, 0, anorm, 0, nblk, a, lda, info);
    mb01pd("S", "G", n, m, 0, 0, bnorm, 0, nblk, b, ldb, info);

    fnrm = SLC_DLANGE("F", &n, &m, b, &ldb, dwork);

    toldef = tol;
    if (toldef <= ZERO) {
        toldef = (f64)(n * n) * SLC_DLAMCH("Precision");
    }

    if (fnrm < toldef) {
        fnrm = ONE;
    }

    wrkopt = 1;
    ni = 0;
    itau = 0;
    ncrt = n;
    mcrt = m;
    iqr = 0;

    while (1) {
        mb03oy(ncrt, mcrt, &b[iqr], ldb, toldef, fnrm, &rank, sval, iwork, &tau[itau], dwork, info);

        if (rank != 0) {
            nj = ni;
            ni = *ncont;
            *ncont = *ncont + rank;
            (*indcon)++;
            nblk[*indcon - 1] = rank;

            i32 ncrt_loc = ncrt;
            i32 rank_loc = rank;
            SLC_DORMQR("Left", "Transpose", &ncrt_loc, &ncrt_loc, &rank_loc,
                       &b[iqr], &ldb, &tau[itau], &a[ni + ni * lda], &lda,
                       dwork, &ldwork, info);
            wrkopt = imax(wrkopt, (i32)dwork[0]);

            i32 n_loc = n;
            SLC_DORMQR("Right", "No transpose", &n_loc, &ncrt_loc, &rank_loc,
                       &b[iqr], &ldb, &tau[itau], &a[ni * lda], &lda,
                       dwork, &ldwork, info);
            wrkopt = imax(wrkopt, (i32)dwork[0]);

            i32 p_loc = p;
            SLC_DORMQR("Right", "No transpose", &p_loc, &ncrt_loc, &rank_loc,
                       &b[iqr], &ldb, &tau[itau], &c[ni * ldc], &ldc,
                       dwork, &ldwork, info);
            wrkopt = imax(wrkopt, (i32)dwork[0]);

            if (ljobz && ncrt > 1) {
                i32 ncrt_m1 = ncrt - 1;
                i32 min_rank_ncrt_m1 = imin(rank, ncrt - 1);
                SLC_DLACPY("L", &ncrt_m1, &min_rank_ncrt_m1,
                           &b[iqr + 1], &ldb, &z[(ni + 1) + itau * ldz], &ldz);
            }

            if (rank > 1) {
                i32 rank_m1 = rank - 1;
                SLC_DLASET("L", &rank_m1, &rank_m1, &ZERO, &ZERO, &b[iqr + 1], &ldb);
            }

            if (*indcon == 1) {
                i32 forwrd = 0;
                SLC_DLAPMT(&forwrd, &rank, &m, &b[iqr], &ldb, iwork);
                iqr = rank;
                fnrm = SLC_DLANGE("F", &n, &n, a, &lda, dwork);
            } else {
                for (j = 0; j < mcrt; j++) {
                    i32 ipiv_j = iwork[j] - 1;
                    SLC_DCOPY(&rank, &b[iqr + j * ldb], &one_i, &a[ni + (nj + ipiv_j) * lda], &one_i);
                }
            }

            itau = itau + rank;
            if (rank != ncrt) {
                mcrt = rank;
                ncrt = ncrt - rank;
                i32 mcrt_loc = mcrt;
                SLC_DLACPY("G", &ncrt, &mcrt_loc, &a[*ncont + ni * lda], &lda, &b[iqr], &ldb);
                SLC_DLASET("G", &ncrt, &mcrt_loc, &ZERO, &ZERO, &a[*ncont + ni * lda], &lda);
                continue;
            }
        }
        break;
    }

    if (ljobi) {
        i32 itau_m1 = itau;
        SLC_DORGQR(&n, &n, &itau_m1, z, &ldz, tau, dwork, &ldwork, info);
        wrkopt = imax(wrkopt, (i32)dwork[0]);
    }

    if (iqr < n) {
        i32 n_minus_iqr = n - iqr;
        SLC_DLASET("G", &n_minus_iqr, &m, &ZERO, &ZERO, &b[iqr], &ldb);
    }

    if (ljobf) {
        for (j = itau; j < n; j++) {
            tau[j] = ZERO;
        }
    }

    if (*indcon < n) {
        nbl = *indcon + 1;
        nblk[nbl - 1] = n - *ncont;
    } else {
        nbl = 0;
    }
    mb01pd("U", "H", n, n, 0, 0, anorm, nbl, nblk, a, lda, info);

    i32 nblk0 = nblk[0];
    mb01pd("U", "G", nblk0, m, 0, 0, bnorm, 0, nblk, b, ldb, info);

    dwork[0] = (f64)wrkopt;
}
