/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <ctype.h>

static inline i32 max_i32(i32 a, i32 b) {
    return a > b ? a : b;
}

static inline i32 min_i32(i32 a, i32 b) {
    return a < b ? a : b;
}

void ab01od(const char* stages, const char* jobu, const char* jobv,
            i32 n, i32 m, f64* a, i32 lda, f64* b, i32 ldb,
            f64* u, i32 ldu, f64* v, i32 ldv,
            i32* ncont, i32* indcon, i32* kstair, f64 tol,
            i32* iwork, f64* dwork, i32 ldwork, i32* info) {

    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    char stages_upper = (char)toupper((unsigned char)stages[0]);
    char jobu_upper = (char)toupper((unsigned char)jobu[0]);
    char jobv_upper = (char)toupper((unsigned char)jobv[0]);

    bool lstagb = (stages_upper == 'B');
    bool lstgab = (stages_upper == 'A') || lstagb;
    bool ljobui = (jobu_upper == 'I');
    bool ljobvi = false;

    *info = 0;

    if (lstgab) {
        ljobvi = (jobv_upper == 'I');
    }

    if (!lstgab && stages_upper != 'F') {
        *info = -1;
    } else if (!ljobui && jobu_upper != 'N') {
        *info = -2;
    } else if (n < 0) {
        *info = -4;
    } else if (m < 0) {
        *info = -5;
    } else if (lda < max_i32(1, n)) {
        *info = -7;
    } else if (ldb < max_i32(1, n)) {
        *info = -9;
    } else if (ldu < 1 || (ljobui && ldu < n)) {
        *info = -11;
    } else if (!lstagb && ldwork < max_i32(1, n + max_i32(n, 3 * m))) {
        *info = -20;
    } else if (lstagb && ldwork < max_i32(1, m + max_i32(n, m))) {
        *info = -20;
    } else if (lstagb && *ncont > n) {
        *info = -14;
    } else if (lstagb && *indcon > n) {
        *info = -15;
    } else if (lstgab) {
        if (!ljobvi && jobv_upper != 'N') {
            *info = -3;
        } else if (ldv < 1 || (ljobvi && ldv < m)) {
            *info = -13;
        }
    }

    if (*info != 0) {
        return;
    }

    if (min_i32(n, m) == 0) {
        *ncont = 0;
        *indcon = 0;
        if (n > 0 && ljobui) {
            SLC_DLASET("F", &n, &n, &ZERO, &ONE, u, &ldu);
        }
        if (lstgab) {
            if (m > 0 && ljobvi) {
                SLC_DLASET("F", &m, &m, &ZERO, &ONE, v, &ldv);
            }
        }
        dwork[0] = ONE;
        return;
    }

    i32 itau = 0;
    i32 wrkopt = 1;

    if (!lstagb) {
        i32 jwork = n;
        i32 ldwork_remaining = ldwork - jwork;
        i32 info_tmp = 0;

        ab01nd(jobu, n, m, a, lda, b, ldb, ncont, indcon,
               kstair, u, ldu, &dwork[itau], tol, iwork,
               &dwork[jwork], ldwork_remaining, &info_tmp);

        wrkopt = (i32)dwork[jwork] + jwork;
    }

    if (!lstgab) {
        dwork[0] = (f64)wrkopt;
        return;
    }

    if (*ncont == 0 || *indcon == 0) {
        if (ljobvi) {
            SLC_DLASET("F", &m, &m, &ZERO, &ONE, v, &ldv);
        }
        dwork[0] = (f64)wrkopt;
        return;
    }

    i32 mcrt = kstair[*indcon - 1];
    i32 i0 = *ncont - mcrt;
    i32 jwork = m;
    i32 info_tmp = 0;

    for (i32 ibstep = *indcon - 1; ibstep >= 1; ibstep--) {
        i32 ncrt = kstair[ibstep - 1];
        i32 j0 = i0 - ncrt;
        i32 mm = min_i32(ncrt, mcrt);

        i32 ldwork_remaining = ldwork - jwork;
        SLC_DGERQF(&mcrt, &ncrt, &a[i0 + j0 * lda], &lda, &dwork[itau],
                   &dwork[jwork], &ldwork_remaining, &info_tmp);
        wrkopt = max_i32(wrkopt, (i32)dwork[jwork] + jwork);

        i32 jini;
        if (ibstep > 1) {
            jini = j0 - kstair[ibstep - 2];
        } else {
            jini = 0;

            ldwork_remaining = ldwork - jwork;
            SLC_DORMRQ("Left", "No transpose", &ncrt, &m, &mm,
                       &a[i0 + j0 * lda], &lda, &dwork[itau],
                       b, &ldb, &dwork[jwork], &ldwork_remaining, &info_tmp);
            wrkopt = max_i32(wrkopt, (i32)dwork[jwork] + jwork);
        }

        i32 cols = n - jini;
        ldwork_remaining = ldwork - jwork;
        SLC_DORMRQ("Left", "No transpose", &ncrt, &cols, &mm,
                   &a[i0 + j0 * lda], &lda, &dwork[itau],
                   &a[j0 + jini * lda], &lda, &dwork[jwork], &ldwork_remaining, &info_tmp);
        wrkopt = max_i32(wrkopt, (i32)dwork[jwork] + jwork);

        ldwork_remaining = ldwork - jwork;
        SLC_DORMRQ("Right", "Transpose", &i0, &ncrt, &mm,
                   &a[i0 + j0 * lda], &lda, &dwork[itau],
                   &a[j0 * lda], &lda, &dwork[jwork], &ldwork_remaining, &info_tmp);
        wrkopt = max_i32(wrkopt, (i32)dwork[jwork] + jwork);

        if (ljobui) {
            ldwork_remaining = ldwork - jwork;
            SLC_DORMRQ("Right", "Transpose", &n, &ncrt, &mm,
                       &a[i0 + j0 * lda], &lda, &dwork[itau],
                       &u[j0 * ldu], &ldu, &dwork[jwork], &ldwork_remaining, &info_tmp);
            wrkopt = max_i32(wrkopt, (i32)dwork[jwork] + jwork);
        }

        i32 cols_zero = ncrt - mcrt;
        if (cols_zero > 0) {
            SLC_DLASET("F", &mcrt, &cols_zero, &ZERO, &ZERO, &a[i0 + j0 * lda], &lda);
        }
        if (i0 < n - 1) {
            i32 rows_zero = mcrt - 1;
            if (rows_zero > 0) {
                i32 col_start = i0 - mcrt;
                SLC_DLASET("L", &rows_zero, &rows_zero, &ZERO, &ZERO,
                           &a[(i0 + 1) + col_start * lda], &lda);
            }
        }

        mcrt = ncrt;
        i0 = j0;
    }

    i32 ldwork_remaining = ldwork - jwork;
    SLC_DGERQF(&mcrt, &m, b, &ldb, &dwork[itau],
               &dwork[jwork], &ldwork_remaining, &info_tmp);
    wrkopt = max_i32(wrkopt, (i32)dwork[jwork] + jwork);

    if (ljobvi) {
        i32 cols_copy = m - mcrt;
        if (cols_copy > 0) {
            SLC_DLACPY("F", &mcrt, &cols_copy, b, &ldb, &v[(m - mcrt) + 0 * ldv], &ldv);
        }
        if (mcrt > 1) {
            i32 rows_copy = mcrt - 1;
            SLC_DLACPY("L", &rows_copy, &rows_copy, &b[1 + (m - mcrt) * ldb], &ldb,
                       &v[(m - mcrt + 1) + (m - mcrt) * ldv], &ldv);
        }

        ldwork_remaining = ldwork - jwork;
        SLC_DORGRQ(&m, &m, &mcrt, v, &ldv, &dwork[itau],
                   &dwork[jwork], &ldwork_remaining, &info_tmp);

        i32 one = 1;
        for (i32 i = 1; i < m; i++) {
            SLC_DSWAP(&i, &v[i], &ldv, &v[i * ldv], &one);
        }
        wrkopt = max_i32(wrkopt, (i32)dwork[jwork] + jwork);
    }

    i32 cols_zero = m - mcrt;
    if (cols_zero > 0) {
        SLC_DLASET("F", &mcrt, &cols_zero, &ZERO, &ZERO, b, &ldb);
    }
    if (mcrt > 1) {
        i32 rows_zero = mcrt - 1;
        SLC_DLASET("L", &rows_zero, &rows_zero, &ZERO, &ZERO,
                   &b[1 + (m - mcrt) * ldb], &ldb);
    }

    dwork[0] = (f64)wrkopt;
}
