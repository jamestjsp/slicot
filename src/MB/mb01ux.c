/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <ctype.h>
#include <stdbool.h>

void mb01ux(const char* side, const char* uplo, const char* trans,
            i32 m, i32 n, f64 alpha, f64* t, i32 ldt, f64* a, i32 lda,
            f64* dwork, i32 ldwork, i32* info)
{
    const f64 zero = 0.0;
    const f64 one = 1.0;
    i32 int1 = 1;

    char side_c = (char)toupper((unsigned char)side[0]);
    char uplo_c = (char)toupper((unsigned char)uplo[0]);
    char trans_c = (char)toupper((unsigned char)trans[0]);

    bool lside = (side_c == 'L');
    bool lup = (uplo_c == 'U');
    bool ltran = (trans_c == 'T' || trans_c == 'C');

    i32 k = lside ? m : n;

    i32 wrkmin = 1;
    if (k > 1) {
        wrkmin = 2 * (k - 1);
    }

    bool lquery = (ldwork == -1);

    *info = 0;
    if (!lside && side_c != 'R') {
        *info = -1;
    } else if (!lup && uplo_c != 'L') {
        *info = -2;
    } else if (!ltran && trans_c != 'N') {
        *info = -3;
    } else if (m < 0) {
        *info = -4;
    } else if (n < 0) {
        *info = -5;
    } else if (ldt < (k > 1 ? k : 1)) {
        *info = -8;
    } else if (lda < (m > 1 ? m : 1)) {
        *info = -10;
    } else {
        i32 mn = (m < n) ? m : n;
        if (alpha != zero && mn > 0) {
            if (lquery) {
                i32 wrkopt;
                if (lside) {
                    wrkopt = (m / 2) * n + m - 1;
                } else {
                    wrkopt = (n / 2) * m + n - 1;
                }
                if (wrkopt < wrkmin) wrkopt = wrkmin;
                dwork[0] = (f64)wrkopt;
                return;
            } else if (ldwork < wrkmin) {
                dwork[0] = (f64)wrkmin;
                *info = -12;
            }
        }
    }

    if (*info != 0) {
        return;
    }

    i32 mn = (m < n) ? m : n;
    if (mn == 0) {
        return;
    }

    if (alpha == zero) {
        SLC_DLASET("Full", &m, &n, &zero, &zero, a, &lda);
        return;
    }

    i32 km1 = k - 1;
    i32 ldtp1 = ldt + 1;
    if (lup) {
        SLC_DCOPY(&km1, &t[1], &ldtp1, dwork, &int1);
    } else {
        SLC_DCOPY(&km1, &t[ldt], &ldtp1, dwork, &int1);
    }

    i32 noff = 0;
    for (i32 i = 0; i < km1; i++) {
        if (dwork[i] != zero) {
            noff++;
        }
    }

    i32 wrkopt;
    if (lside) {
        wrkopt = noff * n + m - 1;
    } else {
        wrkopt = noff * m + n - 1;
    }

    i32 psav = k - 1;
    i32 xdif;
    if (!ltran) {
        xdif = 0;
    } else {
        xdif = 1;
    }
    if (!lup) {
        xdif = 1 - xdif;
    }
    if (!lside) {
        xdif = 1 - xdif;
    }

    if (ldwork >= wrkopt) {
        i32 pdw = psav;
        if (lside) {
            for (i32 j = 0; j < n; j++) {
                for (i32 i = 0; i < m - 1; i++) {
                    if (dwork[i] != zero) {
                        dwork[pdw] = a[(i + xdif) + j * lda];
                        pdw++;
                    }
                }
            }
        } else {
            for (i32 j = 0; j < n - 1; j++) {
                if (dwork[j] != zero) {
                    SLC_DCOPY(&m, &a[(j + xdif) * lda], &int1, &dwork[pdw], &int1);
                    pdw += m;
                }
            }
        }

        SLC_DTRMM(&side_c, &uplo_c, &trans_c, "Non-unit", &m, &n, &alpha, t, &ldt, a, &lda);

        pdw = psav;
        xdif = 1 - xdif;
        if (lside) {
            for (i32 j = 0; j < n; j++) {
                for (i32 i = 0; i < m - 1; i++) {
                    f64 temp = dwork[i];
                    if (temp != zero) {
                        a[(i + xdif) + j * lda] += alpha * temp * dwork[pdw];
                        pdw++;
                    }
                }
            }
        } else {
            for (i32 j = 0; j < n - 1; j++) {
                f64 temp = dwork[j] * alpha;
                if (temp != zero) {
                    SLC_DAXPY(&m, &temp, &dwork[pdw], &int1, &a[(j + xdif) * lda], &int1);
                    pdw += m;
                }
            }
        }
    } else {
        if (lside) {
            for (i32 j = 0; j < n; j++) {
                for (i32 i = 0; i < m - 1; i++) {
                    dwork[psav + i] = dwork[i] * a[(i + xdif) + j * lda];
                }
                SLC_DTRMV(&uplo_c, &trans_c, "Non-unit", &m, t, &ldt, &a[j * lda], &int1);
                i32 mm1 = m - 1;
                SLC_DAXPY(&mm1, &one, &dwork[psav], &int1, &a[(1 - xdif) + j * lda], &int1);
            }
        } else {
            char atran;
            if (ltran) {
                atran = 'N';
            } else {
                atran = 'T';
            }
            for (i32 i = 0; i < m; i++) {
                for (i32 j = 0; j < n - 1; j++) {
                    dwork[psav + j] = a[i + (j + xdif) * lda] * dwork[j];
                }
                SLC_DTRMV(&uplo_c, &atran, "Non-unit", &n, t, &ldt, &a[i], &lda);
                i32 nm1 = n - 1;
                SLC_DAXPY(&nm1, &one, &dwork[psav], &int1, &a[i + (1 - xdif) * lda], &lda);
            }
        }

        if (alpha != one) {
            i32 info_scl;
            i32 zero_i = 0;
            SLC_DLASCL("General", &zero_i, &zero_i, &one, &alpha, &m, &n, a, &lda, &info_scl);
        }
    }

    i32 max_wrk = wrkmin > wrkopt ? wrkmin : wrkopt;
    dwork[0] = (f64)max_wrk;
}
