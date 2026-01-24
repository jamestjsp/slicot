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

i32 slicot_tb05ad(char baleig, char inita, i32 n, i32 m, i32 p, c128 freq,
                  f64* a, i32 lda, f64* b, i32 ldb, f64* c, i32 ldc,
                  f64* rcond, c128* g, i32 ldg, f64* evre, f64* evim,
                  c128* hinvb, i32 ldhinv, f64* dwork, i32 ldwork,
                  c128* zwork, i32 lzwork) {
    i32 info = 0;
    char baleig_up = (char)toupper((unsigned char)baleig);
    char inita_up = (char)toupper((unsigned char)inita);

    bool lbalec = (baleig_up == 'C');
    bool lbaleb = (baleig_up == 'B' || baleig_up == 'E');
    bool lbalea = (baleig_up == 'A');
    bool lbalba = lbaleb || lbalea;
    bool linita = (inita_up == 'G');

    i32 max1n = n > 1 ? n : 1;

    if (!lbalec && !lbalba && baleig_up != 'N') {
        info = -1;
    } else if (!linita && inita_up != 'H') {
        info = -2;
    } else if (n < 0) {
        info = -3;
    } else if (m < 0) {
        info = -4;
    } else if (p < 0) {
        info = -5;
    } else if (lda < max1n) {
        info = -8;
    } else if (ldb < max1n) {
        info = -10;
    } else if (ldc < (p > 1 ? p : 1)) {
        info = -12;
    } else if (ldg < (p > 1 ? p : 1)) {
        info = -15;
    } else if (ldhinv < max1n) {
        info = -19;
    } else {
        i32 max_nmp = n > m ? n : m;
        max_nmp = max_nmp > p ? max_nmp : p;
        i32 max_nm1p1 = n > (m - 1) ? n : (m - 1);
        max_nm1p1 = max_nm1p1 > (p - 1) ? max_nm1p1 : (p - 1);

        bool ldwork_ok = (ldwork >= 1);
        if (linita && !lbalec && !lbalea) {
            ldwork_ok = (ldwork >= n - 1 + max_nmp) || (ldwork >= 1 && n == 0);
        } else if (linita && (lbalec || lbalea)) {
            ldwork_ok = (ldwork >= n + max_nm1p1) || (ldwork >= 1 && n == 0);
        } else if (!linita && (lbalec || lbalea)) {
            ldwork_ok = (ldwork >= 2 * n) || (ldwork >= 1 && n == 0);
        }

        if (!ldwork_ok) {
            info = -22;
        } else {
            i32 zwork_needed = (lbalec || lbalea) ? n * (n + 2) : (n > 0 ? n * n : 1);
            if (lzwork < zwork_needed) {
                info = -24;
            }
        }
    }

    if (info != 0) {
        i32 xinfo = -info;
        SLC_XERBLA("TB05AD", &xinfo);
        return info;
    }

    if (n == 0) {
        i32 minmp = m < p ? m : p;
        if (minmp > 0) {
            c128 czero = 0.0;
            SLC_ZLASET("Full", &p, &m, &czero, &czero, g, &ldg);
        }
        *rcond = 1.0;
        dwork[0] = 1.0;
        return 0;
    }

    i32 wrkopt = 1;
    i32 low = 0, igh = 0;

    if (linita) {
        char balanc = 'N';
        if (lbalba) {
            balanc = 'B';
        }

        SLC_DGEBAL(&balanc, &n, a, &lda, &low, &igh, dwork, &info);
        low--;  // Convert to 0-based
        igh--;

        if (lbalba) {
            for (i32 j = 0; j < n; j++) {
                i32 jj = j;
                if (jj < low || jj > igh) {
                    if (jj < low) {
                        jj = low - jj - 1;
                    }
                    if (jj < 0 || jj >= n) continue;  // Bounds check
                    i32 jp = (i32)dwork[jj] - 1;  // Convert to 0-based
                    if (jp < 0 || jp >= n) continue;  // Bounds check
                    if (jp != jj) {
                        if (m > 0) {
                            SLC_DSWAP(&m, &b[jj], &ldb, &b[jp], &ldb);
                        }
                        if (p > 0) {
                            i32 inc1 = 1;
                            SLC_DSWAP(&p, &c[jj * ldc], &inc1, &c[jp * ldc], &inc1);
                        }
                    }
                }
            }

            if (igh != low) {
                for (i32 j = low; j <= igh; j++) {
                    f64 t = dwork[j];
                    if (m > 0) {
                        f64 invt = 1.0 / t;
                        SLC_DSCAL(&m, &invt, &b[j], &ldb);
                    }
                    if (p > 0) {
                        i32 inc1 = 1;
                        SLC_DSCAL(&p, &t, &c[j * ldc], &inc1);
                    }
                }
            }
        }

        i32 itau = 0;
        i32 jwork = itau + n - 1;
        i32 low1 = low + 1;  // LAPACK uses 1-based
        i32 igh1 = igh + 1;
        i32 lwork_remain = ldwork - jwork;
        if (lwork_remain < 1) lwork_remain = 1;

        SLC_DGEHRD(&n, &low1, &igh1, a, &lda, &dwork[itau], &dwork[jwork],
                   &lwork_remain, &info);
        wrkopt = wrkopt > (i32)dwork[jwork] + jwork ? wrkopt : (i32)dwork[jwork] + jwork;

        SLC_DORMHR("Left", "Transpose", &n, &m, &low1, &igh1, a, &lda,
                   &dwork[itau], b, &ldb, &dwork[jwork], &lwork_remain, &info);
        wrkopt = wrkopt > (i32)dwork[jwork] + jwork ? wrkopt : (i32)dwork[jwork] + jwork;

        SLC_DORMHR("Right", "No transpose", &p, &n, &low1, &igh1, a, &lda,
                   &dwork[itau], c, &ldc, &dwork[jwork], &lwork_remain, &info);
        wrkopt = wrkopt > (i32)dwork[jwork] + jwork ? wrkopt : (i32)dwork[jwork] + jwork;

        if (lbalba) {
            i32 ij = 0;
            for (i32 j = 0; j < n; j++) {
                for (i32 i = 0; i < n; i++) {
                    zwork[ij] = a[i + j * lda];
                    ij++;
                }
            }

            i32 one = 1;
            SLC_DHSEQR("Eigenvalues", "No Schur", &n, &low1, &igh1, a, &lda,
                       evre, evim, dwork, &one, dwork, &ldwork, &info);

            ij = 0;
            for (i32 j = 0; j < n; j++) {
                for (i32 i = 0; i < n; i++) {
                    a[i + j * lda] = creal(zwork[ij]);
                    ij++;
                }
            }

            if (info > 0) {
                info = 1;
            }
        }
    }

    i32 ij = 0;
    i32 jj = 0;
    for (i32 j = 0; j < n; j++) {
        for (i32 i = 0; i < n; i++) {
            zwork[ij] = -a[i + j * lda];
            ij++;
        }
        zwork[jj] = freq + zwork[jj];
        jj += n + 1;
    }

    f64 hnorm = 0.0;
    if (lbalec || lbalea) {
        jj = 0;
        for (i32 j = 0; j < n; j++) {
            f64 t = fabs(creal(zwork[jj])) + fabs(cimag(zwork[jj]));
            i32 inc1 = 1;
            t += SLC_DASUM(&j, &a[j * lda], &inc1);
            if (j < n - 1) {
                t += fabs(a[j + 1 + j * lda]);
            }
            hnorm = hnorm > t ? hnorm : t;
            jj += n + 1;
        }
    }

    i32* iwork = (i32*)malloc((n > 0 ? n : 1) * sizeof(i32));
    if (!iwork) {
        return -100;  // Memory allocation failure
    }

    i32 mb02sz_info = slicot_mb02sz(n, zwork, n, iwork);
    if (mb02sz_info != 0) {
        info = 2;
    }

    if (lbalec || lbalea) {
        f64 rcond_est;
        f64* cnorm = (f64*)malloc((n > 0 ? n : 1) * sizeof(f64));
        if (!cnorm) {
            free(iwork);
            return -100;
        }
        slicot_mb02tz('1', n, hnorm, zwork, n, iwork, &rcond_est, cnorm, &zwork[n * n]);
        free(cnorm);
        *rcond = rcond_est;
        wrkopt = wrkopt > 2 * n ? wrkopt : 2 * n;

        f64 eps = SLC_DLAMCH("Epsilon");
        if (rcond_est < eps) {
            info = 2;
        }
    }

    if (info != 0 && info != 1) {
        free(iwork);
        return info;
    }

    for (i32 j = 0; j < m; j++) {
        for (i32 i = 0; i < n; i++) {
            hinvb[i + j * ldhinv] = b[i + j * ldb];
        }
    }

    slicot_mb02rz('N', n, m, zwork, n, iwork, hinvb, ldhinv);
    free(iwork);

    for (i32 j = 0; j < m; j++) {
        for (i32 i = 0; i < p; i++) {
            g[i + j * ldg] = 0.0;
        }

        for (i32 k = 0; k < n; k++) {
            for (i32 i = 0; i < p; i++) {
                g[i + j * ldg] = g[i + j * ldg] + c[i + k * ldc] * hinvb[k + j * ldhinv];
            }
        }
    }

    dwork[0] = (f64)wrkopt;

    return info;
}
