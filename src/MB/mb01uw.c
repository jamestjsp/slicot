/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <ctype.h>
#include <stdbool.h>

void mb01uw(const char* side, const char* trans, i32 m, i32 n,
            f64 alpha, f64* h, i32 ldh, f64* a, i32 lda,
            f64* dwork, i32 ldwork, i32* info)
{
    const f64 zero = 0.0;
    const f64 one = 1.0;
    i32 int1 = 1;

    char side_c = (char)toupper((unsigned char)side[0]);
    char trans_c = (char)toupper((unsigned char)trans[0]);

    bool lside = (side_c == 'L');
    bool ltrans = (trans_c == 'T' || trans_c == 'C');

    *info = 0;

    if (!lside && side_c != 'R') {
        *info = -1;
    } else if (!ltrans && trans_c != 'N') {
        *info = -2;
    } else if (m < 0) {
        *info = -3;
    } else if (n < 0) {
        *info = -4;
    } else if (ldh < 1 || (lside && ldh < m) || (!lside && ldh < n)) {
        *info = -7;
    } else if (lda < (m > 1 ? m : 1)) {
        *info = -9;
    } else {
        i32 mn = (m < n) ? m : n;
        i32 wrkmin = 0;
        if (alpha != zero && mn > 0) {
            wrkmin = lside ? (m - 1) : (n - 1);
        }
        if (ldwork < wrkmin) {
            *info = -11;
        }
    }

    if (*info != 0) {
        return;
    }

    i32 mn = (m < n) ? m : n;
    if (mn == 0) {
        return;
    }

    if (lside) {
        if (m == 1) {
            f64 scale = alpha * h[0];
            SLC_DSCAL(&n, &scale, a, &lda);
            return;
        }
    } else {
        if (n == 1) {
            f64 scale = alpha * h[0];
            SLC_DSCAL(&m, &scale, a, &int1);
            return;
        }
    }

    if (alpha == zero) {
        SLC_DLASET("Full", &m, &n, &zero, &zero, a, &lda);
        return;
    }

    if (ldwork >= m * n) {
        SLC_DLACPY("Full", &m, &n, a, &lda, dwork, &m);
        SLC_DTRMM(&side_c, "Upper", &trans_c, "Non-unit", &m, &n, &alpha, h, &ldh, a, &lda);

        if (lside) {
            if (m > 2) {
                i32 len = m - 2;
                i32 ldh1 = ldh + 1;
                SLC_DSWAP(&len, &h[2 + ldh], &ldh1, &h[2], &int1);
            }

            if (ltrans) {
                i32 jw = 1;
                for (i32 j = 0; j < n; j++) {
                    jw++;
                    for (i32 i = 0; i < m - 1; i++) {
                        a[i + j * lda] += alpha * h[i + 1] * dwork[jw - 1];
                        jw++;
                    }
                }
            } else {
                i32 jw = 0;
                for (i32 j = 0; j < n; j++) {
                    jw++;
                    for (i32 i = 1; i < m; i++) {
                        a[i + j * lda] += alpha * h[i] * dwork[jw - 1];
                        jw++;
                    }
                }
            }

            if (m > 2) {
                i32 len = m - 2;
                i32 ldh1 = ldh + 1;
                SLC_DSWAP(&len, &h[2 + ldh], &ldh1, &h[2], &int1);
            }
        } else {
            if (ltrans) {
                i32 jw = 1;
                for (i32 j = 0; j < n - 1; j++) {
                    f64 hval = h[j + 1 + j * ldh];
                    if (hval != zero) {
                        f64 scale = alpha * hval;
                        SLC_DAXPY(&m, &scale, &dwork[jw - 1], &int1, &a[(j + 1) * lda], &int1);
                    }
                    jw += m;
                }
            } else {
                i32 jw = m + 1;
                for (i32 j = 0; j < n - 1; j++) {
                    f64 hval = h[j + 1 + j * ldh];
                    if (hval != zero) {
                        f64 scale = alpha * hval;
                        SLC_DAXPY(&m, &scale, &dwork[jw - 1], &int1, &a[j * lda], &int1);
                    }
                    jw += m;
                }
            }
        }
    } else {
        if (lside) {
            if (m > 2) {
                i32 len = m - 2;
                i32 ldh1 = ldh + 1;
                SLC_DSWAP(&len, &h[2 + ldh], &ldh1, &h[2], &int1);
            }

            if (ltrans) {
                for (i32 j = 0; j < n; j++) {
                    for (i32 i = 0; i < m - 1; i++) {
                        dwork[i] = h[i + 1] * a[i + 1 + j * lda];
                    }
                    SLC_DTRMV("Upper", &trans_c, "Non-unit", &m, h, &ldh, &a[j * lda], &int1);
                    i32 mm1 = m - 1;
                    SLC_DAXPY(&mm1, &one, dwork, &int1, &a[j * lda], &int1);
                }
            } else {
                for (i32 j = 0; j < n; j++) {
                    for (i32 i = 0; i < m - 1; i++) {
                        dwork[i] = h[i + 1] * a[i + j * lda];
                    }
                    SLC_DTRMV("Upper", &trans_c, "Non-unit", &m, h, &ldh, &a[j * lda], &int1);
                    i32 mm1 = m - 1;
                    SLC_DAXPY(&mm1, &one, dwork, &int1, &a[1 + j * lda], &int1);
                }
            }

            if (m > 2) {
                i32 len = m - 2;
                i32 ldh1 = ldh + 1;
                SLC_DSWAP(&len, &h[2 + ldh], &ldh1, &h[2], &int1);
            }
        } else {
            if (n > 2) {
                i32 len = n - 2;
                i32 ldh1 = ldh + 1;
                SLC_DSWAP(&len, &h[2 + ldh], &ldh1, &h[2], &int1);
            }

            if (ltrans) {
                for (i32 i = 0; i < m; i++) {
                    for (i32 j = 0; j < n - 1; j++) {
                        dwork[j] = a[i + j * lda] * h[j + 1];
                    }
                    SLC_DTRMV("Upper", "NoTranspose", "Non-unit", &n, h, &ldh, &a[i], &lda);
                    i32 nm1 = n - 1;
                    SLC_DAXPY(&nm1, &one, dwork, &int1, &a[i + lda], &lda);
                }
            } else {
                for (i32 i = 0; i < m; i++) {
                    for (i32 j = 0; j < n - 1; j++) {
                        dwork[j] = a[i + (j + 1) * lda] * h[j + 1];
                    }
                    SLC_DTRMV("Upper", "Transpose", "Non-unit", &n, h, &ldh, &a[i], &lda);
                    i32 nm1 = n - 1;
                    SLC_DAXPY(&nm1, &one, dwork, &int1, &a[i], &lda);
                }
            }

            if (n > 2) {
                i32 len = n - 2;
                i32 ldh1 = ldh + 1;
                SLC_DSWAP(&len, &h[2 + ldh], &ldh1, &h[2], &int1);
            }
        }

        if (alpha != one) {
            i32 info_scl;
            SLC_DLASCL("General", &int1, &int1, &one, &alpha, &m, &n, a, &lda, &info_scl);
        }
    }
}
