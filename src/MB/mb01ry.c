// SPDX-License-Identifier: BSD-3-Clause

#include "slicot.h"
#include "slicot_blas.h"
#include <ctype.h>
#include <stdbool.h>

void mb01ry(const char* side, const char* uplo, const char* trans, i32 m,
            f64 alpha, f64 beta, f64* r, i32 ldr, f64* h, i32 ldh,
            const f64* b, i32 ldb, f64* dwork, i32* info)
{
    const f64 zero = 0.0;
    const f64 one = 1.0;
    i32 int1 = 1;
    i32 int0 = 0;

    char side_c = (char)toupper((unsigned char)side[0]);
    char uplo_c = (char)toupper((unsigned char)uplo[0]);
    char trans_c = (char)toupper((unsigned char)trans[0]);

    bool lside = (side_c == 'L');
    bool luplo = (uplo_c == 'U');
    bool ltrans = (trans_c == 'T' || trans_c == 'C');

    *info = 0;

    if (!lside && side_c != 'R') {
        *info = -1;
    } else if (!luplo && uplo_c != 'L') {
        *info = -2;
    } else if (!ltrans && trans_c != 'N') {
        *info = -3;
    } else if (m < 0) {
        *info = -4;
    } else if (ldr < (m > 1 ? m : 1)) {
        *info = -8;
    } else if (ldh < (m > 1 ? m : 1)) {
        *info = -10;
    } else if (ldb < (m > 1 ? m : 1)) {
        *info = -12;
    }

    if (*info != 0) {
        return;
    }

    if (m == 0) {
        return;
    }

    if (beta == zero) {
        if (alpha == zero) {
            SLC_DLASET(&uplo_c, &m, &m, &zero, &zero, r, &ldr);
        } else if (alpha != one) {
            SLC_DLASCL(&uplo_c, &int0, &int0, &one, &alpha, &m, &m, r, &ldr, info);
        }
        return;
    }

    if (lside) {
        if (m > 2) {
            i32 len = m - 2;
            i32 ldh1 = ldh + 1;
            SLC_DSWAP(&len, &h[2 + ldh], &ldh1, &h[2], &int1);
        }

        if (luplo) {
            if (ltrans) {
                for (i32 j = 0; j < m; j++) {
                    i32 len = j + 1;
                    SLC_DCOPY(&len, &b[j * ldb], &int1, dwork, &int1);
                    SLC_DTRMV("Upper", "Transpose", "Non-unit", &len, h, &ldh, dwork, &int1);

                    i32 lim = (j < m - 1) ? j + 1 : m - 1;
                    for (i32 i = 0; i < lim; i++) {
                        r[i + j * ldr] = alpha * r[i + j * ldr] +
                                         beta * (dwork[i] + h[i + 1] * b[i + 1 + j * ldb]);
                    }
                }
                r[(m - 1) + (m - 1) * ldr] = alpha * r[(m - 1) + (m - 1) * ldr] +
                                              beta * dwork[m - 1];
            } else {
                for (i32 j = 0; j < m; j++) {
                    i32 len = j + 1;
                    SLC_DCOPY(&len, &b[j * ldb], &int1, dwork, &int1);
                    SLC_DTRMV("Upper", "NoTranspose", "Non-unit", &len, h, &ldh, dwork, &int1);

                    if (j < m - 1) {
                        i32 cols = m - j - 1;
                        SLC_DGEMV("NoTranspose", &len, &cols, &beta, &h[(j + 1) * ldh], &ldh,
                                  &b[j + 1 + j * ldb], &int1, &alpha, &r[j * ldr], &int1);
                    } else {
                        SLC_DSCAL(&m, &alpha, &r[(m - 1) * ldr], &int1);
                    }

                    r[j * ldr] = r[j * ldr] + beta * dwork[0];

                    for (i32 i = 1; i <= j; i++) {
                        r[i + j * ldr] = r[i + j * ldr] +
                                         beta * (dwork[i] + h[i] * b[i - 1 + j * ldb]);
                    }
                }
            }
        } else {
            if (ltrans) {
                for (i32 j = m - 1; j >= 0; j--) {
                    i32 len = m - j;
                    SLC_DCOPY(&len, &b[j + j * ldb], &int1, &dwork[j], &int1);
                    SLC_DTRMV("Upper", "Transpose", "Non-unit", &len, &h[j + j * ldh], &ldh,
                              &dwork[j], &int1);

                    if (j > 0) {
                        i32 cols = j;
                        SLC_DGEMV("Transpose", &cols, &len, &beta, &h[j * ldh], &ldh,
                                  &b[j * ldb], &int1, &alpha, &r[j + j * ldr], &int1);
                    } else {
                        SLC_DSCAL(&m, &alpha, r, &int1);
                    }

                    for (i32 i = j; i < m - 1; i++) {
                        r[i + j * ldr] = r[i + j * ldr] +
                                         beta * (dwork[i] + h[i + 1] * b[i + 1 + j * ldb]);
                    }
                    r[(m - 1) + j * ldr] = r[(m - 1) + j * ldr] + beta * dwork[m - 1];
                }
            } else {
                for (i32 j = m - 1; j >= 0; j--) {
                    i32 len = m - j;
                    SLC_DCOPY(&len, &b[j + j * ldb], &int1, &dwork[j], &int1);
                    SLC_DTRMV("Upper", "NoTranspose", "Non-unit", &len, &h[j + j * ldh], &ldh,
                              &dwork[j], &int1);

                    i32 istart = (j > 1) ? j : 1;
                    for (i32 i = istart; i < m; i++) {
                        r[i + j * ldr] = alpha * r[i + j * ldr] +
                                         beta * (dwork[i] + h[i] * b[i - 1 + j * ldb]);
                    }
                }
                r[0] = alpha * r[0] + beta * dwork[0];
            }
        }

        if (m > 2) {
            i32 len = m - 2;
            i32 ldh1 = ldh + 1;
            SLC_DSWAP(&len, &h[2 + ldh], &ldh1, &h[2], &int1);
        }
    } else {
        if (luplo) {
            if (ltrans) {
                r[0] = alpha * r[0] + beta * SLC_DDOT(&m, b, &ldb, h, &ldh);

                for (i32 j = 1; j < m; j++) {
                    i32 len = j + 1;
                    i32 cols = m - j + 1;
                    SLC_DGEMV("NoTranspose", &len, &cols, &beta,
                              &b[(j - 1) * ldb], &ldb, &h[j + (j - 1) * ldh], &ldh,
                              &alpha, &r[j * ldr], &int1);
                }
            } else {
                for (i32 j = 0; j < m - 1; j++) {
                    i32 len = j + 1;
                    i32 cols = j + 2;
                    SLC_DGEMV("NoTranspose", &len, &cols, &beta, b, &ldb,
                              &h[j * ldh], &int1, &alpha, &r[j * ldr], &int1);
                }
                SLC_DGEMV("NoTranspose", &m, &m, &beta, b, &ldb,
                          &h[(m - 1) * ldh], &int1, &alpha, &r[(m - 1) * ldr], &int1);
            }
        } else {
            if (ltrans) {
                SLC_DGEMV("NoTranspose", &m, &m, &beta, b, &ldb, h, &ldh,
                          &alpha, r, &int1);

                for (i32 j = 1; j < m; j++) {
                    i32 len = m - j;
                    i32 cols = m - j + 1;
                    SLC_DGEMV("NoTranspose", &len, &cols, &beta,
                              &b[j + (j - 1) * ldb], &ldb, &h[j + (j - 1) * ldh], &ldh,
                              &alpha, &r[j + j * ldr], &int1);
                }
            } else {
                for (i32 j = 0; j < m - 1; j++) {
                    i32 len = m - j;
                    i32 cols = j + 2;
                    SLC_DGEMV("NoTranspose", &len, &cols, &beta,
                              &b[j], &ldb, &h[j * ldh], &int1, &alpha, &r[j + j * ldr], &int1);
                }
                r[(m - 1) + (m - 1) * ldr] = alpha * r[(m - 1) + (m - 1) * ldr] +
                    beta * SLC_DDOT(&m, &b[(m - 1)], &ldb, &h[(m - 1) * ldh], &int1);
            }
        }
    }
}
