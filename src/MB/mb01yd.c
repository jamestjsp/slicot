/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"

void mb01yd(
    const char* uplo,
    const char* trans,
    const i32 n,
    const i32 k,
    const i32 l,
    const f64 alpha,
    const f64 beta,
    const f64* a,
    const i32 lda,
    f64* c,
    const i32 ldc,
    i32* info
)
{
    const f64 zero = 0.0;
    const f64 one = 1.0;
    const i32 inc1 = 1;

    *info = 0;

    bool upper = (*uplo == 'U' || *uplo == 'u');
    bool transp = (*trans == 'T' || *trans == 't' || *trans == 'C' || *trans == 'c');

    i32 nrowa, ncola;
    if (transp) {
        nrowa = k;
        ncola = n;
    } else {
        nrowa = n;
        ncola = k;
    }

    i32 m_bound;
    if (upper) {
        m_bound = nrowa;
    } else {
        m_bound = ncola;
    }

    if (!upper && *uplo != 'L' && *uplo != 'l') {
        *info = -1;
    } else if (!transp && *trans != 'N' && *trans != 'n') {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    } else if (k < 0) {
        *info = -4;
    } else if (l < 0 || l > (m_bound > 1 ? m_bound - 1 : 0)) {
        *info = -5;
    } else if (lda < (nrowa > 1 ? nrowa : 1)) {
        *info = -9;
    } else if (ldc < (n > 1 ? n : 1)) {
        *info = -11;
    }

    if (*info != 0) {
        return;
    }

    if (n == 0 || ((alpha == zero || k == 0) && beta == one)) {
        return;
    }

    if (alpha == zero) {
        if (beta == zero) {
            SLC_DLASET(uplo, &n, &n, &zero, &zero, c, &ldc);
        } else {
            i32 zero_int = 0;
            SLC_DLASCL(uplo, &zero_int, &zero_int, &one, &beta, &n, &n, c, &ldc, info);
        }
        return;
    }

    if (!transp) {
        if (upper) {
            for (i32 j = 0; j < n; j++) {
                if (beta == zero) {
                    for (i32 i = 0; i <= j; i++) {
                        c[i + j * ldc] = zero;
                    }
                } else if (beta != one) {
                    i32 len = j + 1;
                    SLC_DSCAL(&len, &beta, &c[j * ldc], &inc1);
                }

                i32 m_start = (j - l > 0) ? (j - l) : 0;
                for (i32 m_idx = m_start; m_idx < k; m_idx++) {
                    i32 len = (j < l + m_idx) ? (j + 1) : (l + m_idx + 1);
                    f64 scale = alpha * a[j + m_idx * lda];
                    SLC_DAXPY(&len, &scale, &a[m_idx * lda], &inc1, &c[j * ldc], &inc1);
                }
            }
        } else {
            for (i32 j = 0; j < n; j++) {
                if (beta == zero) {
                    for (i32 i = j; i < n; i++) {
                        c[i + j * ldc] = zero;
                    }
                } else if (beta != one) {
                    i32 len = n - j;
                    SLC_DSCAL(&len, &beta, &c[j + j * ldc], &inc1);
                }

                i32 m_end = (j + l < k - 1) ? (j + l + 1) : k;
                for (i32 m_idx = 0; m_idx < m_end; m_idx++) {
                    i32 len = n - j;
                    f64 scale = alpha * a[j + m_idx * lda];
                    SLC_DAXPY(&len, &scale, &a[j + m_idx * lda], &inc1, &c[j + j * ldc], &inc1);
                }
            }
        }
    } else {
        if (upper) {
            for (i32 j = 0; j < n; j++) {
                for (i32 i = 0; i <= j; i++) {
                    i32 len = (j + l < k - 1) ? (j + l + 1) : k;
                    f64 temp = alpha * SLC_DDOT(&len, &a[i * lda], &inc1, &a[j * lda], &inc1);
                    if (beta == zero) {
                        c[i + j * ldc] = temp;
                    } else {
                        c[i + j * ldc] = temp + beta * c[i + j * ldc];
                    }
                }
            }
        } else {
            for (i32 j = 0; j < n; j++) {
                for (i32 i = j; i < n; i++) {
                    i32 m_start = (i - l > 0) ? (i - l) : 0;
                    i32 len = k - m_start;
                    f64 temp = alpha * SLC_DDOT(&len, &a[m_start + i * lda], &inc1,
                                                 &a[m_start + j * lda], &inc1);
                    if (beta == zero) {
                        c[i + j * ldc] = temp;
                    } else {
                        c[i + j * ldc] = temp + beta * c[i + j * ldc];
                    }
                }
            }
        }
    }
}
