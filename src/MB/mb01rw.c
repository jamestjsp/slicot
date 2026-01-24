/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <ctype.h>

void mb01rw(const char* uplo, const char* trans, i32 m, i32 n,
            f64* a, i32 lda, const f64* z, i32 ldz,
            f64* dwork, i32* info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    char uplo_c = (char)toupper((unsigned char)uplo[0]);
    char trans_c = (char)toupper((unsigned char)trans[0]);

    bool nottra = (trans_c == 'N');
    bool upper = (uplo_c == 'U');

    *info = 0;
    if (!upper && uplo_c != 'L') {
        *info = -1;
    } else if (!nottra && trans_c != 'T') {
        *info = -2;
    } else if (m < 0) {
        *info = -3;
    } else if (n < 0) {
        *info = -4;
    } else if (lda < 1 || lda < m || lda < n) {
        *info = -6;
    } else if ((nottra && ldz < (m > 1 ? m : 1)) ||
               (!nottra && ldz < (n > 1 ? n : 1))) {
        *info = -8;
    }

    if (*info != 0) {
        return;
    }

    if (n == 0 || m == 0) {
        return;
    }

    if (nottra) {
        if (upper) {
            for (i32 j = 0; j < n; j++) {
                for (i32 i = 0; i < j; i++) {
                    dwork[i] = a[i + j * lda];
                }
                for (i32 i = j; i < n; i++) {
                    dwork[i] = a[j + i * lda];
                }

                char tr = 'N';
                i32 mm = m;
                i32 nn = n;
                f64 al = ONE;
                f64 be = ZERO;
                i32 incx = 1;
                i32 incy = 1;
                SLC_DGEMV(&tr, &mm, &nn, &al, z, &ldz, dwork, &incx, &be, &a[0 + j * lda], &incy);
            }

            for (i32 i = 0; i < m; i++) {
                for (i32 k = 0; k < n; k++) {
                    dwork[k] = a[i + k * lda];
                }

                char tr = 'N';
                i32 mm = m - i;
                i32 nn = n;
                f64 al = ONE;
                f64 be = ZERO;
                i32 incx = 1;
                i32 incy = lda;
                SLC_DGEMV(&tr, &mm, &nn, &al, &z[i + 0 * ldz], &ldz, dwork, &incx, &be, &a[i + i * lda], &incy);
            }
        } else {
            for (i32 i = 0; i < n; i++) {
                for (i32 k = 0; k < i; k++) {
                    dwork[k] = a[i + k * lda];
                }
                for (i32 k = i; k < n; k++) {
                    dwork[k] = a[k + i * lda];
                }

                char tr = 'N';
                i32 mm = m;
                i32 nn = n;
                f64 al = ONE;
                f64 be = ZERO;
                i32 incx = 1;
                i32 incy = lda;
                SLC_DGEMV(&tr, &mm, &nn, &al, z, &ldz, dwork, &incx, &be, &a[i + 0 * lda], &incy);
            }

            for (i32 j = 0; j < m; j++) {
                for (i32 k = 0; k < n; k++) {
                    dwork[k] = a[k + j * lda];
                }

                char tr = 'N';
                i32 mm = m - j;
                i32 nn = n;
                f64 al = ONE;
                f64 be = ZERO;
                i32 incx = 1;
                i32 incy = 1;
                SLC_DGEMV(&tr, &mm, &nn, &al, &z[j + 0 * ldz], &ldz, dwork, &incx, &be, &a[j + j * lda], &incy);
            }
        }
    } else {
        if (upper) {
            for (i32 j = 0; j < n; j++) {
                for (i32 i = 0; i < j; i++) {
                    dwork[i] = a[i + j * lda];
                }
                for (i32 i = j; i < n; i++) {
                    dwork[i] = a[j + i * lda];
                }

                char tr = 'T';
                i32 mm = n;
                i32 nn = m;
                f64 al = ONE;
                f64 be = ZERO;
                i32 incx = 1;
                i32 incy = 1;
                SLC_DGEMV(&tr, &mm, &nn, &al, z, &ldz, dwork, &incx, &be, &a[0 + j * lda], &incy);
            }

            for (i32 i = 0; i < m; i++) {
                for (i32 k = 0; k < n; k++) {
                    dwork[k] = a[i + k * lda];
                }

                char tr = 'T';
                i32 mm = n;
                i32 nn = m - i;
                f64 al = ONE;
                f64 be = ZERO;
                i32 incx = 1;
                i32 incy = lda;
                SLC_DGEMV(&tr, &mm, &nn, &al, &z[0 + i * ldz], &ldz, dwork, &incx, &be, &a[i + i * lda], &incy);
            }
        } else {
            for (i32 i = 0; i < n; i++) {
                for (i32 k = 0; k < i; k++) {
                    dwork[k] = a[i + k * lda];
                }
                for (i32 k = i; k < n; k++) {
                    dwork[k] = a[k + i * lda];
                }

                char tr = 'T';
                i32 mm = n;
                i32 nn = m;
                f64 al = ONE;
                f64 be = ZERO;
                i32 incx = 1;
                i32 incy = lda;
                SLC_DGEMV(&tr, &mm, &nn, &al, z, &ldz, dwork, &incx, &be, &a[i + 0 * lda], &incy);
            }

            for (i32 j = 0; j < m; j++) {
                for (i32 k = 0; k < n; k++) {
                    dwork[k] = a[k + j * lda];
                }

                char tr = 'T';
                i32 mm = n;
                i32 nn = m - j;
                f64 al = ONE;
                f64 be = ZERO;
                i32 incx = 1;
                i32 incy = 1;
                SLC_DGEMV(&tr, &mm, &nn, &al, &z[0 + j * ldz], &ldz, dwork, &incx, &be, &a[j + j * lda], &incy);
            }
        }
    }
}
