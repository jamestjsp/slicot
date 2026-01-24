// SPDX-License-Identifier: BSD-3-Clause

#include "slicot.h"
#include "slicot_blas.h"

#include <stdbool.h>

void SLC_MB04ND(const char* uplo, i32 n, i32 m, i32 p,
                f64* r, i32 ldr, f64* a, i32 lda,
                f64* b, i32 ldb, f64* c, i32 ldc,
                f64* tau, f64* dwork) {
    if (n <= 0 || p <= 0) {
        return;
    }

    bool luplo = (*uplo == 'U' || *uplo == 'u');

    if (luplo) {
        for (i32 i = n - 1; i >= 0; i--) {
            i32 im = (n - i < p) ? (n - i) : p;
            i32 ip = (p - n + i > 0) ? (p - n + i) : 0;

            i32 im1 = im + 1;
            SLC_DLARFG(&im1, &r[i + i * ldr], &a[i + ip * lda], &lda, &tau[i]);

            if (i > 0) {
                SLC_MB04NY(i, im, &a[i + ip * lda], lda, tau[i],
                           &r[0 + i * ldr], ldr, &a[0 + ip * lda], lda, dwork);
            }

            if (m > 0) {
                SLC_MB04NY(m, im, &a[i + ip * lda], lda, tau[i],
                           &b[0 + i * ldb], ldb, &c[0 + ip * ldc], ldc, dwork);
            }
        }
    } else {
        i32 p1 = p + 1;
        for (i32 i = n - 1; i >= 1; i--) {
            SLC_DLARFG(&p1, &r[i + i * ldr], &a[i + 0 * lda], &lda, &tau[i]);

            SLC_MB04NY(i, p, &a[i + 0 * lda], lda, tau[i],
                       &r[0 + i * ldr], ldr, a, lda, dwork);
        }

        SLC_DLARFG(&p1, &r[0 + 0 * ldr], &a[0 + 0 * lda], &lda, &tau[0]);

        if (m > 0) {
            for (i32 i = n - 1; i >= 0; i--) {
                SLC_MB04NY(m, p, &a[i + 0 * lda], lda, tau[i],
                           &b[0 + i * ldb], ldb, c, ldc, dwork);
            }
        }
    }
}
