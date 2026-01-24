// SPDX-License-Identifier: BSD-3-Clause

#include "slicot.h"
#include "slicot_blas.h"
#include <stdbool.h>
#include <string.h>

void mb04ld(const char uplo, i32 n, i32 m, i32 p,
            f64 *l, i32 ldl,
            f64 *a, i32 lda,
            f64 *b, i32 ldb,
            f64 *c, i32 ldc,
            f64 *tau,
            f64 *dwork)
{
    const f64 zero = 0.0;
    const f64 one = 1.0;
    const i32 inc1 = 1;

    if (m <= 0 || n <= 0) {
        return;
    }

    bool luplo = (uplo == 'L' || uplo == 'l');
    i32 im = m;

    for (i32 i = 0; i < n; i++) {
        if (luplo) {
            im = (i + 1 < m) ? (i + 1) : m;
        }

        i32 im_plus_1 = im + 1;
        SLC_DLARFG(&im_plus_1, &l[i + i*ldl], &a[i + 0*lda], &lda, &tau[i]);

        if (tau[i] != zero) {
            if (i < n - 1) {
                i32 n_minus_i_minus_1 = n - i - 1;
                SLC_DCOPY(&n_minus_i_minus_1, &l[(i+1) + i*ldl], &inc1, dwork, &inc1);

                SLC_DGEMV("N", &n_minus_i_minus_1, &im, &one,
                         &a[(i+1) + 0*lda], &lda,
                         &a[i + 0*lda], &lda,
                         &one, dwork, &inc1);
            }

            SLC_DGEMV("N", &p, &im, &one, b, &ldb,
                     &a[i + 0*lda], &lda,
                     &zero, &c[0 + i*ldc], &inc1);

            if (i < n - 1) {
                f64 neg_tau = -tau[i];
                i32 n_minus_i_minus_1 = n - i - 1;
                SLC_DAXPY(&n_minus_i_minus_1, &neg_tau, dwork, &inc1,
                         &l[(i+1) + i*ldl], &inc1);

                SLC_DGER(&n_minus_i_minus_1, &im, &neg_tau,
                        dwork, &inc1,
                        &a[i + 0*lda], &lda,
                        &a[(i+1) + 0*lda], &lda);
            }

            f64 neg_tau = -tau[i];
            SLC_DSCAL(&p, &neg_tau, &c[0 + i*ldc], &inc1);

            SLC_DGER(&p, &im, &one,
                    &c[0 + i*ldc], &inc1,
                    &a[i + 0*lda], &lda,
                    b, &ldb);
        }
    }
}
