// SPDX-License-Identifier: BSD-3-Clause

#include "slicot.h"
#include "slicot_blas.h"
#include <stdbool.h>
#include <string.h>

void mb04kd(const char uplo, i32 n, i32 m, i32 p,
            f64 *r, i32 ldr,
            f64 *a, i32 lda,
            f64 *b, i32 ldb,
            f64 *c, i32 ldc,
            f64 *tau,
            f64 *dwork)
{
    const f64 zero = 0.0;
    const f64 one = 1.0;
    const i32 inc1 = 1;

    if (n <= 0 || p <= 0) {
        return;
    }

    bool luplo = (uplo == 'U' || uplo == 'u');
    i32 im = p;

    for (i32 i = 0; i < n; i++) {
        // Annihilate the i-th column of A and apply transformations

        if (luplo) {
            im = (i + 1 < p) ? (i + 1) : p;
        }

        if (im <= 0) continue;

        // Generate Householder reflector to annihilate A(0:im-1, i)
        // Vector: [R(i,i), A(0:im-1,i)]^T has length im+1
        i32 im_plus_1 = im + 1;
        SLC_DLARFG(&im_plus_1, &r[i + i*ldr], &a[0 + i*lda], &inc1, &tau[i]);

        if (tau[i] != zero) {
            // Compute w and C(i,:)
            // w = R(i, i+1:n-1)
            // C(i,:) = 0
            // w += A(0:im-1, i+1:n-1)^T * A(0:im-1, i)
            // C(i,:) = B(0:im-1, :)^T * A(0:im-1, i)

            if (i < n - 1) {
                // Copy R(i, i+1:n-1) to dwork
                SLC_DCOPY(&(i32){n - i - 1}, &r[i + (i+1)*ldr], &ldr, dwork, &inc1);

                // dwork += A(0:im-1, i+1:n-1)^T * A(0:im-1, i)
                SLC_DGEMV("T", &im, &(i32){n - i - 1}, &one,
                         &a[0 + (i+1)*lda], &lda,
                         &a[0 + i*lda], &inc1,
                         &one, dwork, &inc1);
            }

            // C(i,:) = B(0:im-1, :)^T * A(0:im-1, i)
            SLC_DGEMV("T", &im, &m, &one, b, &ldb,
                     &a[0 + i*lda], &inc1,
                     &zero, &c[i + 0*ldc], &ldc);

            // Update R(i, i+1:n-1) and A(0:im-1, i+1:n-1)
            if (i < n - 1) {
                // R(i, i+1:n-1) -= tau[i] * dwork
                f64 neg_tau = -tau[i];
                SLC_DAXPY(&(i32){n - i - 1}, &neg_tau, dwork, &inc1,
                         &r[i + (i+1)*ldr], &ldr);

                // A(0:im-1, i+1:n-1) -= tau[i] * A(0:im-1,i) * dwork^T
                SLC_DGER(&im, &(i32){n - i - 1}, &neg_tau,
                        &a[0 + i*lda], &inc1,
                        dwork, &inc1,
                        &a[0 + (i+1)*lda], &lda);
            }

            // C(i,:) *= -tau[i]
            f64 neg_tau = -tau[i];
            SLC_DSCAL(&m, &neg_tau, &c[i + 0*ldc], &ldc);

            // B(0:im-1,:) += A(0:im-1,i) * C(i,:)^T
            SLC_DGER(&im, &m, &one,
                    &a[0 + i*lda], &inc1,
                    &c[i + 0*ldc], &ldc,
                    b, &ldb);
        }
    }
}
