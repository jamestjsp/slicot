// SPDX-License-Identifier: BSD-3-Clause

#include "slicot.h"
#include "slicot_blas.h"
#include <complex.h>
#include <math.h>

void mb03rw(i32 m, i32 n, f64 pmax, const c128 *a, i32 lda, const c128 *b,
            i32 ldb, c128 *c, i32 ldc, i32 *info) {
    const f64 one = 1.0;
    const c128 cone = 1.0 + 0.0 * I;
    const c128 negcone = -1.0 + 0.0 * I;

    *info = 0;

    if (m == 0 || n == 0) {
        return;
    }

    f64 eps = SLC_DLAMCH("P");
    f64 smlnum = SLC_DLAMCH("S");
    f64 bignum = one / smlnum;
    SLC_DLABAD(&smlnum, &bignum);
    smlnum = smlnum * (f64)(m * n) / eps;
    bignum = one / smlnum;

    f64 dum[1];
    f64 anrm = SLC_ZLANTR("M", "U", "N", &m, &m, a, &lda, dum);
    f64 bnrm = SLC_ZLANTR("M", "U", "N", &n, &n, b, &ldb, dum);
    f64 smin = smlnum;
    if (eps * anrm > smin) {
        smin = eps * anrm;
    }
    if (eps * bnrm > smin) {
        smin = eps * bnrm;
    }

    i32 int1 = 1;

    for (i32 l = 0; l < n; l++) {
        i32 lm1 = l;

        if (lm1 > 0) {
            SLC_ZGEMV("N", &m, &lm1, &negcone, c, &ldc, &b[l * ldb], &int1,
                      &cone, &c[l * ldc], &int1);
        }

        for (i32 k = m - 1; k >= 0; k--) {
            c128 c11 = c[k + l * ldc];

            if (k < m - 1) {
                c128 dot = 0.0 + 0.0 * I;
                for (i32 i = 0; i < m - k - 1; i++) {
                    dot += a[k + (k + 1 + i) * lda] * c[(k + 1 + i) + l * ldc];
                }
                c11 = c11 + dot;
            }

            c128 a11 = b[l + l * ldb] - a[k + k * lda];
            f64 aa11 = fabs(creal(a11)) + fabs(cimag(a11));

            if (aa11 <= smin) {
                a11 = smin;
                aa11 = smin;
                *info = 2;
            }

            f64 ac11 = fabs(creal(c11)) + fabs(cimag(c11));

            if (aa11 < one && ac11 > one) {
                if (ac11 > bignum * aa11) {
                    *info = 1;
                    return;
                }
            }

            c128 x11 = c11 / a11;

            if (cabs(x11) > pmax) {
                *info = 1;
                return;
            }

            c[k + l * ldc] = x11;
        }
    }
}
