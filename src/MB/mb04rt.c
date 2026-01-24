/**
 * @file mb04rt.c
 * @brief Blocked solver for generalized real Sylvester equation using Level 3 BLAS.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "slicot.h"
#include "slicot_blas.h"

void mb04rt(
    const i32 m,
    const i32 n,
    const f64 pmax,
    const f64* a, const i32 lda,
    const f64* b, const i32 ldb,
    f64* c, const i32 ldc,
    const f64* d, const i32 ldd,
    const f64* e, const i32 lde,
    f64* f, const i32 ldf,
    f64* scale,
    i32* iwork,
    i32* info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    i32 int1 = 1;
    f64 dbl1 = 1.0, dbl_neg1 = -1.0;

    i32 i, ie, is, j, je, js, k, mb, nb, p, q;
    f64 scaloc;

    *info = 0;
    *scale = ONE;

    if (m == 0 || n == 0)
        return;

    mb = SLC_ILAENV(&(i32){2}, "DTGSYL", "NoTran", &m, &n, &(i32){-1}, &(i32){-1});
    nb = SLC_ILAENV(&(i32){5}, "DTGSYL", "NoTran", &m, &n, &(i32){-1}, &(i32){-1});

    if ((mb <= 1 && nb <= 1) || (mb >= m && nb >= n)) {
        mb04rs(m, n, pmax, a, lda, b, ldb, c, ldc, d, ldd, e, lde, f, ldf, scale, iwork, info);
        return;
    }

    p = 0;
    i = 1;

    while (i <= m) {
        p++;
        iwork[p - 1] = i;
        i += mb;
        if (i >= m)
            break;
        if (a[(i - 1) + (i - 2) * lda] != ZERO)
            i++;
    }

    iwork[p] = m + 1;
    if (iwork[p - 1] == iwork[p])
        p--;

    q = p + 1;
    j = 1;

    while (j <= n) {
        q++;
        iwork[q - 1] = j;
        j += nb;
        if (j >= n)
            break;
        if (b[(j - 1) + (j - 2) * ldb] != ZERO)
            j++;
    }

    iwork[q] = n + 1;
    if (iwork[q - 1] == iwork[q])
        q--;

    for (j = p + 2; j <= q; j++) {
        js = iwork[j - 1];
        je = iwork[j] - 1;
        nb = je - js + 1;

        for (i = p; i >= 1; i--) {
            is = iwork[i - 1];
            ie = iwork[i] - 1;
            mb = ie - is + 1;

            mb04rs(mb, nb, pmax,
                   &a[(is - 1) + (is - 1) * lda], lda,
                   &b[(js - 1) + (js - 1) * ldb], ldb,
                   &c[(is - 1) + (js - 1) * ldc], ldc,
                   &d[(is - 1) + (is - 1) * ldd], ldd,
                   &e[(js - 1) + (js - 1) * lde], lde,
                   &f[(is - 1) + (js - 1) * ldf], ldf,
                   &scaloc, &iwork[q + 1], info);

            if (*info > 0)
                return;

            if (scaloc != ONE) {
                for (k = 0; k < js - 1; k++) {
                    SLC_DSCAL(&m, &scaloc, &c[k * ldc], &int1);
                    SLC_DSCAL(&m, &scaloc, &f[k * ldf], &int1);
                }

                i32 len = is - 1;
                for (k = js - 1; k <= je - 1; k++) {
                    SLC_DSCAL(&len, &scaloc, &c[k * ldc], &int1);
                    SLC_DSCAL(&len, &scaloc, &f[k * ldf], &int1);
                }

                i32 m_minus_ie = m - ie;
                for (k = js - 1; k <= je - 1; k++) {
                    SLC_DSCAL(&m_minus_ie, &scaloc, &c[ie + k * ldc], &int1);
                    SLC_DSCAL(&m_minus_ie, &scaloc, &f[ie + k * ldf], &int1);
                }

                for (k = je; k < n; k++) {
                    SLC_DSCAL(&m, &scaloc, &c[k * ldc], &int1);
                    SLC_DSCAL(&m, &scaloc, &f[k * ldf], &int1);
                }

                *scale *= scaloc;
            }

            if (i > 1) {
                i32 len = is - 1;
                SLC_DGEMM("N", "N", &len, &nb, &mb, &dbl_neg1,
                          &a[(is - 1) * lda], &lda,
                          &c[(is - 1) + (js - 1) * ldc], &ldc,
                          &dbl1, &c[(js - 1) * ldc], &ldc);
                SLC_DGEMM("N", "N", &len, &nb, &mb, &dbl_neg1,
                          &d[(is - 1) * ldd], &ldd,
                          &c[(is - 1) + (js - 1) * ldc], &ldc,
                          &dbl1, &f[(js - 1) * ldf], &ldf);
            }

            if (j < q) {
                i32 len = n - je;
                SLC_DGEMM("N", "N", &mb, &len, &nb, &dbl1,
                          &f[(is - 1) + (js - 1) * ldf], &ldf,
                          &b[(js - 1) + je * ldb], &ldb,
                          &dbl1, &c[(is - 1) + je * ldc], &ldc);
                SLC_DGEMM("N", "N", &mb, &len, &nb, &dbl1,
                          &f[(is - 1) + (js - 1) * ldf], &ldf,
                          &e[(js - 1) + je * lde], &lde,
                          &dbl1, &f[(is - 1) + je * ldf], &ldf);
            }
        }
    }
}
