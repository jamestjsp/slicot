/*
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "slicot.h"
#include "slicot_blas.h"

void tb01md(const char* jobu, const char* uplo, i32 n, i32 m,
            f64* a, i32 lda, f64* b, i32 ldb,
            f64* u, i32 ldu, f64* dwork, i32* info)
{
    const f64 dbl0 = 0.0;
    const f64 dbl1 = 1.0;
    const i32 int1 = 1;

    bool ljoba, ljobi, luplo;
    i32 ii, j, m1, n1, nj;
    i32 par1, par2, par3, par4, par5, par6;
    f64 dz;

    *info = 0;
    luplo = (*uplo == 'U' || *uplo == 'u');
    ljobi = (*jobu == 'I' || *jobu == 'i');
    ljoba = ljobi || (*jobu == 'U' || *jobu == 'u');

    if (!ljoba && !(*jobu == 'N' || *jobu == 'n')) {
        *info = -1;
    } else if (!luplo && !(*uplo == 'L' || *uplo == 'l')) {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    } else if (m < 0) {
        *info = -4;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -6;
    } else if (ldb < (n > 1 ? n : 1)) {
        *info = -8;
    } else if ((!ljoba && ldu < 1) || (ljoba && ldu < (n > 1 ? n : 1))) {
        *info = -10;
    }

    if (*info != 0) {
        return;
    }

    if (ljobi) {
        SLC_DLASET("Full", &n, &n, &dbl0, &dbl1, u, &ldu);
    }

    if (n == 0 || m == 0) {
        return;
    }

    m1 = m + 1;
    n1 = n - 1;

    i32 min_mn1 = (m < n1) ? m : n1;
    for (j = 1; j <= min_mn1; j++) {
        nj = n - j;
        if (luplo) {
            par1 = j;
            par2 = j;
            par3 = j + 1;
            par4 = m;
            par5 = n;
        } else {
            par1 = m - j + 1;
            par2 = nj + 1;
            par3 = 1;
            par4 = m - j;
            par5 = nj;
        }

        i32 nj1 = nj + 1;
        SLC_DLARFG(&nj1, &b[(par2 - 1) + (par1 - 1) * ldb],
                   &b[(par3 - 1) + (par1 - 1) * ldb], &int1, &dz);

        SLC_DLATZM("Left", &nj1, &n, &b[(par3 - 1) + (par1 - 1) * ldb], &int1, &dz,
                   &a[(par2 - 1) + 0 * lda], &a[(par3 - 1) + 0 * lda], &lda, dwork);
        SLC_DLATZM("Right", &n, &nj1, &b[(par3 - 1) + (par1 - 1) * ldb], &int1, &dz,
                   &a[0 + (par2 - 1) * lda], &a[0 + (par3 - 1) * lda], &lda, dwork);

        if (ljoba) {
            SLC_DLATZM("Right", &n, &nj1, &b[(par3 - 1) + (par1 - 1) * ldb], &int1, &dz,
                       &u[0 + (par2 - 1) * ldu], &u[0 + (par3 - 1) * ldu], &ldu, dwork);
        }

        if (j != m) {
            i32 ncols = par4 - par3 + 1;
            SLC_DLATZM("Left", &nj1, &ncols, &b[(par3 - 1) + (par1 - 1) * ldb], &int1, &dz,
                       &b[(par2 - 1) + (par3 - 1) * ldb], &b[(par3 - 1) + (par3 - 1) * ldb],
                       &ldb, dwork);
        }

        for (ii = par3; ii <= par5; ii++) {
            b[(ii - 1) + (par1 - 1) * ldb] = dbl0;
        }
    }

    for (j = m1; j <= n1; j++) {
        nj = n - j;
        if (luplo) {
            par1 = j - m;
            par2 = j;
            par3 = j + 1;
            par4 = n;
            par5 = j - m + 1;
            par6 = n;
        } else {
            par1 = n + m1 - j;
            par2 = nj + 1;
            par3 = 1;
            par4 = nj;
            par5 = 1;
            par6 = n + m - j;
        }

        i32 nj1 = nj + 1;
        SLC_DLARFG(&nj1, &a[(par2 - 1) + (par1 - 1) * lda],
                   &a[(par3 - 1) + (par1 - 1) * lda], &int1, &dz);

        i32 ncols = par6 - par5 + 1;
        SLC_DLATZM("Left", &nj1, &ncols, &a[(par3 - 1) + (par1 - 1) * lda], &int1, &dz,
                   &a[(par2 - 1) + (par5 - 1) * lda], &a[(par3 - 1) + (par5 - 1) * lda],
                   &lda, dwork);
        SLC_DLATZM("Right", &n, &nj1, &a[(par3 - 1) + (par1 - 1) * lda], &int1, &dz,
                   &a[0 + (par2 - 1) * lda], &a[0 + (par3 - 1) * lda], &lda, dwork);

        if (ljoba) {
            SLC_DLATZM("Right", &n, &nj1, &a[(par3 - 1) + (par1 - 1) * lda], &int1, &dz,
                       &u[0 + (par2 - 1) * ldu], &u[0 + (par3 - 1) * ldu], &ldu, dwork);
        }

        for (ii = par3; ii <= par4; ii++) {
            a[(ii - 1) + (par1 - 1) * lda] = dbl0;
        }
    }
}
