/**
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * SB04OW - Solve periodic Sylvester equation with matrices in periodic Schur form
 *
 * Solves:
 *     A * R - L * B = scale * C
 *     D * L - R * E = scale * F
 *
 * where (A, D), (B, E) are in periodic Schur form.
 */

#include "slicot.h"
#include "slicot_blas.h"

void sb04ow(
    const i32 m,
    const i32 n,
    const f64* a,
    const i32 lda,
    const f64* b,
    const i32 ldb,
    f64* c,
    const i32 ldc,
    const f64* d,
    const i32 ldd,
    const f64* e,
    const i32 lde,
    f64* f,
    const i32 ldf,
    f64* scale,
    i32* iwork,
    i32* info)
{
    const i32 ldz = 8;
    const f64 zero = 0.0;
    const f64 one = 1.0;

    i32 i, ie, ierr, ii, is, isp1, j, je, jj, js, jsp1;
    i32 k, mb, nb, p, q, zdim;
    f64 scaloc;

    i32 ipiv[8], jpiv[8];
    f64 rhs[8], z[64];

    i32 int1 = 1;
    f64 done = 1.0;
    f64 dneone = -1.0;

    *info = 0;
    ierr = 0;

    if (m <= 0) {
        *info = -1;
    } else if (n <= 0) {
        *info = -2;
    } else if (lda < (m > 1 ? m : 1)) {
        *info = -4;
    } else if (ldb < (n > 1 ? n : 1)) {
        *info = -6;
    } else if (ldc < (m > 1 ? m : 1)) {
        *info = -8;
    } else if (ldd < (m > 1 ? m : 1)) {
        *info = -10;
    } else if (lde < (n > 1 ? n : 1)) {
        *info = -12;
    } else if (ldf < (m > 1 ? m : 1)) {
        *info = -14;
    }

    if (*info != 0) {
        return;
    }

    p = 0;
    i = 0;
    while (i < m) {
        iwork[p] = i;
        p++;
        if (i == m - 1) {
            break;
        }
        if (a[(i + 1) + i * lda] != zero) {
            i += 2;
        } else {
            i += 1;
        }
    }
    iwork[p] = m;

    q = p;
    j = 0;
    while (j < n) {
        q++;
        iwork[q] = j;
        if (j == n - 1) {
            break;
        }
        if (b[(j + 1) + j * ldb] != zero) {
            j += 2;
        } else {
            j += 1;
        }
    }
    iwork[q + 1] = n;

    *scale = one;
    scaloc = one;

    for (j = p + 1; j <= q; j++) {
        js = iwork[j];
        jsp1 = js + 1;
        je = iwork[j + 1] - 1;
        nb = je - js + 1;

        for (i = p - 1; i >= 0; i--) {
            is = iwork[i];
            isp1 = is + 1;
            ie = iwork[i + 1] - 1;
            mb = ie - is + 1;
            zdim = mb * nb * 2;

            if ((mb == 1) && (nb == 1)) {
                z[0] = a[is + is * lda];
                z[1] = -e[js + js * lde];
                z[0 + 1 * ldz] = -b[js + js * ldb];
                z[1 + 1 * ldz] = d[is + is * ldd];

                rhs[0] = c[is + js * ldc];
                rhs[1] = f[is + js * ldf];

                SLC_DGETC2(&zdim, z, &ldz, ipiv, jpiv, &ierr);
                if (ierr > 0) {
                    *info = ierr;
                }

                SLC_DGESC2(&zdim, z, &ldz, rhs, ipiv, jpiv, &scaloc);
                if (scaloc != one) {
                    for (k = 0; k < n; k++) {
                        SLC_DSCAL(&m, &scaloc, &c[k * ldc], &int1);
                        SLC_DSCAL(&m, &scaloc, &f[k * ldf], &int1);
                    }
                    *scale *= scaloc;
                }

                c[is + js * ldc] = rhs[0];
                f[is + js * ldf] = rhs[1];

                if (i > 0) {
                    f64 neg_rhs0 = -rhs[0];
                    f64 neg_rhs1 = -rhs[1];
                    SLC_DAXPY(&is, &neg_rhs0, &a[is * lda], &int1, &c[js * ldc], &int1);
                    SLC_DAXPY(&is, &neg_rhs1, &d[is * ldd], &int1, &f[js * ldf], &int1);
                }
                if (j < q) {
                    i32 nremain = n - je - 1;
                    SLC_DAXPY(&nremain, &rhs[1], &b[js + (je + 1) * ldb], &ldb, &c[is + (je + 1) * ldc], &ldc);
                    SLC_DAXPY(&nremain, &rhs[0], &e[js + (je + 1) * lde], &lde, &f[is + (je + 1) * ldf], &ldf);
                }

            } else if ((mb == 1) && (nb == 2)) {
                z[0] = a[is + is * lda];
                z[1] = zero;
                z[2] = -e[js + js * lde];
                z[3] = -e[js + jsp1 * lde];

                z[0 + 1 * ldz] = zero;
                z[1 + 1 * ldz] = a[is + is * lda];
                z[2 + 1 * ldz] = zero;
                z[3 + 1 * ldz] = -e[jsp1 + jsp1 * lde];

                z[0 + 2 * ldz] = -b[js + js * ldb];
                z[1 + 2 * ldz] = -b[js + jsp1 * ldb];
                z[2 + 2 * ldz] = d[is + is * ldd];
                z[3 + 2 * ldz] = zero;

                z[0 + 3 * ldz] = -b[jsp1 + js * ldb];
                z[1 + 3 * ldz] = -b[jsp1 + jsp1 * ldb];
                z[2 + 3 * ldz] = zero;
                z[3 + 3 * ldz] = d[is + is * ldd];

                rhs[0] = c[is + js * ldc];
                rhs[1] = c[is + jsp1 * ldc];
                rhs[2] = f[is + js * ldf];
                rhs[3] = f[is + jsp1 * ldf];

                SLC_DGETC2(&zdim, z, &ldz, ipiv, jpiv, &ierr);
                if (ierr > 0) {
                    *info = ierr;
                }

                SLC_DGESC2(&zdim, z, &ldz, rhs, ipiv, jpiv, &scaloc);
                if (scaloc != one) {
                    for (k = 0; k < n; k++) {
                        SLC_DSCAL(&m, &scaloc, &c[k * ldc], &int1);
                        SLC_DSCAL(&m, &scaloc, &f[k * ldf], &int1);
                    }
                    *scale *= scaloc;
                }

                c[is + js * ldc] = rhs[0];
                c[is + jsp1 * ldc] = rhs[1];
                f[is + js * ldf] = rhs[2];
                f[is + jsp1 * ldf] = rhs[3];

                if (i > 0) {
                    SLC_DGER(&is, &nb, &dneone, &a[is * lda], &int1, &rhs[0], &int1, &c[js * ldc], &ldc);
                    SLC_DGER(&is, &nb, &dneone, &d[is * ldd], &int1, &rhs[2], &int1, &f[js * ldf], &ldf);
                }
                if (j < q) {
                    i32 nremain = n - je - 1;
                    SLC_DAXPY(&nremain, &rhs[2], &b[js + (je + 1) * ldb], &ldb, &c[is + (je + 1) * ldc], &ldc);
                    SLC_DAXPY(&nremain, &rhs[0], &e[js + (je + 1) * lde], &lde, &f[is + (je + 1) * ldf], &ldf);
                    SLC_DAXPY(&nremain, &rhs[3], &b[jsp1 + (je + 1) * ldb], &ldb, &c[is + (je + 1) * ldc], &ldc);
                    SLC_DAXPY(&nremain, &rhs[1], &e[jsp1 + (je + 1) * lde], &lde, &f[is + (je + 1) * ldf], &ldf);
                }

            } else if ((mb == 2) && (nb == 1)) {
                z[0] = a[is + is * lda];
                z[1] = a[isp1 + is * lda];
                z[2] = -e[js + js * lde];
                z[3] = zero;

                z[0 + 1 * ldz] = a[is + isp1 * lda];
                z[1 + 1 * ldz] = a[isp1 + isp1 * lda];
                z[2 + 1 * ldz] = zero;
                z[3 + 1 * ldz] = -e[js + js * lde];

                z[0 + 2 * ldz] = -b[js + js * ldb];
                z[1 + 2 * ldz] = zero;
                z[2 + 2 * ldz] = d[is + is * ldd];
                z[3 + 2 * ldz] = zero;

                z[0 + 3 * ldz] = zero;
                z[1 + 3 * ldz] = -b[js + js * ldb];
                z[2 + 3 * ldz] = d[is + isp1 * ldd];
                z[3 + 3 * ldz] = d[isp1 + isp1 * ldd];

                rhs[0] = c[is + js * ldc];
                rhs[1] = c[isp1 + js * ldc];
                rhs[2] = f[is + js * ldf];
                rhs[3] = f[isp1 + js * ldf];

                SLC_DGETC2(&zdim, z, &ldz, ipiv, jpiv, &ierr);
                if (ierr > 0) {
                    *info = ierr;
                }

                SLC_DGESC2(&zdim, z, &ldz, rhs, ipiv, jpiv, &scaloc);
                if (scaloc != one) {
                    for (k = 0; k < n; k++) {
                        SLC_DSCAL(&m, &scaloc, &c[k * ldc], &int1);
                        SLC_DSCAL(&m, &scaloc, &f[k * ldf], &int1);
                    }
                    *scale *= scaloc;
                }

                c[is + js * ldc] = rhs[0];
                c[isp1 + js * ldc] = rhs[1];
                f[is + js * ldf] = rhs[2];
                f[isp1 + js * ldf] = rhs[3];

                if (i > 0) {
                    SLC_DGEMV("N", &is, &mb, &dneone, &a[is * lda], &lda, &rhs[0], &int1, &done, &c[js * ldc], &int1);
                    SLC_DGEMV("N", &is, &mb, &dneone, &d[is * ldd], &ldd, &rhs[2], &int1, &done, &f[js * ldf], &int1);
                }
                if (j < q) {
                    i32 nremain = n - je - 1;
                    SLC_DGER(&mb, &nremain, &done, &rhs[2], &int1, &b[js + (je + 1) * ldb], &ldb, &c[is + (je + 1) * ldc], &ldc);
                    SLC_DGER(&mb, &nremain, &done, &rhs[0], &int1, &e[js + (je + 1) * lde], &lde, &f[is + (je + 1) * ldf], &ldf);
                }

            } else { /* (mb == 2) && (nb == 2) */
                SLC_DLASET("All", &ldz, &ldz, &zero, &zero, z, &ldz);

                z[0] = a[is + is * lda];
                z[1] = a[isp1 + is * lda];
                z[4] = -e[js + js * lde];
                z[6] = -e[js + jsp1 * lde];

                z[0 + 1 * ldz] = a[is + isp1 * lda];
                z[1 + 1 * ldz] = a[isp1 + isp1 * lda];
                z[5 + 1 * ldz] = -e[js + js * lde];
                z[7 + 1 * ldz] = -e[js + jsp1 * lde];

                z[2 + 2 * ldz] = a[is + is * lda];
                z[3 + 2 * ldz] = a[isp1 + is * lda];
                z[6 + 2 * ldz] = -e[jsp1 + jsp1 * lde];

                z[2 + 3 * ldz] = a[is + isp1 * lda];
                z[3 + 3 * ldz] = a[isp1 + isp1 * lda];
                z[7 + 3 * ldz] = -e[jsp1 + jsp1 * lde];

                z[0 + 4 * ldz] = -b[js + js * ldb];
                z[2 + 4 * ldz] = -b[js + jsp1 * ldb];
                z[4 + 4 * ldz] = d[is + is * ldd];

                z[1 + 5 * ldz] = -b[js + js * ldb];
                z[3 + 5 * ldz] = -b[js + jsp1 * ldb];
                z[4 + 5 * ldz] = d[is + isp1 * ldd];
                z[5 + 5 * ldz] = d[isp1 + isp1 * ldd];

                z[0 + 6 * ldz] = -b[jsp1 + js * ldb];
                z[2 + 6 * ldz] = -b[jsp1 + jsp1 * ldb];
                z[6 + 6 * ldz] = d[is + is * ldd];

                z[1 + 7 * ldz] = -b[jsp1 + js * ldb];
                z[3 + 7 * ldz] = -b[jsp1 + jsp1 * ldb];
                z[6 + 7 * ldz] = d[is + isp1 * ldd];
                z[7 + 7 * ldz] = d[isp1 + isp1 * ldd];

                k = 0;
                ii = mb * nb;
                for (jj = 0; jj < nb; jj++) {
                    SLC_DCOPY(&mb, &c[is + (js + jj) * ldc], &int1, &rhs[k], &int1);
                    SLC_DCOPY(&mb, &f[is + (js + jj) * ldf], &int1, &rhs[ii], &int1);
                    k += mb;
                    ii += mb;
                }

                SLC_DGETC2(&zdim, z, &ldz, ipiv, jpiv, &ierr);
                if (ierr > 0) {
                    *info = ierr;
                }

                SLC_DGESC2(&zdim, z, &ldz, rhs, ipiv, jpiv, &scaloc);
                if (scaloc != one) {
                    for (k = 0; k < n; k++) {
                        SLC_DSCAL(&m, &scaloc, &c[k * ldc], &int1);
                        SLC_DSCAL(&m, &scaloc, &f[k * ldf], &int1);
                    }
                    *scale *= scaloc;
                }

                k = 0;
                ii = mb * nb;
                for (jj = 0; jj < nb; jj++) {
                    SLC_DCOPY(&mb, &rhs[k], &int1, &c[is + (js + jj) * ldc], &int1);
                    SLC_DCOPY(&mb, &rhs[ii], &int1, &f[is + (js + jj) * ldf], &int1);
                    k += mb;
                    ii += mb;
                }

                k = mb * nb;
                if (i > 0) {
                    SLC_DGEMM("N", "N", &is, &nb, &mb, &dneone, &a[is * lda], &lda, &rhs[0], &mb, &done, &c[js * ldc], &ldc);
                    SLC_DGEMM("N", "N", &is, &nb, &mb, &dneone, &d[is * ldd], &ldd, &rhs[k], &mb, &done, &f[js * ldf], &ldf);
                }
                if (j < q) {
                    i32 nremain = n - je - 1;
                    SLC_DGEMM("N", "N", &mb, &nremain, &nb, &done, &rhs[k], &mb, &b[js + (je + 1) * ldb], &ldb, &done, &c[is + (je + 1) * ldc], &ldc);
                    SLC_DGEMM("N", "N", &mb, &nremain, &nb, &done, &rhs[0], &mb, &e[js + (je + 1) * lde], &lde, &done, &f[is + (je + 1) * ldf], &ldf);
                }
            }
        }
    }
}
