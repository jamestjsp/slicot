// SPDX-License-Identifier: BSD-3-Clause

#include "slicot.h"
#include "slicot_blas.h"
#include <complex.h>
#include <math.h>
#include <stdbool.h>

void mb3jzp(const char *compq, i32 n, c128 *a, i32 lda, c128 *d, i32 ldd,
            c128 *b, i32 ldb, c128 *f, i32 ldf, c128 *q, i32 ldq, i32 *neig,
            f64 tol, f64 *dwork, c128 *zwork, i32 *info) {
    const f64 zero = 0.0;
    const f64 ten = 10.0;
    const c128 czero = 0.0 + 0.0 * I;
    const c128 cone = 1.0 + 0.0 * I;

    i32 nb = *info;
    i32 m = n / 2;
    i32 m1 = (m > 1) ? m : 1;
    *neig = 0;

    bool liniq = (compq[0] == 'I' || compq[0] == 'i');
    bool lupdq = (compq[0] == 'U' || compq[0] == 'u');
    bool lcmpq = liniq || lupdq;

    *info = 0;
    bool compq_n = (compq[0] == 'N' || compq[0] == 'n');

    if (!compq_n && !lcmpq) {
        *info = -1;
    } else if (n < 0 || (n % 2) != 0) {
        *info = -2;
    } else if (lda < m1) {
        *info = -4;
    } else if (ldd < m1) {
        *info = -6;
    } else if (ldb < m1) {
        *info = -8;
    } else if (ldf < m1) {
        *info = -10;
    } else if (ldq < 1 || (lcmpq && ldq < n)) {
        *info = -12;
    }

    if (*info != 0) {
        return;
    }

    if (n == 0) {
        return;
    }

    // Block algorithm: estimate NB if not provided
    if (nb <= 0) {
        c128 zwork_query;
        i32 info_query = 0;
        i32 neg1 = -1;
        SLC_ZGEQRF(&m, &m, a, &lda, zwork, &zwork_query, &neg1, &info_query);
        i32 opt_lwork = (i32)creal(zwork_query);
        nb = opt_lwork / m1;
        if (nb < 2) nb = 2;
        if (nb > m) nb = m;
    }

    f64 eps = tol;
    if (eps <= zero) {
        f64 n_dbl = (f64)n;
        f64 min_val = (n_dbl < ten) ? n_dbl : ten;
        eps = min_val * SLC_DLAMCH("P");
    }

    i32 iupd = 0, upds = 0;
    if (liniq) {
        iupd = m;
        upds = m;
        SLC_ZLASET("F", &n, &n, &czero, &cone, q, &ldq);
    } else if (lupdq) {
        iupd = 0;
        upds = n;
    }

    // STEP 1. Reorder the eigenvalues in the subpencil aA - bB.
    i32 mm = 0;
    i32 mp = m + 1;
    i32 int1 = 1;

    f64 co1, co2;
    c128 si1, si2;
    c128 cjf, tmp;

    // I. Reorder the eigenvalues with negative real parts to the top.
    for (i32 k = 0; k < m; k++) {
        c128 akk = a[k + k * lda];
        c128 bkk = b[k + k * ldb];
        f64 test_val = creal(akk) * creal(bkk) + cimag(akk) * cimag(bkk);
        f64 threshold = -cabs(akk) * cabs(bkk) * eps;

        if (test_val < threshold) {
            i32 ic = 0;
            for (i32 j = k - 1; j >= mm; j--) {
                mb03dz(&a[j + j * lda], lda, &b[j + j * ldb], ldb, &co1, &si1,
                       &co2, &si2);
                dwork[ic] = co2;
                zwork[ic] = -si2;
                ic++;

                // Update A and D
                i32 len_j = j;
                SLC_ZROT(&len_j, &a[(j + 1) * lda], &int1, &a[j * lda], &int1,
                         &co1, &si1);
                a[j + j * lda] =
                    co2 * a[j + j * lda] + si2 * a[(j + 1) + (j + 1) * lda] * conj(si1);
                a[(j + 1) + (j + 1) * lda] = co1 * a[(j + 1) + (j + 1) * lda];

                cjf = -conj(d[j + (j + 1) * ldd]);
                tmp = co2 * cjf - conj(si2) * d[(j + 1) + (j + 1) * ldd];
                SLC_ZROT(&len_j, &d[(j + 1) * ldd], &int1, &d[j * ldd], &int1,
                         &co2, &si2);
                d[j + j * ldd] = co2 * d[j + j * ldd] - si2 * tmp;
                d[(j + 1) + (j + 1) * ldd] =
                    co2 * d[(j + 1) + (j + 1) * ldd] + si2 * cjf;

                // Update B and F
                SLC_ZROT(&len_j, &b[(j + 1) * ldb], &int1, &b[j * ldb], &int1,
                         &co1, &si1);
                b[j + j * ldb] =
                    co2 * b[j + j * ldb] + si2 * b[(j + 1) + (j + 1) * ldb] * conj(si1);
                b[(j + 1) + (j + 1) * ldb] = co1 * b[(j + 1) + (j + 1) * ldb];

                cjf = conj(f[j + (j + 1) * ldf]);
                tmp = co2 * cjf - conj(si2) * f[(j + 1) + (j + 1) * ldf];
                SLC_ZROT(&len_j, &f[(j + 1) * ldf], &int1, &f[j * ldf], &int1,
                         &co2, &si2);
                f[j + j * ldf] = co2 * f[j + j * ldf] - si2 * tmp;
                f[(j + 1) + (j + 1) * ldf] =
                    co2 * f[(j + 1) + (j + 1) * ldf] + si2 * cjf;

                if (lcmpq) {
                    SLC_ZROT(&upds, &q[(j + 1) * ldq], &int1, &q[j * ldq], &int1,
                             &co1, &si1);
                    SLC_ZROT(&upds, &q[iupd + (m + j + 1) * ldq], &int1,
                             &q[iupd + (m + j) * ldq], &int1, &co2, &si2);
                }
            }

            // Panel Updates for A
            i32 ics = 0;
            i32 je = k - 1;
            while (je >= mm) {
                i32 nc = 0;
                i32 ic_loc = ics;
                ics += nb;
                for (i32 j = je; j >= mm; j--) {
                    nc++;
                    if (nc > nb) nc = nb;
                    i32 js = je - nc + 1;
                    SLC_ZROT(&nc, &a[j + (js + 1) * lda], &lda,
                             &a[(j + 1) + (js + 1) * lda], &lda,
                             &dwork[ic_loc], &zwork[ic_loc]);
                    ic_loc++;
                }
                je -= nb;
            }

            for (i32 js = k; js < m - 1; js += nb) {
                je = js + nb - 1;
                if (je > m - 2) je = m - 2;
                i32 nc = je - js + 1;
                ic = 0;
                for (i32 j = k - 1; j >= mm; j--) {
                    SLC_ZROT(&nc, &a[j + (js + 1) * lda], &lda,
                             &a[(j + 1) + (js + 1) * lda], &lda,
                             &dwork[ic], &zwork[ic]);
                    ic++;
                }
            }

            // Panel Updates for B
            ics = 0;
            je = k - 1;
            while (je >= mm) {
                i32 nc = 0;
                i32 ic_loc = ics;
                ics += nb;
                for (i32 j = je; j >= mm; j--) {
                    nc++;
                    if (nc > nb) nc = nb;
                    i32 js = je - nc + 1;
                    SLC_ZROT(&nc, &b[j + (js + 1) * ldb], &ldb,
                             &b[(j + 1) + (js + 1) * ldb], &ldb,
                             &dwork[ic_loc], &zwork[ic_loc]);
                    ic_loc++;
                }
                je -= nb;
            }

            for (i32 js = k; js < m - 1; js += nb) {
                je = js + nb - 1;
                if (je > m - 2) je = m - 2;
                i32 nc = je - js + 1;
                ic = 0;
                for (i32 j = k - 1; j >= mm; j--) {
                    SLC_ZROT(&nc, &b[j + (js + 1) * ldb], &ldb,
                             &b[(j + 1) + (js + 1) * ldb], &ldb,
                             &dwork[ic], &zwork[ic]);
                    ic++;
                }
            }

            // Panel Updates for D
            ics = 0;
            je = k - 1;
            while (je >= mm) {
                i32 nc = 0;
                i32 ic_loc = ics;
                ics += nb;
                for (i32 j = je; j >= mm; j--) {
                    nc++;
                    if (nc > nb) nc = nb;
                    i32 js = je - nc + 1;
                    SLC_ZROT(&nc, &d[j + (js + 1) * ldd], &ldd,
                             &d[(j + 1) + (js + 1) * ldd], &ldd,
                             &dwork[ic_loc], &zwork[ic_loc]);
                    ic_loc++;
                }
                je -= nb;
            }

            for (i32 js = k; js < m - 1; js += nb) {
                je = js + nb - 1;
                if (je > m - 2) je = m - 2;
                i32 nc = je - js + 1;
                ic = 0;
                for (i32 j = k - 1; j >= mm; j--) {
                    SLC_ZROT(&nc, &d[j + (js + 1) * ldd], &ldd,
                             &d[(j + 1) + (js + 1) * ldd], &ldd,
                             &dwork[ic], &zwork[ic]);
                    ic++;
                }
            }

            // Panel Updates for F
            ics = 0;
            je = k - 1;
            while (je >= mm) {
                i32 nc = 0;
                i32 ic_loc = ics;
                ics += nb;
                for (i32 j = je; j >= mm; j--) {
                    nc++;
                    if (nc > nb) nc = nb;
                    i32 js = je - nc + 1;
                    SLC_ZROT(&nc, &f[j + (js + 1) * ldf], &ldf,
                             &f[(j + 1) + (js + 1) * ldf], &ldf,
                             &dwork[ic_loc], &zwork[ic_loc]);
                    ic_loc++;
                }
                je -= nb;
            }

            for (i32 js = k; js < m - 1; js += nb) {
                je = js + nb - 1;
                if (je > m - 2) je = m - 2;
                i32 nc = je - js + 1;
                ic = 0;
                for (i32 j = k - 1; j >= mm; j--) {
                    SLC_ZROT(&nc, &f[j + (js + 1) * ldf], &ldf,
                             &f[(j + 1) + (js + 1) * ldf], &ldf,
                             &dwork[ic], &zwork[ic]);
                    ic++;
                }
            }

            mm++;
        }
    }

    // II. Reorder the eigenvalues with positive real parts to the bottom.
    for (i32 k = m - 1; k >= mm; k--) {
        c128 akk = a[k + k * lda];
        c128 bkk = b[k + k * ldb];
        f64 test_val = creal(akk) * creal(bkk) + cimag(akk) * cimag(bkk);
        f64 threshold = cabs(akk) * cabs(bkk) * eps;

        if (test_val > threshold) {
            i32 ic = 0;
            for (i32 j = k; j < mp - 2; j++) {
                mb03dz(&a[j + j * lda], lda, &b[j + j * ldb], ldb, &co1, &si1,
                       &co2, &si2);
                dwork[ic] = co2;
                zwork[ic] = -si2;
                ic++;

                // Update A and D
                i32 len_j = j;
                SLC_ZROT(&len_j, &a[(j + 1) * lda], &int1, &a[j * lda], &int1,
                         &co1, &si1);
                a[j + j * lda] =
                    co2 * a[j + j * lda] + si2 * a[(j + 1) + (j + 1) * lda] * conj(si1);
                a[(j + 1) + (j + 1) * lda] = co1 * a[(j + 1) + (j + 1) * lda];
                i32 len_mpj = mp - j - 2;
                c128 neg_si2 = -si2;
                SLC_ZROT(&len_mpj, &a[j + (j + 1) * lda], &lda,
                         &a[(j + 1) + (j + 1) * lda], &lda, &co2, &neg_si2);

                cjf = -conj(d[j + (j + 1) * ldd]);
                tmp = co2 * cjf - conj(si2) * d[(j + 1) + (j + 1) * ldd];
                SLC_ZROT(&len_j, &d[(j + 1) * ldd], &int1, &d[j * ldd], &int1,
                         &co2, &si2);
                d[j + j * ldd] = co2 * d[j + j * ldd] - si2 * tmp;
                d[(j + 1) + (j + 1) * ldd] =
                    co2 * d[(j + 1) + (j + 1) * ldd] + si2 * cjf;
                SLC_ZROT(&len_mpj, &d[j + (j + 1) * ldd], &ldd,
                         &d[(j + 1) + (j + 1) * ldd], &ldd, &co2, &neg_si2);

                // Update B and F
                SLC_ZROT(&len_j, &b[(j + 1) * ldb], &int1, &b[j * ldb], &int1,
                         &co1, &si1);
                b[j + j * ldb] =
                    co2 * b[j + j * ldb] + si2 * b[(j + 1) + (j + 1) * ldb] * conj(si1);
                b[(j + 1) + (j + 1) * ldb] = co1 * b[(j + 1) + (j + 1) * ldb];
                SLC_ZROT(&len_mpj, &b[j + (j + 1) * ldb], &ldb,
                         &b[(j + 1) + (j + 1) * ldb], &ldb, &co2, &neg_si2);

                cjf = conj(f[j + (j + 1) * ldf]);
                tmp = co2 * cjf - conj(si2) * f[(j + 1) + (j + 1) * ldf];
                SLC_ZROT(&len_j, &f[(j + 1) * ldf], &int1, &f[j * ldf], &int1,
                         &co2, &si2);
                f[j + j * ldf] = co2 * f[j + j * ldf] - si2 * tmp;
                f[(j + 1) + (j + 1) * ldf] =
                    co2 * f[(j + 1) + (j + 1) * ldf] + si2 * cjf;
                SLC_ZROT(&len_mpj, &f[j + (j + 1) * ldf], &ldf,
                         &f[(j + 1) + (j + 1) * ldf], &ldf, &co2, &neg_si2);

                if (lcmpq) {
                    SLC_ZROT(&upds, &q[(j + 1) * ldq], &int1, &q[j * ldq], &int1,
                             &co1, &si1);
                    SLC_ZROT(&upds, &q[iupd + (m + j + 1) * ldq], &int1,
                             &q[iupd + (m + j) * ldq], &int1, &co2, &si2);
                }
            }

            // Panel Updates for A
            i32 ics = 0;
            for (i32 js = mp - 1; js < m; js += nb) {
                ic = ics;
                i32 je = js + nb;
                if (je > m) je = m;
                i32 nc = je - js;
                if (nc > nb) nc = nb;
                for (i32 j = k; j < mp - 2; j++) {
                    SLC_ZROT(&nc, &a[j + js * lda], &lda, &a[(j + 1) + js * lda],
                             &lda, &dwork[ic], &zwork[ic]);
                    ic++;
                }
            }

            // Panel Updates for D
            ics = 0;
            for (i32 js = mp - 1; js < m; js += nb) {
                ic = ics;
                i32 je = js + nb;
                if (je > m) je = m;
                i32 nc = je - js;
                if (nc > nb) nc = nb;
                for (i32 j = k; j < mp - 2; j++) {
                    SLC_ZROT(&nc, &d[j + js * ldd], &ldd, &d[(j + 1) + js * ldd],
                             &ldd, &dwork[ic], &zwork[ic]);
                    ic++;
                }
            }

            // Panel Updates for B
            ics = 0;
            for (i32 js = mp - 1; js < m; js += nb) {
                ic = ics;
                i32 je = js + nb;
                if (je > m) je = m;
                i32 nc = je - js;
                if (nc > nb) nc = nb;
                for (i32 j = k; j < mp - 2; j++) {
                    SLC_ZROT(&nc, &b[j + js * ldb], &ldb, &b[(j + 1) + js * ldb],
                             &ldb, &dwork[ic], &zwork[ic]);
                    ic++;
                }
            }

            // Panel Updates for F
            ics = 0;
            for (i32 js = mp - 1; js < m; js += nb) {
                ic = ics;
                i32 je = js + nb;
                if (je > m) je = m;
                i32 nc = je - js;
                if (nc > nb) nc = nb;
                for (i32 j = k; j < mp - 2; j++) {
                    SLC_ZROT(&nc, &f[j + js * ldf], &ldf, &f[(j + 1) + js * ldf],
                             &ldf, &dwork[ic], &zwork[ic]);
                    ic++;
                }
            }

            mp--;
        }
    }

    // STEP 2. Reorder the remaining M-MP+1 eigenvalues with negative real parts.
    for (i32 k = m - 1; k >= mp - 1; k--) {
        // I. Exchange the eigenvalues between two diagonal blocks.
        i32 mlast = m - 1;
        mb03hz(a[mlast + mlast * lda], d[mlast + mlast * ldd],
               b[mlast + mlast * ldb], f[mlast + mlast * ldf], &co1, &si1);

        // Update A and D
        tmp = conj(a[mlast + mlast * lda]);
        SLC_ZROT(&m, &d[mlast * ldd], &int1, &a[mlast * lda], &int1, &co1, &si1);
        a[mlast + mlast * lda] =
            a[mlast + mlast * lda] * co1 + tmp * conj(si1) * conj(si1);
        d[mlast + mlast * ldd] =
            d[mlast + mlast * ldd] * co1 - tmp * conj(si1) * co1;

        // Update B and F
        tmp = -conj(b[mlast + mlast * ldb]);
        SLC_ZROT(&m, &f[mlast * ldf], &int1, &b[mlast * ldb], &int1, &co1, &si1);
        b[mlast + mlast * ldb] =
            b[mlast + mlast * ldb] * co1 + tmp * conj(si1) * conj(si1);
        f[mlast + mlast * ldf] =
            f[mlast + mlast * ldf] * co1 - tmp * conj(si1) * co1;

        if (lcmpq) {
            SLC_ZROT(&n, &q[(n - 1) * ldq], &int1, &q[mlast * ldq], &int1, &co1,
                     &si1);
        }

        // II. Move the eigenvalue in the M-th diagonal position to the (MM+1)-th position.
        mm++;
        i32 ic = 0;
        for (i32 j = m - 2; j >= mm - 1; j--) {
            mb03dz(&a[j + j * lda], lda, &b[j + j * ldb], ldb, &co1, &si1, &co2,
                   &si2);
            dwork[ic] = co2;
            zwork[ic] = -si2;
            ic++;

            // Update A and D
            i32 len_j = j;
            SLC_ZROT(&len_j, &a[(j + 1) * lda], &int1, &a[j * lda], &int1, &co1,
                     &si1);
            a[j + j * lda] =
                co2 * a[j + j * lda] + si2 * a[(j + 1) + (j + 1) * lda] * conj(si1);
            a[(j + 1) + (j + 1) * lda] = co1 * a[(j + 1) + (j + 1) * lda];

            cjf = -conj(d[j + (j + 1) * ldd]);
            tmp = co2 * cjf - conj(si2) * d[(j + 1) + (j + 1) * ldd];
            SLC_ZROT(&len_j, &d[(j + 1) * ldd], &int1, &d[j * ldd], &int1, &co2,
                     &si2);
            d[j + j * ldd] = co2 * d[j + j * ldd] - si2 * tmp;
            d[(j + 1) + (j + 1) * ldd] =
                co2 * d[(j + 1) + (j + 1) * ldd] + si2 * cjf;

            // Update B and F
            SLC_ZROT(&len_j, &b[(j + 1) * ldb], &int1, &b[j * ldb], &int1, &co1,
                     &si1);
            b[j + j * ldb] =
                co2 * b[j + j * ldb] + si2 * b[(j + 1) + (j + 1) * ldb] * conj(si1);
            b[(j + 1) + (j + 1) * ldb] = co1 * b[(j + 1) + (j + 1) * ldb];

            cjf = conj(f[j + (j + 1) * ldf]);
            tmp = co2 * cjf - conj(si2) * f[(j + 1) + (j + 1) * ldf];
            SLC_ZROT(&len_j, &f[(j + 1) * ldf], &int1, &f[j * ldf], &int1, &co2,
                     &si2);
            f[j + j * ldf] = co2 * f[j + j * ldf] - si2 * tmp;
            f[(j + 1) + (j + 1) * ldf] =
                co2 * f[(j + 1) + (j + 1) * ldf] + si2 * cjf;

            if (lcmpq) {
                SLC_ZROT(&n, &q[(j + 1) * ldq], &int1, &q[j * ldq], &int1, &co1,
                         &si1);
                SLC_ZROT(&n, &q[(m + j + 1) * ldq], &int1, &q[(m + j) * ldq],
                         &int1, &co2, &si2);
            }
        }

        // Panel Updates for A
        i32 ics = 0;
        i32 je = m - 2;
        while (je >= mm - 1) {
            i32 nc = 0;
            i32 ic_loc = ics;
            ics += nb;
            for (i32 j = je; j >= mm - 1; j--) {
                nc++;
                if (nc > nb) nc = nb;
                i32 js = je - nc + 1;
                SLC_ZROT(&nc, &a[j + (js + 1) * lda], &lda,
                         &a[(j + 1) + (js + 1) * lda], &lda,
                         &dwork[ic_loc], &zwork[ic_loc]);
                ic_loc++;
            }
            je -= nb;
        }

        // Panel Updates for B
        ics = 0;
        je = m - 2;
        while (je >= mm - 1) {
            i32 nc = 0;
            i32 ic_loc = ics;
            ics += nb;
            for (i32 j = je; j >= mm - 1; j--) {
                nc++;
                if (nc > nb) nc = nb;
                i32 js = je - nc + 1;
                SLC_ZROT(&nc, &b[j + (js + 1) * ldb], &ldb,
                         &b[(j + 1) + (js + 1) * ldb], &ldb,
                         &dwork[ic_loc], &zwork[ic_loc]);
                ic_loc++;
            }
            je -= nb;
        }

        // Panel Updates for D
        ics = 0;
        je = m - 2;
        while (je >= mm - 1) {
            i32 nc = 0;
            i32 ic_loc = ics;
            ics += nb;
            for (i32 j = je; j >= mm - 1; j--) {
                nc++;
                if (nc > nb) nc = nb;
                i32 js = je - nc + 1;
                SLC_ZROT(&nc, &d[j + (js + 1) * ldd], &ldd,
                         &d[(j + 1) + (js + 1) * ldd], &ldd,
                         &dwork[ic_loc], &zwork[ic_loc]);
                ic_loc++;
            }
            je -= nb;
        }

        // Panel Updates for F
        ics = 0;
        je = m - 2;
        while (je >= mm - 1) {
            i32 nc = 0;
            i32 ic_loc = ics;
            ics += nb;
            for (i32 j = je; j >= mm - 1; j--) {
                nc++;
                if (nc > nb) nc = nb;
                i32 js = je - nc + 1;
                SLC_ZROT(&nc, &f[j + (js + 1) * ldf], &ldf,
                         &f[(j + 1) + (js + 1) * ldf], &ldf,
                         &dwork[ic_loc], &zwork[ic_loc]);
                ic_loc++;
            }
            je -= nb;
        }
    }

    *neig = mm;
}
