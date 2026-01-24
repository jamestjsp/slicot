/*
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

void mb03cd(const char *uplo, i32 *n1, i32 *n2, f64 prec, f64 *a, i32 lda,
            f64 *b, i32 ldb, f64 *d, i32 ldd, f64 *q1, i32 ldq1, f64 *q2,
            i32 ldq2, f64 *q3, i32 ldq3, f64 *dwork, i32 ldwork, i32 *info) {

    const f64 zero = 0.0, one = 1.0, ten = 10.0, hund = 100.0;
    i32 int1 = 1, int0 = 0;

    bool luplo = (*uplo == 'U' || *uplo == 'u');
    *info = 0;

    i32 loc_n1 = *n1;
    i32 loc_n2 = *n2;
    i32 m = loc_n1 + loc_n2;

    if (m > 2) {
        i32 ievs = 3 * loc_n1;
        i32 iaev = ievs + 3 * loc_n1;

        if (luplo) {
            SLC_DGEMM("N", "N", &loc_n1, &loc_n1, &loc_n1, &one, a, &lda, b, &ldb, &zero, q2, &ldq2);
            SLC_DLASET("F", &loc_n2, &loc_n1, &zero, &zero, &q2[loc_n1], &ldq2);
            SLC_DGEMM("N", "N", &loc_n1, &loc_n2, &m, &one, a, &lda, &b[loc_n1 * ldb], &ldb, &zero, &q2[loc_n1 * ldq2], &ldq2);
            SLC_DGEMM("N", "N", &loc_n2, &loc_n2, &loc_n2, &one, &a[loc_n1 + loc_n1 * lda], &lda, &b[loc_n1 + loc_n1 * ldb], &ldb, &zero, &q2[loc_n1 + loc_n1 * ldq2], &ldq2);
        } else {
            SLC_DGEMM("N", "N", &loc_n2, &loc_n2, &loc_n2, &one, &a[loc_n1 + loc_n1 * lda], &lda, &b[loc_n1 + loc_n1 * ldb], &ldb, &zero, q2, &ldq2);
            SLC_DLASET("F", &loc_n1, &loc_n2, &zero, &zero, &q2[loc_n2], &ldq2);
            SLC_DGEMM("N", "N", &loc_n2, &loc_n1, &m, &one, &a[loc_n1], &lda, b, &ldb, &zero, &q2[loc_n2 * ldq2], &ldq2);
            SLC_DGEMM("N", "N", &loc_n1, &loc_n1, &loc_n1, &one, a, &lda, b, &ldb, &zero, &q2[loc_n2 + loc_n2 * ldq2], &ldq2);

            if (loc_n1 == 1) {
                f64 dum0 = d[0];
                f64 dum1 = d[1];
                d[0] = d[1 + ldd];
                d[1] = d[2 + ldd];
                d[ldd] = d[1 + 2 * ldd];
                d[1 + ldd] = d[2 + 2 * ldd];
                d[2 * ldd] = dum1;
                d[1 + 2 * ldd] = d[2];
                d[2 + 2 * ldd] = dum0;
                d[2] = zero;
                d[2 + ldd] = zero;
            } else if (loc_n2 == 1) {
                f64 dum0 = d[2 + ldd];
                f64 dum1 = d[2 + 2 * ldd];
                d[1 + 2 * ldd] = d[ldd];
                d[2 + 2 * ldd] = d[1 + ldd];
                d[1 + ldd] = d[0];
                d[2 + ldd] = d[1];
                d[0] = dum1;
                d[ldd] = d[2];
                d[2 * ldd] = dum0;
                d[1] = zero;
                d[2] = zero;
            } else {
                for (i32 j = 0; j < loc_n1; j++) {
                    SLC_DSWAP(&loc_n1, &d[j * ldd], &int1, &d[loc_n1 + (loc_n1 + j) * ldd], &int1);
                    SLC_DSWAP(&loc_n1, &d[(loc_n1 + j) * ldd], &int1, &d[loc_n1 + j * ldd], &int1);
                }
            }

            i32 itmp = loc_n1;
            loc_n1 = loc_n2;
            loc_n2 = itmp;

            ievs = 3 * loc_n1;
            iaev = ievs + 3 * loc_n1;
        }

        SLC_DLACPY("F", &m, &m, d, &ldd, q1, &ldq1);
        SLC_DLACPY("F", &m, &m, q2, &ldq2, q3, &ldq3);

        f64 dum[2];
        i32 lwork_dggev = ldwork - ievs;
        SLC_DGGEV("N", "N", &loc_n1, q1, &ldq1, q3, &ldq3,
                  dwork, &dwork[loc_n1], &dwork[2 * loc_n1],
                  dum, &int1, dum, &int1,
                  &dwork[ievs], &lwork_dggev, info);

        if (*info >= 1 && *info <= loc_n1) {
            *info = 1;
            return;
        } else if (*info > loc_n1) {
            *info = 2;
            return;
        }

        i32 itmp_iaev = iaev + 3 * m;
        SLC_DCOPY(&(i32){3 * loc_n1}, dwork, &int1, &dwork[ievs], &int1);

        i32 bwork_val = 0;
        i32 idum = 0;
        i32 lwork_dgges = ldwork - itmp_iaev;
        SLC_DGGES("V", "V", "N", sb02ow, &m, d, &ldd, q2, &ldq2, &idum,
                  &dwork[iaev], &dwork[iaev + m], &dwork[iaev + 2 * m],
                  q3, &ldq3, q1, &ldq1,
                  &dwork[itmp_iaev], &lwork_dgges, &bwork_val, info);

        if (*info != 0) {
            if (*info >= 1 && *info <= m) {
                *info = 3;
                return;
            } else if (*info != m + 2) {
                *info = 4;
                return;
            } else {
                *info = 0;
            }
        }

        f64 tol = prec;
        f64 tolb = ten * prec;
        i32 evsel = 0;

        i32 *slct = (i32 *)malloc(4 * sizeof(i32));
        bool out[2];

        for (i32 i = 0; i < m; i++) {
            slct[i] = 1;
        }

        while (evsel == 0) {
            i32 cnt = 0;
            out[0] = false;
            out[1] = false;

            for (i32 i = 0; i < m; i++) {
                i32 idx_i = iaev + i;
                f64 aev_r = dwork[idx_i];
                f64 aev_i = dwork[iaev + m + i];
                f64 aev_b = dwork[iaev + 2 * m + i];

                bool aevinf = fabs(aev_b) < prec * (fabs(aev_r) + fabs(aev_i));
                if (fabs(aev_b) == 0.0) aevinf = true;

                for (i32 j = 0; j < loc_n1; j++) {
                    f64 ev_r = dwork[j];
                    f64 ev_i = dwork[loc_n1 + j];
                    f64 ev_b = dwork[2 * loc_n1 + j];

                    bool evinf = fabs(ev_b) < prec * (fabs(ev_r) + fabs(ev_i));
                    if (fabs(ev_b) == 0.0) evinf = true;

                    if ((!evinf || aevinf) && (!aevinf || evinf) && !out[j]) {
                        if (!evinf || !aevinf) {
                            f64 adif = fabs(ev_r / ev_b - aev_r / aev_b) +
                                       fabs(ev_i / ev_b - aev_i / aev_b);
                            f64 absev = fabs(ev_r / ev_b) + fabs(ev_i / ev_b);
                            f64 absaev = fabs(aev_r / aev_b) + fabs(aev_i / aev_b);
                            f64 maxval = tolb;
                            if (absev > maxval) maxval = absev;
                            if (absaev > maxval) maxval = absaev;

                            if (adif <= tol * maxval) {
                                slct[i] = 0;
                                out[j] = true;
                                cnt++;
                            }
                        } else {
                            slct[i] = 0;
                            out[j] = true;
                            cnt++;
                        }
                    }
                }
            }

            if (cnt == loc_n1) {
                evsel = 1;
            } else {
                tol = ten * tol;
                SLC_DCOPY(&(i32){3 * loc_n1}, &dwork[ievs], &int1, dwork, &int1);
            }
        }

        i32 itmp_dtgsen = 3 * m;
        i32 lwork_dtgsen = ldwork - itmp_dtgsen;
        i32 liwork = 1;
        i32 idm[1];
        f64 pl_val, pr_val;
        f64 dif_val[2];

        i32 wantq = 1, wantz = 1;
        SLC_DTGSEN(&int0, &wantq, &wantz, slct, &m, d, &ldd, q2, &ldq2,
                   dwork, &dwork[m], &dwork[2 * m],
                   q3, &ldq3, q1, &ldq1, idm, &pl_val, &pr_val, dif_val,
                   &dwork[itmp_dtgsen], &lwork_dtgsen, idm, &liwork, info);

        free(slct);

        if (*info == 1) {
            *info = 5;
            return;
        }

        i32 new_n1 = loc_n1;
        i32 new_n2 = loc_n2;
        loc_n1 = new_n2;
        loc_n2 = new_n1;

        if (!luplo) {
            if (loc_n1 == 1) {
                for (i32 j = 0; j < m; j++) {
                    f64 tmp = q1[2 + j * ldq1];
                    q1[2 + j * ldq1] = q1[1 + j * ldq1];
                    q1[1 + j * ldq1] = q1[j * ldq1];
                    q1[j * ldq1] = tmp;

                    tmp = q3[2 + j * ldq3];
                    q3[2 + j * ldq3] = q3[1 + j * ldq3];
                    q3[1 + j * ldq3] = q3[j * ldq3];
                    q3[j * ldq3] = tmp;
                }
            } else if (loc_n2 == 1) {
                for (i32 j = 0; j < m; j++) {
                    f64 tmp = q1[j * ldq1];
                    q1[j * ldq1] = q1[1 + j * ldq1];
                    q1[1 + j * ldq1] = q1[2 + j * ldq1];
                    q1[2 + j * ldq1] = tmp;

                    tmp = q3[j * ldq3];
                    q3[j * ldq3] = q3[1 + j * ldq3];
                    q3[1 + j * ldq3] = q3[2 + j * ldq3];
                    q3[2 + j * ldq3] = tmp;
                }
            } else {
                for (i32 j = 0; j < m; j++) {
                    SLC_DSWAP(&loc_n1, &q1[j * ldq1], &int1, &q1[loc_n1 + j * ldq1], &int1);
                    SLC_DSWAP(&loc_n1, &q3[j * ldq3], &int1, &q3[loc_n1 + j * ldq3], &int1);
                }
            }
        }

        if (luplo) {
            SLC_DGEMM("N", "N", &loc_n2, &m, &m, &one, b, &ldb, q1, &ldq1, &zero, q2, &ldq2);
            SLC_DGEMM("N", "N", &loc_n1, &m, &loc_n1, &one, &b[loc_n2 + loc_n2 * ldb], &ldb, &q1[loc_n2], &ldq1, &zero, &q2[loc_n2], &ldq2);
        } else {
            SLC_DGEMM("N", "N", &loc_n1, &m, &loc_n1, &one, b, &ldb, q1, &ldq1, &zero, q2, &ldq2);
            SLC_DGEMM("N", "N", &loc_n2, &m, &m, &one, &b[loc_n1], &ldb, q1, &ldq1, &zero, &q2[loc_n1], &ldq2);
        }

        SLC_DGEQR2(&m, &m, q2, &ldq2, dwork, &dwork[m], info);
        SLC_DORG2R(&m, &m, &m, q2, &ldq2, dwork, &dwork[m], info);

    } else {
        if (!luplo) {
            f64 tmp = a[0];
            a[0] = a[1 + lda];
            a[1 + lda] = tmp;
            a[lda] = -a[1];
            a[1] = zero;

            tmp = b[0];
            b[0] = b[1 + ldb];
            b[1 + ldb] = tmp;
            b[ldb] = -b[1];
            b[1] = zero;

            tmp = d[0];
            d[0] = d[1 + ldd];
            d[1 + ldd] = tmp;
            d[ldd] = -d[1];
            d[1] = zero;
        }

        f64 tmp_val = a[1 + lda] * b[1 + ldb] * d[0];
        f64 g = a[0] * b[0] * d[1 + ldd] - tmp_val;

        if (fabs(g) < hund * prec * fabs(tmp_val)) {
            if (luplo) {
                SLC_DLASET("F", &m, &m, &zero, &one, q1, &ldq1);
                SLC_DLASET("F", &m, &m, &zero, &one, q2, &ldq2);
                SLC_DLASET("F", &m, &m, &zero, &one, q3, &ldq3);
            } else {
                f64 neg_one = -one;
                q1[0] = zero;
                q1[1] = neg_one;
                q1[ldq1] = one;
                q1[1 + ldq1] = zero;

                q2[0] = zero;
                q2[1] = neg_one;
                q2[ldq2] = one;
                q2[1 + ldq2] = zero;

                q3[0] = zero;
                q3[1] = neg_one;
                q3[ldq3] = one;
                q3[1 + ldq3] = zero;
            }
        } else {
            f64 e, co1, si1, co2, si2, co3, si3, r_tmp;

            e = (a[0] * b[lda] + a[lda] * b[1 + ldb]) * d[1 + ldd]
                - a[1 + lda] * b[1 + ldb] * d[ldd];
            SLC_DLARTG(&e, &g, &co1, &si1, &r_tmp);

            e = (a[lda] * d[1 + ldd] - a[1 + lda] * d[ldd]) * b[0]
                + a[1 + lda] * d[0] * b[lda];
            SLC_DLARTG(&e, &g, &co2, &si2, &r_tmp);

            e = (b[lda] * d[0] - b[0] * d[ldd]) * a[0]
                + a[lda] * b[1 + ldb] * d[0];
            SLC_DLARTG(&e, &g, &co3, &si3, &r_tmp);

            if (luplo) {
                q1[0] = co1;
                q1[1] = -si1;
                q1[ldq1] = si1;
                q1[1 + ldq1] = co1;

                q2[0] = co2;
                q2[1] = -si2;
                q2[ldq2] = si2;
                q2[1 + ldq2] = co2;

                q3[0] = co3;
                q3[1] = -si3;
                q3[ldq3] = si3;
                q3[1 + ldq3] = co3;
            } else {
                q1[0] = -si1;
                q1[1] = -co1;
                q1[ldq1] = co1;
                q1[1 + ldq1] = -si1;

                q2[0] = -si2;
                q2[1] = -co2;
                q2[ldq2] = co2;
                q2[1 + ldq2] = -si2;

                q3[0] = -si3;
                q3[1] = -co3;
                q3[ldq3] = co3;
                q3[1 + ldq3] = -si3;
            }
        }
    }

    *n1 = loc_n1;
    *n2 = loc_n2;

    *info = 0;
}
