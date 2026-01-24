/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>

void mb03dd(const char *uplo, i32 *n1, i32 *n2, f64 prec, f64 *a, i32 lda,
            f64 *b, i32 ldb, f64 *q1, i32 ldq1, f64 *q2, i32 ldq2,
            f64 *dwork, i32 ldwork, i32 *info) {

    const f64 zero = 0.0, one = 1.0, ten = 10.0, hund = 100.0;
    i32 int1 = 1, int0 = 0;

    bool ltriu = (*uplo == 'T' || *uplo == 't');
    bool luplo = (*uplo == 'U' || *uplo == 'u') || ltriu;

    *info = 0;

    i32 loc_n1 = *n1;
    i32 loc_n2 = *n2;
    i32 m = loc_n1 + loc_n2;

    if (m > 2) {
        if (!luplo) {
            f64 norm_a21 = SLC_DLANGE("1", &loc_n2, &loc_n1, &a[loc_n1], &lda, dwork);
            f64 norm_b21 = SLC_DLANGE("1", &loc_n2, &loc_n1, &b[loc_n1], &ldb, dwork);

            if (norm_a21 == zero && norm_b21 == zero) {
                if (loc_n1 == 2) {
                    i32 idum;
                    i32 bwork_val = 0;
                    i32 lwork_dgges = ldwork - 2 * m;
                    SLC_DGGES("V", "V", "N", sb02ow, &loc_n1, a, &lda, b, &ldb,
                              &idum, dwork, &dwork[m], &dwork[2 * m], q2, &ldq2,
                              q1, &ldq1, &dwork[3 * m], &lwork_dgges, &bwork_val, info);
                    if (*info != 0) {
                        if (*info >= 1 && *info <= loc_n1) {
                            *info = 3;
                            return;
                        } else {
                            *info = 4;
                            return;
                        }
                    }
                    if (loc_n2 == 1) {
                        q1[2 + 2 * ldq1] = one;
                        q2[2 + 2 * ldq2] = one;
                    }
                }
                if (loc_n2 == 2) {
                    i32 idum;
                    i32 bwork_val = 0;
                    i32 lwork_dgges = ldwork - 2 * m;
                    SLC_DGGES("V", "V", "N", sb02ow, &loc_n2,
                              &a[loc_n1 + loc_n1 * lda], &lda,
                              &b[loc_n1 + loc_n1 * ldb], &ldb,
                              &idum, &dwork[loc_n1], &dwork[m + loc_n1],
                              &dwork[2 * m + loc_n1], &q2[loc_n1 + loc_n1 * ldq2],
                              &ldq2, &q1[loc_n1 + loc_n1 * ldq1], &ldq1,
                              &dwork[3 * m], &lwork_dgges, &bwork_val, info);
                    if (*info != 0) {
                        if (*info >= 1 && *info <= loc_n2) {
                            *info = 3;
                            return;
                        } else {
                            *info = 4;
                            return;
                        }
                    }
                    if (loc_n1 == 1) {
                        q1[0] = one;
                        q2[0] = one;
                    }
                }
                SLC_DLASET("F", &loc_n2, &loc_n1, &zero, &zero, &q1[loc_n1], &ldq1);
                SLC_DLASET("F", &loc_n1, &loc_n2, &zero, &zero, &q1[loc_n1 * ldq1], &ldq1);
                SLC_DLASET("F", &loc_n2, &loc_n1, &zero, &zero, &q2[loc_n1], &ldq2);
                SLC_DLASET("F", &loc_n1, &loc_n2, &zero, &zero, &q2[loc_n1 * ldq2], &ldq2);
                return;
            }

            if (loc_n1 == 1) {
                f64 dum0 = a[0];
                f64 dum1 = a[1];
                a[0] = a[1 + lda];
                a[1] = a[2 + lda];
                a[lda] = a[1 + 2 * lda];
                a[1 + lda] = a[2 + 2 * lda];
                a[2 * lda] = dum1;
                a[1 + 2 * lda] = a[2];
                a[2 + 2 * lda] = dum0;
                a[2] = zero;
                a[2 + lda] = zero;

                dum0 = b[0];
                dum1 = b[1];
                b[0] = b[1 + ldb];
                b[1] = b[2 + ldb];
                b[ldb] = b[1 + 2 * ldb];
                b[1 + ldb] = b[2 + 2 * ldb];
                b[2 * ldb] = dum1;
                b[1 + 2 * ldb] = b[2];
                b[2 + 2 * ldb] = dum0;
                b[2] = zero;
                b[2 + ldb] = zero;
            } else if (loc_n2 == 1) {
                f64 dum0 = a[2 + lda];
                f64 dum1 = a[2 + 2 * lda];
                a[1 + 2 * lda] = a[lda];
                a[2 + 2 * lda] = a[1 + lda];
                a[1 + lda] = a[0];
                a[2 + lda] = a[1];
                a[0] = dum1;
                a[lda] = a[2];
                a[2 * lda] = dum0;
                a[1] = zero;
                a[2] = zero;

                dum0 = b[2 + ldb];
                dum1 = b[2 + 2 * ldb];
                b[1 + 2 * ldb] = b[ldb];
                b[2 + 2 * ldb] = b[1 + ldb];
                b[1 + ldb] = b[0];
                b[2 + ldb] = b[1];
                b[0] = dum1;
                b[ldb] = b[2];
                b[2 * ldb] = dum0;
                b[1] = zero;
                b[2] = zero;
            } else {
                for (i32 j = 0; j < loc_n1; j++) {
                    SLC_DSWAP(&loc_n1, &a[j * lda], &int1, &a[loc_n1 + (loc_n1 + j) * lda], &int1);
                    SLC_DSWAP(&loc_n1, &a[(loc_n1 + j) * lda], &int1, &a[loc_n1 + j * lda], &int1);
                    SLC_DSWAP(&loc_n1, &b[j * ldb], &int1, &b[loc_n1 + (loc_n1 + j) * ldb], &int1);
                    SLC_DSWAP(&loc_n1, &b[(loc_n1 + j) * ldb], &int1, &b[loc_n1 + j * ldb], &int1);
                }
            }

            i32 itmp = loc_n1;
            loc_n1 = loc_n2;
            loc_n2 = itmp;
        }

        i32 ievs = 3 * loc_n1;
        i32 iaev = ievs + 3 * loc_n1;

        if (loc_n1 == 1) {
            f64 sign_b = b[0] >= 0.0 ? one : -one;
            dwork[0] = a[0] * sign_b;
            dwork[1] = zero;
            dwork[2] = fabs(b[0]);
        } else {
            f64 sfmin = SLC_DLAMCH("S");
            q1[0] = a[0];
            q1[1] = a[1];
            q1[ldq1] = a[lda];
            q1[1 + ldq1] = a[1 + lda];
            q2[0] = b[0];
            q2[1] = b[1];
            q2[ldq2] = b[ldb];
            q2[1 + ldq2] = b[1 + ldb];

            if (!ltriu && b[1] != zero) {
                f64 a11 = fabs(q1[0]);
                f64 a22 = fabs(q1[1 + ldq1]);
                f64 b11 = fabs(q2[0]);
                f64 b22 = fabs(q2[1 + ldq2]);
                f64 mx = a11 + fabs(q1[1]);
                if (a22 + fabs(q1[ldq1]) > mx) mx = a22 + fabs(q1[ldq1]);
                if (b11 + fabs(q2[1]) > mx) mx = b11 + fabs(q2[1]);
                if (b22 + fabs(q2[ldq2]) > mx) mx = b22 + fabs(q2[ldq2]);
                if (sfmin > mx) mx = sfmin;

                q1[0] /= mx;
                q1[1] /= mx;
                q1[ldq1] /= mx;
                q1[1 + ldq1] /= mx;
                q2[0] /= mx;
                q2[1] /= mx;
                q2[ldq2] /= mx;
                q2[1 + ldq2] /= mx;

                f64 co, si, e, g;
                f64 co1, si1;
                SLC_DLARTG(&q2[0], &q2[1], &co, &si, &e);
                SLC_DLARTG(&q2[1 + ldq2], &q2[1], &co1, &si1, &g);

                if (fabs(co * b[1] - si * b[0]) <=
                    fabs(co1 * b[1] - si1 * b[1 + ldb])) {
                    SLC_DROT(&(i32){2}, &q1[0], &ldq1, &q1[1], &ldq1, &co, &si);
                    q2[0] = e;
                    f64 tmp = q2[ldq2];
                    q2[ldq2] = si * q2[1 + ldq2] + co * tmp;
                    q2[1 + ldq2] = co * q2[1 + ldq2] - si * tmp;
                } else {
                    SLC_DROT(&(i32){2}, &q1[ldq1], &int1, &q1[0], &int1, &co1, &si1);
                    q2[1 + ldq2] = g;
                    f64 tmp = q2[ldq2];
                    q2[ldq2] = si1 * q2[0] + co1 * tmp;
                    q2[0] = co1 * q2[0] - si1 * tmp;
                }
                q2[1] = zero;
            }

            i32 ldq1_loc = ldq1;
            i32 ldq2_loc = ldq2;
            SLC_DLAG2(q1, &ldq1_loc, q2, &ldq2_loc, &(f64){sfmin * hund},
                      &dwork[2 * loc_n1], &dwork[2 * loc_n1 + 1],
                      &dwork[0], &dwork[1], &dwork[loc_n1]);
            dwork[loc_n1 + 1] = -dwork[loc_n1];
        }

        i32 itmp_iaev = iaev + 3 * m;
        SLC_DCOPY(&(i32){3 * loc_n1}, dwork, &int1, &dwork[ievs], &int1);

        if (ltriu) {
            i32 lwork_dhgeqz = ldwork - itmp_iaev + 1;
            SLC_DHGEQZ("S", "I", "I", &m, &int1, &m, a, &lda, b, &ldb,
                       &dwork[iaev], &dwork[iaev + m], &dwork[iaev + 2 * m],
                       q2, &ldq2, q1, &ldq1, &dwork[itmp_iaev], &lwork_dhgeqz, info);
            if (*info >= 1 && *info <= m) {
                *info = 3;
                return;
            } else if (*info != 0) {
                *info = 4;
                return;
            }
        } else {
            i32 idum;
            i32 bwork_val = 0;
            i32 lwork_dgges = ldwork - itmp_iaev + 1;
            SLC_DGGES("V", "V", "N", sb02ow, &m, a, &lda, b, &ldb, &idum,
                      &dwork[iaev], &dwork[iaev + m], &dwork[iaev + 2 * m],
                      q2, &ldq2, q1, &ldq1, &dwork[itmp_iaev], &lwork_dgges,
                      &bwork_val, info);
            if (*info != 0) {
                if (*info >= 1 && *info <= m) {
                    *info = 3;
                    return;
                } else {
                    *info = 4;
                    return;
                }
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
                f64 aev_i = dwork[m + idx_i];
                f64 aev_b = dwork[2 * m + idx_i];

                bool aevinf = fabs(aev_b) < prec * (fabs(aev_r) + fabs(aev_i));

                for (i32 j = 0; j < loc_n1; j++) {
                    f64 ev_r = dwork[j];
                    f64 ev_i = dwork[loc_n1 + j];
                    f64 ev_b = dwork[2 * loc_n1 + j];

                    bool evinf = fabs(ev_b) < prec * (fabs(ev_r) + fabs(ev_i));

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
        i32 idm[2];
        f64 pl_val, pr_val;
        f64 dif_val[2];

        f64 nra = SLC_DLANHS("1", &m, a, &lda, dwork);
        f64 nrb = SLC_DLANHS("1", &m, b, &ldb, dwork);

        idm[0] = 2;
        idm[1] = 2;
        mb01qd('H', m, m, 0, 0, nra, one, 2, idm, a, lda, info);
        mb01qd('H', m, m, 0, 0, nrb, one, 2, idm, b, ldb, info);

        i32 wantq = 1, wantz = 1;
        SLC_DTGSEN(&int0, &wantq, &wantz, slct, &m, a, &lda, b, &ldb,
                   dwork, &dwork[m], &dwork[2 * m],
                   q2, &ldq2, q1, &ldq1, idm, &pl_val, &pr_val, dif_val,
                   &dwork[itmp_dtgsen], &lwork_dtgsen, idm, &liwork, info);

        free(slct);

        if (*info == 1) {
            *info = 5;
            return;
        }

        mb01qd('H', m, m, 0, 0, one, nra, 0, idm, a, lda, info);
        mb01qd('H', m, m, 0, 0, one, nrb, 0, idm, b, ldb, info);

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

                    tmp = q2[2 + j * ldq2];
                    q2[2 + j * ldq2] = q2[1 + j * ldq2];
                    q2[1 + j * ldq2] = q2[j * ldq2];
                    q2[j * ldq2] = tmp;
                }
            } else if (loc_n2 == 1) {
                for (i32 j = 0; j < m; j++) {
                    f64 tmp = q1[j * ldq1];
                    q1[j * ldq1] = q1[1 + j * ldq1];
                    q1[1 + j * ldq1] = q1[2 + j * ldq1];
                    q1[2 + j * ldq1] = tmp;

                    tmp = q2[j * ldq2];
                    q2[j * ldq2] = q2[1 + j * ldq2];
                    q2[1 + j * ldq2] = q2[2 + j * ldq2];
                    q2[2 + j * ldq2] = tmp;
                }
            } else {
                for (i32 j = 0; j < m; j++) {
                    SLC_DSWAP(&loc_n1, &q1[j * ldq1], &int1, &q1[loc_n1 + j * ldq1], &int1);
                    SLC_DSWAP(&loc_n1, &q2[j * ldq2], &int1, &q2[loc_n1 + j * ldq2], &int1);
                }
            }
        }
    } else {
        if (!luplo && a[1] == zero && b[1] == zero) {
            SLC_DLASET("F", &m, &m, &zero, &one, q1, &ldq1);
            SLC_DLASET("F", &m, &m, &zero, &one, q2, &ldq2);
            return;
        } else if (luplo) {
            SLC_DLASET("F", &m, &m, &zero, &one, q1, &ldq1);
            SLC_DLASET("F", &m, &m, &zero, &one, q2, &ldq2);
        } else {
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

            q1[0] = zero;
            q1[1] = -one;
            q1[ldq1] = one;
            q1[1 + ldq1] = zero;
            q2[0] = zero;
            q2[1] = -one;
            q2[ldq2] = one;
            q2[1 + ldq2] = zero;
        }

        f64 a11 = a[0];
        f64 a22 = a[1 + lda];
        f64 b11 = b[0];
        f64 b22 = b[1 + ldb];
        f64 sfmin = SLC_DLAMCH("S");
        f64 mx = fabs(a11);
        if (fabs(a22) > mx) mx = fabs(a22);
        if (fabs(a[lda]) > mx) mx = fabs(a[lda]);
        if (fabs(b11) > mx) mx = fabs(b11);
        if (fabs(b22) > mx) mx = fabs(b22);
        if (fabs(b[ldb]) > mx) mx = fabs(b[ldb]);
        if (sfmin > mx) mx = sfmin;

        f64 as[4], bs[4];
        as[0] = a11 / mx;
        as[1] = zero;
        as[2] = a[lda] / mx;
        as[3] = a22 / mx;
        bs[0] = b11 / mx;
        bs[1] = zero;
        bs[2] = b[ldb] / mx;
        bs[3] = b22 / mx;

        SLC_DLACPY("F", &m, &m, a, &lda, as, &(i32){2});
        SLC_DLACPY("F", &m, &m, b, &ldb, bs, &(i32){2});

        f64 dum[8];
        i32 itmp_info;
        i32 two = 2;
        SLC_DTGEX2(&(i32){1}, &(i32){1}, &m, as, &two, bs, &two, q2, &ldq2,
                   q1, &ldq1, &int1, &int1, &int1, dum, &(i32){8}, &itmp_info);
    }

    *n1 = loc_n1;
    *n2 = loc_n2;
    *info = 0;
}
