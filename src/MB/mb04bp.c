/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 *
 * MB04BP - Eigenvalues of skew-Hamiltonian/Hamiltonian pencil (block algorithm)
 *
 * Purpose:
 *   Computes the eigenvalues of a real N-by-N skew-Hamiltonian/Hamiltonian
 *   pencil aS - bH using a block algorithm for efficiency on large matrices.
 *   For small N, delegates to MB04BD directly.
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <ctype.h>
#include <math.h>
#include <complex.h>
#include <stdlib.h>
#include <string.h>

#define NX 250

void mb04bp(const char *job, const char *compq1, const char *compq2,
            i32 n, f64 *a, i32 lda, f64 *de, i32 ldde, f64 *c1, i32 ldc1,
            f64 *vw, i32 ldvw, f64 *q1, i32 ldq1, f64 *q2, i32 ldq2,
            f64 *b, i32 ldb, f64 *f, i32 ldf, f64 *c2, i32 ldc2,
            f64 *alphar, f64 *alphai, f64 *beta,
            i32 *iwork, i32 liwork, f64 *dwork, i32 ldwork, i32 *info)
{
    const f64 ZERO = 0.0;
    const f64 HALF = 0.5;
    const f64 ONE = 1.0;
    const f64 TWO = 2.0;
    const f64 FIVE = 5.0;

    i32 nb = *info;
    i32 m = n / 2;
    i32 mm = m * m;
    i32 m1 = m > 1 ? m : 1;

    char job_upper = (char)toupper((unsigned char)job[0]);
    char cq1_upper = (char)toupper((unsigned char)compq1[0]);
    char cq2_upper = (char)toupper((unsigned char)compq2[0]);

    bool ltri = (job_upper == 'T');
    bool liniq1 = (cq1_upper == 'I');
    bool liniq2 = (cq2_upper == 'I');
    bool lupdq1 = (cq1_upper == 'U');
    bool lupdq2 = (cq2_upper == 'U');
    bool lcmpq1 = lupdq1 || liniq1;
    bool lcmpq2 = lupdq2 || liniq2;

    *info = 0;

    if (!(job_upper == 'E' || ltri)) {
        *info = -1;
    } else if (!(cq1_upper == 'N' || lcmpq1)) {
        *info = -2;
    } else if (!(cq2_upper == 'N' || lcmpq2)) {
        *info = -3;
    } else if ((liniq2 && !liniq1) || (lupdq2 && !lupdq1)) {
        *info = -3;
    } else if (n < 0 || (n % 2) != 0) {
        *info = -4;
    } else if (lda < m1) {
        *info = -6;
    } else if (ldde < m1) {
        *info = -8;
    } else if (ldc1 < m1) {
        *info = -10;
    } else if (ldvw < m1) {
        *info = -12;
    } else if (ldq1 < 1 || (lcmpq1 && ldq1 < n)) {
        *info = -14;
    } else if (ldq2 < 1 || (lcmpq2 && ldq2 < n)) {
        *info = -16;
    } else if (ldb < m1) {
        *info = -18;
    } else if (ldf < m1) {
        *info = -20;
    } else if (ldc2 < m1) {
        *info = -22;
    } else if (liwork < n + 12) {
        *info = -27;
    } else {
        i32 wsize;
        if ((m % 2) == 0) {
            wsize = (4 * n > 32 ? 4 * n : 32) + 4;
        } else {
            wsize = (4 * n > 36 ? 4 * n : 36);
        }
        i32 optdw;
        if (ltri || lcmpq1 || lcmpq2) {
            optdw = 8 * mm + wsize;
        } else {
            optdw = 4 * mm + wsize;
        }
        if (ldwork < optdw) {
            *info = -29;
        }
    }

    if (*info != 0) {
        return;
    }

    if (n == 0) {
        iwork[0] = 0;
        dwork[0] = FIVE;
        dwork[1] = ZERO;
        dwork[2] = ZERO;
        dwork[3] = ZERO;
        dwork[4] = ZERO;
        return;
    }

    if (nb == 0 || (nb < 0 && n <= NX)) {
        mb04bd(job, compq1, compq2, n, a, lda, de, ldde, c1, ldc1, vw, ldvw,
               q1, ldq1, q2, ldq2, b, ldb, f, ldf, c2, ldc2,
               alphar, alphai, beta, iwork, liwork, dwork, ldwork, info);
        return;
    } else if (nb < 0) {
        f64 work_query;
        i32 info_qr = 0;
        i32 neg1 = -1;
        SLC_DGEQRF(&m, &m, a, &lda, dwork, &work_query, &neg1, &info_qr);
        i32 opt_nb = (i32)(work_query) / m1;
        nb = opt_nb < 2 ? 2 : opt_nb;
        nb = nb > m ? m : nb;
    }

    f64 base = SLC_DLAMCH("Base");
    i32 emin = (i32)SLC_DLAMCH("Minimum Exponent");
    i32 emax = (i32)SLC_DLAMCH("Largest Exponent");

    i32 ninf = 0;
    f64 temp;
    if (m == 1) {
        temp = ZERO;
    } else {
        i32 m_1 = m - 1;
        temp = SLC_DLANTR("Max", "Lower", "No-diag", &m_1, &m_1, &de[1], &ldde, dwork) +
               SLC_DLANTR("Max", "Upper", "No-diag", &m_1, &m_1, &de[2 * ldde], &ldde, dwork);
    }

    if (temp == ZERO) {
        if (m == 1) {
            if (a[0] == ZERO) {
                ninf = 1;
            }
        } else {
            i32 m_1 = m - 1;
            f64 t1 = SLC_DLANTR("Max", "Lower", "No-diag", &m_1, &m_1, &a[1], &lda, dwork);
            f64 t2 = SLC_DLANTR("Max", "Upper", "No-diag", &m_1, &m_1, &a[lda], &lda, dwork);
            if (t1 == ZERO && t2 == ZERO) {
                for (i32 j = 0; j < m; j++) {
                    if (a[j + j * lda] == ZERO) {
                        ninf++;
                    }
                }
            } else {
                i32 ii, jj;
                ma02pd(m, m, a, lda, &ii, &jj);
                ninf = (ii > jj ? ii : jj) / 2;
            }
        }
    } else {
        ninf = ma02od("Skew", m, a, lda, de, ldde);
        if ((ninf % 2) > 0) {
            ninf++;
        }
        ninf = ninf / 2;
    }

    if (liniq1) {
        SLC_DLASET("Full", &n, &n, &ZERO, &ONE, q1, &ldq1);
    }

    f64 dum[4] = {ZERO, ZERO, ZERO, ZERO};
    f64 nu, mu, co, si, tmp1, tmp2;
    i32 int1 = 1;
    i32 int0 = 0;

    for (i32 k = 0; k < m - 1; k++) {
        i32 mk2 = k + 2 < m ? k + 2 : m;
        i32 mk3 = mk2 + 1;
        i32 len_h = m - k - 1;

        tmp1 = de[(k + 1) + k * ldde];
        SLC_DLARFG(&len_h, &tmp1, &de[mk2 + k * ldde], &int1, &nu);

        if (nu != ZERO) {
            de[(k + 1) + k * ldde] = ONE;

            i32 info_mb01md;
            mb01md('L', m - k - 1, nu, &de[(k + 1) + (k + 1) * ldde], ldde,
                   &de[(k + 1) + k * ldde], 1, ZERO, dwork, 1, &info_mb01md);

            mu = -HALF * nu * SLC_DDOT(&len_h, dwork, &int1, &de[(k + 1) + k * ldde], &int1);
            SLC_DAXPY(&len_h, &mu, &de[(k + 1) + k * ldde], &int1, dwork, &int1);

            i32 info_mb01nd;
            mb01nd('L', m - k - 1, ONE, &de[(k + 1) + k * ldde], 1, dwork, 1,
                   &de[(k + 1) + (k + 1) * ldde], ldde, &info_mb01nd);

            i32 kk = k + 1;
            SLC_DLARF("Left", &len_h, &kk, &de[(k + 1) + k * ldde], &int1, &nu,
                      &vw[k + 1], &ldvw, dwork);

            i32 len_w = m - k - 1;
            SLC_DSYMV("Lower", &len_w, &nu, &vw[(k + 1) + (k + 1) * ldvw], &ldvw,
                      &de[(k + 1) + k * ldde], &int1, &ZERO, dwork, &int1);

            mu = -HALF * nu * SLC_DDOT(&len_w, dwork, &int1, &de[(k + 1) + k * ldde], &int1);
            SLC_DAXPY(&len_w, &mu, &de[(k + 1) + k * ldde], &int1, dwork, &int1);

            f64 neg1 = -ONE;
            SLC_DSYR2("Lower", &len_w, &neg1, &de[(k + 1) + k * ldde], &int1, dwork, &int1,
                      &vw[(k + 1) + (k + 1) * ldvw], &ldvw);

            i32 len_r = m - k - 1;
            SLC_DLARF("Right", &m, &len_r, &de[(k + 1) + k * ldde], &int1, &nu,
                      &a[(k + 1) * lda], &lda, dwork);
            SLC_DLARF("Right", &m, &len_r, &de[(k + 1) + k * ldde], &int1, &nu,
                      &c1[(k + 1) * ldc1], &ldc1, dwork);

            if (lcmpq1) {
                SLC_DLARF("Right", &n, &len_r, &de[(k + 1) + k * ldde], &int1, &nu,
                          &q1[(m + k + 1) * ldq1], &ldq1, dwork);
            }
            de[(k + 1) + k * ldde] = tmp1;
        }

        tmp2 = a[(k + 1) + k * lda];
        SLC_DLARTG(&tmp2, &de[(k + 1) + k * ldde], &co, &si, &a[(k + 1) + k * lda]);

        i32 len_rot = m - k - 2;
        if (len_rot > 0) {
            SLC_DROT(&len_rot, &de[mk2 + (k + 1) * ldde], &int1, &a[(k + 1) + mk2 * lda], &lda, &co, &si);
        }
        i32 kk = k + 1;
        SLC_DROT(&kk, &a[(k + 1) * lda], &int1, &de[(k + 2) * ldde], &int1, &co, &si);
        if (len_rot > 0) {
            SLC_DROT(&len_rot, &de[(k + 1) + mk3 * ldde], &ldde, &a[mk2 + (k + 1) * lda], &int1, &co, &si);
        }

        f64 neg_si = -si;
        SLC_DROT(&kk, &vw[(k + 1)], &ldvw, &c1[(k + 1)], &ldc1, &co, &neg_si);
        if (len_rot > 0) {
            SLC_DROT(&len_rot, &vw[mk2 + (k + 1) * ldvw], &int1, &c1[(k + 1) + mk2 * ldc1], &ldc1, &co, &neg_si);
        }
        SLC_DROT(&kk, &c1[(k + 1) * ldc1], &int1, &vw[(k + 2) * ldvw], &int1, &co, &si);
        if (len_rot > 0) {
            SLC_DROT(&len_rot, &vw[(k + 1) + mk3 * ldvw], &ldvw, &c1[mk2 + (k + 1) * ldc1], &int1, &co, &neg_si);
        }

        tmp1 = c1[(k + 1) + (k + 1) * ldc1];
        tmp2 = vw[(k + 1) + (k + 2) * ldvw];
        c1[(k + 1) + (k + 1) * ldc1] = (co - si) * (co + si) * tmp1 +
                                        co * si * (vw[(k + 1) + (k + 1) * ldvw] + tmp2);
        tmp1 = TWO * co * si * tmp1;
        vw[(k + 1) + (k + 2) * ldvw] = co * co * tmp2 - si * si * vw[(k + 1) + (k + 1) * ldvw] - tmp1;
        vw[(k + 1) + (k + 1) * ldvw] = co * co * vw[(k + 1) + (k + 1) * ldvw] - si * si * tmp2 - tmp1;

        if (lcmpq1) {
            SLC_DROT(&n, &q1[(k + 1) * ldq1], &int1, &q1[(m + k + 1) * ldq1], &int1, &co, &si);
        }

        i32 len_p = m - k;
        tmp1 = a[k + k * lda];
        SLC_DLARFG(&len_p, &tmp1, &a[(k + 1) + k * lda], &int1, &nu);

        if (nu != ZERO) {
            a[k + k * lda] = ONE;

            i32 len_l = m - k;
            i32 len_r2 = m - k - 1;
            SLC_DLARF("Left", &len_l, &len_r2, &a[k + k * lda], &int1, &nu,
                      &a[k + (k + 1) * lda], &lda, dwork);

            if (k > 0) {
                SLC_DLARF("Right", &k, &len_l, &a[k + k * lda], &int1, &nu,
                          &de[(k + 1) * ldde], &ldde, dwork);
            }

            i32 info_mb01md2;
            mb01md('U', m - k, nu, &de[k + (k + 1) * ldde], ldde,
                   &a[k + k * lda], 1, ZERO, dwork, 1, &info_mb01md2);

            mu = -HALF * nu * SLC_DDOT(&len_l, dwork, &int1, &a[k + k * lda], &int1);
            SLC_DAXPY(&len_l, &mu, &a[k + k * lda], &int1, dwork, &int1);

            i32 info_mb01nd2;
            mb01nd('U', m - k, ONE, &a[k + k * lda], 1, dwork, 1,
                   &de[k + (k + 1) * ldde], ldde, &info_mb01nd2);

            SLC_DLARF("Left", &len_l, &m, &a[k + k * lda], &int1, &nu,
                      &c1[k], &ldc1, dwork);

            if (k > 0) {
                SLC_DLARF("Right", &k, &len_l, &a[k + k * lda], &int1, &nu,
                          &vw[(k + 1) * ldvw], &ldvw, dwork);
            }

            SLC_DSYMV("Upper", &len_l, &nu, &vw[k + (k + 1) * ldvw], &ldvw,
                      &a[k + k * lda], &int1, &ZERO, dwork, &int1);

            mu = -HALF * nu * SLC_DDOT(&len_l, dwork, &int1, &a[k + k * lda], &int1);
            SLC_DAXPY(&len_l, &mu, &a[k + k * lda], &int1, dwork, &int1);

            f64 neg1 = -ONE;
            SLC_DSYR2("Upper", &len_l, &neg1, &a[k + k * lda], &int1, dwork, &int1,
                      &vw[k + (k + 1) * ldvw], &ldvw);

            if (lcmpq1) {
                SLC_DLARF("Right", &n, &len_l, &a[k + k * lda], &int1, &nu,
                          &q1[k * ldq1], &ldq1, dwork);
            }
            a[k + k * lda] = tmp1;
        }

        i32 len_z = m - k - 1;
        SLC_DCOPY(&len_z, dum, &int0, &a[(k + 1) + k * lda], &int1);
    }

    i32 imat = 4 * mm;
    SLC_DLACPY("Full", &m, &m, a, &lda, b, &ldb);
    SLC_DLACPY("Upper", &m, &m, &de[ldde], &ldde, f, &ldf);

    for (i32 j = 0; j < m; j++) {
        for (i32 i = 0; i < m; i++) {
            c2[i + j * ldc2] = -c1[i + j * ldc1];
        }
    }

    SLC_DLACPY("Lower", &m, &m, vw, &ldvw, dwork, &m);
    ma02ad("Lower", m, m, vw, ldvw, dwork, m);
    ma02ad("Upper", m, m, &vw[ldvw], ldvw, &vw[ldvw], ldvw);

    if (lcmpq2) {
        SLC_DLACPY("Full", &m, &m, &q1[(m + 1) * ldq1 + m], &ldq1, q2, &ldq2);

        for (i32 j = 0; j < m; j++) {
            for (i32 i = m; i < n; i++) {
                q2[i + j * ldq2] = -q1[(i - m) + (j + m) * ldq1];
            }
        }
        for (i32 j = m; j < n; j++) {
            for (i32 i = 0; i < m; i++) {
                q2[i + j * ldq2] = -q1[(i + m) + (j - m) * ldq1];
            }
        }
        SLC_DLACPY("Full", &m, &m, q1, &ldq1, &q2[m + m * ldq2], &ldq2);
    }

    i32 jb, ic, jc, n1, n2, n3, nc, ja;
    i32 js, je;

    for (i32 k = 0; k < m; k++) {
        i32 mk1 = k + 1 < m ? k + 1 : m;

        js = k;
        je = (k + nb - 1) < m ? (k + nb - 1) : (m - 1);
        jb = je;
        ic = imat;
        jc = 2 * (m - k - 1) + imat;
        n1 = (m - k - 1) < (je - js + 1) ? (m - k - 1) : (je - js + 1);
        n2 = (m - k) < (je - js + 1) ? (m - k) : (je - js + 1);
        n3 = m < nb ? m : nb;

        for (i32 j = k; j < m - 1; j++) {
            i32 mj3 = j + 3 < m + 1 ? j + 3 : m + 1;

            SLC_DLARTG(&dwork[(k) * m + j + 1], &dwork[(k) * m + j], &co, &si, &tmp1);
            dwork[ic] = co;
            dwork[ic + 1] = si;
            ic += 2;

            SLC_DROT(&m, &c2[j + 1], &ldc2, &c2[j], &ldc2, &co, &si);
            dwork[(k) * m + j + 1] = tmp1;
            dwork[(k) * m + j] = ZERO;
            if (n1 > 0) {
                SLC_DROT(&n1, &dwork[(k + 1) * m + j + 1], &m, &dwork[(k + 1) * m + j], &m, &co, &si);
            }

            if (j == je) {
                js = je + 1;
                je = js + nb - 1 < m - 1 ? js + nb - 1 : m - 1;
                nc = je - js + 1;
                ja = 2 * (m - k - 1) + imat;
                for (i32 ii = k; ii < j; ii++) {
                    if (nc > 0) {
                        SLC_DROT(&nc, &a[ii + js * lda], &lda, &a[ii + 1 + js * lda], &lda,
                                 &dwork[ja], &dwork[ja + 1]);
                    }
                    ja += 2;
                }
                ja = 2 * (m - k - 1) + imat;
                for (i32 ii = k; ii < j; ii++) {
                    if (nc > 0) {
                        SLC_DROT(&nc, &de[ii + (js + 1) * ldde], &ldde, &de[ii + 1 + (js + 1) * ldde], &ldde,
                                 &dwork[ja], &dwork[ja + 1]);
                    }
                    ja += 2;
                }
            }

            i32 jj = j + 1;
            SLC_DROT(&jj, &a[j + 1], &lda, &a[j], &lda, &co, &si);
            tmp1 = -si * a[(j + 1) + (j + 1) * lda];
            a[(j + 1) + (j + 1) * lda] = co * a[(j + 1) + (j + 1) * lda];

            if (lcmpq1) {
                SLC_DROT(&n, &q1[(m + j + 1) * ldq1], &int1, &q1[(m + j) * ldq1], &int1, &co, &si);
            }

            SLC_DLARTG(&a[j + j * lda], &tmp1, &co, &si, &tmp2);
            dwork[jc] = co;
            dwork[jc + 1] = si;
            jc += 2;

            nc = je - j < je - js + 1 ? je - j : je - js + 1;
            a[j + j * lda] = tmp2;
            if (nc > 0) {
                SLC_DROT(&nc, &a[j + (j + 1) * lda], &lda, &a[(j + 1) + (j + 1) * lda], &lda, &co, &si);
            }
            if (j > 0) {
                SLC_DROT(&j, &de[(j + 1) * ldde], &int1, &de[(j + 2) * ldde], &int1, &co, &si);
            }
            i32 nc2 = nc - 1;
            if (nc2 > 0) {
                SLC_DROT(&nc2, &de[j + mj3 * ldde], &ldde, &de[(j + 1) + mj3 * ldde], &ldde, &co, &si);
            }

            SLC_DROT(&n2, &c1[j + k * ldc1], &ldc1, &c1[(j + 1) + k * ldc1], &ldc1, &co, &si);
            SLC_DROT(&n3, &vw[j + ldvw], &ldvw, &vw[(j + 1) + ldvw], &ldvw, &co, &si);

            if (lcmpq1) {
                SLC_DROT(&n, &q1[j * ldq1], &int1, &q1[(j + 1) * ldq1], &int1, &co, &si);
            }
        }

        for (js = jb + 1; js < m; js += nb) {
            je = js + nb - 1 < m - 1 ? js + nb - 1 : m - 1;
            nc = je - js + 1;
            ic = imat;
            for (i32 j = k; j < m - 1; j++) {
                SLC_DROT(&nc, &dwork[js * m + j + 1], &m, &dwork[js * m + j], &m,
                         &dwork[ic], &dwork[ic + 1]);
                ic += 2;
            }
            jc = 2 * (m - k - 1) + imat;
            for (i32 j = k; j < m - 1; j++) {
                SLC_DROT(&nc, &c1[j + js * ldc1], &ldc1, &c1[(j + 1) + js * ldc1], &ldc1,
                         &dwork[jc], &dwork[jc + 1]);
                jc += 2;
            }
        }

        for (js = 1 + nb; js < m + 1; js += nb) {
            je = js + nb - 1 < m ? js + nb - 1 : m;
            nc = je - js + 1;
            jc = 2 * (m - k - 1) + imat;
            for (i32 j = k; j < m - 1; j++) {
                SLC_DROT(&nc, &vw[j + js * ldvw], &ldvw, &vw[(j + 1) + js * ldvw], &ldvw,
                         &dwork[jc], &dwork[jc + 1]);
                jc += 2;
            }
        }

        SLC_DLARTG(&c1[(m - 1) + k * ldc1], &dwork[m * (k + 1) - 1], &co, &si, &tmp1);

        c1[(m - 1) + k * ldc1] = tmp1;
        dwork[m * (k + 1) - 1] = ZERO;
        i32 len_c1_rot = m - k - 1;
        if (len_c1_rot > 0) {
            SLC_DROT(&len_c1_rot, &c1[(m - 1) + mk1 * ldc1], &ldc1, &dwork[m * mk1 + m - 1], &m, &co, &si);
        }
        SLC_DROT(&m, &vw[(m - 1) + ldvw], &ldvw, &c2[m - 1], &ldc2, &co, &si);

        i32 m_1 = m - 1;
        SLC_DROT(&m_1, &a[m - 1], &lda, &de[m * ldde], &int1, &co, &si);

        if (lcmpq1) {
            SLC_DROT(&n, &q1[(m - 1) * ldq1], &int1, &q1[(n - 1) * ldq1], &int1, &co, &si);
        }

        js = k;
        je = k + nb - 1 < m - 1 ? k + nb - 1 : m - 1;
        ic = imat;
        jc = 2 * (m - k - 1) + imat;
        n1 = (m - k - 1) < (je - js + 1) ? (m - k - 1) : (je - js + 1);
        n2 = (m - k) < (je - js + 1) ? (m - k) : (je - js + 1);
        n3 = m < nb ? m : nb;

        for (i32 j = m - 1; j >= k + 1; j--) {
            SLC_DLARTG(&c1[(j - 1) + k * ldc1], &c1[j + k * ldc1], &co, &si, &tmp1);
            dwork[ic] = co;
            dwork[ic + 1] = si;
            ic += 2;

            c1[(j - 1) + k * ldc1] = tmp1;
            c1[j + k * ldc1] = ZERO;
            if (n1 > 0) {
                SLC_DROT(&n1, &c1[(j - 1) + mk1 * ldc1], &ldc1, &c1[j + mk1 * ldc1], &ldc1, &co, &si);
            }
            SLC_DROT(&n3, &vw[(j - 1) + ldvw], &ldvw, &vw[j + ldvw], &ldvw, &co, &si);

            tmp1 = -si * a[(j - 1) + (j - 1) * lda];
            a[(j - 1) + (j - 1) * lda] = co * a[(j - 1) + (j - 1) * lda];
            i32 len_a = m - j;
            SLC_DROT(&len_a, &a[(j - 1) + j * lda], &lda, &a[j + j * lda], &lda, &co, &si);
            i32 j1 = j - 1;
            if (j1 > 0) {
                SLC_DROT(&j1, &de[j * ldde], &int1, &de[(j + 1) * ldde], &int1, &co, &si);
            }

            if (lcmpq1) {
                SLC_DROT(&n, &q1[(j - 1) * ldq1], &int1, &q1[j * ldq1], &int1, &co, &si);
            }

            SLC_DLARTG(&a[j + j * lda], &tmp1, &co, &si, &tmp2);
            dwork[jc] = co;
            dwork[jc + 1] = si;
            jc += 2;

            a[j + j * lda] = tmp2;
            i32 jj = j;
            SLC_DROT(&jj, &a[j], &lda, &a[j - 1], &lda, &co, &si);

            SLC_DROT(&m, &c2[j], &ldc2, &c2[j - 1], &ldc2, &co, &si);
            i32 len_w = m - k;
            SLC_DROT(&len_w, &dwork[(k) * m + j], &m, &dwork[(k) * m + j - 1], &m, &co, &si);

            if (lcmpq1) {
                SLC_DROT(&n, &q1[(m + j) * ldq1], &int1, &q1[(m + j - 1) * ldq1], &int1, &co, &si);
            }
        }

        for (js = mk1 + n1; js < m; js += nb) {
            je = js + nb - 1 < m - 1 ? js + nb - 1 : m - 1;
            nc = je - js + 1;
            ic = imat;
            for (i32 j = m - 1; j >= k + 1; j--) {
                SLC_DROT(&nc, &c1[(j - 1) + js * ldc1], &ldc1, &c1[j + js * ldc1], &ldc1,
                         &dwork[ic], &dwork[ic + 1]);
                ic += 2;
            }
        }

        for (js = k - 1 + n2; js < m; js += nb) {
            je = js + nb - 1 < m - 1 ? js + nb - 1 : m - 1;
            nc = je - js + 1;
            jc = 2 * (m - k - 1) + imat;
            for (i32 j = m - 1; j >= k + 1; j--) {
                SLC_DROT(&nc, &dwork[js * m + j], &m, &dwork[js * m + j - 1], &m,
                         &dwork[jc], &dwork[jc + 1]);
                jc += 2;
            }
        }

        for (js = 1 + nb; js < m + 1; js += nb) {
            je = js + nb - 1 < m ? js + nb - 1 : m;
            nc = je - js + 1;
            ic = imat;
            for (i32 j = m - 1; j >= k + 1; j--) {
                SLC_DROT(&nc, &vw[(j - 1) + js * ldvw], &ldvw, &vw[j + js * ldvw], &ldvw,
                         &dwork[ic], &dwork[ic + 1]);
                ic += 2;
            }
        }

        i32 ics = imat + 2;
        je = m;
        while (je > 2) {
            nc = 0;
            ic = ics;
            ics += 2 * nb;
            for (i32 j = je - 1; j >= k + 1; j--) {
                nc = nc + 1 < nb ? nc + 1 : nb;
                js = je - nc;
                if (nc > 0) {
                    SLC_DROT(&nc, &a[(j - 1) + js * lda], &lda, &a[j + js * lda], &lda,
                             &dwork[ic], &dwork[ic + 1]);
                }
                ic += 2;
            }
            je -= nb;
        }

        ics = imat + 2;
        je = m + 1;
        while (je > 3) {
            nc = 0;
            ic = ics;
            ics += 2 * nb;
            for (i32 j = je - 2; j >= k + 1; j--) {
                nc = nc + 1 < nb ? nc + 1 : nb;
                js = je - nc;
                if (nc > 0) {
                    SLC_DROT(&nc, &de[(j - 1) + js * ldde], &ldde, &de[j + js * ldde], &ldde,
                             &dwork[ic], &dwork[ic + 1]);
                }
                ic += 2;
            }
            je -= nb;
        }

        js = k + 1;
        je = js + nb - 1 < m - 1 ? js + nb - 1 : m - 1;
        jc = imat;
        n2 = (m - k) < (je - js + 1) ? (m - k) : (je - js + 1);

        for (i32 j = k + 1; j < m - 1; j++) {
            i32 mj2 = j + 2 < m ? j + 2 : m;

            SLC_DLARTG(&dwork[j * m + k], &dwork[(j - 1) * m + k], &co, &si, &tmp1);

            SLC_DROT(&m, &c1[j + 1], &ldc1, &c1[j], &ldc1, &co, &si);
            dwork[(j - 1) * m + k] = ZERO;
            dwork[j * m + k] = tmp1;
            i32 len_w = m - k - 1;
            if (len_w > 0) {
                SLC_DROT(&len_w, &dwork[j * m + mk1], &int1, &dwork[(j - 1) * m + k + 1], &int1, &co, &si);
            }

            if (j == je) {
                js = je + 1;
                je = js + nb - 1 < m - 1 ? js + nb - 1 : m - 1;
                nc = je - js + 1;
                ja = imat;
                for (i32 ii = k + 1; ii < j; ii++) {
                    if (nc > 0) {
                        SLC_DROT(&nc, &b[ii + js * ldb], &ldb, &b[ii + 1 + js * ldb], &ldb,
                                 &dwork[ja], &dwork[ja + 1]);
                    }
                    ja += 2;
                }
                ja = imat;
                for (i32 ii = k + 1; ii < j; ii++) {
                    if (nc > 0) {
                        SLC_DROT(&nc, &f[ii + js * ldf], &ldf, &f[ii + 1 + js * ldf], &ldf,
                                 &dwork[ja], &dwork[ja + 1]);
                    }
                    ja += 2;
                }
            }

            i32 jj = j + 1;
            SLC_DROT(&jj, &b[j + 1], &ldb, &b[j], &ldb, &co, &si);
            tmp1 = -si * b[(j + 1) + (j + 1) * ldb];
            b[(j + 1) + (j + 1) * ldb] = co * b[(j + 1) + (j + 1) * ldb];

            if (lcmpq2) {
                SLC_DROT(&n, &q2[(j + 1) * ldq2], &int1, &q2[j * ldq2], &int1, &co, &si);
            }

            SLC_DLARTG(&b[j + j * ldb], &tmp1, &co, &si, &tmp2);
            dwork[jc] = co;
            dwork[jc + 1] = si;
            jc += 2;

            nc = je - j < je - js + 1 ? je - j : je - js + 1;
            b[j + j * ldb] = tmp2;
            if (nc > 0) {
                SLC_DROT(&nc, &b[j + (j + 1) * ldb], &ldb, &b[(j + 1) + (j + 1) * ldb], &ldb, &co, &si);
            }
            if (j > 0) {
                SLC_DROT(&j, &f[j * ldf], &int1, &f[(j + 1) * ldf], &int1, &co, &si);
            }
            i32 nc2 = nc - 1;
            if (nc2 > 0) {
                SLC_DROT(&nc2, &f[j + mj2 * ldf], &ldf, &f[(j + 1) + mj2 * ldf], &ldf, &co, &si);
            }

            SLC_DROT(&n2, &c2[j + k * ldc2], &ldc2, &c2[(j + 1) + k * ldc2], &ldc2, &co, &si);
            SLC_DROT(&m, &vw[j * ldvw], &int1, &vw[(j + 1) * ldvw], &int1, &co, &si);

            if (lcmpq2) {
                SLC_DROT(&n, &q2[(m + j) * ldq2], &int1, &q2[(m + j + 1) * ldq2], &int1, &co, &si);
            }
        }

        for (js = k + n2; js < m; js += nb) {
            je = js + nb - 1 < m - 1 ? js + nb - 1 : m - 1;
            nc = je - js + 1;
            jc = imat;
            for (i32 j = k + 1; j < m - 1; j++) {
                SLC_DROT(&nc, &c2[j + js * ldc2], &ldc2, &c2[(j + 1) + js * ldc2], &ldc2,
                         &dwork[jc], &dwork[jc + 1]);
                jc += 2;
            }
        }

        if (k < m - 1) {
            SLC_DLARTG(&c2[(m - 1) + k * ldc2], &dwork[(m - 2) * m + k], &co, &si, &tmp1);

            SLC_DROT(&m, &vw[m * ldvw], &int1, &c1[m - 1], &ldc1, &co, &si);
            c2[(m - 1) + k * ldc2] = tmp1;
            dwork[(m - 2) * m + k] = ZERO;
            i32 len_rot = m - k - 1;
            if (len_rot > 0) {
                SLC_DROT(&len_rot, &c2[(m - 1) + (k + 1) * ldc2], &ldc2, &dwork[(m - 2) * m + k + 1], &int1, &co, &si);
            }

            i32 m_1 = m - 1;
            SLC_DROT(&m_1, &f[m - 1], &ldf, &b[m - 1], &ldb, &co, &si);

            if (lcmpq2) {
                SLC_DROT(&n, &q2[(n - 1) * ldq2], &int1, &q2[(m - 1) * ldq2], &int1, &co, &si);
            }
        } else {
            SLC_DLARTG(&c1[(m - 1) + (m - 1) * ldc1], &dwork[mm - 1], &co, &si, &tmp1);

            c1[(m - 1) + (m - 1) * ldc1] = tmp1;
            dwork[mm - 1] = ZERO;
            SLC_DROT(&m, &vw[(m - 1) + ldvw], &ldvw, &c2[m - 1], &ldc2, &co, &si);

            i32 m_1 = m - 1;
            SLC_DROT(&m_1, &a[m - 1], &lda, &de[m * ldde], &int1, &co, &si);

            if (lcmpq1) {
                SLC_DROT(&n, &q1[(m - 1) * ldq1], &int1, &q1[(n - 1) * ldq1], &int1, &co, &si);
            }
        }

        js = k;
        je = k + nb - 1 < m - 1 ? k + nb - 1 : m - 1;
        ic = imat;
        n1 = (m - k - 1) < (je - js + 1) ? (m - k - 1) : (je - js + 1);

        for (i32 j = m - 1; j >= k + 2; j--) {
            SLC_DLARTG(&c2[(j - 1) + k * ldc2], &c2[j + k * ldc2], &co, &si, &tmp1);
            dwork[ic] = co;
            dwork[ic + 1] = si;
            ic += 2;

            c2[(j - 1) + k * ldc2] = tmp1;
            c2[j + k * ldc2] = ZERO;
            if (n1 > 0) {
                SLC_DROT(&n1, &c2[(j - 1) + mk1 * ldc2], &ldc2, &c2[j + mk1 * ldc2], &ldc2, &co, &si);
            }
            SLC_DROT(&m, &vw[j * ldvw], &int1, &vw[(j + 1) * ldvw], &int1, &co, &si);

            i32 len_b = m - j;
            SLC_DROT(&len_b, &b[(j - 1) + j * ldb], &ldb, &b[j + j * ldb], &ldb, &co, &si);
            tmp1 = -si * b[(j - 1) + (j - 1) * ldb];
            b[(j - 1) + (j - 1) * ldb] = co * b[(j - 1) + (j - 1) * ldb];
            i32 j1 = j - 1;
            if (j1 > 0) {
                SLC_DROT(&j1, &f[(j - 1) * ldf], &int1, &f[j * ldf], &int1, &co, &si);
            }

            if (lcmpq2) {
                SLC_DROT(&n, &q2[(m + j - 1) * ldq2], &int1, &q2[(m + j) * ldq2], &int1, &co, &si);
            }

            SLC_DLARTG(&b[j + j * ldb], &tmp1, &co, &si, &tmp2);
            b[j + j * ldb] = tmp2;

            i32 jj = j;
            SLC_DROT(&jj, &b[j], &ldb, &b[j - 1], &ldb, &co, &si);

            SLC_DROT(&m, &c1[j], &ldc1, &c1[j - 1], &ldc1, &co, &si);
            i32 len_w = m - k;
            SLC_DROT(&len_w, &dwork[(j - 1) * m + k], &int1, &dwork[(j - 2) * m + k], &int1, &co, &si);

            if (lcmpq2) {
                SLC_DROT(&n, &q2[j * ldq2], &int1, &q2[(j - 1) * ldq2], &int1, &co, &si);
            }
        }

        for (js = mk1 + n1; js < m; js += nb) {
            je = js + nb - 1 < m - 1 ? js + nb - 1 : m - 1;
            nc = je - js + 1;
            ic = imat;
            for (i32 j = m - 1; j >= k + 2; j--) {
                SLC_DROT(&nc, &c2[(j - 1) + js * ldc2], &ldc2, &c2[j + js * ldc2], &ldc2,
                         &dwork[ic], &dwork[ic + 1]);
                ic += 2;
            }
        }

        ics = imat + 2;
        je = m;
        while (je > 2) {
            nc = 0;
            ic = ics;
            ics += 2 * nb;
            for (i32 j = je - 1; j >= k + 2; j--) {
                nc = nc + 1 < nb ? nc + 1 : nb;
                js = je - nc;
                if (nc > 0) {
                    SLC_DROT(&nc, &b[(j - 1) + js * ldb], &ldb, &b[j + js * ldb], &ldb,
                             &dwork[ic], &dwork[ic + 1]);
                }
                ic += 2;
            }
            je -= nb;
        }

        ics = imat + 2;
        je = m;
        while (je > 3) {
            nc = 0;
            ic = ics;
            ics += 2 * nb;
            for (i32 j = je - 1; j >= k + 2; j--) {
                nc = nc + 1 < nb ? nc + 1 : nb;
                js = je - nc;
                if (nc > 0) {
                    SLC_DROT(&nc, &f[(j - 1) + js * ldf], &ldf, &f[j + js * ldf], &ldf,
                             &dwork[ic], &dwork[ic + 1]);
                }
                ic += 2;
            }
            je -= nb;
        }
    }

    const char *cmpq;
    const char *cmpsc;
    i32 iwrk;

    if (ltri || lcmpq1 || lcmpq2) {
        cmpq = "Initialize";
        imat = 4 * mm;
        iwrk = 8 * mm;
    } else {
        cmpq = "No Computation";
        imat = 0;
        iwrk = 4 * mm;
    }

    if (ltri) {
        cmpsc = "Schur Form";
    } else {
        cmpsc = "Eigenvalues Only";
    }

    SLC_DLACPY("Full", &m, &m, c2, &ldc2, &dwork[imat], &m);
    SLC_DLACPY("Full", &m, &m, a, &lda, &dwork[imat + mm], &m);
    SLC_DLACPY("Full", &m, &m, c1, &ldc1, &dwork[imat + 2 * mm], &m);
    SLC_DLACPY("Full", &m, &m, b, &ldb, &dwork[imat + 3 * mm], &m);

    iwork[0] = 1;
    iwork[1] = -1;
    iwork[2] = 1;
    iwork[3] = -1;

    i32 idum[1] = {0};
    i32 k_mb03bd = 4;
    i32 h_mb03bd = 1;
    i32 ilo_mb03bd = 1;
    i32 ihi_mb03bd = m;
    i32 liwork_mb03bd = liwork - (m + 4);
    i32 ldwork_mb03bd = ldwork - iwrk;

    i32 iwarn = 0;
    i32 info_mb03bd = 0;
    i32 iwarn_mb03bd = 0;
    mb03bd(cmpsc, "Careful", cmpq, idum, k_mb03bd, m, h_mb03bd, ilo_mb03bd, ihi_mb03bd,
           iwork, &dwork[imat], m, m, dwork, m, m,
           alphar, alphai, beta, &iwork[4],
           &iwork[m + 4], liwork_mb03bd, &dwork[iwrk], ldwork_mb03bd, &iwarn_mb03bd, &info_mb03bd);

    iwarn = info_mb03bd;
    if (iwarn > 0 && iwarn < m) {
        *info = 1;
        return;
    } else if (iwarn == m + 1) {
        *info = 3;
    } else if (info_mb03bd > 0) {
        *info = 2;
        return;
    }

    i32 optdw_result;
    if ((m % 2) == 0) {
        optdw_result = (4 * n > 32 ? 4 * n : 32) + 4;
    } else {
        optdw_result = (4 * n > 36 ? 4 * n : 36);
    }
    if (ltri || lcmpq1 || lcmpq2) {
        optdw_result = 8 * mm + optdw_result;
    } else {
        optdw_result = 4 * mm + optdw_result;
    }
    if ((i32)dwork[iwrk] + iwrk > optdw_result) {
        optdw_result = (i32)dwork[iwrk] + iwrk;
    }

    i32 nbeta0 = 0;
    i32 i11 = 0;
    i32 i22 = 0;
    i32 i2x2 = 0;

    for (i32 idx = 0; idx < m; idx++) {
        if (ninf > 0) {
            if (beta[idx] == ZERO) {
                nbeta0++;
            }
        }
        if (iwork[idx + 4] >= 2 * emin && iwork[idx + 4] <= 2 * emax) {
            beta[idx] = beta[idx] / pow(base, HALF * iwork[idx + 4]);
            if (beta[idx] != ZERO) {
                if (iwork[m + idx + 4] < 0) {
                    i22++;
                } else if (iwork[m + idx + 4] > 0) {
                    i11++;
                }
                double complex eig = csqrt(alphar[idx] + I * alphai[idx]);
                alphar[idx] = cimag(eig);
                alphai[idx] = creal(eig);
                if (alphar[idx] < ZERO) {
                    alphar[idx] = -alphar[idx];
                }
                if (alphai[idx] < ZERO) {
                    alphai[idx] = -alphai[idx];
                }
                if (alphar[idx] != ZERO && alphai[idx] != ZERO) {
                    if (idx + 1 < m) {
                        alphar[idx + 1] = -alphar[idx];
                        alphai[idx + 1] = alphai[idx];
                        beta[idx + 1] = beta[idx];
                    }
                    i2x2++;
                    idx++;
                } else if (iwork[m + idx + 4] < 0) {
                    i2x2++;
                }
            }
        } else if (iwork[idx + 4] < 2 * emin) {
            alphar[idx] = ZERO;
            alphai[idx] = ZERO;
            i11++;
        } else {
            if (ninf > 0) {
                nbeta0++;
            }
            beta[idx] = ZERO;
            i11++;
        }
    }

    iwork[0] = i11 + i22;

    i32 l = 0;
    if (ninf > 0) {
        for (i32 j = 0; j < ninf - nbeta0; j++) {
            tmp1 = ZERO;
            tmp2 = ONE;
            i32 p = 0;
            for (i32 idx = 0; idx < m; idx++) {
                if (beta[idx] > ZERO) {
                    temp = SLC_DLAPY2(&alphar[idx], &alphai[idx]);
                    if (temp > tmp1 && tmp2 >= beta[idx]) {
                        tmp1 = temp;
                        tmp2 = beta[idx];
                        p = idx;
                    }
                }
            }
            l++;
            beta[p] = ZERO;
        }
        if (l == iwork[0]) {
            *info = 0;
            i11 = 0;
            i22 = 0;
            iwork[0] = 0;
        }
    }

    SLC_DCOPY(&int1, &dwork[iwrk + 1], &int1, &dum[0], &int1);
    SLC_DCOPY(&int1, &dwork[iwrk + 2], &int1, &dum[1], &int1);
    SLC_DCOPY(&int1, &dwork[iwrk + 3], &int1, &dum[2], &int1);
    SLC_DCOPY(&int1, &dwork[iwrk + 4], &int1, &dum[3], &int1);

    i32 kk = iwrk;
    i32 iw = iwork[0];
    i32 idx_i = 0;
    i32 idx_j = 0;
    i32 ll = 4 * (m - 2 * i2x2) + kk;
    bool unrel = false;

    while (idx_i < m) {
        if (idx_j < iw) {
            unrel = (idx_i == abs(iwork[m + idx_i + 4]) - 1);
        } else {
            unrel = false;
        }
        if (alphar[idx_i] != ZERO && beta[idx_i] != ZERO &&
            (alphai[idx_i] != ZERO || iwork[m + idx_i + 4] < 0)) {
            if (unrel) {
                idx_j++;
                iwork[idx_j] = iwork[m + idx_i + 4];
                iwork[iw + idx_j] = ll - iwrk + 6;
                unrel = false;
            }
            i32 int2 = 2;
            SLC_DLACPY("Full", &int2, &int2, &dwork[imat + (m + 1) * idx_i], &m, &dwork[ll], &int2);
            SLC_DLACPY("Full", &int2, &int2, &dwork[imat + (m + 1) * idx_i + mm], &m, &dwork[ll + 4], &int2);
            SLC_DLACPY("Full", &int2, &int2, &dwork[imat + (m + 1) * idx_i + 2 * mm], &m, &dwork[ll + 8], &int2);
            SLC_DLACPY("Full", &int2, &int2, &dwork[imat + (m + 1) * idx_i + 3 * mm], &m, &dwork[ll + 12], &int2);
            ll += 16;
            idx_i += 2;
        } else {
            if (unrel) {
                idx_j++;
                iwork[idx_j] = idx_i + 1;
                iwork[iw + idx_j] = kk - iwrk + 6;
                unrel = false;
            }
            SLC_DCOPY(&int1, &dwork[imat + (m + 1) * idx_i], &mm, &dwork[kk], &int1);
            SLC_DCOPY(&int1, &dwork[imat + (m + 1) * idx_i + mm], &mm, &dwork[kk + 1], &int1);
            SLC_DCOPY(&int1, &dwork[imat + (m + 1) * idx_i + 2 * mm], &mm, &dwork[kk + 2], &int1);
            SLC_DCOPY(&int1, &dwork[imat + (m + 1) * idx_i + 3 * mm], &mm, &dwork[kk + 3], &int1);
            kk += 4;
            idx_i++;
        }
    }

    iwork[2 * iw + 1] = i11;
    iwork[2 * iw + 2] = i22;
    iwork[2 * iw + 3] = i2x2;

    if (ltri) {
        SLC_DLACPY("Upper", &m, &m, &dwork[imat + 2 * mm], &m, c1, &ldc1);
        SLC_DLACPY("Full", &m, &m, &dwork[imat], &m, c2, &ldc2);

        SLC_DGEMM("Transpose", "No Transpose", &m, &m, &m, &ONE,
                  &dwork[2 * mm], &m, &vw[ldvw], &ldvw, &ZERO, &dwork[imat], &m);
        SLC_DGEMM("No Transpose", "No Transpose", &m, &m, &m, &ONE,
                  &dwork[imat], &m, dwork, &m, &ZERO, &vw[ldvw], &ldvw);

        SLC_DLACPY("Upper", &m, &m, &dwork[imat + mm], &m, a, &lda);

        i32 iw_temp;
        mb01ld("Upper", "Transpose", m, m, ZERO, ONE, &de[ldde], ldde,
               &dwork[2 * mm], m, &de[ldde], ldde, &dwork[imat], ldwork - imat, &iw_temp);

        SLC_DLACPY("Upper", &m, &m, &dwork[imat + 3 * mm], &m, b, &ldb);

        mb01ld("Upper", "Transpose", m, m, ZERO, ONE, f, ldf,
               dwork, m, f, ldf, &dwork[imat], ldwork - imat, &iw_temp);

        if (lcmpq1) {
            SLC_DGEMM("No Transpose", "No Transpose", &n, &m, &m, &ONE,
                      q1, &ldq1, &dwork[2 * mm], &m, &ZERO, &dwork[imat], &n);
            SLC_DLACPY("Full", &n, &m, &dwork[imat], &n, q1, &ldq1);
            SLC_DGEMM("No Transpose", "No Transpose", &n, &m, &m, &ONE,
                      &q1[m * ldq1], &ldq1, &dwork[mm], &m, &ZERO, &dwork[imat], &n);
            SLC_DLACPY("Full", &n, &m, &dwork[imat], &n, &q1[m * ldq1], &ldq1);
        }

        if (lcmpq2) {
            SLC_DGEMM("No Transpose", "No Transpose", &n, &m, &m, &ONE,
                      q2, &ldq2, &dwork[3 * mm], &m, &ZERO, &dwork[imat], &n);
            SLC_DLACPY("Full", &n, &m, &dwork[imat], &n, q2, &ldq2);
            SLC_DGEMM("No Transpose", "No Transpose", &n, &m, &m, &ONE,
                      &q2[m * ldq2], &ldq2, dwork, &m, &ZERO, &dwork[imat], &n);
            SLC_DLACPY("Full", &n, &m, &dwork[imat], &n, &q2[m * ldq2], &ldq2);
        }
    }

    kk = 4 * (m - 2 * i2x2) + 16 * i2x2;
    i32 int6 = kk;
    SLC_DCOPY(&int6, &dwork[iwrk], &int1, &dwork[5], &int1);
    i32 int4 = 4;
    SLC_DCOPY(&int4, dum, &int1, &dwork[1], &int1);

    dwork[0] = (f64)optdw_result;
}
