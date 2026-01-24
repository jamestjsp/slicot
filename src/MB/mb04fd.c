/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 *
 * MB04FD - Eigenvalues and orthogonal decomposition of a real
 *          skew-Hamiltonian/skew-Hamiltonian pencil
 *
 * Computes eigenvalues of aS - bT where:
 *   S = [[A, D], [E, A']] with D, E skew-symmetric
 *   T = [[B, F], [G, B']] with F, G skew-symmetric
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <ctype.h>
#include <stdlib.h>

void mb04fd(const char *job, const char *compq, i32 n,
            f64 *a, i32 lda, f64 *de, i32 ldde, f64 *b, i32 ldb, f64 *fg, i32 ldfg,
            f64 *q, i32 ldq, f64 *alphar, f64 *alphai, f64 *beta,
            i32 *iwork, f64 *dwork, i32 ldwork, i32 *info) {

    const f64 ZERO = 0.0;
    const f64 HALF = 0.5;
    const f64 ONE = 1.0;
    const f64 TWO = 2.0;
    const f64 THREE = 3.0;
    const f64 TEN = 10.0;

    char job_upper = (char)toupper((unsigned char)job[0]);
    char compq_upper = (char)toupper((unsigned char)compq[0]);

    bool ltri = (job_upper == 'T');
    bool liniq = (compq_upper == 'I');
    bool lupdq = (compq_upper == 'U');
    bool lcmpq = liniq || lupdq;

    i32 m = n / 2;
    i32 mm = m * m;
    i32 m1 = (1 > m) ? 1 : m;

    i32 iq1 = 0;
    i32 iq2, iwrk, mindw;

    if (lcmpq) {
        iq2 = iq1 + mm;
        iwrk = iq2 + mm;
        mindw = (3 > iwrk - 1 + mm) ? 3 : iwrk - 1 + mm;
    } else if (ltri) {
        iq2 = 0;
        iwrk = iq2 + mm;
        mindw = (3 > iwrk - 1 + m) ? 3 : iwrk - 1 + m;
    } else {
        iq2 = 0;
        iwrk = 0;
        mindw = (3 > m) ? 3 : m;
    }
    i32 k_work = iwrk - 1;

    bool lquery = (ldwork == -1);

    *info = 0;
    if (!(job_upper == 'E' || ltri)) {
        *info = -1;
    } else if (!(compq_upper == 'N' || lcmpq)) {
        *info = -2;
    } else if (n < 0 || (n % 2) != 0) {
        *info = -3;
    } else if (lda < m1) {
        *info = -5;
    } else if (ldde < m1) {
        *info = -7;
    } else if (ldb < m1) {
        *info = -9;
    } else if (ldfg < m1) {
        *info = -11;
    } else if (ldq < 1 || (lcmpq && ldq < n)) {
        *info = -13;
    } else if (!lquery && ldwork < mindw) {
        dwork[0] = (f64)mindw;
        *info = -19;
    }

    if (*info != 0) {
        return;
    }

    if (n > 0 && lquery) {
        const char *cmpsc = ltri ? "S" : "E";
        const char *cmpq_str = lcmpq ? "I" : "N";
        f64 dum;
        i32 lwork_query = -1;
        i32 info_dhgeqz;
        SLC_DHGEQZ(cmpsc, cmpq_str, cmpq_str, &m, &(i32){1}, &m, b, &ldb, a, &lda,
                   alphar, alphai, beta, dwork, &m1, dwork, &m1,
                   &dum, &lwork_query, &info_dhgeqz);

        i32 optdw;
        if (lcmpq) {
            i32 temp = k_work > (i32)dum ? k_work : (i32)dum;
            optdw = k_work + temp;
        } else if (ltri) {
            i32 temp = k_work - m > (i32)dum ? k_work - m : (i32)dum;
            optdw = k_work + temp;
        } else {
            optdw = (i32)dum;
        }
        dwork[0] = (f64)((optdw > mindw) ? optdw : mindw);
        return;
    } else if (lquery) {
        dwork[0] = (f64)mindw;
        return;
    }

    if (n == 0) {
        iwork[0] = 0;
        dwork[0] = THREE;
        dwork[1] = ZERO;
        dwork[2] = ZERO;
        return;
    }

    i32 ninf = 0;
    f64 nrms, nrmt, nrm;

    if (m == 1) {
        nrm = ZERO;
    } else {
        i32 mm1 = m - 1;
        nrm = SLC_DLANTR("M", "L", "N", &mm1, &mm1, &de[1], &ldde, dwork) +
              SLC_DLANTR("M", "U", "N", &mm1, &mm1, &de[ldde], &ldde, dwork);
    }

    if (nrm == ZERO) {
        if (m == 1) {
            nrms = fabs(a[0]);
            if (nrms == ZERO) {
                ninf = 1;
            }
        } else {
            i32 mm1 = m - 1;
            f64 norm_lower = SLC_DLANTR("M", "L", "N", &mm1, &mm1, &a[1], &lda, dwork);
            f64 norm_upper = SLC_DLANTR("M", "U", "N", &mm1, &mm1, &a[lda], &lda, dwork);
            if (norm_lower == ZERO && norm_upper == ZERO) {
                f64 tmp1 = ZERO;
                f64 tmp2 = ONE;
                i32 ldap1 = lda + 1;
                SLC_DLASSQ(&m, a, &ldap1, &tmp1, &tmp2);
                nrms = tmp1 * sqrt(tmp2);
                for (i32 j = 0; j < m; j++) {
                    if (a[j + j * lda] == ZERO) {
                        ninf++;
                    }
                }
            } else {
                i32 nzr, nzc;
                ma02pd(m, m, a, lda, &nzr, &nzc);
                ninf = (nzr > nzc) ? nzr : nzc;
                nrms = SLC_DLANGE("F", &m, &m, a, &lda, dwork);
            }
        }
        nrms = nrms * sqrt(TWO);
    } else {
        ninf = ma02od("S", m, a, lda, de, ldde);
        if ((ninf % 2) > 0) {
            ninf++;
        }
        ninf = ninf / 2;
        nrms = ma02id("S", "F", m, a, lda, de, ldde, dwork);
    }

    nrmt = ma02id("S", "F", m, b, ldb, fg, ldfg, dwork);

    if (liniq) {
        SLC_DLASET("F", &n, &n, &ZERO, &ONE, q, &ldq);
    }

    f64 dum0 = ZERO;

    for (i32 k = 0; k < m - 1; k++) {
        i32 mk2 = (k + 2 < m) ? k + 2 : m - 1;
        i32 mk3 = mk2 + 1;
        f64 tmp1 = de[(k + 1) + k * ldde];
        i32 len = m - k - 1;
        f64 nu;
        SLC_DLARFG(&len, &tmp1, &de[mk2 + k * ldde], &(i32){1}, &nu);

        if (nu != ZERO) {
            de[(k + 1) + k * ldde] = ONE;
            i32 info_dummy;

            mb01md('L', m - k - 1, nu, &de[(k + 1) + (k + 1) * ldde], ldde,
                   &de[(k + 1) + k * ldde], 1, ZERO, dwork, 1, &info_dummy);

            f64 dot;
            i32 one = 1;
            dot = SLC_DDOT(&len, dwork, &one, &de[(k + 1) + k * ldde], &one);
            f64 mu = -HALF * nu * dot;
            SLC_DAXPY(&len, &mu, &de[(k + 1) + k * ldde], &one, dwork, &one);

            mb01nd('L', m - k - 1, ONE, &de[(k + 1) + k * ldde], 1, dwork, 1,
                   &de[(k + 1) + (k + 1) * ldde], ldde, &info_dummy);

            i32 kp1 = k + 1;
            SLC_DLARF("L", &len, &kp1, &de[(k + 1) + k * ldde], &one, &nu,
                      &fg[(k + 1)], &ldfg, dwork);

            mb01md('L', m - k - 1, nu, &fg[(k + 1) + (k + 1) * ldfg], ldfg,
                   &de[(k + 1) + k * ldde], 1, ZERO, dwork, 1, &info_dummy);

            dot = SLC_DDOT(&len, dwork, &one, &de[(k + 1) + k * ldde], &one);
            mu = -HALF * nu * dot;
            SLC_DAXPY(&len, &mu, &de[(k + 1) + k * ldde], &one, dwork, &one);

            mb01nd('L', m - k - 1, ONE, &de[(k + 1) + k * ldde], 1, dwork, 1,
                   &fg[(k + 1) + (k + 1) * ldfg], ldfg, &info_dummy);

            SLC_DLARF("R", &m, &len, &de[(k + 1) + k * ldde], &one, &nu,
                      &a[(k + 1) * lda], &lda, dwork);
            SLC_DLARF("R", &m, &len, &de[(k + 1) + k * ldde], &one, &nu,
                      &b[(k + 1) * ldb], &ldb, dwork);

            if (lcmpq) {
                SLC_DLARF("R", &n, &len, &de[(k + 1) + k * ldde], &one, &nu,
                          &q[(k + 1) * ldq], &ldq, dwork);
            }
            de[(k + 1) + k * ldde] = tmp1;
        }

        f64 tmp2 = a[(k + 1) + k * lda];
        f64 co, si;
        SLC_DLARTG(&tmp2, &tmp1, &co, &si, &a[(k + 1) + k * lda]);

        i32 len2 = m - k - 2;
        if (len2 > 0) {
            i32 one = 1;
            SLC_DROT(&len2, &de[mk2 + (k + 1) * ldde], &one, &a[(k + 1) + mk2 * lda], &lda, &co, &si);
        }
        i32 kp1 = k + 1;
        i32 one = 1;
        SLC_DROT(&kp1, &a[(k + 1) * lda], &one, &de[(k + 2) * ldde], &one, &co, &si);
        if (len2 > 0) {
            SLC_DROT(&len2, &de[(k + 1) + mk3 * ldde], &ldde, &a[mk2 + (k + 1) * lda], &one, &co, &si);
        }

        SLC_DROT(&kp1, &fg[(k + 1)], &ldfg, &b[(k + 1)], &ldb, &co, &(f64){-si});
        if (len2 > 0) {
            SLC_DROT(&len2, &fg[mk2 + (k + 1) * ldfg], &one, &b[(k + 1) + mk2 * ldb], &ldb, &co, &si);
        }
        SLC_DROT(&kp1, &b[(k + 1) * ldb], &one, &fg[(k + 2) * ldfg], &one, &co, &si);
        if (len2 > 0) {
            SLC_DROT(&len2, &fg[(k + 1) + mk3 * ldfg], &ldfg, &b[mk2 + (k + 1) * ldb], &one, &co, &si);
        }

        if (lcmpq) {
            SLC_DROT(&n, &q[(m + k + 1) * ldq], &one, &q[(k + 1) * ldq], &one, &co, &(f64){-si});
        }

        tmp1 = a[k + k * lda];
        i32 len_p = m - k;
        SLC_DLARFG(&len_p, &tmp1, &a[(k + 1) + k * lda], &one, &nu);

        if (nu != ZERO) {
            a[k + k * lda] = ONE;

            i32 one = 1;
            SLC_DLARF("L", &len_p, &len, &a[k + k * lda], &one, &nu,
                      &a[k + (k + 1) * lda], &lda, dwork);

            if (k > 0) {
                i32 km1 = k;
                SLC_DLARF("R", &km1, &len_p, &a[k + k * lda], &one, &nu,
                          &de[(k + 1) * ldde], &ldde, dwork);
            }

            i32 info_dummy;
            mb01md('U', m - k, nu, &de[k + (k + 1) * ldde], ldde,
                   &a[k + k * lda], 1, ZERO, dwork, 1, &info_dummy);

            f64 dot = SLC_DDOT(&len_p, dwork, &one, &a[k + k * lda], &one);
            f64 mu = -HALF * nu * dot;
            SLC_DAXPY(&len_p, &mu, &a[k + k * lda], &one, dwork, &one);

            mb01nd('U', m - k, ONE, &a[k + k * lda], 1, dwork, 1,
                   &de[k + (k + 1) * ldde], ldde, &info_dummy);

            SLC_DLARF("L", &len_p, &m, &a[k + k * lda], &one, &nu,
                      &b[k], &ldb, dwork);

            if (k > 0) {
                i32 km1 = k;
                SLC_DLARF("R", &km1, &len_p, &a[k + k * lda], &one, &nu,
                          &fg[(k + 1) * ldfg], &ldfg, dwork);
            }

            mb01md('U', m - k, nu, &fg[k + (k + 1) * ldfg], ldfg,
                   &a[k + k * lda], 1, ZERO, dwork, 1, &info_dummy);

            dot = SLC_DDOT(&len_p, dwork, &one, &a[k + k * lda], &one);
            mu = -HALF * nu * dot;
            SLC_DAXPY(&len_p, &mu, &a[k + k * lda], &one, dwork, &one);

            mb01nd('U', m - k, ONE, &a[k + k * lda], 1, dwork, 1,
                   &fg[k + (k + 1) * ldfg], ldfg, &info_dummy);

            if (lcmpq) {
                SLC_DLARF("R", &n, &len_p, &a[k + k * lda], &one, &nu,
                          &q[(m + k) * ldq], &ldq, dwork);
            }
            a[k + k * lda] = tmp1;
        }

        i32 zero_inc = 0;
        SLC_DCOPY(&len, &dum0, &zero_inc, &a[(k + 1) + k * lda], &one);
    }

    for (i32 k = 0; k < m - 1; k++) {
        for (i32 j = k + 1; j < m - 1; j++) {
            i32 mj2 = (j + 2 < m) ? j + 2 : m - 1;
            i32 mj3 = mj2 + 1;

            f64 co, si, tmp1;
            SLC_DLARTG(&fg[(j + 1) + k * ldfg], &fg[j + k * ldfg], &co, &si, &tmp1);

            i32 one = 1;
            SLC_DROT(&m, &b[(j + 1) * ldb], &one, &b[j * ldb], &one, &co, &si);
            fg[(j + 1) + k * ldfg] = tmp1;
            i32 len = m - j - 2;
            if (len > 0) {
                SLC_DROT(&len, &fg[mj2 + (j + 1) * ldfg], &one, &fg[mj2 + j * ldfg], &one, &co, &si);
            }
            len = j - k - 1;
            if (len > 0) {
                SLC_DROT(&len, &fg[(j + 1) + (k + 1) * ldfg], &ldfg, &fg[j + (k + 1) * ldfg], &ldfg, &co, &si);
            }

            i32 jp1 = j + 1;
            SLC_DROT(&jp1, &a[(j + 1) * lda], &one, &a[j * lda], &one, &co, &si);
            tmp1 = -si * a[(j + 1) + (j + 1) * lda];
            a[(j + 1) + (j + 1) * lda] = co * a[(j + 1) + (j + 1) * lda];

            if (lcmpq) {
                SLC_DROT(&n, &q[(j + 1) * ldq], &one, &q[j * ldq], &one, &co, &si);
            }

            f64 tmp2;
            SLC_DLARTG(&a[j + j * lda], &tmp1, &co, &si, &tmp2);

            a[j + j * lda] = tmp2;
            len = m - j - 1;
            SLC_DROT(&len, &a[j + (j + 1) * lda], &lda, &a[(j + 1) + (j + 1) * lda], &lda, &co, &si);
            i32 jm1 = j;
            if (jm1 > 0) {
                SLC_DROT(&jm1, &de[(j + 1) * ldde], &one, &de[(j + 2) * ldde], &one, &co, &si);
            }
            len = m - j - 2;
            if (len > 0) {
                SLC_DROT(&len, &de[j + mj3 * ldde], &ldde, &de[(j + 1) + mj3 * ldde], &ldde, &co, &si);
            }

            len = m - k;
            SLC_DROT(&len, &b[j + k * ldb], &ldb, &b[(j + 1) + k * ldb], &ldb, &co, &si);
            if (jm1 > 0) {
                SLC_DROT(&jm1, &fg[(j + 1) * ldfg], &one, &fg[(j + 2) * ldfg], &one, &co, &si);
            }
            len = m - j - 2;
            if (len > 0) {
                SLC_DROT(&len, &fg[j + mj3 * ldfg], &ldfg, &fg[(j + 1) + mj3 * ldfg], &ldfg, &co, &si);
            }

            if (lcmpq) {
                SLC_DROT(&n, &q[(m + j) * ldq], &one, &q[(m + j + 1) * ldq], &one, &co, &si);
            }
        }

        f64 co, si, tmp1;
        f64 neg_fg = -fg[(m - 1) + k * ldfg];
        SLC_DLARTG(&b[(m - 1) + k * ldb], &neg_fg, &co, &si, &tmp1);

        b[(m - 1) + k * ldb] = tmp1;
        fg[(m - 1) + k * ldfg] = ZERO;
        i32 one = 1;
        i32 mm1 = m - 1;
        SLC_DROT(&mm1, &fg[m * ldfg], &one, &b[(m - 1) * ldb], &one, &co, &si);
        i32 len = m - k - 2;
        if (len > 0) {
            SLC_DROT(&len, &fg[(m - 1) + (k + 1) * ldfg], &ldfg, &b[(m - 1) + (k + 1) * ldb], &ldb, &co, &si);
        }

        SLC_DROT(&mm1, &de[m * ldde], &one, &a[(m - 1) * lda], &one, &co, &si);

        if (lcmpq) {
            SLC_DROT(&n, &q[(n - 1) * ldq], &one, &q[(m - 1) * ldq], &one, &co, &si);
        }

        for (i32 j = m - 1; j >= k + 2; j--) {
            i32 mj1 = (j + 2 < m) ? j + 1 : m - 1;
            i32 mj2 = mj1 + 1;

            f64 co_inner, si_inner, tmp1_inner;
            SLC_DLARTG(&b[(j - 1) + k * ldb], &b[j + k * ldb], &co_inner, &si_inner, &tmp1_inner);

            i32 one_inner = 1;
            b[(j - 1) + k * ldb] = tmp1_inner;
            b[j + k * ldb] = ZERO;
            i32 len_inner = j - 1;
            if (len_inner > 0) {
                SLC_DROT(&len_inner, &fg[j * ldfg], &one_inner, &fg[(j + 1) * ldfg], &one_inner, &co_inner, &si_inner);
            }
            len_inner = m - k - 1;
            SLC_DROT(&len_inner, &b[(j - 1) + (k + 1) * ldb], &ldb, &b[j + (k + 1) * ldb], &ldb, &co_inner, &si_inner);
            len_inner = m - j - 1;
            if (len_inner > 0) {
                SLC_DROT(&len_inner, &fg[(j - 1) + mj2 * ldfg], &ldfg, &fg[j + mj2 * ldfg], &ldfg, &co_inner, &si_inner);
            }

            tmp1_inner = -si_inner * a[(j - 1) + (j - 1) * lda];
            a[(j - 1) + (j - 1) * lda] = co_inner * a[(j - 1) + (j - 1) * lda];
            len_inner = m - j;
            SLC_DROT(&len_inner, &a[(j - 1) + j * lda], &lda, &a[j + j * lda], &lda, &co_inner, &si_inner);
            len_inner = j - 1;
            if (len_inner > 0) {
                SLC_DROT(&len_inner, &de[j * ldde], &one_inner, &de[(j + 1) * ldde], &one_inner, &co_inner, &si_inner);
            }
            len_inner = m - j - 1;
            if (len_inner > 0) {
                SLC_DROT(&len_inner, &de[(j - 1) + mj2 * ldde], &ldde, &de[j + mj2 * ldde], &ldde, &co_inner, &si_inner);
            }

            if (lcmpq) {
                SLC_DROT(&n, &q[(m + j - 1) * ldq], &one_inner, &q[(m + j) * ldq], &one_inner, &co_inner, &si_inner);
            }

            f64 tmp2_inner;
            SLC_DLARTG(&a[j + j * lda], &tmp1_inner, &co_inner, &si_inner, &tmp2_inner);

            a[j + j * lda] = tmp2_inner;
            i32 jval = j;
            SLC_DROT(&jval, &a[j * lda], &one_inner, &a[(j - 1) * lda], &one_inner, &co_inner, &si_inner);

            SLC_DROT(&m, &b[j * ldb], &one_inner, &b[(j - 1) * ldb], &one_inner, &co_inner, &si_inner);
            len_inner = j - k - 1;
            if (len_inner > 0) {
                SLC_DROT(&len_inner, &fg[j + k * ldfg], &ldfg, &fg[(j - 1) + k * ldfg], &ldfg, &co_inner, &si_inner);
            }
            len_inner = m - j - 1;
            if (len_inner > 0) {
                SLC_DROT(&len_inner, &fg[mj1 + j * ldfg], &one_inner, &fg[mj1 + (j - 1) * ldfg], &one_inner, &co_inner, &si_inner);
            }

            if (lcmpq) {
                SLC_DROT(&n, &q[j * ldq], &one_inner, &q[(j - 1) * ldq], &one_inner, &co_inner, &si_inner);
            }
        }
    }

    const char *cmpsc = ltri ? "S" : "E";
    const char *cmpq_str = lcmpq ? "I" : "N";
    const char *cmpz_str = ltri ? "I" : "N";

    i32 lwork_avail = ldwork - iwrk;
    if (lwork_avail < 1) lwork_avail = 1;
    i32 info_dhgeqz = 0;

    SLC_DHGEQZ(cmpsc, cmpq_str, cmpz_str, &m, &(i32){1}, &m, b, &ldb, a, &lda,
               alphar, alphai, beta, &dwork[iq1], &m, &dwork[iq2], &m,
               &dwork[iwrk], &lwork_avail, &info_dhgeqz);

    if (info_dhgeqz > 0) {
        *info = 1;
        return;
    }

    i32 optdw = (mindw > (i32)dwork[iwrk] + iwrk - 1) ? mindw : (i32)dwork[iwrk] + iwrk - 1;

    i32 j = 0;
    i32 nbeta0 = 0;
    while (j < m) {
        if (alphai[j] != ZERO) {
            f64 tmp1, tmp2;
            if (beta[j] >= beta[j + 1]) {
                tmp2 = beta[j + 1] / beta[j];
                tmp1 = (alphar[j] * tmp2 + alphar[j + 1]) / TWO;
                tmp2 = (alphai[j] * tmp2 - alphai[j + 1]) / TWO;
                beta[j] = beta[j + 1];
            } else {
                tmp2 = beta[j] / beta[j + 1];
                tmp1 = (alphar[j + 1] * tmp2 + alphar[j]) / TWO;
                tmp2 = (alphai[j + 1] * tmp2 - alphai[j]) / TWO;
                beta[j + 1] = beta[j];
            }
            alphar[j] = tmp1;
            alphar[j + 1] = tmp1;
            alphai[j] = tmp2;
            alphai[j + 1] = -tmp2;
            j += 2;
        } else {
            if (ninf > 0) {
                if (beta[j] == ZERO) {
                    nbeta0++;
                }
            }
            j++;
        }
    }
    if (j == m && ninf > 0) {
        if (beta[m - 1] == ZERO) {
            nbeta0++;
        }
    }

    if (ninf > 0) {
        for (i32 jj = 0; jj < ninf - nbeta0; jj++) {
            f64 tmp1 = ZERO;
            f64 tmp2 = ONE;
            i32 p = 0;
            for (i32 kk = 0; kk < m; kk++) {
                if (beta[kk] > ZERO) {
                    if (fabs(alphar[kk]) * tmp2 > tmp1 * beta[kk]) {
                        tmp1 = fabs(alphar[kk]);
                        tmp2 = beta[kk];
                        p = kk;
                    }
                }
            }
            beta[p] = ZERO;
        }
    }

    if (ltri) {
        i32 info_mb01ld = 0;
        mb01ld("U", "T", m, m, ZERO, ONE, &de[ldde], ldde, &dwork[iq1], m,
               &de[ldde], ldde, &dwork[iwrk], ldwork - iwrk, &info_mb01ld);

        mb01ld("U", "T", m, m, ZERO, ONE, &fg[ldfg], ldfg, &dwork[iq1], m,
               &fg[ldfg], ldfg, &dwork[iwrk], ldwork - iwrk, &info_mb01ld);
    }

    if (lcmpq) {
        if (ldwork >= n * n) {
            SLC_DGEMM("N", "N", &n, &m, &m, &ONE, q, &ldq, &dwork[iq2], &m, &ZERO, &dwork[iwrk], &n);
            SLC_DLACPY("F", &n, &m, &dwork[iwrk], &n, q, &ldq);
            SLC_DGEMM("N", "N", &n, &m, &m, &ONE, &q[m * ldq], &ldq, &dwork[iq1], &m, &ZERO, &dwork[iwrk], &n);
            SLC_DLACPY("F", &n, &m, &dwork[iwrk], &n, &q[m * ldq], &ldq);
        } else {
            SLC_DGEMM("N", "N", &m, &m, &m, &ONE, q, &ldq, &dwork[iq2], &m, &ZERO, &dwork[iwrk], &m);
            SLC_DLACPY("F", &m, &m, &dwork[iwrk], &m, q, &ldq);
            SLC_DGEMM("N", "N", &m, &m, &m, &ONE, &q[m], &ldq, &dwork[iq2], &m, &ZERO, &dwork[iwrk], &m);
            SLC_DLACPY("F", &m, &m, &dwork[iwrk], &m, &q[m], &ldq);
            SLC_DGEMM("N", "N", &m, &m, &m, &ONE, &q[m * ldq], &ldq, &dwork[iq1], &m, &ZERO, &dwork[iwrk], &m);
            SLC_DLACPY("F", &m, &m, &dwork[iwrk], &m, &q[m * ldq], &ldq);
            SLC_DGEMM("N", "N", &m, &m, &m, &ONE, &q[m + m * ldq], &ldq, &dwork[iq1], &m, &ZERO, &dwork[iwrk], &m);
            SLC_DLACPY("F", &m, &m, &dwork[iwrk], &m, &q[m + m * ldq], &ldq);
        }
    }

    f64 tols = TEN * SLC_DLAMCH("P") * nrms;
    f64 tolt = TEN * SLC_DLAMCH("P") * nrmt;
    i32 p = 0;
    i32 kk = 0;
    bool sing, psng;

    while (kk < m) {
        if (beta[kk] != ZERO) {
            if (alphai[kk] == ZERO) {
                sing = fabs(b[kk + kk * ldb]) < tolt;
                if (sing) {
                    p++;
                    iwork[p] = kk + 1;
                }
                if (fabs(a[kk + kk * lda]) < tols) {
                    if (sing) {
                        *info = 2;
                    } else {
                        p++;
                        iwork[p] = kk + 1;
                    }
                }
            } else {
                f64 x1 = b[kk + kk * ldb];
                f64 x2 = b[(kk + 1) + kk * ldb];
                f64 x3 = b[kk + (kk + 1) * ldb];
                f64 x4 = b[(kk + 1) + (kk + 1) * ldb];
                i32 two = 2;
                f64 norm_b = SLC_DLANGE("F", &two, &two, &b[kk + kk * ldb], &ldb, dwork);

                f64 sdet = (fabs(x1) > fabs(x4) ? fabs(x1) : fabs(x4)) / norm_b *
                           (fabs(x1) < fabs(x4) ? fabs(x1) : fabs(x4)) *
                           (x1 >= 0 ? ONE : -ONE) * (x4 >= 0 ? ONE : -ONE) -
                           (fabs(x2) > fabs(x3) ? fabs(x2) : fabs(x3)) / norm_b *
                           (fabs(x2) < fabs(x3) ? fabs(x2) : fabs(x3)) *
                           (x2 >= 0 ? ONE : -ONE) * (x3 >= 0 ? ONE : -ONE);

                if (norm_b > ONE) {
                    psng = fabs(sdet) < tolt / norm_b;
                } else {
                    psng = fabs(sdet) * norm_b < tolt;
                }

                if (psng) {
                    f64 co, si, tmp_val;
                    if (fabs(x1) >= fabs(x4)) {
                        SLC_DLARTG(&x1, &x2, &co, &si, &tmp_val);
                        x1 = tmp_val;
                        tmp_val = co * x3 + si * x4;
                        x4 = co * x4 - si * x3;
                        x3 = tmp_val;
                    } else {
                        SLC_DLARTG(&x4, &x2, &co, &si, &tmp_val);
                        x4 = tmp_val;
                        tmp_val = co * x3 + si * x1;
                        x1 = co * x1 - si * x3;
                        x3 = tmp_val;
                    }
                    f64 tmp1, tmp2_sv;
                    SLC_DLAS2(&x1, &x3, &x4, &tmp1, &tmp2_sv);
                    sing = tmp1 < tolt;
                    if (sing) {
                        p++;
                        iwork[p] = -(kk + 1);
                        *info = 2;
                    }
                }

                x1 = a[kk + kk * lda];
                x4 = a[(kk + 1) + (kk + 1) * lda];
                norm_b = sqrt(x1 * x1 + x4 * x4);
                f64 max_x = (fabs(x1) > fabs(x4)) ? fabs(x1) : fabs(x4);
                f64 min_x = (fabs(x1) < fabs(x4)) ? fabs(x1) : fabs(x4);
                sdet = (max_x / norm_b) * min_x;
                if (fabs(sdet) < tols) {
                    if (sing) {
                        *info = 2;
                    } else {
                        p++;
                        iwork[p] = -(kk + 1);
                    }
                }
                kk++;
            }
        } else {
            iwork[kk + 1] = 0;
        }
        kk++;
    }
    iwork[0] = p;

    dwork[0] = (f64)optdw;
    dwork[1] = nrms;
    dwork[2] = nrmt;
}
