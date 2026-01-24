/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 *
 * MB04ED - Eigenvalues and orthogonal decomposition of a real
 *          skew-Hamiltonian/skew-Hamiltonian pencil in factored form
 *
 * Computes eigenvalues of aS - bT where:
 *   S = J*Z'*J'*Z
 *   T = [[B, F], [G, B']]
 *   J = [[0, I], [-I, 0]]
 */

/* #define MB04ED_DEBUG 1 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>

void mb04ed(const char *job, const char *compq, const char *compu,
            i32 n, f64 *z, i32 ldz, f64 *b, i32 ldb, f64 *fg, i32 ldfg,
            f64 *q, i32 ldq, f64 *u1, i32 ldu1, f64 *u2, i32 ldu2,
            f64 *alphar, f64 *alphai, f64 *beta,
            i32 *iwork, i32 liwork, f64 *dwork, i32 ldwork, i32 *info) {

    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const f64 FOUR = 4.0;
    const i32 zero_inc = 0;

    char job_upper = (char)toupper((unsigned char)job[0]);
    char compq_upper = (char)toupper((unsigned char)compq[0]);
    char compu_upper = (char)toupper((unsigned char)compu[0]);

    bool ltri = (job_upper == 'T');
    bool lcmpq = (compq_upper == 'I');
    bool liniu = (compu_upper == 'I');
    bool lupdu = (compu_upper == 'U');
    bool lcmpu = liniu || lupdu;

    i32 m = n / 2;
    i32 mm = m * m;

    i32 mindw;
    if (n == 0) {
        mindw = 4;
    } else if (ltri || lcmpq || lcmpu) {
        mindw = 6 * mm + (3 * n > 27 ? 3 * n : 27);
    } else {
        mindw = 3 * mm + (3 * n > 27 ? 3 * n : 27);
    }

    bool lquery = (ldwork == -1);

    *info = 0;
    if (!(job_upper == 'E' || ltri)) {
        *info = -1;
    } else if (!(compq_upper == 'N' || lcmpq)) {
        *info = -2;
    } else if (!(compu_upper == 'N' || lcmpu)) {
        *info = -3;
    } else if (n < 0 || (n % 2) != 0) {
        *info = -4;
    } else if (ldz < (1 > n ? 1 : n)) {
        *info = -6;
    } else if (ldb < (1 > m ? 1 : m)) {
        *info = -8;
    } else if (ldfg < (1 > m ? 1 : m)) {
        *info = -10;
    } else if (ldq < (1 > n ? 1 : n)) {
        *info = -12;
    } else if (ldu1 < 1 || (lcmpu && ldu1 < m)) {
        *info = -14;
    } else if (ldu2 < 1 || (lcmpu && ldu2 < m)) {
        *info = -16;
    } else if (liwork < n + 9) {
        *info = -21;
    } else if (!lquery && ldwork < mindw) {
        dwork[0] = (f64)mindw;
        *info = -23;
    }

    if (*info != 0) {
        return;
    }

    if (n > 0 && lquery) {
        i32 lwork_query = -1;
        f64 dum[4];
        f64 *tmp_dwork = dwork;

        SLC_DGEQRF(&n, &m, tmp_dwork, &n, tmp_dwork, dum, &lwork_query, info);
        f64 d1 = dum[0];

        SLC_DORMQR("R", "N", &n, &n, &m, tmp_dwork, &n, tmp_dwork, q, &ldq, &dum[1], &lwork_query, info);
        f64 d2 = dum[1];

        SLC_DGERQF(&m, &m, z, &ldz, tmp_dwork, &dum[2], &lwork_query, info);
        f64 d3 = dum[2];

        SLC_DORMRQ("R", "T", &n, &m, &m, z, &ldz, tmp_dwork, q, &ldq, &dum[3], &lwork_query, info);
        f64 d4 = dum[3];

        i32 j1 = (i32)d1 > (i32)d2 ? (i32)d1 : (i32)d2;
        i32 j2 = n * m + j1;
        j2 = j2 > (i32)d3 ? j2 : (i32)d3;
        j2 = j2 > (i32)d4 ? j2 : (i32)d4;
        j2 = j2 + m;
        j2 = j2 > 3 * mm ? j2 : 3 * mm;
        dwork[0] = (f64)(mindw > j2 ? mindw : j2);
        return;
    }

    if (n == 0) {
        iwork[0] = 0;
        dwork[0] = FOUR;
        dwork[1] = ZERO;
        dwork[2] = ZERO;
        dwork[3] = ZERO;
        return;
    }

    char param = 'B';
    f64 base = SLC_DLAMCH(&param);
    param = 'M';
    i32 emin = (i32)SLC_DLAMCH(&param);
    param = 'L';
    i32 emax = (i32)SLC_DLAMCH(&param);

    i32 ninf = 0;
    if (n == 1) {
        if (z[0] == ZERO) {
            ninf = 1;
        }
    } else {
        i32 nm1 = n - 1;
        f64 norm_lower = SLC_DLANTR("M", "L", "N", &nm1, &nm1, &z[1], &ldz, dwork);
        f64 norm_upper = SLC_DLANTR("M", "U", "N", &nm1, &nm1, &z[ldz], &ldz, dwork);

        if (norm_lower == ZERO && norm_upper == ZERO) {
            for (i32 j = 0; j < m; j++) {
                if (z[j + j * ldz] == ZERO || z[(j + m) + (j + m) * ldz] == ZERO) {
                    ninf++;
                }
            }
        } else {
            for (i32 j = 0; j < m; j++) {
                i32 one = 1;
                i32 i = SLC_IDAMAX(&n, &z[j * ldz], &one) - 1;
                i32 k = SLC_IDAMAX(&n, &z[(m + j) * ldz], &one) - 1;
                i32 l = SLC_IDAMAX(&n, &z[j], &ldz) - 1;
                i32 p = SLC_IDAMAX(&n, &z[m + j], &ldz) - 1;
                if (z[i + j * ldz] == ZERO || z[k + (m + j) * ldz] == ZERO ||
                    z[j + l * ldz] == ZERO || z[(m + j) + p * ldz] == ZERO) {
                    ninf++;
                }
            }
        }
    }

    SLC_DLASET("F", &n, &n, &ZERO, &ONE, q, &ldq);

    if (liniu) {
        SLC_DLASET("F", &m, &m, &ZERO, &ONE, u1, &ldu1);
        SLC_DLASET("F", &m, &m, &ZERO, &ZERO, u2, &ldu2);
    }

    i32 itau = n * m;
    i32 iwrk = itau + m;

    ma02ad("F", m, n, &z[m], ldz, dwork, n);

    i32 lwork_avail = ldwork - iwrk;
    SLC_DGEQRF(&n, &m, dwork, &n, &dwork[itau], &dwork[iwrk], &lwork_avail, info);
    i32 optdw = mindw > (i32)dwork[iwrk] + iwrk ? mindw : (i32)dwork[iwrk] + iwrk;

    SLC_DORMQR("R", "N", &m, &n, &m, dwork, &n, &dwork[itau], z, &ldz, &dwork[iwrk], &lwork_avail, info);

    ma02ad("U", m, m, dwork, n, &z[m + m * ldz], ldz);

    if (m > 1) {
        i32 mm1 = m - 1;
        SLC_DLASET("U", &mm1, &mm1, &ZERO, &ZERO, &z[m + (m + 1) * ldz], &ldz);
    }

    SLC_DORMQR("R", "N", &n, &n, &m, dwork, &n, &dwork[itau], q, &ldq, &dwork[iwrk], &lwork_avail, info);
    optdw = optdw > (i32)dwork[iwrk] + iwrk ? optdw : (i32)dwork[iwrk] + iwrk;

    for (i32 i = 0; i < m; i++) {
        i32 one = 1;
        SLC_DSWAP(&n, &q[i * ldq], &one, &q[(m + i) * ldq], &one);
    }

    itau = 0;
    iwrk = itau + m;

    lwork_avail = ldwork - iwrk;
    SLC_DGERQF(&m, &m, &z[m * ldz], &ldz, &dwork[itau], &dwork[iwrk], &lwork_avail, info);
    optdw = optdw > (i32)dwork[iwrk] + iwrk ? optdw : (i32)dwork[iwrk] + iwrk;

    SLC_DORMRQ("R", "T", &n, &m, &m, &z[m * ldz], &ldz, &dwork[itau], q, &ldq, &dwork[iwrk], &lwork_avail, info);
    optdw = optdw > (i32)dwork[iwrk] + iwrk ? optdw : (i32)dwork[iwrk] + iwrk;

    f64 dum0 = ZERO;
    for (i32 j = 0; j < m - 1; j++) {
        i32 one = 1;
        SLC_DSWAP(&m, &z[j * ldz], &one, &z[(m + j) * ldz], &one);
        i32 len = m - j - 1;
        SLC_DCOPY(&len, &dum0, &zero_inc, &z[(j + 1) + j * ldz], &one);
    }
    {
        i32 one = 1;
        SLC_DSWAP(&m, &z[(m - 1) * ldz], &one, &z[(n - 1) * ldz], &one);
    }

    i32 icf = 0;
    i32 icg = icf + mm;
    iwrk = icg + mm;

    if (m > 1) {
        i32 mm1 = m - 1;
        SLC_DLACPY("U", &mm1, &mm1, &fg[2 * ldfg], &ldfg, &dwork[icf + m], &m);
        ma02ad("L", mm1, mm1, &fg[1], ldfg, &dwork[icg + m], m);
    }

    mb01ld("U", "T", m, m, ZERO, ONE, &dwork[icf], m, &q[m + m * ldq], ldq, &dwork[icf], m, &dwork[iwrk], ldwork - iwrk, info);
    mb01ld("U", "T", m, m, ONE, ONE, &dwork[icf], m, &q[m * ldq], ldq, &dwork[icg], m, &dwork[iwrk], ldwork - iwrk, info);
    SLC_DGEMM("N", "N", &m, &m, &m, &ONE, b, &ldb, &q[m * ldq], &ldq, &ZERO, &dwork[iwrk], &m);
    mb01kd("U", "T", m, m, ONE, &q[m + m * ldq], ldq, &dwork[iwrk], m, ONE, &dwork[icf], m, info);

    if (m > 1) {
        i32 mm1 = m - 1;
        SLC_DLACPY("L", &mm1, &mm1, &fg[1], &ldfg, &dwork[icg + 1], &m);
        ma02ad("U", mm1, mm1, &fg[2 * ldfg], ldfg, &dwork[icf + 1], m);
    }

    mb01ld("L", "T", m, m, ZERO, ONE, &dwork[icg], m, q, ldq, &dwork[icg], m, &dwork[iwrk], ldwork - iwrk, info);
    mb01ld("L", "T", m, m, ONE, ONE, &dwork[icg], m, &q[m], ldq, &dwork[icf], m, &dwork[iwrk], ldwork - iwrk, info);
    SLC_DGEMM("T", "N", &m, &m, &m, &ONE, b, &ldb, &q[m], &ldq, &ZERO, &dwork[iwrk], &m);
    mb01kd("L", "T", m, m, ONE, q, ldq, &dwork[iwrk], m, ONE, &dwork[icg], m, info);

    i32 info_dummy;
    for (i32 i = 0; i < m; i++) {
        i32 one = 1;
        mb01md('U', m, ONE, &fg[ldfg], ldfg, &q[m + i * ldq], one, ZERO, &dwork[iwrk + i * m], one, &info_dummy);
    }
    if (m > 1) {
        i32 mm1 = m - 1;
        SLC_DLACPY("U", &mm1, &mm1, &dwork[icf + m], &m, &fg[2 * ldfg], &ldfg);
    }
    SLC_DGEMM("N", "N", &m, &m, &m, &ONE, b, &ldb, q, &ldq, &ONE, &dwork[iwrk], &m);
    SLC_DGEMM("T", "N", &m, &m, &m, &ONE, &q[m + m * ldq], &ldq, &dwork[iwrk], &m, &ZERO, &dwork[icf], &m);

    for (i32 i = 0; i < m; i++) {
        i32 one = 1;
        mb01md('L', m, ONE, fg, ldfg, &q[i * ldq], one, ZERO, &dwork[iwrk + i * m], one, &info_dummy);
    }
    SLC_DGEMM("T", "N", &m, &m, &m, &ONE, b, &ldb, &q[m], &ldq, &ONE, &dwork[iwrk], &m);
    f64 neg_one = -ONE;
    SLC_DGEMM("T", "N", &m, &m, &m, &neg_one, &q[m * ldq], &ldq, &dwork[iwrk], &m, &ONE, &dwork[icf], &m);

    if (m > 1) {
        i32 mm1 = m - 1;
        SLC_DLACPY("L", &mm1, &mm1, &dwork[icg + 1], &m, &fg[1], &ldfg);
    }
    SLC_DLACPY("F", &m, &m, &dwork[icf], &m, b, &ldb);

#ifdef MB04ED_DEBUG
    fprintf(stderr, "MB04ED: B after preprocessing (before reduction loops):\n");
    for (i32 i = 0; i < m; i++) {
        fprintf(stderr, "  ");
        for (i32 j_ = 0; j_ < m; j_++) {
            fprintf(stderr, "%+.8e ", b[i + j_ * ldb]);
        }
        fprintf(stderr, "\n");
    }
#endif

    for (i32 k = 0; k < m - 1; k++) {
        for (i32 j = k + 1; j < m - 1; j++) {
            i32 mj2 = (j + 2 < m) ? j + 2 : m;
            i32 mj3 = mj2 + 1;

            f64 co, si, tmp1;
            SLC_DLARTG(&fg[(j + 1) + k * ldfg], &fg[j + k * ldfg], &co, &si, &tmp1);
#ifdef MB04ED_DEBUG
            fprintf(stderr, "MB04ED: k=%d, j=%d: DLARTG input fg[%d]=%e, fg[%d]=%e\n",
                    k, j, (j+1)+k*ldfg, fg[(j+1)+k*ldfg], j+k*ldfg, fg[j+k*ldfg]);
            fprintf(stderr, "MB04ED: k=%d, j=%d: DLARTG output co=%e, si=%e, tmp1=%e\n", k, j, co, si, tmp1);
#endif

            i32 one = 1;
            i32 one_col = 1;
            SLC_DROT(&m, &b[(j + 1) * ldb], &one_col, &b[j * ldb], &one_col, &co, &si);
#ifdef MB04ED_DEBUG
            fprintf(stderr, "MB04ED: k=%d, j=%d: B after DROT(B columns j+1=%d, j=%d):\n", k, j, j+1, j);
            for (i32 ii = 0; ii < m; ii++) {
                fprintf(stderr, "  ");
                for (i32 jj = 0; jj < m; jj++) {
                    fprintf(stderr, "%+.8e ", b[ii + jj * ldb]);
                }
                fprintf(stderr, "\n");
            }
#endif
            i32 len = m - j - 2;
            if (len > 0) {
                SLC_DROT(&len, &fg[mj2 + (j + 1) * ldfg], &one, &fg[mj2 + j * ldfg], &one, &co, &si);
            }
            fg[(j + 1) + k * ldfg] = tmp1;
            len = j - k - 1;
            if (len > 0) {
                SLC_DROT(&len, &fg[(j + 1) + (k + 1) * ldfg], &ldfg, &fg[j + (k + 1) * ldfg], &ldfg, &co, &si);
            }

            i32 jp1 = j + 1;
            SLC_DROT(&jp1, &z[(j + 1) * ldz], &one, &z[j * ldz], &one, &co, &si);
            tmp1 = -si * z[(j + 1) + (j + 1) * ldz];
            z[(j + 1) + (j + 1) * ldz] = co * z[(j + 1) + (j + 1) * ldz];

            if (lcmpq) {
                SLC_DROT(&n, &q[(j + 1) * ldq], &one, &q[j * ldq], &one, &co, &si);
            }

            f64 tmp2;
            SLC_DLARTG(&z[j + j * ldz], &tmp1, &co, &si, &tmp2);

            z[j + j * ldz] = tmp2;
            z[(j + 1) + j * ldz] = ZERO;
            len = n - j - 1;
            SLC_DROT(&len, &z[j + (j + 1) * ldz], &ldz, &z[(j + 1) + (j + 1) * ldz], &ldz, &co, &si);
            SLC_DROT(&jp1, &z[(m + j) + m * ldz], &ldz, &z[(m + j + 1) + m * ldz], &ldz, &co, &si);
            tmp1 = si * z[(m + j + 1) + (m + j + 1) * ldz];
            z[(m + j + 1) + (m + j + 1) * ldz] = co * z[(m + j + 1) + (m + j + 1) * ldz];

            if (lcmpu) {
                SLC_DROT(&m, &u1[j * ldu1], &one, &u1[(j + 1) * ldu1], &one, &co, &si);
                SLC_DROT(&m, &u2[j * ldu2], &one, &u2[(j + 1) * ldu2], &one, &co, &si);
            }

            SLC_DLARTG(&z[(m + j) + (m + j) * ldz], &tmp1, &co, &si, &tmp2);

            SLC_DROT(&m, &z[(m + j) * ldz], &one, &z[(m + j + 1) * ldz], &one, &co, &si);
            z[(m + j) + (m + j) * ldz] = tmp2;
            len = m - j - 1;
            SLC_DROT(&len, &z[(m + j + 1) + (m + j) * ldz], &one, &z[(m + j + 1) + (m + j + 1) * ldz], &one, &co, &si);

            len = m - k;
            SLC_DROT(&len, &b[j + k * ldb], &ldb, &b[(j + 1) + k * ldb], &ldb, &co, &si);
#ifdef MB04ED_DEBUG
            fprintf(stderr, "MB04ED: k=%d, j=%d: 2nd B DROT len=%d, co=%e, si=%e\n", k, j, len, co, si);
            fprintf(stderr, "MB04ED: k=%d, j=%d: B after 2nd DROT (B rows j=%d, j+1=%d):\n", k, j, j, j+1);
            for (i32 ii = 0; ii < m; ii++) {
                fprintf(stderr, "  ");
                for (i32 jj = 0; jj < m; jj++) {
                    fprintf(stderr, "%+.8e ", b[ii + jj * ldb]);
                }
                fprintf(stderr, "\n");
            }
#endif
            len = j;
            if (len > 0) {
#ifdef MB04ED_DEBUG
                fprintf(stderr, "MB04ED: j=%d, DROT fg cols %d and %d, len=%d, ldfg=%d, m=%d, total_fg=%d\n",
                        j, j+1, j+2, len, ldfg, m, ldfg*(m+2));
                if ((j + 2) * ldfg + len - 1 >= ldfg * (m + 2)) {
                    fprintf(stderr, "MB04ED ERROR: out of bounds! max_offset=%d, limit=%d\n",
                            (j + 2) * ldfg + len - 1, ldfg * (m + 2) - 1);
                }
#endif
                SLC_DROT(&len, &fg[(j + 1) * ldfg], &one, &fg[(j + 2) * ldfg], &one, &co, &si);
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
#ifdef MB04ED_DEBUG
        fprintf(stderr, "MB04ED: k=%d: Annihilate G(k,m) - DLARTG inputs: b[%d]=%e, -fg[%d]=%e\n",
                k, (m-1)+k*ldb, b[(m-1)+k*ldb], (m-1)+k*ldfg, -fg[(m-1)+k*ldfg]);
#endif
        SLC_DLARTG(&b[(m - 1) + k * ldb], &(f64){-fg[(m - 1) + k * ldfg]}, &co, &si, &tmp1);
#ifdef MB04ED_DEBUG
        fprintf(stderr, "MB04ED: k=%d: DLARTG outputs: co=%e, si=%e, tmp1=%e\n", k, co, si, tmp1);
#endif

        i32 one = 1;
        i32 mm1 = m - 1;
#ifdef MB04ED_DEBUG
        fprintf(stderr, "MB04ED: k=%d: BEFORE 1st DROT, fg col %d: ", k, m);
        for (i32 ii = 0; ii < mm1; ii++) {
            fprintf(stderr, "%+.8e ", fg[m * ldfg + ii]);
        }
        fprintf(stderr, "\n");
        fprintf(stderr, "MB04ED: k=%d: BEFORE 1st DROT, b col %d: ", k, m-1);
        for (i32 ii = 0; ii < mm1; ii++) {
            fprintf(stderr, "%+.8e ", b[(m - 1) * ldb + ii]);
        }
        fprintf(stderr, "\n");
#endif
        SLC_DROT(&mm1, &fg[m * ldfg], &one, &b[(m - 1) * ldb], &one, &co, &si);
#ifdef MB04ED_DEBUG
        fprintf(stderr, "MB04ED: k=%d: after 1st DROT(fg col %d, b col %d, len=%d), B:\n", k, m, m-1, mm1);
        for (i32 ii = 0; ii < m; ii++) {
            fprintf(stderr, "  ");
            for (i32 jj = 0; jj < m; jj++) {
                fprintf(stderr, "%+.8e ", b[ii + jj * ldb]);
            }
            fprintf(stderr, "\n");
        }
#endif
        b[(m - 1) + k * ldb] = tmp1;
        i32 len = m - k - 2;
        if (len > 0) {
            SLC_DROT(&len, &fg[(m - 1) + (k + 1) * ldfg], &ldfg, &b[(m - 1) + (k + 1) * ldb], &ldb, &co, &si);
#ifdef MB04ED_DEBUG
            fprintf(stderr, "MB04ED: k=%d: after 2nd DROT(fg row %d col %d, b row %d col %d, len=%d), B:\n",
                    k, m-1, k+1, m-1, k+1, len);
            for (i32 ii = 0; ii < m; ii++) {
                fprintf(stderr, "  ");
                for (i32 jj = 0; jj < m; jj++) {
                    fprintf(stderr, "%+.8e ", b[ii + jj * ldb]);
                }
                fprintf(stderr, "\n");
            }
#endif
        }

        SLC_DROT(&m, &z[(n - 1) * ldz], &one, &z[(m - 1) * ldz], &one, &co, &si);
        tmp1 = -si * z[(n - 1) + (n - 1) * ldz];
        z[(n - 1) + (n - 1) * ldz] = co * z[(n - 1) + (n - 1) * ldz];

        if (lcmpq) {
            SLC_DROT(&n, &q[(n - 1) * ldq], &one, &q[(m - 1) * ldq], &one, &co, &si);
        }

        f64 tmp2;
        SLC_DLARTG(&z[(m - 1) + (m - 1) * ldz], &tmp1, &co, &si, &tmp2);

        z[(m - 1) + (m - 1) * ldz] = tmp2;
        SLC_DROT(&m, &z[(m - 1) + m * ldz], &ldz, &z[(n - 1) + m * ldz], &ldz, &co, &si);

        if (lcmpu) {
            SLC_DROT(&m, &u1[(m - 1) * ldu1], &one, &u2[(m - 1) * ldu2], &one, &co, &si);
        }

#ifdef MB04ED_DEBUG
        fprintf(stderr, "MB04ED: k=%d: BEFORE reverse loop, B matrix:\n", k);
        for (i32 ii = 0; ii < m; ii++) {
            fprintf(stderr, "  ");
            for (i32 jj = 0; jj < m; jj++) {
                fprintf(stderr, "%+.8e ", b[ii + jj * ldb]);
            }
            fprintf(stderr, "\n");
        }
#endif
        for (i32 j = m - 1; j >= k + 2; j--) {
            i32 mj1 = (j + 1 < m) ? j + 1 : m;
            i32 mj2 = mj1 + 1;

            f64 co_inner, si_inner, tmp1_inner;
            SLC_DLARTG(&b[(j - 1) + k * ldb], &b[j + k * ldb], &co_inner, &si_inner, &tmp1_inner);

            i32 one_inner = 1;
            i32 len_inner = j - 1;
            if (len_inner > 0) {
                SLC_DROT(&len_inner, &fg[j * ldfg], &one_inner, &fg[(j + 1) * ldfg], &one_inner, &co_inner, &si_inner);
            }
            b[(j - 1) + k * ldb] = tmp1_inner;
            b[j + k * ldb] = ZERO;
            len_inner = m - k - 1;
            SLC_DROT(&len_inner, &b[(j - 1) + (k + 1) * ldb], &ldb, &b[j + (k + 1) * ldb], &ldb, &co_inner, &si_inner);
            len_inner = m - j - 1;
            if (len_inner > 0) {
                SLC_DROT(&len_inner, &fg[(j - 1) + mj2 * ldfg], &ldfg, &fg[j + mj2 * ldfg], &ldfg, &co_inner, &si_inner);
            }

            SLC_DROT(&m, &z[(m + j - 1) * ldz], &one_inner, &z[(m + j) * ldz], &one_inner, &co_inner, &si_inner);
            tmp1_inner = -si_inner * z[(m + j - 1) + (m + j - 1) * ldz];
            z[(m + j - 1) + (m + j - 1) * ldz] = co_inner * z[(m + j - 1) + (m + j - 1) * ldz];
            len_inner = m - j;
            SLC_DROT(&len_inner, &z[(m + j) + (m + j - 1) * ldz], &one_inner, &z[(m + j) + (m + j) * ldz], &one_inner, &co_inner, &si_inner);

            if (lcmpq) {
                SLC_DROT(&n, &q[(m + j - 1) * ldq], &one_inner, &q[(m + j) * ldq], &one_inner, &co_inner, &si_inner);
            }

            f64 tmp2_inner;
            SLC_DLARTG(&z[(m + j) + (m + j) * ldz], &tmp1_inner, &co_inner, &si_inner, &tmp2_inner);

            tmp1_inner = si_inner * z[(j - 1) + (j - 1) * ldz];
            z[(j - 1) + (j - 1) * ldz] = co_inner * z[(j - 1) + (j - 1) * ldz];
            len_inner = n - j;
            SLC_DROT(&len_inner, &z[j + j * ldz], &ldz, &z[(j - 1) + j * ldz], &ldz, &co_inner, &si_inner);
            SLC_DROT(&j, &z[(m + j) + m * ldz], &ldz, &z[(m + j - 1) + m * ldz], &ldz, &co_inner, &si_inner);
            z[(m + j) + (m + j) * ldz] = tmp2_inner;

            if (lcmpu) {
                SLC_DROT(&m, &u1[j * ldu1], &one_inner, &u1[(j - 1) * ldu1], &one_inner, &co_inner, &si_inner);
                SLC_DROT(&m, &u2[j * ldu2], &one_inner, &u2[(j - 1) * ldu2], &one_inner, &co_inner, &si_inner);
            }

            SLC_DLARTG(&z[j + j * ldz], &tmp1_inner, &co_inner, &si_inner, &tmp2_inner);

            z[j + j * ldz] = tmp2_inner;
            SLC_DROT(&j, &z[j * ldz], &one_inner, &z[(j - 1) * ldz], &one_inner, &co_inner, &si_inner);

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
#ifdef MB04ED_DEBUG
        fprintf(stderr, "MB04ED: k=%d: AFTER reverse loop (end of outer iter), B matrix:\n", k);
        for (i32 ii = 0; ii < m; ii++) {
            fprintf(stderr, "  ");
            for (i32 jj = 0; jj < m; jj++) {
                fprintf(stderr, "%+.8e ", b[ii + jj * ldb]);
            }
            fprintf(stderr, "\n");
        }
#endif
    }

#ifdef MB04ED_DEBUG
    fprintf(stderr, "MB04ED: After Hessenberg reduction (before DLACPY):\n");
    fprintf(stderr, "MB04ED: B matrix (m x m):\n");
    for (i32 i = 0; i < m; i++) {
        fprintf(stderr, "  ");
        for (i32 j = 0; j < m; j++) {
            fprintf(stderr, "%+.8e ", b[i + j * ldb]);
        }
        fprintf(stderr, "\n");
    }
    fprintf(stderr, "MB04ED: Z(0:m, 0:m) top-left block:\n");
    for (i32 i = 0; i < m; i++) {
        fprintf(stderr, "  ");
        for (i32 j = 0; j < m; j++) {
            fprintf(stderr, "%+.8e ", z[i + j * ldz]);
        }
        fprintf(stderr, "\n");
    }
    fprintf(stderr, "MB04ED: Z(m:n, m:n) bottom-right block:\n");
    for (i32 i = m; i < n; i++) {
        fprintf(stderr, "  ");
        for (i32 j = m; j < n; j++) {
            fprintf(stderr, "%+.8e ", z[i + j * ldz]);
        }
        fprintf(stderr, "\n");
    }
#endif

    i32 iq2 = 0;
    i32 iq1, iu, ib1;
    if (ltri || lcmpq || lcmpu) {
        iq1 = iq2 + mm;
        iu = iq1 + mm;
        ib1 = iu + mm;
    } else {
        ib1 = 0;
    }
    i32 iz11 = ib1 + mm;
    i32 iz22 = iz11 + mm;
    iwrk = iz22 + mm;

    const char *cmpq_str = (ltri || lcmpq || lcmpu) ? "I" : "N";
    const char *cmpsc_str = ltri ? "S" : "E";

    iwork[0] = 1;
    iwork[1] = -1;
    iwork[2] = -1;

    SLC_DLACPY("F", &m, &m, b, &ldb, &dwork[ib1], &m);
    SLC_DLACPY("F", &m, &m, z, &ldz, &dwork[iz11], &m);
    ma02ad("F", m, m, &z[m + m * ldz], ldz, &dwork[iz22], m);

    i32 iwarn_mb03bd = 0;
    i32 info_mb03bd = 0;
    i32 liwork_mb03bd = liwork - (m + 3);
    lwork_avail = ldwork - iwrk;
    i32 three = 3;
    i32 one_int = 1;

#ifdef MB04ED_DEBUG
    fprintf(stderr, "MB04ED: Before mb03bd, m=%d, ib1=%d\n", m, ib1);
    fprintf(stderr, "MB04ED: Input A to mb03bd (3 factors, m x m each):\n");
    for (i32 k = 0; k < 3; k++) {
        fprintf(stderr, "  Factor %d:\n", k);
        for (i32 i = 0; i < m; i++) {
            fprintf(stderr, "    ");
            for (i32 j = 0; j < m; j++) {
                fprintf(stderr, "%+.16e ", dwork[ib1 + i + j*m + k*m*m]);
            }
            fprintf(stderr, "\n");
        }
    }
#endif

    mb03bd(cmpsc_str, "C", cmpq_str, &one_int, three, m, one_int, one_int, m,
           iwork, &dwork[ib1], m, m, &dwork[iq2], m, m,
           alphar, alphai, beta, &iwork[3], &iwork[m + 3],
           liwork_mb03bd, &dwork[iwrk], lwork_avail, &iwarn_mb03bd, &info_mb03bd);

#ifdef MB04ED_DEBUG
    fprintf(stderr, "MB04ED: After mb03bd, iwarn=%d, info=%d, m=%d\n", iwarn_mb03bd, info_mb03bd, m);
    fprintf(stderr, "MB04ED: alphar from mb03bd: ");
    for (i32 i = 0; i < m; i++) fprintf(stderr, "%.6e ", alphar[i]);
    fprintf(stderr, "\nMB04ED: alphai from mb03bd: ");
    for (i32 i = 0; i < m; i++) fprintf(stderr, "%.6e ", alphai[i]);
    fprintf(stderr, "\nMB04ED: beta from mb03bd: ");
    for (i32 i = 0; i < m; i++) fprintf(stderr, "%.6e ", beta[i]);
    fprintf(stderr, "\nMB04ED: scal (iwork[3:]): ");
    for (i32 i = 0; i < m; i++) fprintf(stderr, "%d ", iwork[3 + i]);
    fprintf(stderr, "\nMB04ED: norms (iwork[m+3:]): ");
    for (i32 i = 0; i < m; i++) fprintf(stderr, "%d ", iwork[m + 3 + i]);
    fprintf(stderr, "\n");
#endif

    if (iwarn_mb03bd > 0 && iwarn_mb03bd < m) {
        *info = 1;
        return;
    } else if (iwarn_mb03bd == m + 1) {
        *info = 3;
    } else if (info_mb03bd > 0) {
        *info = 2;
        return;
    }

    i32 nbeta0 = 0;
    i32 i11 = 0;
    i32 i22 = 0;
    i32 i2x2 = 0;

    i32 i = 0;
    while (i < m) {
        if (ninf > 0) {
            if (beta[i] == ZERO) {
                nbeta0++;
            }
        }
        if (iwork[i + 3] >= emin && iwork[i + 3] <= emax) {
            beta[i] = beta[i] / pow(base, (f64)iwork[i + 3]);
            if (beta[i] != ZERO) {
                if (iwork[m + i + 4] < 0) {
                    i22++;
                } else if (iwork[m + i + 4] > 0) {
                    i11++;
                }
                if (alphai[i] < ZERO) {
                    alphai[i] = -alphai[i];
                }
                if (alphar[i] != ZERO && alphai[i] != ZERO) {
                    alphai[i + 1] = -alphai[i];
                    beta[i + 1] = beta[i];
                    i2x2++;
                    i++;
                }
            }
        } else if (iwork[i + 3] < emin) {
            alphar[i] = ZERO;
            alphai[i] = ZERO;
            i11++;
        } else {
            if (ninf > 0) {
                nbeta0++;
            }
            beta[i] = ZERO;
            i11++;
        }
        i++;
    }

    iwork[0] = i11 + i22;

    i32 l = 0;
    if (ninf > 0) {
        for (i32 j = 0; j < ninf - nbeta0; j++) {
            f64 tmp1 = ZERO;
            f64 tmp2 = ONE;
            i32 p = 0;
            for (i32 ii = 0; ii < m; ii++) {
                if (beta[ii] > ZERO) {
                    f64 temp = sqrt(alphar[ii] * alphar[ii] + alphai[ii] * alphai[ii]);
                    if (temp > tmp1 && tmp2 >= beta[ii]) {
                        tmp1 = temp;
                        tmp2 = beta[ii];
                        p = ii;
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

    f64 dum[4];
    dum[0] = dwork[iwrk + 1];
    dum[1] = dwork[iwrk + 2];
    dum[2] = dwork[iwrk + 3];

    i32 kk = iwrk;
    i32 pp = iwrk;
    i32 iw = iwork[0];
    i = 0;
    i32 jj = 0;
    i32 ll = 3 * (m - 2 * i2x2) + kk;

    bool unrel = false;
    while (i < m) {
        if (jj < iw) {
            unrel = (i == abs(iwork[m + i + 4]));
        }
        if (alphar[i] != ZERO && beta[i] != ZERO && alphai[i] != ZERO) {
            if (unrel) {
                jj++;
                iwork[jj] = iwork[m + i + 4];
                iwork[iw + jj] = ll - iwrk + 1;
                unrel = false;
            }
            i32 two = 2;
            SLC_DLACPY("F", &two, &two, &dwork[ib1 + (m + 1) * i], &m, &dwork[ll], &two);
            SLC_DLACPY("F", &two, &two, &dwork[ib1 + (m + 1) * i + mm], &m, &dwork[ll + 4], &two);
            SLC_DLACPY("F", &two, &two, &dwork[ib1 + (m + 1) * i + 2 * mm], &m, &dwork[ll + 8], &two);
            ll += 12;
            i += 2;
        } else {
            if (unrel) {
                jj++;
                iwork[jj] = i;
                iwork[iw + jj] = kk - iwrk + 1;
                unrel = false;
            }
            i32 one_int = 1;
            SLC_DCOPY(&three, &dwork[ib1 + (m + 1) * i], &mm, &dwork[kk], &one_int);
            kk += 3;
            i++;
        }
    }

    iwork[2 * iw + 1] = i11;
    iwork[2 * iw + 2] = i22;
    iwork[2 * iw + 3] = i2x2;

    if (ltri) {
        iwrk = iz11;
        SLC_DLACPY("U", &m, &m, &dwork[iz11], &m, z, &ldz);
        SLC_DGEMM("T", "N", &m, &m, &m, &ONE, &dwork[iu], &m, &z[m * ldz], &ldz, &ZERO, &dwork[iwrk], &m);
        SLC_DGEMM("N", "N", &m, &m, &m, &ONE, &dwork[iwrk], &m, &dwork[iq2], &m, &ZERO, &z[m * ldz], &ldz);
        ma02ad("U", m, m, &dwork[iz22], m, &z[m + m * ldz], ldz);

        SLC_DLACPY("F", &m, &m, &dwork[ib1], &m, b, &ldb);
        iwrk = ib1;

        lwork_avail = ldwork - iwrk;
        i32 itau_dummy;
        mb01ld("U", "T", m, m, ZERO, ONE, &fg[ldfg], ldfg, &dwork[iq2], m, &fg[ldfg], ldfg, &dwork[iwrk], lwork_avail, &itau_dummy);

        if (lcmpq) {
            SLC_DGEMM("N", "N", &n, &m, &m, &ONE, q, &ldq, &dwork[iq1], &m, &ZERO, &dwork[iwrk], &n);
            SLC_DLACPY("F", &n, &m, &dwork[iwrk], &n, q, &ldq);
            SLC_DGEMM("N", "N", &n, &m, &m, &ONE, &q[m * ldq], &ldq, &dwork[iq2], &m, &ZERO, &dwork[iwrk], &n);
            SLC_DLACPY("F", &n, &m, &dwork[iwrk], &n, &q[m * ldq], &ldq);
        }

        if (lcmpu) {
            SLC_DGEMM("N", "N", &m, &m, &m, &ONE, u1, &ldu1, &dwork[iu], &m, &ZERO, &dwork[iwrk], &m);
            SLC_DLACPY("F", &m, &m, &dwork[iwrk], &m, u1, &ldu1);
            SLC_DGEMM("N", "N", &m, &m, &m, &ONE, u2, &ldu2, &dwork[iu], &m, &ZERO, &dwork[iwrk], &m);
            SLC_DLACPY("F", &m, &m, &dwork[iwrk], &m, u2, &ldu2);
        }
    }

    kk = 3 * (m - 2 * i2x2) + 12 * i2x2;
    i32 one_int_copy = 1;
    SLC_DCOPY(&kk, &dwork[pp], &one_int_copy, &dwork[4], &one_int_copy);
    dwork[1] = dum[0];
    dwork[2] = dum[1];
    dwork[3] = dum[2];

    dwork[0] = (f64)optdw;
}
