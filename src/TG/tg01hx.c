/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <stdbool.h>

void tg01hx(
    const char* compq, const char* compz,
    const i32 l, const i32 n, const i32 m, const i32 p,
    const i32 n1, const i32 lbe,
    f64* a, const i32 lda,
    f64* e, const i32 lde,
    f64* b, const i32 ldb,
    f64* c, const i32 ldc,
    f64* q, const i32 ldq,
    f64* z, const i32 ldz,
    i32* nr, i32* nrblck, i32* rtau,
    const f64 tol,
    i32* iwork, f64* dwork,
    i32* info
)
{
    const i32 IMAX = 1, IMIN = 2;
    const f64 one = 1.0, zero = 0.0;

    bool ilq, ilz, withc;
    i32 i, ic, icol, icompq, icompz, irow, ismax, ismin, j, k, mn, nf, nr1, rank, tauim1;
    f64 c1, c2, co, nrma, rcond, s1, s2, si, smax, smaxpr, smin, sminpr, svlmax, t, tolz, tt;

    i32 int1 = 1;

    if (compq[0] == 'N' || compq[0] == 'n') {
        ilq = false;
        icompq = 1;
    } else if (compq[0] == 'U' || compq[0] == 'u') {
        ilq = true;
        icompq = 2;
    } else if (compq[0] == 'I' || compq[0] == 'i') {
        ilq = true;
        icompq = 3;
    } else {
        icompq = 0;
    }

    if (compz[0] == 'N' || compz[0] == 'n') {
        ilz = false;
        icompz = 1;
    } else if (compz[0] == 'U' || compz[0] == 'u') {
        ilz = true;
        icompz = 2;
    } else if (compz[0] == 'I' || compz[0] == 'i') {
        ilz = true;
        icompz = 3;
    } else {
        icompz = 0;
    }

    *info = 0;
    i32 min_ln = (l < n) ? l : n;

    if (icompq <= 0) {
        *info = -1;
    } else if (icompz <= 0) {
        *info = -2;
    } else if (l < 0) {
        *info = -3;
    } else if (n < 0) {
        *info = -4;
    } else if (m < 0) {
        *info = -5;
    } else if (p < 0) {
        *info = -6;
    } else if (n1 < 0 || n1 > min_ln) {
        *info = -7;
    } else if (lbe < 0 || lbe > ((n1 > 0) ? n1 - 1 : 0)) {
        *info = -8;
    } else if (lda < ((1 > l) ? 1 : l)) {
        *info = -10;
    } else if (lde < ((1 > l) ? 1 : l)) {
        *info = -12;
    } else if (ldb < ((1 > l) ? 1 : l)) {
        *info = -14;
    } else if (ldc < ((1 > p) ? 1 : p)) {
        *info = -16;
    } else if ((ilq && ldq < l) || ldq < 1) {
        *info = -18;
    } else if ((ilz && ldz < n) || ldz < 1) {
        *info = -20;
    } else if (tol >= one) {
        *info = -24;
    }

    if (*info != 0) {
        return;
    }

    if (icompq == 3) {
        SLC_DLASET("F", &l, &l, &zero, &one, q, &ldq);
    }
    if (icompz == 3) {
        SLC_DLASET("F", &n, &n, &zero, &one, z, &ldz);
    }

    *nr = 0;
    *nrblck = 0;

    if (m == 0 || n1 == 0) {
        return;
    }

    tolz = sqrt(SLC_DLAMCH("E"));
    withc = (p > 0);
    svlmax = zero;
    nrma = SLC_DLANGE("F", &l, &n, a, &lda, dwork);
    rcond = tol;
    if (rcond <= zero) {
        rcond = (f64)(l * n) * SLC_DLAMCH("E");
    }

    if (lbe > 0) {
        for (i = 0; i < n1 - 1; i++) {
            k = (lbe < n1 - i - 1) ? lbe : (n1 - i - 1);
            k = k + 1;
            SLC_DLARFG(&k, &e[i + i * lde], &e[(i + 1) + i * lde], &int1, &tt);
            t = e[i + i * lde];
            e[i + i * lde] = one;

            i32 n_minus_i_minus_1 = n - i - 1;
            SLC_DLARF("L", &k, &n_minus_i_minus_1, &e[i + i * lde], &int1, &tt,
                      &e[i + (i + 1) * lde], &lde, dwork);

            SLC_DLARF("L", &k, &n, &e[i + i * lde], &int1, &tt,
                      &a[i], &lda, dwork);

            SLC_DLARF("L", &k, &m, &e[i + i * lde], &int1, &tt,
                      &b[i], &ldb, dwork);

            if (ilq) {
                SLC_DLARF("R", &l, &k, &e[i + i * lde], &int1, &tt,
                          &q[i * ldq], &ldq, dwork);
            }

            e[i + i * lde] = t;
        }
        i32 n1_m1 = n1 - 1;
        SLC_DLASET("L", &n1_m1, &n1_m1, &zero, &zero, &e[1], &lde);
    }

    ismin = 0;
    ismax = ismin + m;
    ic = -m;
    tauim1 = m;
    nf = n1;

    while (1) {
        (*nrblck)++;
        rank = 0;

        if (nf > 0) {
            icol = ic + tauim1;
            irow = *nr;
            nr1 = *nr;

            if (*nr > 0) {
                SLC_DLACPY("F", &nf, &tauim1, &a[nr1 + ic * lda], &lda,
                           &b[nr1], &ldb);
                if (svlmax == zero) {
                    svlmax = nrma;
                }
            }

            if (tauim1 > 1) {
                for (j = 0; j < tauim1; j++) {
                    dwork[j] = SLC_DNRM2(&nf, &b[nr1 + j * ldb], &int1);
                    dwork[m + j] = dwork[j];
                    iwork[j] = j + 1;
                }
            }

            mn = (nf < tauim1) ? nf : tauim1;

            while (rank < mn) {
                j = rank;
                irow++;

                if (j != tauim1 - 1) {
                    i32 len = tauim1 - j;
                    k = j + SLC_IDAMAX(&len, &dwork[j], &int1) - 1;
                    if (k != j) {
                        SLC_DSWAP(&nf, &b[nr1 + j * ldb], &int1, &b[nr1 + k * ldb], &int1);
                        i32 tmp = iwork[k];
                        iwork[k] = iwork[j];
                        iwork[j] = tmp;
                        dwork[k] = dwork[j];
                        dwork[m + k] = dwork[m + j];
                    }
                }

                for (i = n1 - 2; i >= irow - 1; i--) {
                    t = b[i + j * ldb];
                    SLC_DLARTG(&t, &b[(i + 1) + j * ldb], &co, &si, &b[i + j * ldb]);
                    b[(i + 1) + j * ldb] = zero;

                    i32 n_minus_i = n - i;
                    SLC_DROT(&n_minus_i, &e[i + i * lde], &lde, &e[(i + 1) + i * lde], &lde, &co, &si);

                    if (j < tauim1 - 1) {
                        i32 tauim1_minus_j_minus_1 = tauim1 - j - 1;
                        SLC_DROT(&tauim1_minus_j_minus_1, &b[i + (j + 1) * ldb], &ldb,
                                 &b[(i + 1) + (j + 1) * ldb], &ldb, &co, &si);
                    }

                    i32 n_minus_icol = n - icol;
                    SLC_DROT(&n_minus_icol, &a[i + icol * lda], &lda,
                             &a[(i + 1) + icol * lda], &lda, &co, &si);

                    if (ilq) {
                        SLC_DROT(&l, &q[i * ldq], &int1, &q[(i + 1) * ldq], &int1, &co, &si);
                    }

                    t = e[(i + 1) + (i + 1) * lde];
                    SLC_DLARTG(&t, &e[(i + 1) + i * lde], &co, &si, &e[(i + 1) + (i + 1) * lde]);
                    e[(i + 1) + i * lde] = zero;

                    i32 i_plus_1 = i + 1;
                    SLC_DROT(&i_plus_1, &e[(i + 1) * lde], &int1, &e[i * lde], &int1, &co, &si);
                    SLC_DROT(&n1, &a[(i + 1) * lda], &int1, &a[i * lda], &int1, &co, &si);

                    if (ilz) {
                        SLC_DROT(&n, &z[(i + 1) * ldz], &int1, &z[i * ldz], &int1, &co, &si);
                    }
                    if (withc) {
                        SLC_DROT(&p, &c[(i + 1) * ldc], &int1, &c[i * ldc], &int1, &co, &si);
                    }
                }

                if (rank == 0) {
                    smax = fabs(b[nr1 + 0 * ldb]);
                    if (smax <= rcond) goto label_80;
                    smin = smax;
                    smaxpr = smax;
                    sminpr = smin;
                    c1 = one;
                    c2 = one;
                } else {
                    SLC_DLAIC1(&IMIN, &rank, &dwork[ismin], &smin,
                               &b[nr1 + j * ldb], &b[(irow - 1) + j * ldb], &sminpr, &s1, &c1);
                    SLC_DLAIC1(&IMAX, &rank, &dwork[ismax], &smax,
                               &b[nr1 + j * ldb], &b[(irow - 1) + j * ldb], &smaxpr, &s2, &c2);
                }

                if (svlmax * rcond <= smaxpr) {
                    if (svlmax * rcond <= sminpr) {
                        if (smaxpr * rcond < sminpr) {
                            if (irow == n1) {
                                rank++;
                                goto label_80;
                            }

                            for (i = j + 1; i < tauim1; i++) {
                                if (dwork[i] != zero) {
                                    t = fabs(b[(irow - 1) + i * ldb]) / dwork[i];
                                    t = ((one + t) * (one - t) > zero) ? (one + t) * (one - t) : zero;
                                    tt = t * (dwork[i] / dwork[m + i]) * (dwork[i] / dwork[m + i]);
                                    if (tt > tolz) {
                                        dwork[i] = dwork[i] * sqrt(t);
                                    } else {
                                        i32 nf_minus_j_minus_1 = nf - j - 1;
                                        dwork[i] = SLC_DNRM2(&nf_minus_j_minus_1, &b[irow + i * ldb], &int1);
                                        dwork[m + i] = dwork[i];
                                    }
                                }
                            }

                            for (i = 0; i < rank; i++) {
                                dwork[ismin + i] = s1 * dwork[ismin + i];
                                dwork[ismax + i] = s2 * dwork[ismax + i];
                            }

                            dwork[ismin + rank] = c1;
                            dwork[ismax + rank] = c2;
                            smin = sminpr;
                            smax = smaxpr;
                            rank++;
                            continue;
                        }
                    }
                }

                if (*nr > 0) {
                    i32 n1_minus_irow = n1 - irow + 1;
                    i32 tauim1_minus_j = tauim1 - j;
                    SLC_DLASET("F", &n1_minus_irow, &tauim1_minus_j, &zero, &zero, &b[(irow - 1) + j * ldb], &ldb);
                }
                goto label_80;
            }
        }

label_80:
        if (rank > 0) {
            rtau[*nrblck - 1] = rank;

            if (tauim1 > 1) {
                for (j = 0; j < tauim1; j++) {
                    if (iwork[j] > 0) {
                        k = iwork[j] - 1;
                        iwork[j] = -(k + 1);
                        while (k != j) {
                            if (k < 0 || k >= tauim1) break;
                            SLC_DSWAP(&rank, &b[nr1 + j * ldb], &int1, &b[nr1 + k * ldb], &int1);
                            i32 next_k_val = iwork[k];
                            iwork[k] = -iwork[k];
                            k = next_k_val - 1;
                        }
                    }
                }
            }
        }

        if (*nr > 0) {
            SLC_DLACPY("F", &nf, &tauim1, &b[nr1], &ldb, &a[nr1 + ic * lda], &lda);
        }

        if (rank > 0) {
            *nr = *nr + rank;
            nf = nf - rank;
            ic = ic + tauim1;
            tauim1 = rank;
        } else {
            (*nrblck)--;
            break;
        }
    }

    if (*nrblck > 0) {
        rank = rtau[0];
    }
    if (rank < n1) {
        i32 n1_minus_rank = n1 - rank;
        SLC_DLASET("F", &n1_minus_rank, &m, &zero, &zero, &b[rank], &ldb);
    }
}
