/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <stdbool.h>

void tg01hu(
    const char* compq, const char* compz,
    const i32 l, const i32 n, const i32 m1, const i32 m2, const i32 p,
    const i32 n1, const i32 lbe,
    f64* a, const i32 lda,
    f64* e, const i32 lde,
    f64* b, const i32 ldb,
    f64* c, const i32 ldc,
    f64* q, const i32 ldq,
    f64* z, const i32 ldz,
    i32* nr, i32* nrblck, i32* rtau,
    const f64 tol,
    i32* iwork, f64* dwork, const i32 ldwork,
    i32* info
)
{
    const i32 IMAX = 1, IMIN = 2;
    const f64 one = 1.0, zero = 0.0;

    bool ilq, ilz, withc, b1red, onecol, lquery;
    i32 i, ic, icol, icompq, icompz, irow, ismax, ismin, j, jb2, k, m, mcrt;
    i32 mcrt1, mcrt2, minwrk, mn, nb, nf, nr1, nx, rank, wrkopt;
    f64 c1, c2, co, rcond, s1, s2, si, smax, smaxpr, smin, sminpr;
    f64 svlmax, svma, svmr, t, tolz, tt;

    i32 int1 = 1;
    i32 min_ln = (l < n) ? l : n;

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
    m = m1 + m2;
    wrkopt = 1;

    if (icompq <= 0) {
        *info = -1;
    } else if (icompz <= 0) {
        *info = -2;
    } else if (l < 0) {
        *info = -3;
    } else if (n < 0) {
        *info = -4;
    } else if (m1 < 0) {
        *info = -5;
    } else if (m2 < 0) {
        *info = -6;
    } else if (p < 0) {
        *info = -7;
    } else if (n1 < 0 || n1 > min_ln) {
        *info = -8;
    } else if (lbe < 0 || lbe > ((n1 > 0) ? n1 - 1 : 0)) {
        *info = -9;
    } else if (lda < ((1 > l) ? 1 : l)) {
        *info = -11;
    } else if (lde < ((1 > l) ? 1 : l)) {
        *info = -13;
    } else if (ldb < ((1 > l) ? 1 : l)) {
        *info = -15;
    } else if (ldc < ((1 > p) ? 1 : p)) {
        *info = -17;
    } else if ((ilq && ldq < l) || ldq < 1) {
        *info = -19;
    } else if ((ilz && ldz < n) || ldz < 1) {
        *info = -21;
    } else if (tol >= one) {
        *info = -25;
    } else {
        i32 min_n1_m = (n1 < m) ? n1 : m;
        if (min_n1_m == 0) {
            minwrk = 1;
        } else if (lbe > 0 && n1 > 2) {
            i32 max_lnm = l;
            if (n > max_lnm) max_lnm = n;
            if (m > max_lnm) max_lnm = m;
            minwrk = n1 + max_lnm;
            if (2 * m > minwrk) minwrk = 2 * m;
        } else {
            minwrk = 1;
            if (l > minwrk) minwrk = l;
            if (n > minwrk) minwrk = n;
            if (2 * m > minwrk) minwrk = 2 * m;
        }

        lquery = (ldwork == -1);
        if (lquery) {
            if (lbe > 0 && n1 > 2) {
                SLC_DGEQRF(&n1, &n1, e, &lde, dwork, dwork, &int1, info);
                wrkopt = minwrk;
                if (n1 + (i32)dwork[0] > wrkopt) wrkopt = n1 + (i32)dwork[0];

                SLC_DORMQR("L", "T", &n1, &n, &n1, e, &lde, dwork, a, &lda, dwork, &int1, info);
                if (n1 + (i32)dwork[0] > wrkopt) wrkopt = n1 + (i32)dwork[0];

                SLC_DORMQR("L", "T", &n1, &m, &n1, e, &lde, dwork, b, &ldb, dwork, &int1, info);
                if (n1 + (i32)dwork[0] > wrkopt) wrkopt = n1 + (i32)dwork[0];

                if (ilq) {
                    SLC_DORMQR("R", "N", &l, &n1, &n1, e, &lde, dwork, q, &ldq, dwork, &int1, info);
                    if (n1 + (i32)dwork[0] > wrkopt) wrkopt = n1 + (i32)dwork[0];
                }
            } else {
                wrkopt = minwrk;
            }
        } else if (ldwork < minwrk) {
            *info = -28;
        }
    }

    if (*info != 0) {
        return;
    } else if (lquery) {
        dwork[0] = (f64)wrkopt;
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

    i32 min_n1_m = (n1 < m) ? n1 : m;
    if (min_n1_m == 0) {
        dwork[0] = one;
        return;
    }

    tolz = SLC_DLAMCH("E");
    withc = (p > 0);
    svlmax = SLC_DLANGE("F", &l, &m, b, &ldb, dwork);
    rcond = tol;
    if (rcond <= zero) {
        rcond = (f64)(l * n) * SLC_DLAMCH("E");
    }
    tolz = sqrt(tolz);

    if (svlmax < rcond) {
        svlmax = one;
    }
    svmr = svlmax * rcond;
    svma = SLC_DLANGE("F", &l, &n, a, &lda, dwork);
    if (svma < one) svma = one;
    svma = svma * rcond;
    if (svma > svmr * tolz) {
        svma = SLC_DLAPY2(&svmr, &svma);
    }

    nx = SLC_ILAENV(&(i32){3}, "DGEQRF", " ", &n1, &n1, &(i32){-1}, &(i32){-1});
    nb = ldwork / n1;

    if (lbe > nx / 2 && (nb < nx ? nb : nx) >= nx && n1 >= nx) {
        i32 ldw_minus_n1 = ldwork - n1;
        SLC_DGEQRF(&n1, &n1, e, &lde, dwork, &dwork[n1], &ldw_minus_n1, info);
        wrkopt = n1 + (i32)dwork[n1];
        if (wrkopt < minwrk) wrkopt = minwrk;

        SLC_DORMQR("L", "T", &n1, &n, &n1, e, &lde, dwork, a, &lda, &dwork[n1], &ldw_minus_n1, info);
        if (n1 + (i32)dwork[n1] > wrkopt) wrkopt = n1 + (i32)dwork[n1];

        SLC_DORMQR("L", "T", &n1, &m, &n1, e, &lde, dwork, b, &ldb, &dwork[n1], &ldw_minus_n1, info);
        if (n1 + (i32)dwork[n1] > wrkopt) wrkopt = n1 + (i32)dwork[n1];

        if (ilq) {
            SLC_DORMQR("R", "N", &l, &n1, &n1, e, &lde, dwork, q, &ldq, &dwork[n1], &ldw_minus_n1, info);
            if (n1 + (i32)dwork[n1] > wrkopt) wrkopt = n1 + (i32)dwork[n1];
        }

        i32 n1_m1 = n1 - 1;
        SLC_DLASET("L", &n1_m1, &n1_m1, &zero, &zero, &e[1], &lde);
    } else if (lbe > 0 && n1 > 1) {
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

    mcrt1 = m1;
    mcrt2 = m2;
    mcrt = mcrt1;
    b1red = true;
    ismin = 0;
    ismax = ismin + m;

    ic = 0;
    nf = n1;
    jb2 = m;

    while (1) {
        if (nf == 0 && b1red) break;

        (*nrblck)++;
        rank = 0;

        if (nf > 0) {
            icol = ic;
            irow = *nr;
            nr1 = *nr;

            if (*nrblck == 2) {
                SLC_DLACPY("F", &nf, &m2, &b[nr1 + m1 * ldb], &ldb, &b[nr1], &ldb);
                jb2 = mcrt;
            } else if (*nrblck > 2) {
                SLC_DLACPY("F", &nf, &mcrt, &a[nr1 + ic * lda], &lda, &b[nr1], &ldb);
                icol = ic + mcrt;
                svmr = svma;
                jb2 = mcrt;
            }

            onecol = (mcrt == 1);

            if (onecol) {
                mn = 1;
            } else {
                mn = (nf < mcrt) ? nf : mcrt;
                for (j = 0; j < mcrt; j++) {
                    dwork[j] = SLC_DNRM2(&nf, &b[nr1 + j * ldb], &int1);
                    dwork[m + j] = dwork[j];
                    iwork[j] = j + 1;
                }
            }

            while (rank < mn) {
                j = rank;
                irow++;

                if (j != mcrt - 1) {
                    i32 len = mcrt - j;
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

                for (i = n1 - 2; i >= irow; i--) {
                    t = b[i + j * ldb];
                    SLC_DLARTG(&t, &b[(i + 1) + j * ldb], &co, &si, &b[i + j * ldb]);
                    b[(i + 1) + j * ldb] = zero;

                    i32 n_minus_i = n - i;
                    SLC_DROT(&n_minus_i, &e[i + i * lde], &lde, &e[(i + 1) + i * lde], &lde, &co, &si);

                    if (j < jb2 - 1) {
                        i32 jb2_minus_j_minus_1 = jb2 - j - 1;
                        SLC_DROT(&jb2_minus_j_minus_1, &b[i + (j + 1) * ldb], &ldb,
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
                    if (smax <= svmr) {
                        goto label_80;
                    } else if (onecol) {
                        rank++;
                        goto label_80;
                    }
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

                if (svmr <= smaxpr) {
                    if (smaxpr * rcond < sminpr) {
                        if (irow == n1) {
                            rank++;
                            goto label_80;
                        }

                        for (i = j + 1; i < mcrt; i++) {
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
                goto label_80;
            }
        }

label_80:
        if (rank > 0) {
            rtau[*nrblck - 1] = rank;

            if (!onecol) {
                for (j = 0; j < mcrt; j++) {
                    if (iwork[j] > 0) {
                        k = iwork[j] - 1;
                        iwork[j] = -(k + 1);
                        while (k != j) {
                            if (k < 0 || k >= mcrt) break;
                            SLC_DSWAP(&rank, &b[nr1 + j * ldb], &int1, &b[nr1 + k * ldb], &int1);
                            i32 next_k_val = iwork[k];
                            iwork[k] = -iwork[k];
                            k = next_k_val - 1;
                        }
                    }
                }
            }
        }

        if (*nrblck == 2) {
            for (j = m2 - 1; j >= 0; j--) {
                SLC_DCOPY(&nf, &b[nr1 + j * ldb], &int1, &b[nr1 + (m1 + j) * ldb], &int1);
            }
        } else if (*nrblck > 2) {
            SLC_DLACPY("F", &nf, &mcrt, &b[nr1], &ldb, &a[nr1 + ic * lda], &lda);
        }

        if (rank > 0) {
            *nr = *nr + rank;
            nf = nf - rank;
            if (*nrblck > 2) {
                ic = ic + mcrt;
            }
            if (b1red) {
                mcrt1 = rank;
                mcrt = mcrt2;
            } else {
                mcrt2 = rank;
                mcrt = mcrt1;
            }
            b1red = !b1red;
        } else {
            if (b1red) {
                if (mcrt2 > 0) {
                    b1red = !b1red;
                    rtau[*nrblck - 1] = 0;
                    if (*nrblck > 2) {
                        ic = ic + mcrt;
                    }
                    mcrt1 = 0;
                    mcrt = mcrt2;
                    continue;
                }
                (*nrblck)--;
            } else {
                if (mcrt1 > 0) {
                    b1red = !b1red;
                    rtau[*nrblck - 1] = 0;
                    if (*nrblck > 2) {
                        ic = ic + mcrt;
                    }
                    mcrt2 = 0;
                    mcrt = mcrt1;
                    continue;
                }
                *nrblck = *nrblck - 2;
            }
            break;
        }
    }

    if (*nrblck > 0) {
        rank = rtau[0];
        if (rank < n1) {
            i32 n1_minus_rank = n1 - rank;
            SLC_DLASET("F", &n1_minus_rank, &m1, &zero, &zero, &b[rank], &ldb);
        }
        rank = rank + rtau[1];
        if (rank < n1) {
            i32 n1_minus_rank = n1 - rank;
            SLC_DLASET("F", &n1_minus_rank, &m2, &zero, &zero, &b[rank + m1 * ldb], &ldb);
        }
    }

    dwork[0] = (f64)wrkopt;
}
