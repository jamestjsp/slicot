/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <stdbool.h>

static int delctg_select(const f64* ar, const f64* ai, const f64* b)
{
    (void)ar; (void)ai; (void)b;
    return 0;
}

void tg01jy(
    const char* job, const char* systyp, const char* equil,
    const char* cksing, const char* restor,
    const i32 n, const i32 m, const i32 p,
    f64* a, const i32 lda,
    f64* e, const i32 lde,
    f64* b, const i32 ldb,
    f64* c, const i32 ldc,
    i32* nr, i32* infred,
    const f64* tol,
    i32* iwork, f64* dwork, const i32 ldwork,
    i32* info
)
{
    const f64 one = 1.0, zero = 0.0, ten = 10.0;
    const f64 tolrc = 1.0e-10;

    bool fincon, finobs, infcon, infobs, lequil, ljobc, ljobir, ljobo;
    bool lquery, lsing, lspace, lsysp, lsysr, lsyss, maxacc, singa, singe;

    i32 i, j, k, kwa, kwb, kwc, kwe, kwr, lba, lbas, lbe, lbes;
    i32 ldq, ldz, lwa, lwb, lwc, lwe, m1, maxmp, maxwrk, minwrk, n1;
    i32 nb, nblck, nc, nn, nx, p1;
    f64 anorm, enorm, rcond, t, tl, tt, tzer, thresh;
    f64 dum[1];
    i32 bwork[1];
    i32 int1 = 1, intm1 = -1, int3 = 3;

    *info = 0;
    maxmp = (m > p) ? m : p;
    n1 = (n > 1) ? n : 1;

    ljobir = (job[0] == 'I' || job[0] == 'i');
    ljobc = ljobir || (job[0] == 'C' || job[0] == 'c');
    ljobo = ljobir || (job[0] == 'O' || job[0] == 'o');

    lsysr = (systyp[0] == 'R' || systyp[0] == 'r');
    lsyss = lsysr || (systyp[0] == 'S' || systyp[0] == 's');
    lsysp = lsysr || (systyp[0] == 'P' || systyp[0] == 'p');

    lequil = (equil[0] == 'S' || equil[0] == 's');
    lsing = (cksing[0] == 'C' || cksing[0] == 'c');
    maxacc = (restor[0] == 'R' || restor[0] == 'r');

    if (!ljobc && !ljobo) {
        *info = -1;
    } else if (!lsyss && !lsysp) {
        *info = -2;
    } else if (!lequil && !(equil[0] == 'N' || equil[0] == 'n')) {
        *info = -3;
    } else if (!lsing && !(cksing[0] == 'N' || cksing[0] == 'n')) {
        *info = -4;
    } else if (!maxacc && !(restor[0] == 'N' || restor[0] == 'n')) {
        *info = -5;
    } else if (n < 0) {
        *info = -6;
    } else if (m < 0) {
        *info = -7;
    } else if (p < 0) {
        *info = -8;
    } else if (lda < n1) {
        *info = -10;
    } else if (lde < n1) {
        *info = -12;
    } else if (ldb < n1) {
        *info = -14;
    } else if (ldc < 1 || (n > 0 && ldc < maxmp)) {
        *info = -16;
    } else if (tol[0] >= one) {
        *info = -19;
    } else if (tol[1] >= one) {
        *info = -19;
    } else if (tol[2] >= one) {
        *info = -19;
    } else {
        nn = n * n;
        k = n * (2 * n + m + p);
        if (maxacc) {
            i32 tmp1 = 2 * (k + maxmp + n - 1);
            i32 tmp2 = nn + 4 * n;
            minwrk = (tmp1 > tmp2) ? tmp1 : tmp2;
            minwrk = (minwrk > 1) ? minwrk : 1;
        } else {
            i32 tmp1 = 2 * (maxmp + n - 1);
            i32 tmp2 = nn + 4 * n;
            minwrk = (tmp1 > tmp2) ? tmp1 : tmp2;
            minwrk = (minwrk > 1) ? minwrk : 1;
        }
        if (lequil) {
            if (8 * n > minwrk) minwrk = 8 * n;
        }
        if (lsing) {
            i32 tmp = 2 * nn + 10 * n + ((n > 23) ? n : 23);
            if (tmp > minwrk) minwrk = tmp;
        }

        fincon = ljobc && lsyss;
        infcon = ljobc && lsysp;
        finobs = ljobo && lsyss;
        infobs = ljobo && lsysp;

        maxwrk = k + 2 * (maxmp + n - 1);
        lspace = ldwork >= maxwrk;
        if (ljobir) {
            maxwrk = maxwrk + k;
        }
        lquery = ldwork == -1;
        if (maxwrk < minwrk) maxwrk = minwrk;

        tl = tol[0];
        if (lquery) {
            if (lsing) {
                SLC_DGGES("N", "N", "N", delctg_select, &n, dwork, &n1,
                          dwork, &n1, &kwa, dwork, dwork, dwork,
                          dwork, &int1, dwork, &int1, dwork, &intm1, bwork, info);
                i32 tmp = (i32)dwork[0] + 2 * nn + 3 * n;
                if (tmp > maxwrk) maxwrk = tmp;
            }
            if (n > 1) {
                if (fincon || infcon) {
                    i32 n_minus_1 = n - 1;
                    tg01hx("N", "N", n, n, m, p, n, n_minus_1, a, lda,
                           e, lde, b, ldb, c, ldc, dum, int1, dum, int1,
                           nr, &nblck, iwork, tl, iwork, dwork, info);
                    i32 tmp = (i32)dwork[0] + k;
                    if (tmp > maxwrk) maxwrk = tmp;
                }
                if (finobs || infobs) {
                    i32 n_minus_1 = n - 1;
                    tg01hx("N", "N", n, n, p, m, n, n_minus_1, a, lda,
                           e, lde, b, ldb, c, ldc, dum, int1, dum, int1,
                           nr, &nblck, iwork, tl, iwork, dwork, info);
                    i32 tmp = (i32)dwork[0] + k;
                    if (tmp > maxwrk) maxwrk = tmp;
                }
            }
        } else if (ldwork < minwrk) {
            *info = -22;
        }
    }

    if (*info != 0) {
        return;
    } else if (lquery) {
        dwork[0] = (f64)maxwrk;
        return;
    }

    infred[0] = -1;
    infred[1] = -1;
    infred[2] = -1;
    infred[3] = -1;
    infred[4] = 0;
    infred[5] = 0;
    infred[6] = 0;

    i32 max_n_maxmp = (n > maxmp) ? n : maxmp;
    if (max_n_maxmp == 0) {
        *nr = 0;
        dwork[0] = one;
        return;
    }

    lba = (n > 1) ? n - 1 : 0;
    lbe = lba;
    tzer = tol[1];

    if (tl <= zero || tzer <= zero || lequil) {
        t = SLC_DLAMCH("P");
        if (tl <= zero) tl = nn * t;
        if (tzer <= zero) tzer = ten * t;
        if (lequil) {
            thresh = tol[2];
            if (thresh < zero) {
                anorm = SLC_DLANGE("1", &n, &n, a, &lda, dwork);
                enorm = SLC_DLANGE("1", &n, &n, e, &lde, dwork);
                f64 bnorm = SLC_DLANGE("1", &n, &m, b, &ldb, dwork);
                f64 cnorm = SLC_DLANGE("1", &p, &n, c, &ldc, dwork);
                thresh = anorm;
                if (enorm > thresh) thresh = enorm;
                if (bnorm > thresh) thresh = bnorm;
                if (cnorm > thresh) thresh = cnorm;
                thresh = thresh * t;
            }
        }
    }

    singa = false;
    singe = false;

    j = nn + 1;

    if (lsing || fincon || finobs) {
        SLC_DLACPY("F", &n, &n, a, &lda, dwork, &n1);
        SLC_DGETRF(&n, &n, dwork, &n1, iwork, &i);
        if (i > 0) {
            singa = true;
        } else {
            if (!lequil || tol[2] >= zero) {
                anorm = SLC_DLANGE("1", &n, &n, a, &lda, dwork);
            }
            SLC_DGECON("1", &n, dwork, &n1, &anorm, &rcond, &dwork[j - 1], iwork, &i);
            if (rcond > tolrc) {
                fincon = false;
                finobs = false;
            }
            if (rcond <= tzer) {
                singa = true;
            }
        }
    }

    if (lsing || infcon || infobs) {
        SLC_DLACPY("F", &n, &n, e, &lde, dwork, &n1);
        SLC_DGETRF(&n, &n, dwork, &n1, iwork, &i);
        if (i > 0) {
            singe = true;
        } else {
            if (!lequil || tol[2] >= zero) {
                enorm = SLC_DLANGE("1", &n, &n, e, &lde, dwork);
            }
            SLC_DGECON("1", &n, dwork, &n1, &enorm, &rcond, &dwork[j - 1], iwork, &i);
            if (rcond <= tzer) {
                singe = true;
            }
        }

        if (!singe) {
            infcon = false;
            infobs = false;
        }
    }

    if (lsing && singa && singe) {
        k = j + nn;
        SLC_DLACPY("F", &n, &n, a, &lda, dwork, &n1);
        SLC_DLACPY("F", &n, &n, e, &lde, &dwork[j - 1], &n1);

        i32 lwork_rem = ldwork - (k + 3 * n) + 1;
        SLC_DGGES("N", "N", "N", delctg_select, &n, dwork, &n1,
                  &dwork[j - 1], &n1, &kwa, &dwork[k - 1], &dwork[k - 1 + n],
                  &dwork[k - 1 + 2 * n], dwork, &int1, dwork, &int1,
                  &dwork[k - 1 + 3 * n], &lwork_rem, bwork, &i);

        i32 tmp = (i32)dwork[k - 1 + 3 * n] + k + 3 * n - 1;
        if (tmp > maxwrk) maxwrk = tmp;

        for (i = k - 1; i < k - 1 + n; i++) {
            f64 beta_abs = dwork[i + 2 * n];
            if (beta_abs < 0) beta_abs = -beta_abs;
            if (beta_abs <= tzer) {
                f64 re = dwork[i];
                f64 im = dwork[i + n];
                f64 hyp = SLC_DLAPY2(&re, &im);
                if (hyp <= tzer) {
                    *info = 1;
                    return;
                }
            }
        }
    }

    m1 = (m > 1) ? m : 1;
    p1 = (p > 1) ? p : 1;

    if (lspace) {
        kwa = 1;
        kwe = j;
        kwb = kwe + nn;
        kwc = kwb + n * m;
        kwr = kwc + n * p;
    } else {
        kwr = 1;
    }

    if (maxacc) {
        lwa = kwr;
        lwe = lwa + nn;
        lwb = lwe + nn;
        lwc = lwb + n * m;
        kwr = lwc + n * p;
    } else {
        lwa = 0; lwe = 0; lwb = 0; lwc = 0;
    }

    if (lequil) {
        tg01ad("A", n, n, m, p, thresh, a, lda, e, lde, b, ldb, c, ldc,
               dwork, &dwork[n], &dwork[2 * n], info);
    }

    ldq = 1;
    ldz = 1;
    nc = n;
    *nr = n;

    if (infcon) {
        if (maxacc) {
            SLC_DLACPY("F", &nc, &nc, a, &lda, &dwork[lwa - 1], &n1);
            SLC_DLACPY("F", &nc, &nc, e, &lde, &dwork[lwe - 1], &n1);
            SLC_DLACPY("F", &nc, &m, b, &ldb, &dwork[lwb - 1], &n1);
            SLC_DLACPY("F", &p, &nc, c, &ldc, &dwork[lwc - 1], &p1);
        }

        if (lspace) {
            if (lba > 0 && infobs) {
                i32 lwork_qr = -1;
                SLC_DGEQRF(&nc, &nc, a, &lda, dwork, dwork, &lwork_qr, info);
                nb = (i32)dwork[0] / nc;
                if (ldwork < nc * nb) nb = ldwork / nc;

                nx = SLC_ILAENV(&int3, "DGEQRF", " ", &nc, &nc, &intm1, &intm1);
                if (lba < nx / 2 || nb < nx || nc < nx) {
                    for (i = 0; i < nc - 1; i++) {
                        k = (lba < nc - i - 1) ? lba + 1 : nc - i;
                        SLC_DLARFG(&k, &a[i + i * lda], &a[i + 1 + i * lda], &int1, &tt);
                        t = a[i + i * lda];
                        a[i + i * lda] = one;

                        i32 n_minus_i = n - i - 1;
                        SLC_DLARF("L", &k, &n_minus_i, &a[i + i * lda], &int1, &tt,
                                  &a[i + (i + 1) * lda], &lda, dwork);
                        SLC_DLARF("L", &k, &n, &a[i + i * lda], &int1, &tt,
                                  &e[i], &lde, dwork);
                        SLC_DLARF("L", &k, &m, &a[i + i * lda], &int1, &tt,
                                  &b[i], &ldb, dwork);
                        a[i + i * lda] = t;
                    }
                } else {
                    i32 lwork_avail = ldwork - nc;
                    SLC_DGEQRF(&nc, &nc, a, &lda, dwork, &dwork[nc], &lwork_avail, info);
                    i32 tmp = (i32)dwork[nc] + nc;
                    if (tmp > maxwrk) maxwrk = tmp;

                    SLC_DORMQR("L", "T", &nc, &n, &nc, a, &lda, dwork, e, &lde,
                               &dwork[nc], &lwork_avail, info);
                    tmp = (i32)dwork[nc] + nc;
                    if (tmp > maxwrk) maxwrk = tmp;

                    SLC_DORMQR("L", "T", &nc, &m, &nc, a, &lda, dwork, b, &ldb,
                               &dwork[nc], &lwork_avail, info);
                    tmp = (i32)dwork[nc] + nc;
                    if (tmp > maxwrk) maxwrk = tmp;
                }
                if (nc > 1) {
                    i32 nc_m1 = nc - 1;
                    SLC_DLASET("L", &nc_m1, &nc_m1, &zero, &zero, &a[1], &lda);
                }
                lba = 0;
            }

            SLC_DLACPY("F", &nc, &nc, a, &lda, &dwork[kwa - 1], &n1);
            SLC_DLACPY("F", &nc, &nc, e, &lde, &dwork[kwe - 1], &n1);
            SLC_DLACPY("F", &nc, &m, b, &ldb, &dwork[kwb - 1], &n1);
            SLC_DLACPY("F", &p, &nc, c, &ldc, &dwork[kwc - 1], &p1);
        }

        tg01hx("N", "N", nc, nc, m, p, nc, lba, e, lde, a, lda, b, ldb, c, ldc,
               dum, ldq, dum, ldz, nr, &nblck, iwork, tl, &iwork[n], &dwork[kwr - 1],
               info);

        i32 tmp = (i32)dwork[kwr - 1] + kwr - 1;
        if (tmp > maxwrk) maxwrk = tmp;

        infred[0] = nc - *nr;
        infred[6] = nblck;

        if (*nr < nc || !lspace) {
            if (nblck > 1) {
                lbe = iwork[0] + iwork[1] - 1;
            } else if (nblck == 1) {
                lbe = iwork[0] - 1;
            } else {
                lbe = 0;
            }
            lba = 0;
            nc = *nr;
        } else if (!maxacc) {
            SLC_DLACPY("F", &nc, &nc, &dwork[kwa - 1], &n1, a, &lda);
            SLC_DLACPY("F", &nc, &nc, &dwork[kwe - 1], &n1, e, &lde);
            SLC_DLACPY("F", &nc, &m, &dwork[kwb - 1], &n1, b, &ldb);
            SLC_DLACPY("F", &p, &nc, &dwork[kwc - 1], &p1, c, &ldc);
        } else {
            SLC_DLACPY("F", &nc, &nc, &dwork[lwa - 1], &n1, a, &lda);
            SLC_DLACPY("F", &nc, &nc, &dwork[lwe - 1], &n1, e, &lde);
            SLC_DLACPY("F", &nc, &m, &dwork[lwb - 1], &n1, b, &ldb);
            SLC_DLACPY("F", &p, &nc, &dwork[lwc - 1], &p1, c, &ldc);
            lba = (n > 1) ? n - 1 : 0;
        }
    }

    if (infobs) {
        i32 nc_m1 = (nc > 1) ? nc - 1 : 0;
        tb01xd("Z", nc, m, p, lba, nc_m1, a, lda, b, ldb, c, ldc, dum, int1, info);
        ma02cd(nc, lbe, nc_m1, e, lde);

        if (lspace) {
            SLC_DLACPY("F", &nc, &nc, a, &lda, &dwork[kwa - 1], &n1);
            SLC_DLACPY("F", &nc, &nc, e, &lde, &dwork[kwe - 1], &n1);
            SLC_DLACPY("F", &nc, &p, b, &ldb, &dwork[kwc - 1], &n1);
            SLC_DLACPY("F", &m, &nc, c, &ldc, &dwork[kwb - 1], &m1);
        }

        tg01hx("N", "N", nc, nc, p, m, nc, lba, e, lde, a, lda, b, ldb, c, ldc,
               dum, ldz, dum, ldq, nr, &nblck, &iwork[n], tl, &iwork[2 * n],
               &dwork[kwr - 1], info);

        i32 tmp2 = (i32)dwork[kwr - 1] + kwr - 1;
        if (tmp2 > maxwrk) maxwrk = tmp2;

        infred[1] = nc - *nr;

        if (*nr < nc || !lspace) {
            infred[6] = nblck;
            for (i = 0; i < nblck; i++) {
                iwork[i] = iwork[n + i];
            }
            if (nblck > 1) {
                lbe = iwork[0] + iwork[1] - 1;
            } else if (nblck == 1) {
                lbe = iwork[0] - 1;
            } else {
                lbe = 0;
            }
            lba = 0;
            nc = *nr;
        } else {
            SLC_DLACPY("F", &nc, &nc, &dwork[kwa - 1], &n1, a, &lda);
            SLC_DLACPY("F", &nc, &nc, &dwork[kwe - 1], &n1, e, &lde);
            SLC_DLACPY("F", &nc, &p, &dwork[kwc - 1], &n1, b, &ldb);
            SLC_DLACPY("F", &m, &nc, &dwork[kwb - 1], &m1, c, &ldc);
        }

        if (fincon || !finobs) {
            i32 nc_m1_2 = (nc > 1) ? nc - 1 : 0;
            tb01xd("Z", nc, p, m, lba, nc_m1_2, a, lda, b, ldb, c, ldc, dum, int1, info);
            ma02cd(nc, lbe, nc_m1_2, e, lde);
        }
    }

    if (fincon) {
        if (maxacc) {
            SLC_DLACPY("F", &nc, &nc, a, &lda, &dwork[lwa - 1], &n1);
            SLC_DLACPY("F", &nc, &nc, e, &lde, &dwork[lwe - 1], &n1);
            SLC_DLACPY("F", &nc, &m, b, &ldb, &dwork[lwb - 1], &n1);
            SLC_DLACPY("F", &p, &nc, c, &ldc, &dwork[lwc - 1], &p1);
            lbas = lba;
            lbes = lbe;
        }

        if (lspace) {
            if (lbe > 0 && finobs) {
                i32 lwork_qr = -1;
                SLC_DGEQRF(&nc, &nc, e, &lde, dwork, dwork, &lwork_qr, info);
                nb = (i32)dwork[0] / nc;
                if (ldwork < nc * nb) nb = ldwork / nc;

                nx = SLC_ILAENV(&int3, "DGEQRF", " ", &nc, &nc, &intm1, &intm1);
                if (lbe < nx / 2 || nb < nx || nc < nx) {
                    for (i = 0; i < nc - 1; i++) {
                        k = (lbe < nc - i - 1) ? lbe + 1 : nc - i;
                        SLC_DLARFG(&k, &e[i + i * lde], &e[i + 1 + i * lde], &int1, &tt);
                        t = e[i + i * lde];
                        e[i + i * lde] = one;

                        i32 n_minus_i = n - i - 1;
                        SLC_DLARF("L", &k, &n_minus_i, &e[i + i * lde], &int1, &tt,
                                  &e[i + (i + 1) * lde], &lde, dwork);
                        SLC_DLARF("L", &k, &n, &e[i + i * lde], &int1, &tt,
                                  &a[i], &lda, dwork);
                        SLC_DLARF("L", &k, &m, &e[i + i * lde], &int1, &tt,
                                  &b[i], &ldb, dwork);
                        e[i + i * lde] = t;
                    }
                } else {
                    i32 lwork_avail = ldwork - nc;
                    SLC_DGEQRF(&nc, &nc, e, &lde, dwork, &dwork[nc], &lwork_avail, info);
                    i32 tmp = (i32)dwork[nc] + nc;
                    if (tmp > maxwrk) maxwrk = tmp;

                    SLC_DORMQR("L", "T", &nc, &n, &nc, e, &lde, dwork, a, &lda,
                               &dwork[nc], &lwork_avail, info);
                    tmp = (i32)dwork[nc] + nc;
                    if (tmp > maxwrk) maxwrk = tmp;

                    SLC_DORMQR("L", "T", &nc, &m, &nc, e, &lde, dwork, b, &ldb,
                               &dwork[nc], &lwork_avail, info);
                    tmp = (i32)dwork[nc] + nc;
                    if (tmp > maxwrk) maxwrk = tmp;
                }
                if (nc > 1) {
                    i32 nc_m1 = nc - 1;
                    SLC_DLASET("L", &nc_m1, &nc_m1, &zero, &zero, &e[1], &lde);
                }
                lbe = 0;
                lba = (nc > 1) ? nc - 1 : 0;
            }

            SLC_DLACPY("F", &nc, &nc, a, &lda, &dwork[kwa - 1], &n1);
            SLC_DLACPY("F", &nc, &nc, e, &lde, &dwork[kwe - 1], &n1);
            SLC_DLACPY("F", &nc, &m, b, &ldb, &dwork[kwb - 1], &n1);
            SLC_DLACPY("F", &p, &nc, c, &ldc, &dwork[kwc - 1], &p1);
        }

        tg01hx("N", "N", nc, nc, m, p, nc, lbe, a, lda, e, lde, b, ldb, c, ldc,
               dum, ldq, dum, ldz, nr, &nblck, &iwork[n], tl, &iwork[2 * n],
               &dwork[kwr - 1], info);

        i32 tmp3 = (i32)dwork[kwr - 1] + kwr - 1;
        if (tmp3 > maxwrk) maxwrk = tmp3;

        infred[2] = nc - *nr;

        if (*nr < nc || !lspace) {
            infred[6] = nblck;
            for (i = 0; i < nblck; i++) {
                iwork[i] = iwork[n + i];
            }
            if (nblck > 1) {
                lba = iwork[0] + iwork[1] - 1;
            } else if (nblck == 1) {
                lba = iwork[0] - 1;
            } else {
                lba = 0;
            }
            lbe = 0;
            nc = *nr;
        } else if (!maxacc) {
            SLC_DLACPY("F", &nc, &nc, &dwork[kwa - 1], &n1, a, &lda);
            SLC_DLACPY("F", &nc, &nc, &dwork[kwe - 1], &n1, e, &lde);
            SLC_DLACPY("F", &nc, &m, &dwork[kwb - 1], &n1, b, &ldb);
            SLC_DLACPY("F", &p, &nc, &dwork[kwc - 1], &p1, c, &ldc);
        } else {
            SLC_DLACPY("F", &nc, &nc, &dwork[lwa - 1], &n1, a, &lda);
            SLC_DLACPY("F", &nc, &nc, &dwork[lwe - 1], &n1, e, &lde);
            SLC_DLACPY("F", &nc, &m, &dwork[lwb - 1], &n1, b, &ldb);
            SLC_DLACPY("F", &p, &nc, &dwork[lwc - 1], &p1, c, &ldc);
            lba = lbas;
            lbe = lbes;
        }

        if (finobs) {
            i32 nc_m1 = (nc > 1) ? nc - 1 : 0;
            tb01xd("Z", nc, m, p, lba, nc_m1, a, lda, b, ldb, c, ldc, dum, int1, info);
            ma02cd(nc, lbe, nc_m1, e, lde);
        }
    }

    if (finobs) {
        if (lspace) {
            SLC_DLACPY("F", &nc, &nc, a, &lda, &dwork[kwa - 1], &n1);
            SLC_DLACPY("F", &nc, &nc, e, &lde, &dwork[kwe - 1], &n1);
            SLC_DLACPY("F", &nc, &p, b, &ldb, &dwork[kwc - 1], &n1);
            SLC_DLACPY("F", &m, &nc, c, &ldc, &dwork[kwb - 1], &m1);
        }

        tg01hx("N", "N", nc, nc, p, m, nc, lbe, a, lda, e, lde, b, ldb, c, ldc,
               dum, ldz, dum, ldq, nr, &nblck, &iwork[n], tl, &iwork[2 * n],
               &dwork[kwr - 1], info);

        i32 tmp4 = (i32)dwork[kwr - 1] + kwr - 1;
        if (tmp4 > maxwrk) maxwrk = tmp4;

        infred[3] = nc - *nr;

        if (*nr < nc || !lspace) {
            infred[6] = nblck;
            for (i = 0; i < nblck; i++) {
                iwork[i] = iwork[n + i];
            }
            if (nblck > 1) {
                lba = iwork[0] + iwork[1] - 1;
            } else if (nblck == 1) {
                lba = iwork[0] - 1;
            } else {
                lba = 0;
            }
            lbe = 0;
            nc = *nr;
        } else {
            SLC_DLACPY("F", &nc, &nc, &dwork[kwa - 1], &n1, a, &lda);
            SLC_DLACPY("F", &nc, &nc, &dwork[kwe - 1], &n1, e, &lde);
            SLC_DLACPY("F", &nc, &p, &dwork[kwc - 1], &n1, b, &ldb);
            SLC_DLACPY("F", &m, &nc, &dwork[kwb - 1], &m1, c, &ldc);
        }

        i32 nc_m1 = (nc > 1) ? nc - 1 : 0;
        tb01xd("Z", nc, p, m, lba, nc_m1, a, lda, b, ldb, c, ldc, dum, int1, info);
        ma02cd(nc, lbe, nc_m1, e, lde);
    }

    infred[4] = lba;
    infred[5] = lbe;
    dwork[0] = (f64)maxwrk;
}
