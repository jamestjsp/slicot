/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>

void sb08fd(
    const char* dico,
    const i32 n,
    const i32 m,
    const i32 p,
    const f64* alpha,
    f64* a,
    const i32 lda,
    f64* b,
    const i32 ldb,
    f64* c,
    const i32 ldc,
    const f64* d,
    const i32 ldd,
    i32* nq,
    i32* nr,
    f64* cr,
    const i32 ldcr,
    f64* dr,
    const i32 lddr,
    const f64 tol,
    f64* dwork,
    const i32 ldwork,
    i32* iwarn,
    i32* info
)
{
    const f64 ONE = 1.0;
    const f64 TEN = 10.0;
    const f64 ZERO = 0.0;

    bool discr = (*dico == 'D' || *dico == 'd');
    *iwarn = 0;
    *info = 0;

    if (!(*dico == 'C' || *dico == 'c') && !discr) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (m < 0) {
        *info = -3;
    } else if (p < 0) {
        *info = -4;
    } else if ((discr && (alpha[0] < ZERO || alpha[0] >= ONE ||
                          alpha[1] < ZERO || alpha[1] >= ONE)) ||
               (!discr && (alpha[0] >= ZERO || alpha[1] >= ZERO))) {
        *info = -5;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -7;
    } else if (ldb < (n > 1 ? n : 1)) {
        *info = -9;
    } else if (ldc < (p > 1 ? p : 1)) {
        *info = -11;
    } else if (ldd < (p > 1 ? p : 1)) {
        *info = -13;
    } else if (ldcr < (m > 1 ? m : 1)) {
        *info = -17;
    } else if (lddr < (m > 1 ? m : 1)) {
        *info = -19;
    } else {
        i32 min1 = n * (n + 5);
        i32 min2 = 5 * m;
        i32 min3 = 4 * p;
        i32 minwrk = min1 > min2 ? min1 : min2;
        minwrk = minwrk > min3 ? minwrk : min3;
        minwrk = minwrk > 1 ? minwrk : 1;
        if (ldwork < minwrk) {
            *info = -22;
        }
    }

    if (*info != 0) {
        i32 neginfo = -(*info);
        SLC_XERBLA("SB08FD", &neginfo);
        return;
    }

    *nr = 0;
    SLC_DLASET("Full", &m, &m, &ZERO, &ONE, dr, &lddr);

    i32 minn = m < n ? m : n;
    if (minn == 0) {
        *nq = 0;
        dwork[0] = ONE;
        return;
    }

    SLC_DLASET("Full", &m, &n, &ZERO, &ZERO, cr, &ldcr);

    f64 bnorm = SLC_DLANGE("1-norm", &n, &m, b, &ldb, dwork);
    f64 toler = tol;
    if (toler <= ZERO) {
        f64 eps = SLC_DLAMCH("Epsilon");
        toler = (f64)n * bnorm * eps;
    }
    if (bnorm <= toler) {
        *nq = 0;
        dwork[0] = ONE;
        return;
    }

    f64 anorm = SLC_DLANGE("1-norm", &n, &n, a, &lda, dwork);
    f64 rmax = TEN * anorm / bnorm;

    i32 kz = 0;
    i32 kwr = kz + n * n;
    i32 kwi = kwr + n;
    i32 kw = kwi + n;

    i32 nfp;
    i32 ldw = ldwork - kw;

    tb01ld(dico, "Stable", "General", n, m, p, alpha[1], a, lda, b, ldb, c, ldc,
           &nfp, &dwork[kz], n, &dwork[kwr], &dwork[kwi], &dwork[kw], ldw, info);
    if (*info != 0) {
        return;
    }

    f64 wrkopt = dwork[kw] + (f64)kw;

    *nq = n;
    if (nfp < n) {
        i32 kg = 0;
        i32 kfi = kg + 2 * m;
        kw = kfi + 2 * m;

        i32 nlow = nfp;
        i32 nsup = n - 1;

        while (nlow <= nsup) {
            i32 ib = 1;
            if (nlow < nsup) {
                if (a[nsup + (nsup - 1) * lda] != ZERO) {
                    ib = 2;
                }
            }
            i32 l = nsup - ib + 1;

            SLC_DLACPY("Full", &ib, &m, &b[l], &ldb, &dwork[kg], &ib);

            f64 gnorm = SLC_DLANGE("1-norm", &ib, &m, &dwork[kg], &ib, &dwork[kw]);
            if (gnorm <= toler) {
                nsup = nsup - ib;
            } else {
                f64 a2[4];
                f64 sm, pr;

                a2[0] = a[l + l * lda];
                if (ib == 1) {
                    sm = alpha[0];
                    if (discr) {
                        sm = a2[0] >= 0 ? alpha[0] : -alpha[0];
                    }
                    pr = alpha[0];
                } else {
                    a2[1] = a[l + 1 + l * lda];
                    a2[2] = a[l + (l + 1) * lda];
                    a2[3] = a[l + 1 + (l + 1) * lda];
                    sm = alpha[0] + alpha[0];
                    pr = alpha[0] * alpha[0];
                    if (discr) {
                        f64 x = a2[0];
                        f64 y = sqrt(fabs(a2[1] * a2[2]));
                        f64 hyp = sqrt(x * x + y * y);
                        sm = sm * x / hyp;
                    } else {
                        pr = pr - a2[1] * a2[2];
                    }
                }

                i32 sb_info;
                sb01by(ib, m, sm, pr, a2, &dwork[kg], &dwork[kfi], toler, &dwork[kw], &sb_info);

                if (sb_info != 0) {
                    f64 cs = dwork[kfi];
                    f64 sn = -dwork[kfi + m];

                    i32 l1 = l + 1;
                    i32 len = nsup - l + 1;
                    SLC_DROT(&len, &a[l1 + l * lda], &lda, &a[l + l * lda], &lda, &cs, &sn);
                    i32 len2 = l1 + 1;
                    i32 int1 = 1;
                    SLC_DROT(&len2, &a[0 + l1 * lda], &int1, &a[0 + l * lda], &int1, &cs, &sn);
                    SLC_DROT(&m, &b[l1], &ldb, &b[l], &ldb, &cs, &sn);
                    if (p > 0) {
                        SLC_DROT(&p, &c[0 + l1 * ldc], &int1, &c[0 + l * ldc], &int1, &cs, &sn);
                    }
                    SLC_DROT(&m, &cr[0 + l1 * ldcr], &int1, &cr[0 + l * ldcr], &int1, &cs, &sn);

                    a[l1 + l * lda] = ZERO;
                    nsup = nsup - 1;
                    continue;
                }

                f64 fnorm = SLC_DLANGE("1-norm", &m, &ib, &dwork[kfi], &m, &dwork[kw]);
                if (fnorm > rmax) {
                    (*iwarn)++;
                }

                i32 k = kfi;
                for (i32 j = l; j < l + ib; j++) {
                    for (i32 i = 0; i < m; i++) {
                        cr[i + j * ldcr] += dwork[k];
                        k++;
                    }
                }

                i32 nrows = nsup + 1;
                SLC_DGEMM("N", "N", &nrows, &ib, &m, &ONE, b, &ldb, &dwork[kfi], &m,
                          &ONE, &a[0 + l * lda], &lda);

                if (ib == 2) {
                    i32 l1 = l + 1;
                    f64 x, y, pr2, sm2, cs, sn;
                    SLC_DLANV2(&a[l + l * lda], &a[l + l1 * lda], &a[l1 + l * lda],
                               &a[l1 + l1 * lda], &x, &y, &pr2, &sm2, &cs, &sn);

                    if (l1 < nsup) {
                        i32 len = nsup - l1;
                        SLC_DROT(&len, &a[l + (l1 + 1) * lda], &lda, &a[l1 + (l1 + 1) * lda],
                                 &lda, &cs, &sn);
                    }
                    i32 int1 = 1;
                    SLC_DROT(&l, &a[0 + l * lda], &int1, &a[0 + l1 * lda], &int1, &cs, &sn);
                    SLC_DROT(&m, &b[l], &ldb, &b[l1], &ldb, &cs, &sn);
                    if (p > 0) {
                        SLC_DROT(&p, &c[0 + l * ldc], &int1, &c[0 + l1 * ldc], &int1, &cs, &sn);
                    }
                    SLC_DROT(&m, &cr[0 + l * ldcr], &int1, &cr[0 + l1 * ldcr], &int1, &cs, &sn);
                }

                if (nlow + ib <= nsup) {
                    i32 ncur1 = nsup - ib;
                    i32 nmoves = 1;
                    i32 ib_cur = ib;

                    if (ib == 2 && a[nsup + (nsup - 1) * lda] == ZERO) {
                        ib_cur = 1;
                        nmoves = 2;
                    }

                    while (nmoves > 0) {
                        i32 ncur = ncur1;

                        while (ncur >= nlow) {
                            i32 ib1 = 1;
                            if (ncur > nlow) {
                                if (a[ncur + (ncur - 1) * lda] != ZERO) {
                                    ib1 = 2;
                                }
                            }
                            i32 nb = ib1 + ib_cur;

                            f64 z[16];
                            i32 int4 = 4;
                            SLC_DLASET("Full", &nb, &nb, &ZERO, &ONE, z, &int4);
                            i32 ll = ncur - ib1 + 1;

                            i32 exc_info;
                            i32 int1 = 1;
                            SLC_DLAEXC(&int1, &nb, &a[ll + ll * lda], &lda, z, &int4,
                                       &int1, &ib1, &ib_cur, dwork, &exc_info);
                            if (exc_info != 0) {
                                *info = 2;
                                return;
                            }

                            i32 l1_ = ll + nb;
                            if (l1_ <= nsup) {
                                i32 cols = nsup - l1_ + 1;
                                SLC_DGEMM("T", "N", &nb, &cols, &nb, &ONE, z, &int4,
                                          &a[ll + l1_ * lda], &lda, &ZERO, dwork, &nb);
                                SLC_DLACPY("Full", &nb, &cols, dwork, &nb, &a[ll + l1_ * lda], &lda);
                            }
                            SLC_DGEMM("N", "N", &ll, &nb, &nb, &ONE, &a[0 + ll * lda], &lda,
                                      z, &int4, &ZERO, dwork, &n);
                            SLC_DLACPY("Full", &ll, &nb, dwork, &n, &a[0 + ll * lda], &lda);

                            SLC_DGEMM("T", "N", &nb, &m, &nb, &ONE, z, &int4,
                                      &b[ll], &ldb, &ZERO, dwork, &nb);
                            SLC_DLACPY("Full", &nb, &m, dwork, &nb, &b[ll], &ldb);

                            if (p > 0) {
                                SLC_DGEMM("N", "N", &p, &nb, &nb, &ONE, &c[0 + ll * ldc], &ldc,
                                          z, &int4, &ZERO, dwork, &p);
                                SLC_DLACPY("Full", &p, &nb, dwork, &p, &c[0 + ll * ldc], &ldc);
                            }

                            SLC_DGEMM("N", "N", &m, &nb, &nb, &ONE, &cr[0 + ll * ldcr], &ldcr,
                                      z, &int4, &ZERO, dwork, &m);
                            SLC_DLACPY("Full", &m, &nb, dwork, &m, &cr[0 + ll * ldcr], &ldcr);

                            ncur = ncur - ib1;
                        }

                        nmoves--;
                        ncur1++;
                        nlow = nlow + ib_cur;
                    }
                } else {
                    nlow = nlow + ib;
                }
            }
        }

        *nq = nsup + 1;
        *nr = nsup + 1 - nfp;

        if (*nq > 2) {
            i32 rows = *nq - 2;
            i32 cols = *nq - 2;
            SLC_DLASET("Lower", &rows, &cols, &ZERO, &ZERO, &a[2 + 0 * lda], &lda);
        }
    }

    SLC_DGEMM("N", "N", &p, nq, &m, &ONE, d, &ldd, cr, &ldcr, &ONE, c, &ldc);

    f64 opt1 = (f64)(5 * m);
    f64 opt2 = (f64)(4 * p);
    f64 opt_max = opt1 > opt2 ? opt1 : opt2;
    dwork[0] = wrkopt > opt_max ? wrkopt : opt_max;
}
