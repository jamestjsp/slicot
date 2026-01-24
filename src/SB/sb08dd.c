/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>

void sb08dd(
    const char* dico,
    const i32 n,
    const i32 m,
    const i32 p,
    f64* a,
    const i32 lda,
    f64* b,
    const i32 ldb,
    f64* c,
    const i32 ldc,
    f64* d,
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
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -6;
    } else if (ldb < (n > 1 ? n : 1)) {
        *info = -8;
    } else if (ldc < (p > 1 ? p : 1)) {
        *info = -10;
    } else if (ldd < (p > 1 ? p : 1)) {
        *info = -12;
    } else if (ldcr < (m > 1 ? m : 1)) {
        *info = -16;
    } else if (lddr < (m > 1 ? m : 1)) {
        *info = -18;
    } else {
        i32 min1 = n * (n + 5);
        i32 min2 = m * (m + 2);
        i32 min3 = 4 * m;
        i32 min4 = 4 * p;
        i32 minwrk = min1 > min2 ? min1 : min2;
        minwrk = minwrk > min3 ? minwrk : min3;
        minwrk = minwrk > min4 ? minwrk : min4;
        minwrk = minwrk > 1 ? minwrk : 1;
        if (ldwork < minwrk) {
            *info = -21;
        }
    }

    if (*info != 0) {
        i32 neginfo = -(*info);
        SLC_XERBLA("SB08DD", &neginfo);
        return;
    }

    *nr = 0;
    i32 minmp = m < p ? m : p;
    if (minmp > 0) {
        SLC_DLASET("Full", &m, &m, &ZERO, &ONE, dr, &lddr);
    }
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

    f64 alpha = discr ? ONE : ZERO;
    i32 nfp;
    i32 ldw = ldwork - kw;

    tb01ld(dico, "Stable", "General", n, m, p, alpha, a, lda, b, ldb, c, ldc,
           &nfp, &dwork[kz], n, &dwork[kwr], &dwork[kwi], &dwork[kw], ldw, info);
    if (*info != 0) {
        return;
    }

    f64 wrkopt = dwork[kw] + (f64)kw;

    *nq = n;
    if (nfp < n) {
        i32 kv = 0;
        i32 kfi = kv + m * m;
        kw = kfi + 2 * m;

        i32 nlow = nfp;
        i32 nsup = n - 1;

        while (nlow <= nsup) {
            i32 ib = 1;
            if (nlow < nsup) {
                if (a[(nsup) + (nsup - 1) * lda] != ZERO) {
                    ib = 2;
                }
            }
            i32 l = nsup - ib + 1;

            f64 bnorm_block = SLC_DLANGE("1-norm", &ib, &m, &b[l], &ldb, &dwork[kw]);
            if (bnorm_block <= toler) {
                nsup = nsup - ib;
            } else {
                i32 sb_info;
                sb01fy(discr, ib, m, &a[l + l * lda], lda, &b[l], ldb,
                       &dwork[kfi], m, &dwork[kv], m, &sb_info);
                if (sb_info == 2) {
                    *info = 3;
                    return;
                }

                f64 fnorm = SLC_DLANGE("1-norm", &m, &ib, &dwork[kfi], &m, &dwork[kw]);
                if (fnorm > rmax) {
                    (*iwarn)++;
                }

                // Fortran uses NSUP (which equals N at this point) as row count
                // In C, nsup = n-1 (0-based index), so use nsup+1 = n as row count
                i32 nrows = nsup + 1;
                SLC_DGEMM("N", "N", &nrows, &ib, &m, &ONE, b, &ldb, &dwork[kfi], &m,
                          &ONE, &a[0 + l * lda], &lda);

                if (discr) {
                    SLC_DTRMM("Left", "Upper", "N", "NonUnit", &m, &ib, &ONE, dr, &lddr,
                              &dwork[kfi], &m);
                }

                i32 k = kfi;
                for (i32 j = l; j < l + ib; j++) {
                    for (i32 i = 0; i < m; i++) {
                        cr[i + j * ldcr] += dwork[k];
                        k++;
                    }
                }

                if (discr) {
                    SLC_DTRMM("Right", "Upper", "N", "NonUnit", &n, &m, &ONE, &dwork[kv], &m,
                              b, &ldb);

                    k = kv;
                    for (i32 i = 0; i < m; i++) {
                        i32 len = m - i;
                        SLC_DTRMV("Upper", "Transpose", "NonUnit", &len, &dwork[k], &m,
                                  &dr[i + i * lddr], &lddr);
                        k = k + m + 1;
                    }
                }

                if (ib == 2) {
                    i32 l1 = l + 1;
                    f64 x, y, pr, sm, cs, sn;
                    SLC_DLANV2(&a[l + l * lda], &a[l + l1 * lda], &a[l1 + l * lda],
                               &a[l1 + l1 * lda], &x, &y, &pr, &sm, &cs, &sn);

                    if (l1 < nsup) {
                        i32 len = nsup - l1;
                        SLC_DROT(&len, &a[l + (l1 + 1) * lda], &lda, &a[l1 + (l1 + 1) * lda],
                                 &lda, &cs, &sn);
                    }
                    SLC_DROT(&l, &a[0 + l * lda], &(i32){1}, &a[0 + l1 * lda], &(i32){1}, &cs, &sn);
                    SLC_DROT(&m, &b[l], &ldb, &b[l1], &ldb, &cs, &sn);
                    if (p > 0) {
                        SLC_DROT(&p, &c[0 + l * ldc], &(i32){1}, &c[0 + l1 * ldc], &(i32){1}, &cs, &sn);
                    }
                    SLC_DROT(&m, &cr[0 + l * ldcr], &(i32){1}, &cr[0 + l1 * ldcr], &(i32){1}, &cs, &sn);
                }

                if (nlow + ib <= nsup) {
                    f64 z[16];
                    i32 ncur = nsup - ib;

                    while (ncur >= nlow) {
                        i32 ib1 = 1;
                        if (ncur > nlow) {
                            if (a[ncur + (ncur - 1) * lda] != ZERO) {
                                ib1 = 2;
                            }
                        }
                        i32 nb = ib1 + ib;
                        SLC_DLASET("Full", &nb, &nb, &ZERO, &ONE, z, &(i32){4});
                        i32 ll = ncur - ib1 + 1;

                        i32 exc_info;
                        SLC_DLAEXC(&(i32){1}, &nb, &a[ll + ll * lda], &lda, z, &(i32){4},
                                   &(i32){1}, &ib1, &ib, dwork, &exc_info);
                        if (exc_info != 0) {
                            *info = 2;
                            return;
                        }

                        i32 l1_ = ll + nb;
                        if (l1_ <= nsup) {
                            i32 cols = nsup - l1_ + 1;
                            SLC_DGEMM("T", "N", &nb, &cols, &nb, &ONE, z, &(i32){4},
                                      &a[ll + l1_ * lda], &lda, &ZERO, dwork, &nb);
                            SLC_DLACPY("Full", &nb, &cols, dwork, &nb, &a[ll + l1_ * lda], &lda);
                        }
                        SLC_DGEMM("N", "N", &ll, &nb, &nb, &ONE, &a[0 + ll * lda], &lda,
                                  z, &(i32){4}, &ZERO, dwork, &n);
                        SLC_DLACPY("Full", &ll, &nb, dwork, &n, &a[0 + ll * lda], &lda);

                        SLC_DGEMM("T", "N", &nb, &m, &nb, &ONE, z, &(i32){4},
                                  &b[ll], &ldb, &ZERO, dwork, &nb);
                        SLC_DLACPY("Full", &nb, &m, dwork, &nb, &b[ll], &ldb);

                        if (p > 0) {
                            SLC_DGEMM("N", "N", &p, &nb, &nb, &ONE, &c[0 + ll * ldc], &ldc,
                                      z, &(i32){4}, &ZERO, dwork, &p);
                            SLC_DLACPY("Full", &p, &nb, dwork, &p, &c[0 + ll * ldc], &ldc);
                        }

                        SLC_DGEMM("N", "N", &m, &nb, &nb, &ONE, &cr[0 + ll * ldcr], &ldcr,
                                  z, &(i32){4}, &ZERO, dwork, &m);
                        SLC_DLACPY("Full", &m, &nb, dwork, &m, &cr[0 + ll * ldcr], &ldcr);

                        ncur = ncur - ib1;
                    }
                }
                nlow = nlow + ib;
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
    if (discr) {
        SLC_DTRMM("Right", "Upper", "N", "NonUnit", &p, &m, &ONE, dr, &lddr, d, &ldd);
    }

    f64 opt1 = (f64)(m * (m + 2));
    f64 opt2 = (f64)(4 * m);
    f64 opt3 = (f64)(4 * p);
    f64 opt_max = opt1 > opt2 ? opt1 : opt2;
    opt_max = opt_max > opt3 ? opt_max : opt3;
    dwork[0] = wrkopt > opt_max ? wrkopt : opt_max;
}
