// SPDX-License-Identifier: BSD-3-Clause

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <stdbool.h>

void tg01ed(
    const char* joba,
    const i32 l, const i32 n, const i32 m, const i32 p,
    f64* a, const i32 lda,
    f64* e, const i32 lde,
    f64* b, const i32 ldb,
    f64* c, const i32 ldc,
    f64* q, const i32 ldq,
    f64* z, const i32 ldz,
    i32* ranke, i32* rnka22,
    const f64 tol,
    f64* dwork, const i32 ldwork,
    i32* info
) {
    const f64 one = 1.0, zero = 0.0;

    bool reda;
    i32 i, ir1, j, kw, la22, ln, ln2, lwr, na22, wrkopt;
    f64 epsm, svemax, svlmax, toldef;
    i32 int1 = 1;

    reda = (joba[0] == 'R' || joba[0] == 'r');

    *info = 0;

    ln = (l < n) ? l : n;
    i32 temp1 = 3 * ln + ((l > n) ? l : n);
    i32 temp2 = 5 * ln;
    temp1 = (temp1 > temp2) ? temp1 : temp2;
    temp1 = (temp1 > m) ? temp1 : m;
    temp1 = (temp1 > p) ? temp1 : p;
    wrkopt = ln + temp1;
    if (wrkopt < 1) wrkopt = 1;

    if (joba[0] != 'N' && joba[0] != 'n' && !reda) {
        *info = -1;
    } else if (l < 0) {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    } else if (m < 0) {
        *info = -4;
    } else if (p < 0) {
        *info = -5;
    } else if (lda < ((1 > l) ? 1 : l)) {
        *info = -7;
    } else if (lde < ((1 > l) ? 1 : l)) {
        *info = -9;
    } else if (ldb < 1 || (m > 0 && ldb < l)) {
        *info = -11;
    } else if (ldc < ((1 > p) ? 1 : p)) {
        *info = -13;
    } else if (ldq < ((1 > l) ? 1 : l)) {
        *info = -15;
    } else if (ldz < ((1 > n) ? 1 : n)) {
        *info = -17;
    } else if (tol >= one) {
        *info = -20;
    } else if (ldwork < ((1 > wrkopt) ? 1 : wrkopt)) {
        *info = -22;
    }

    if (*info != 0) {
        return;
    }

    if (l == 0 || n == 0) {
        if (l > 0)
            SLC_DLASET("F", &l, &l, &zero, &one, q, &ldq);
        if (n > 0)
            SLC_DLASET("F", &n, &n, &zero, &one, z, &ldz);
        dwork[0] = one;
        *ranke = 0;
        if (reda) *rnka22 = 0;
        return;
    }

    ln = (l < n) ? l : n;
    epsm = SLC_DLAMCH("E");

    toldef = tol;
    if (toldef <= zero) {
        toldef = epsm * (f64)(l * n);
    }

    svlmax = SLC_DLANGE("F", &l, &n, e, &lde, dwork);

    lwr = ldwork - ln;
    kw = ln;

    SLC_DGESVD("A", "A", &l, &n, e, &lde, dwork, q, &ldq, z, &ldz,
               &dwork[kw], &lwr, info);
    if (*info > 0)
        return;
    wrkopt = ln + (i32)dwork[kw];
    if (wrkopt < 1) wrkopt = 1;

    *ranke = 0;
    if (dwork[0] > toldef) {
        *ranke = 1;
        svemax = dwork[0];
        for (i = 1; i < ln; i++) {
            if (dwork[i] < svemax * toldef) break;
            (*ranke)++;
        }
    }

    if (*ranke > 0) {
        SLC_DGEMM("T", "N", &l, &n, &l, &one, q, &ldq, a, &lda, &zero, e, &lde);
        SLC_DGEMM("N", "T", &l, &n, &n, &one, e, &lde, z, &ldz, &zero, a, &lda);

        if (lwr > l * m && m > 0) {
            SLC_DGEMM("T", "N", &l, &m, &l, &one, q, &ldq, b, &ldb, &zero, &dwork[kw], &l);
            SLC_DLACPY("F", &l, &m, &dwork[kw], &l, b, &ldb);
        } else {
            for (j = 0; j < m; j++) {
                SLC_DGEMV("T", &l, &l, &one, q, &ldq, &b[j * ldb], &int1, &zero, &dwork[kw], &int1);
                SLC_DCOPY(&l, &dwork[kw], &int1, &b[j * ldb], &int1);
            }
        }

        if (lwr > p * n) {
            i32 maxp1 = (1 > p) ? 1 : p;
            SLC_DGEMM("N", "T", &p, &n, &n, &one, c, &ldc, z, &ldz, &zero, &dwork[kw], &maxp1);
            SLC_DLACPY("F", &p, &n, &dwork[kw], &maxp1, c, &ldc);
        } else {
            for (i = 0; i < p; i++) {
                SLC_DGEMV("N", &n, &n, &one, z, &ldz, &c[i], &ldc, &zero, &dwork[kw], &int1);
                SLC_DCOPY(&n, &dwork[kw], &int1, &c[i], &ldc);
            }
        }
        i32 lm = l * m, pn = p * n;
        lm = (lm > 0) ? lm : 1;
        pn = (pn > 0) ? pn : 1;
        i32 newopt = ln + ((lm > pn) ? lm : pn);
        wrkopt = (wrkopt > newopt) ? wrkopt : newopt;
    }

    if (reda) {
        la22 = l - *ranke;
        na22 = n - *ranke;
        ln2 = (la22 < na22) ? la22 : na22;
        if (ln2 == 0) {
            ir1 = 0;
            *rnka22 = 0;
        } else {
            svlmax = SLC_DLANGE("F", &l, &n, a, &lda, dwork);

            ir1 = *ranke;
            if (la22 >= na22) {
                SLC_DGEQRF(&la22, &na22, &a[ir1 + ir1 * lda], &lda, &dwork[ir1],
                           &dwork[kw], &lwr, info);
                i32 opt = ln + (i32)dwork[kw];
                wrkopt = (wrkopt > opt) ? wrkopt : opt;

                i32 ranke_val = *ranke;
                SLC_DORMQR("L", "T", &la22, &ranke_val, &ln2,
                           &a[ir1 + ir1 * lda], &lda, &dwork[ir1], &a[ir1], &lda,
                           &dwork[kw], &lwr, info);
                opt = ln + (i32)dwork[kw];
                wrkopt = (wrkopt > opt) ? wrkopt : opt;

                if (m > 0) {
                    SLC_DORMQR("L", "T", &la22, &m, &ln2,
                               &a[ir1 + ir1 * lda], &lda, &dwork[ir1], &b[ir1], &ldb,
                               &dwork[kw], &lwr, info);
                    opt = ln + (i32)dwork[kw];
                    wrkopt = (wrkopt > opt) ? wrkopt : opt;
                }

                SLC_DORMQR("R", "N", &l, &la22, &ln2,
                           &a[ir1 + ir1 * lda], &lda, &dwork[ir1], &q[ir1 * ldq], &ldq,
                           &dwork[kw], &lwr, info);
                opt = ln + (i32)dwork[kw];
                wrkopt = (wrkopt > opt) ? wrkopt : opt;

                mb03ud('V', 'V', ln2, &a[ir1 + ir1 * lda], lda,
                       &e[ir1 + ir1 * lde], lde, &dwork[ir1], &dwork[kw], lwr, info);
                if (*info > 0)
                    return;
                opt = ln + (i32)dwork[kw];
                wrkopt = (wrkopt > opt) ? wrkopt : opt;

                *rnka22 = 0;
                if (dwork[ir1] > svlmax * epsm) {
                    *rnka22 = 1;
                    for (i = ir1 + 1; i < ln; i++) {
                        if (dwork[i] <= svlmax * toldef) break;
                        (*rnka22)++;
                    }
                }

                if (*rnka22 > 0) {
                    i32 ranke_val = *ranke;
                    SLC_DGEMM("T", "N", &ln2, &ranke_val, &ln2, &one,
                              &e[ir1 + ir1 * lde], &lde, &a[ir1], &lda,
                              &zero, &e[ir1], &lde);
                    SLC_DLACPY("F", &ln2, &ranke_val, &e[ir1], &lde, &a[ir1], &lda);
                    SLC_DGEMM("N", "T", &ranke_val, &ln2, &ln2, &one,
                              &a[ir1 * lda], &lda, &a[ir1 + ir1 * lda], &lda,
                              &zero, &e[ir1 * lde], &lde);
                    SLC_DLACPY("F", &ranke_val, &ln2, &e[ir1 * lde], &lde, &a[ir1 * lda], &lda);

                    if (lwr > ln2 * m && m > 0) {
                        SLC_DGEMM("T", "N", &ln2, &m, &ln2, &one,
                                  &e[ir1 + ir1 * lde], &lde, &b[ir1], &ldb,
                                  &zero, &dwork[kw], &ln2);
                        SLC_DLACPY("F", &ln2, &m, &dwork[kw], &ln2, &b[ir1], &ldb);
                    } else {
                        for (j = 0; j < m; j++) {
                            SLC_DGEMV("T", &ln2, &ln2, &one,
                                      &e[ir1 + ir1 * lde], &lde, &b[ir1 + j * ldb], &int1,
                                      &zero, &dwork[kw], &int1);
                            SLC_DCOPY(&ln2, &dwork[kw], &int1, &b[ir1 + j * ldb], &int1);
                        }
                    }

                    if (lwr > p * ln2 && p > 0) {
                        SLC_DGEMM("N", "T", &p, &ln2, &ln2, &one,
                                  &c[ir1 * ldc], &ldc, &a[ir1 + ir1 * lda], &lda,
                                  &zero, &dwork[kw], &p);
                        SLC_DLACPY("F", &p, &ln2, &dwork[kw], &p, &c[ir1 * ldc], &ldc);
                    } else {
                        for (i = 0; i < p; i++) {
                            SLC_DGEMV("N", &ln2, &ln2, &one,
                                      &a[ir1 + ir1 * lda], &lda, &c[i + ir1 * ldc], &ldc,
                                      &zero, &dwork[kw], &int1);
                            SLC_DCOPY(&ln2, &dwork[kw], &int1, &c[i + ir1 * ldc], &int1);
                        }
                    }

                    if (lwr > l * ln2) {
                        SLC_DGEMM("N", "N", &l, &ln2, &ln2, &one,
                                  &q[ir1 * ldq], &ldq, &e[ir1 + ir1 * lde], &lde,
                                  &zero, &dwork[kw], &l);
                        SLC_DLACPY("F", &l, &ln2, &dwork[kw], &l, &q[ir1 * ldq], &ldq);
                    } else {
                        for (i = 0; i < l; i++) {
                            SLC_DGEMV("T", &ln2, &ln2, &one,
                                      &e[ir1 + ir1 * lde], &lde, &q[i + ir1 * ldq], &ldq,
                                      &zero, &dwork[kw], &int1);
                            SLC_DCOPY(&ln2, &dwork[kw], &int1, &q[i + ir1 * ldq], &ldq);
                        }
                    }

                    if (lwr > n * ln2) {
                        SLC_DGEMM("N", "N", &ln2, &n, &ln2, &one,
                                  &a[ir1 + ir1 * lda], &lda, &z[ir1], &ldz,
                                  &zero, &dwork[kw], &ln2);
                        SLC_DLACPY("F", &ln2, &n, &dwork[kw], &ln2, &z[ir1], &ldz);
                    } else {
                        for (j = 0; j < n; j++) {
                            SLC_DGEMV("N", &ln2, &ln2, &one,
                                      &a[ir1 + ir1 * lda], &lda, &z[ir1 + j * ldz], &int1,
                                      &zero, &dwork[kw], &int1);
                            SLC_DCOPY(&ln2, &dwork[kw], &int1, &z[ir1 + j * ldz], &int1);
                        }
                    }
                }
            } else {
                SLC_DGELQF(&la22, &na22, &a[ir1 + ir1 * lda], &lda, &dwork[ir1],
                           &dwork[kw], &lwr, info);
                i32 opt = ln + (i32)dwork[kw];
                wrkopt = (wrkopt > opt) ? wrkopt : opt;

                i32 ranke_val = *ranke;
                SLC_DORMLQ("R", "T", &ranke_val, &na22, &ln2,
                           &a[ir1 + ir1 * lda], &lda, &dwork[ir1], &a[ir1 * lda], &lda,
                           &dwork[kw], &lwr, info);
                opt = ln + (i32)dwork[kw];
                wrkopt = (wrkopt > opt) ? wrkopt : opt;

                if (p > 0) {
                    SLC_DORMLQ("R", "T", &p, &na22, &ln2,
                               &a[ir1 + ir1 * lda], &lda, &dwork[ir1], &c[ir1 * ldc], &ldc,
                               &dwork[kw], &lwr, info);
                    opt = ln + (i32)dwork[kw];
                    wrkopt = (wrkopt > opt) ? wrkopt : opt;
                }

                SLC_DORMLQ("L", "N", &na22, &n, &ln2,
                           &a[ir1 + ir1 * lda], &lda, &dwork[ir1], &z[ir1], &ldz,
                           &dwork[kw], &lwr, info);
                opt = ln + (i32)dwork[kw];
                wrkopt = (wrkopt > opt) ? wrkopt : opt;

                ma02ad("L", ln2, ln2, &a[ir1 + ir1 * lda], lda, &e[ir1 + ir1 * lde], lde);
                mb03ud('V', 'V', ln2, &e[ir1 + ir1 * lde], lde,
                       &a[ir1 + ir1 * lda], lda, &dwork[ir1], &dwork[kw], lwr, info);
                if (*info > 0)
                    return;
                opt = ln + (i32)dwork[kw];
                wrkopt = (wrkopt > opt) ? wrkopt : opt;

                *rnka22 = 0;
                if (dwork[ir1] > svlmax * epsm) {
                    *rnka22 = 1;
                    for (i = ir1 + 1; i < ln; i++) {
                        if (dwork[i] <= svlmax * toldef) break;
                        (*rnka22)++;
                    }
                }

                if (*rnka22 > 0) {
                    i32 ranke_val = *ranke;
                    SLC_DGEMM("N", "N", &ln2, &ranke_val, &ln2, &one,
                              &e[ir1 + ir1 * lde], &lde, &a[ir1], &lda,
                              &zero, &e[ir1], &lde);
                    SLC_DLACPY("F", &ln2, &ranke_val, &e[ir1], &lde, &a[ir1], &lda);
                    SLC_DGEMM("N", "N", &ranke_val, &ln2, &ln2, &one,
                              &a[ir1 * lda], &lda, &a[ir1 + ir1 * lda], &lda,
                              &zero, &e[ir1 * lde], &lde);
                    SLC_DLACPY("F", &ranke_val, &ln2, &e[ir1 * lde], &lde, &a[ir1 * lda], &lda);

                    if (lwr > ln2 * m && m > 0) {
                        SLC_DGEMM("N", "N", &ln2, &m, &ln2, &one,
                                  &e[ir1 + ir1 * lde], &lde, &b[ir1], &ldb,
                                  &zero, &dwork[kw], &ln2);
                        SLC_DLACPY("F", &ln2, &m, &dwork[kw], &ln2, &b[ir1], &ldb);
                    } else {
                        for (j = 0; j < m; j++) {
                            SLC_DGEMV("N", &ln2, &ln2, &one,
                                      &e[ir1 + ir1 * lde], &lde, &b[ir1 + j * ldb], &int1,
                                      &zero, &dwork[kw], &int1);
                            SLC_DCOPY(&ln2, &dwork[kw], &int1, &b[ir1 + j * ldb], &int1);
                        }
                    }

                    if (lwr > p * ln2 && p > 0) {
                        SLC_DGEMM("N", "N", &p, &ln2, &ln2, &one,
                                  &c[ir1 * ldc], &ldc, &a[ir1 + ir1 * lda], &lda,
                                  &zero, &dwork[kw], &p);
                        SLC_DLACPY("F", &p, &ln2, &dwork[kw], &p, &c[ir1 * ldc], &ldc);
                    } else {
                        for (i = 0; i < p; i++) {
                            SLC_DGEMV("T", &ln2, &ln2, &one,
                                      &a[ir1 + ir1 * lda], &lda, &c[i + ir1 * ldc], &ldc,
                                      &zero, &dwork[kw], &int1);
                            SLC_DCOPY(&ln2, &dwork[kw], &int1, &c[i + ir1 * ldc], &ldc);
                        }
                    }

                    if (lwr > l * ln2) {
                        SLC_DGEMM("N", "T", &l, &ln2, &ln2, &one,
                                  &q[ir1 * ldq], &ldq, &e[ir1 + ir1 * lde], &lde,
                                  &zero, &dwork[kw], &l);
                        SLC_DLACPY("F", &l, &ln2, &dwork[kw], &l, &q[ir1 * ldq], &ldq);
                    } else {
                        for (i = 0; i < l; i++) {
                            SLC_DGEMV("N", &ln2, &ln2, &one,
                                      &e[ir1 + ir1 * lde], &lde, &q[i + ir1 * ldq], &ldq,
                                      &zero, &dwork[kw], &int1);
                            SLC_DCOPY(&ln2, &dwork[kw], &int1, &q[i + ir1 * ldq], &ldq);
                        }
                    }

                    if (lwr > n * ln2) {
                        SLC_DGEMM("T", "N", &ln2, &n, &ln2, &one,
                                  &a[ir1 + ir1 * lda], &lda, &z[ir1], &ldz,
                                  &zero, &dwork[kw], &ln2);
                        SLC_DLACPY("F", &ln2, &n, &dwork[kw], &ln2, &z[ir1], &ldz);
                    } else {
                        for (j = 0; j < n; j++) {
                            SLC_DGEMV("T", &ln2, &ln2, &one,
                                      &a[ir1 + ir1 * lda], &lda, &z[ir1 + j * ldz], &int1,
                                      &zero, &dwork[kw], &int1);
                            SLC_DCOPY(&ln2, &dwork[kw], &int1, &z[ir1 + j * ldz], &int1);
                        }
                    }
                }
            }
        }
    }

    SLC_DLASET("F", &l, &n, &zero, &zero, e, &lde);
    i32 lde_plus_1 = lde + 1;
    SLC_DCOPY(ranke, dwork, &int1, e, &lde_plus_1);

    if (reda) {
        la22 = l - *ranke;
        na22 = n - *ranke;
        ir1 = *ranke;
        SLC_DLASET("F", &la22, &na22, &zero, &zero, &a[ir1 + ir1 * lda], &lda);
        i32 lda_plus_1 = lda + 1;
        SLC_DCOPY(rnka22, &dwork[ir1], &int1, &a[ir1 + ir1 * lda], &lda_plus_1);
    }

    for (i = 1; i < n; i++) {
        i32 im1 = i;
        SLC_DSWAP(&im1, &z[i * ldz], &int1, &z[i], &ldz);
    }

    dwork[0] = (f64)wrkopt;
}
