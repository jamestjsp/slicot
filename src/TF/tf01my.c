/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"

void tf01my(const i32 n, const i32 m, const i32 p, const i32 ny,
            const f64* a, const i32 lda, const f64* b, const i32 ldb,
            const f64* c, const i32 ldc, const f64* d, const i32 ldd,
            const f64* u, const i32 ldu, f64* x, f64* y, const i32 ldy,
            f64* dwork, const i32 ldwork, i32* info)
{
    const f64 zero = 0.0;
    const f64 one = 1.0;
    const i32 int1 = 1;

    i32 maxn, ik, is, iyl, irem, ns, wrkopt;
    i32 max_mp, lquery, minnyp;
    f64 upd;

    *info = 0;

    maxn = (n > 1) ? n : 1;
    minnyp = (ny < p) ? ny : p;

    if (n < 0) {
        *info = -1;
    } else if (m < 0) {
        *info = -2;
    } else if (p < 0) {
        *info = -3;
    } else if (ny < 0) {
        *info = -4;
    } else if (lda < maxn) {
        *info = -6;
    } else if (ldb < maxn) {
        *info = -8;
    } else if (ldc < ((p > 1) ? p : 1)) {
        *info = -10;
    } else if (ldd < ((p > 1) ? p : 1)) {
        *info = -12;
    } else if (ldu < ((ny > 1) ? ny : 1)) {
        *info = -14;
    } else if (ldy < ((ny > 1) ? ny : 1)) {
        *info = -17;
    } else {
        lquery = (ldwork == -1);
        if (lquery) {
            max_mp = (m > p) ? m : p;
            i32 geqrf_ldy = ldy;
            i32 geqrf_ny = ny;
            i32 geqrf_max_mp = max_mp;
            i32 geqrf_info;
            f64 work_query;
            i32 neg1 = -1;
            SLC_DGEQRF(&geqrf_ny, &geqrf_max_mp, y, &geqrf_ldy, dwork, &work_query, &neg1, &geqrf_info);
            i32 geqrf_opt = (i32)work_query;
            wrkopt = (geqrf_opt > 1) ? geqrf_opt : 1;
            wrkopt = (wrkopt > 2 * n) ? wrkopt : 2 * n;
        } else {
            i32 ik_min;
            if (minnyp == 0) {
                ik_min = 1;
            } else {
                ik_min = maxn;
            }
            if (ldwork < ik_min) {
                *info = -19;
            }
        }
    }

    if (*info != 0) {
        return;
    }

    if (lquery) {
        dwork[0] = (f64)wrkopt;
        return;
    }

    if (minnyp == 0) {
        dwork[0] = one;
        return;
    }

    if (n == 0) {
        if (m == 0) {
            SLC_DLASET("Full", &ny, &p, &zero, &zero, y, &ldy);
        } else {
            SLC_DGEMM("No transpose", "Transpose", &ny, &p, &m, &one,
                      u, &ldu, d, &ldd, &zero, y, &ldy);
        }
        dwork[0] = one;
        return;
    }

    max_mp = (m > p) ? m : p;
    i32 geqrf_ldy = ldy;
    i32 geqrf_ny = ny;
    i32 geqrf_max_mp = max_mp;
    i32 geqrf_info;
    f64 work_query;
    i32 neg1 = -1;
    SLC_DGEQRF(&geqrf_ny, &geqrf_max_mp, y, &geqrf_ldy, dwork, &work_query, &neg1, &geqrf_info);
    wrkopt = (i32)work_query;
    wrkopt = (wrkopt > 1) ? wrkopt : 1;
    wrkopt = (wrkopt > 2 * n) ? wrkopt : 2 * n;

    ns = ldwork / n;
    i32 wrkopt_ns = wrkopt / n;
    ns = (ns < wrkopt_ns) ? ns : wrkopt_ns;
    ns = (ns < ny) ? ns : ny;

    if (ns <= 1 || ny * max_mp <= wrkopt) {
        for (ik = 0; ik < ny; ik++) {
            SLC_DGEMV("No transpose", &p, &n, &one, c, &ldc, x, &int1,
                      &zero, &y[ik], &ldy);

            SLC_DGEMV("No transpose", &n, &n, &one, a, &lda, x, &int1,
                      &zero, dwork, &int1);
            SLC_DGEMV("No transpose", &n, &m, &one, b, &ldb, &u[ik], &ldu,
                      &one, dwork, &int1);

            SLC_DCOPY(&n, dwork, &int1, x, &int1);
        }
    } else {
        iyl = (ny / ns) * ns;
        upd = (m == 0) ? zero : one;

        SLC_DCOPY(&n, x, &int1, dwork, &int1);

        for (ik = 0; ik < iyl; ik += ns) {
            i32 ns_m1 = ns - 1;
            SLC_DGEMM("No transpose", "Transpose", &n, &ns_m1, &m, &one,
                      b, &ldb, &u[ik], &ldu, &zero, &dwork[n], &maxn);

            for (is = 0; is < ns - 1; is++) {
                SLC_DGEMV("No transpose", &n, &n, &one, a, &lda,
                          &dwork[is * n], &int1, &upd, &dwork[(is + 1) * n], &int1);
            }

            SLC_DGEMM("Transpose", "Transpose", &ns, &p, &n, &one, dwork,
                      &maxn, c, &ldc, &zero, &y[ik], &ldy);

            SLC_DGEMV("No transpose", &n, &m, &one, b, &ldb,
                      &u[ik + ns - 1], &ldu, &zero, dwork, &int1);
            SLC_DGEMV("No transpose", &n, &n, &one, a, &lda,
                      &dwork[(ns - 1) * n], &int1, &upd, dwork, &int1);
        }

        irem = ny - iyl;

        if (irem > 1) {
            ik = iyl;
            i32 irem_m1 = irem - 1;
            SLC_DGEMM("No transpose", "Transpose", &n, &irem_m1, &m, &one,
                      b, &ldb, &u[ik], &ldu, &zero, &dwork[n], &maxn);

            for (is = 0; is < irem - 1; is++) {
                SLC_DGEMV("No transpose", &n, &n, &one, a, &lda,
                          &dwork[is * n], &int1, &upd, &dwork[(is + 1) * n], &int1);
            }

            SLC_DGEMM("Transpose", "Transpose", &irem, &p, &n, &one, dwork,
                      &maxn, c, &ldc, &zero, &y[ik], &ldy);

            SLC_DGEMV("No transpose", &n, &m, &one, b, &ldb,
                      &u[ik + irem - 1], &ldu, &zero, dwork, &int1);
            SLC_DGEMV("No transpose", &n, &n, &one, a, &lda,
                      &dwork[(irem - 1) * n], &int1, &upd, dwork, &int1);

        } else if (irem == 1) {
            ik = iyl;
            SLC_DGEMV("No transpose", &p, &n, &one, c, &ldc, dwork, &int1,
                      &zero, &y[ik], &ldy);

            SLC_DCOPY(&n, dwork, &int1, &dwork[n], &int1);
            SLC_DGEMV("No transpose", &n, &m, &one, b, &ldb,
                      &u[ik], &ldu, &zero, dwork, &int1);
            SLC_DGEMV("No transpose", &n, &n, &one, a, &lda,
                      &dwork[n], &int1, &upd, dwork, &int1);
        }

        SLC_DCOPY(&n, dwork, &int1, x, &int1);
    }

    SLC_DGEMM("No transpose", "Transpose", &ny, &p, &m, &one, u, &ldu,
              d, &ldd, &one, y, &ldy);

    dwork[0] = (f64)wrkopt;
}
