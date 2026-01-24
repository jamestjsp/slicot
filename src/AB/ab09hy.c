/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <stdlib.h>

void ab09hy(
    const i32 n,
    const i32 m,
    const i32 p,
    const f64* a,
    const i32 lda,
    const f64* b,
    const i32 ldb,
    const f64* c,
    const i32 ldc,
    const f64* d,
    const i32 ldd,
    f64* scalec,
    f64* scaleo,
    f64* s,
    const i32 lds,
    f64* r,
    const i32 ldr,
    i32* iwork,
    f64* dwork,
    const i32 ldwork,
    i32* bwork,
    i32* info
)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const f64 TWO = 2.0;

    *info = 0;

    i32 max_n = n > 1 ? n : 1;
    i32 max_p = p > 1 ? p : 1;

    i32 max_nmp = n > m ? n : m;
    if (p > max_nmp) max_nmp = p;

    i32 lw = n * (max_nmp + 5);
    i32 lw2 = 2 * n * p + (p * (m + 2) > 10 * n * (n + 1) ? p * (m + 2) : 10 * n * (n + 1));
    if (lw2 > lw) lw = lw2;
    if (lw < 2) lw = 2;

    if (n < 0) {
        *info = -1;
    } else if (m < 0) {
        *info = -2;
    } else if (p < 0 || p > m) {
        *info = -3;
    } else if (lda < max_n) {
        *info = -5;
    } else if (ldb < max_n) {
        *info = -7;
    } else if (ldc < max_p) {
        *info = -9;
    } else if (ldd < max_p) {
        *info = -11;
    } else if (lds < max_n) {
        *info = -15;
    } else if (ldr < max_n) {
        *info = -17;
    } else if (ldwork < lw) {
        *info = -20;
    }

    if (*info != 0) {
        i32 neginfo = -(*info);
        SLC_XERBLA("AB09HY", &neginfo);
        return;
    }

    *scalec = ONE;
    *scaleo = ONE;
    if (n == 0 || m == 0 || p == 0) {
        dwork[0] = TWO;
        dwork[1] = ONE;
        return;
    }

    i32 wrkopt;
    i32 ierr;

    i32 ku = 0;
    i32 max_nm = n > m ? n : m;
    i32 ktau = ku + n * max_nm;
    i32 kw = ktau + n;

    SLC_DLACPY("Full", &n, &m, b, &ldb, dwork, &n);

    i32 ldwork_sb03ou = ldwork - kw;
    sb03ou(false, true, n, m, a, lda, dwork, n, &dwork[ktau], s, lds, scalec,
           &dwork[kw], ldwork_sb03ou, &ierr);

    if (ierr != 0) {
        *info = 1;
        return;
    }
    wrkopt = (i32)dwork[kw] + kw;

    i32 kbw = 0;
    i32 kcw = kbw + p * n;
    i32 kd = kcw + p * n;
    i32 kdw = kd + p * (m - p);
    ktau = kd + p * m;
    kw = ktau + p;

    SLC_DLACPY("Full", &p, &m, d, &ldd, &dwork[kd], &p);

    i32 ldwork_rqf = ldwork - kw;
    SLC_DGERQF(&p, &m, &dwork[kd], &p, &dwork[ktau], &dwork[kw], &ldwork_rqf, &ierr);
    if ((i32)dwork[kw] + kw > wrkopt) {
        wrkopt = (i32)dwork[kw] + kw;
    }

    f64 rtol = (f64)m * SLC_DLAMCH("E") * SLC_DLANGE("1", &p, &m, d, &ldd, dwork);

    for (i32 i = kdw; i < kdw + p * p; i += p + 1) {
        if (fabs(dwork[i]) <= rtol) {
            *info = 6;
            return;
        }
    }

    SLC_DLACPY("Full", &p, &n, c, &ldc, &dwork[kcw], &p);
    SLC_DTRSM("Left", "Upper", "No-transpose", "Non-unit", &p, &n,
              &ONE, &dwork[kdw], &p, &dwork[kcw], &p);

    SLC_DLACPY("Full", &p, &n, &dwork[kcw], &p, &dwork[kbw], &p);
    SLC_DTRMM("Right", "Upper", "No-transpose", "Non-unit", &p, &n,
              &ONE, s, &lds, &dwork[kbw], &p);
    SLC_DTRMM("Right", "Upper", "Transpose", "Non-unit", &p, &n,
              &ONE, s, &lds, &dwork[kbw], &p);

    i32 ldwork_rq = ldwork - kw;
    SLC_DORGRQ(&p, &m, &p, &dwork[kd], &p, &dwork[ktau], &dwork[kw], &ldwork_rq, &ierr);
    if ((i32)dwork[kw] + kw > wrkopt) {
        wrkopt = (i32)dwork[kw] + kw;
    }

    SLC_DGEMM("No-transpose", "Transpose", &p, &n, &m, &ONE,
              &dwork[kd], &p, b, &ldb, &ONE, &dwork[kbw], &p);

    SLC_DLACPY("Full", &n, &n, a, &lda, r, &ldr);
    f64 neg_one = -ONE;
    SLC_DGEMM("Transpose", "No-transpose", &n, &n, &p, &neg_one,
              &dwork[kbw], &p, &dwork[kcw], &p, &ONE, r, &ldr);

    i32 n2 = n + n;
    i32 kg = kd;
    i32 kq = kg + n * n;
    i32 kwr = kq + n * n;
    i32 kwi = kwr + n2;
    i32 ks = kwi + n2;
    i32 ku2 = ks + n2 * n2;
    kw = ku2 + n2 * n2;

    SLC_DSYRK("Upper", "Transpose", &n, &p, &neg_one, &dwork[kbw], &p, &ZERO,
              &dwork[kg], &n);

    SLC_DSYRK("Upper", "Transpose", &n, &p, &ONE, &dwork[kcw], &p, &ZERO,
              &dwork[kq], &n);

    i32 ldwork_sb02md = ldwork - kw;
    f64 rcond_val;
    sb02md("Continuous", "None", "Upper", "General", "Stable",
           n, r, ldr, &dwork[kg], n, &dwork[kq], n, &rcond_val,
           &dwork[kwr], &dwork[kwi], &dwork[ks], n2,
           &dwork[ku2], n2, iwork, &dwork[kw], ldwork_sb02md,
           bwork, info);

    if (*info != 0) {
        return;
    }
    if ((i32)dwork[kw] + kw > wrkopt) {
        wrkopt = (i32)dwork[kw] + kw;
    }

    SLC_DGEMM("No-transpose", "No-transpose", &p, &n, &n, &neg_one,
              &dwork[kbw], &p, &dwork[kq], &n, &ONE, &dwork[kcw], &p);

    i32 max_np = n > p ? n : p;
    ktau = kcw + n * max_np;
    kw = ktau + n;

    i32 ldwork_sb03ou2 = ldwork - kw;
    sb03ou(false, false, n, p, a, lda, &dwork[kcw], p, &dwork[ktau], r, ldr,
           scaleo, &dwork[kw], ldwork_sb03ou2, &ierr);

    if ((i32)dwork[kw] + kw > wrkopt) {
        wrkopt = (i32)dwork[kw] + kw;
    }

    dwork[0] = (f64)wrkopt;
    dwork[1] = rcond_val;
}
