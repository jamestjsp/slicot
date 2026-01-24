/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>


void fb01vd(i32 n, i32 m, i32 l,
            f64* p, i32 ldp,
            const f64* a, i32 lda,
            const f64* b, i32 ldb,
            const f64* c, i32 ldc,
            f64* q, i32 ldq,
            f64* r, i32 ldr,
            f64* k, i32 ldk,
            f64 tol,
            i32* iwork,
            f64* dwork, i32 ldwork,
            i32* info)
{
    const f64 zero = 0.0, one = 1.0, two = 2.0;

    *info = 0;
    i32 n1 = (n > 1) ? n : 1;

    if (n < 0) {
        *info = -1;
    } else if (m < 0) {
        *info = -2;
    } else if (l < 0) {
        *info = -3;
    } else if (ldp < n1) {
        *info = -5;
    } else if (lda < n1) {
        *info = -7;
    } else if (ldb < n1) {
        *info = -9;
    } else if (ldc < ((l > 1) ? l : 1)) {
        *info = -11;
    } else if (ldq < ((m > 1) ? m : 1)) {
        *info = -13;
    } else if (ldr < ((l > 1) ? l : 1)) {
        *info = -15;
    } else if (ldk < n1) {
        *info = -17;
    } else {
        i32 wk1 = l*n + 3*l;
        i32 wk2 = n*n;
        i32 wk3 = n*m;
        i32 ldwork_min = wk1;
        if (wk2 > ldwork_min) ldwork_min = wk2;
        if (wk3 > ldwork_min) ldwork_min = wk3;
        if (ldwork_min < 1) ldwork_min = 1;
        if (ldwork < ldwork_min) {
            *info = -21;
        }
    }

    if (*info != 0) {
        return;
    }

    i32 max_nl = (n > l) ? n : l;
    if (max_nl == 0) {
        dwork[0] = one;
        return;
    }

    i32 ldw = (l > 1) ? l : 1;

    mb01rd("Upper", "No transpose", l, n, one, one, r, ldr, c, ldc, p, ldp, dwork, ldwork, info);

    for (i32 j = 0; j < l; j++) {
        SLC_DCOPY(&n, dwork + j, &ldw, k + j*ldk, &(i32){1});
    }

    SLC_DLACPY("Full", &l, &n, (f64*)c, &ldc, dwork, &ldw);
    SLC_DTRMM("Right", "Upper", "Transpose", "Non-unit", &l, &n, &one, p, &ldp, dwork, &ldw);

    i32 ldp1 = ldp + 1;
    SLC_DSCAL(&n, &two, p, &ldp1);

    for (i32 j = 0; j < l; j++) {
        SLC_DAXPY(&n, &one, k + j*ldk, &(i32){1}, dwork + j, &ldw);
        SLC_DCOPY(&n, dwork + j, &ldw, k + j*ldk, &(i32){1});
    }

    i32 jwork = l*n;
    f64 rnorm = SLC_DLANSY("1-norm", "Upper", &l, r, &ldr, dwork + jwork);

    f64 toldef = tol;
    if (toldef <= zero) {
        toldef = (f64)(l*l) * SLC_DLAMCH("Epsilon");
    }

    SLC_DPOTRF("Upper", &l, r, &ldr, info);
    if (*info != 0) {
        return;
    }

    f64 rcond;
    SLC_DPOCON("Upper", &l, r, &ldr, &rnorm, &rcond, dwork + jwork, iwork, info);

    if (rcond < toldef) {
        *info = l + 1;
        dwork[0] = rcond;
        return;
    }

    if (l > 1) {
        i32 l1 = l - 1;
        SLC_DLASET("Lower", &l1, &l1, &zero, &zero, r + 1, &ldr);
    }

    SLC_DTRSM("Right", "Upper", "No transpose", "Non-unit", &n, &l, &one, r, &ldr, k, &ldk);
    SLC_DTRSM("Right", "Upper", "Transpose", "Non-unit", &n, &l, &one, r, &ldr, k, &ldk);

    jwork = 0;
    f64 neg_one = -one;
    for (i32 j = 0; j < n; j++) {
        i32 jj = j + 1;
        SLC_DGEMV("No transpose", &jj, &l, &neg_one, k, &ldk, dwork + jwork, &(i32){1}, &one, p + j*ldp, &(i32){1});
        jwork += l;
    }

    mb01rd("Upper", "No transpose", n, n, zero, one, p, ldp, a, lda, p, ldp, dwork, ldwork, info);

    mb01rd("Upper", "No transpose", n, m, one, one, p, ldp, b, ldb, q, ldq, dwork, ldwork, info);

    i32 ldq1 = ldq + 1;
    SLC_DSCAL(&m, &two, q, &ldq1);

    dwork[0] = rcond;
}
