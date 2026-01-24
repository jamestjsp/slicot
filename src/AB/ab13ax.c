/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <stdbool.h>
#include <stddef.h>

f64 ab13ax(
    const char* dico,
    const i32 n,
    const i32 m,
    const i32 p,
    const f64* a,
    const i32 lda,
    const f64* b,
    const i32 ldb,
    const f64* c,
    const i32 ldc,
    f64* hsv,
    f64* dwork,
    const i32 ldwork,
    i32* info
)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    *info = 0;
    bool discr = (*dico == 'D' || *dico == 'd');

    i32 max_nmp = n;
    if (m > max_nmp) max_nmp = m;
    if (p > max_nmp) max_nmp = p;

    if (!discr && !(*dico == 'C' || *dico == 'c')) {
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
    } else if (ldwork < (n > 1 ? n * (max_nmp + 5) + (n * (n + 1)) / 2 : 1)) {
        *info = -13;
    }

    if (*info != 0) {
        i32 neginfo = -(*info);
        SLC_XERBLA("AB13AX", &neginfo);
        return ZERO;
    }

    i32 mn = n;
    if (m < mn) mn = m;
    if (p < mn) mn = p;
    if (mn == 0) {
        dwork[0] = ONE;
        return ZERO;
    }

    i32 ku = 0;
    i32 ks = 0;
    i32 ktau = ks + n * max_nmp;
    i32 kr = ktau + n;
    i32 kw = kr;

    SLC_DLACPY("Full", &p, &n, c, &ldc, &dwork[ku], &max_nmp);

    f64 scaleo;
    i32 ierr = 0;
    sb03ou(discr, false, n, p, a, lda, &dwork[ku], max_nmp,
           &dwork[ktau], &dwork[ku], n, &scaleo, &dwork[kw], ldwork - kw, &ierr);

    if (ierr != 0) {
        *info = 1;
        return ZERO;
    }

    f64 wrkopt = dwork[kw] + (f64)(kw);

    i32 pack_size = (n * (n + 1)) / 2;
    ma02dd("Pack", "Upper", n, &dwork[ku], n, &dwork[kr]);

    kw = kr + pack_size;

    SLC_DLACPY("Full", &n, &m, b, &ldb, &dwork[ks], &n);

    f64 scalec;
    sb03ou(discr, true, n, m, a, lda, &dwork[ks], n,
           &dwork[ktau], &dwork[ks], n, &scalec, &dwork[kw], ldwork - kw, &ierr);

    f64 tmp = dwork[kw] + (f64)(kw);
    if (tmp > wrkopt) wrkopt = tmp;

    i32 j = ks;
    for (i32 i = 0; i < n; i++) {
        i32 len = i + 1;
        SLC_DTPMV("Upper", "NoTranspose", "NonUnit", &len, &dwork[kr],
                  &dwork[j], &(i32){1});
        j += n;
    }

    kw = ktau;
    ierr = mb03ud('N', 'N', n, &dwork[ks], n, NULL, 1, hsv, &dwork[kw], ldwork - kw, info);

    if (ierr != 0) {
        *info = 2;
        return ZERO;
    }

    f64 scale_factor = ONE / scalec / scaleo;
    SLC_DSCAL(&n, &scale_factor, hsv, &(i32){1});

    f64 result = hsv[0];

    tmp = dwork[kw] + (f64)(kw);
    if (tmp > wrkopt) wrkopt = tmp;
    dwork[0] = wrkopt;

    return result;
}
