/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <stdbool.h>
#include <stddef.h>

static bool lsame(char ca, char cb) {
    if (ca >= 'a' && ca <= 'z') ca -= 32;
    if (cb >= 'a' && cb <= 'z') cb -= 32;
    return ca == cb;
}

static i32 imin(i32 a, i32 b) {
    return a < b ? a : b;
}

static i32 imax(i32 a, i32 b) {
    return a > b ? a : b;
}

void mb01pd(const char* scun, const char* type, i32 m, i32 n, i32 kl, i32 ku,
            f64 anrm, i32 nbl, const i32* nrows, f64* a, i32 lda, i32* info) {

    const f64 zero = 0.0, one = 1.0;
    bool lscale;
    i32 itype, isum, mn;
    f64 smlnum, bignum;

    *info = 0;

    lscale = lsame(scun[0], 'S');

    if (lsame(type[0], 'G')) {
        itype = 0;
    } else if (lsame(type[0], 'L')) {
        itype = 1;
    } else if (lsame(type[0], 'U')) {
        itype = 2;
    } else if (lsame(type[0], 'H')) {
        itype = 3;
    } else if (lsame(type[0], 'B')) {
        itype = 4;
    } else if (lsame(type[0], 'Q')) {
        itype = 5;
    } else if (lsame(type[0], 'Z')) {
        itype = 6;
    } else {
        itype = -1;
    }

    mn = imin(m, n);

    isum = 0;
    if (nbl > 0 && nrows != NULL) {
        for (i32 i = 0; i < nbl; i++) {
            isum += nrows[i];
        }
    }

    if (!lscale && !lsame(scun[0], 'U')) {
        *info = -1;
    } else if (itype == -1) {
        *info = -2;
    } else if (m < 0) {
        *info = -3;
    } else if (n < 0 || ((itype == 4 || itype == 5) && n != m)) {
        *info = -4;
    } else if (anrm < zero) {
        *info = -7;
    } else if (nbl < 0) {
        *info = -8;
    } else if (nbl > 0 && isum != mn) {
        *info = -9;
    } else if (itype <= 3 && lda < imax(1, m)) {
        *info = -11;
    } else if (itype >= 4) {
        if (kl < 0 || kl > imax(m - 1, 0)) {
            *info = -5;
        } else if (ku < 0 || ku > imax(n - 1, 0) ||
                   ((itype == 4 || itype == 5) && kl != ku)) {
            *info = -6;
        } else if ((itype == 4 && lda < kl + 1) ||
                   (itype == 5 && lda < ku + 1) ||
                   (itype == 6 && lda < 2 * kl + ku + 1)) {
            *info = -11;
        }
    }

    if (*info != 0) {
        return;
    }

    if (mn == 0 || anrm == zero) {
        return;
    }

    smlnum = SLC_DLAMCH("S") / SLC_DLAMCH("P");
    bignum = one / smlnum;
    SLC_DLABAD(&smlnum, &bignum);

    if (lscale) {
        if (anrm < smlnum) {
            mb01qd(type[0], m, n, kl, ku, anrm, smlnum, nbl, nrows, a, lda, info);
        } else if (anrm > bignum) {
            mb01qd(type[0], m, n, kl, ku, anrm, bignum, nbl, nrows, a, lda, info);
        }
    } else {
        if (anrm < smlnum) {
            mb01qd(type[0], m, n, kl, ku, smlnum, anrm, nbl, nrows, a, lda, info);
        } else if (anrm > bignum) {
            mb01qd(type[0], m, n, kl, ku, bignum, anrm, nbl, nrows, a, lda, info);
        }
    }
}
