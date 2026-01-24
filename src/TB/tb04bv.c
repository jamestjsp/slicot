/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 2002-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <stdbool.h>

static bool lsame(char ca, char cb) {
    if (ca >= 'a' && ca <= 'z') ca -= 32;
    if (cb >= 'a' && cb <= 'z') cb -= 32;
    return ca == cb;
}

static i32 imax(i32 a, i32 b) {
    return a > b ? a : b;
}

static i32 imin(i32 a, i32 b) {
    return a < b ? a : b;
}

void tb04bv(const char* order, i32 p, i32 m, i32 md,
            i32* ign, i32 ldign, const i32* igd, i32 ldigd,
            f64* gn, const f64* gd, f64* d, i32 ldd, f64 tol, i32* info)
{
    const f64 ZERO = 0.0;
    const i32 int1 = 1;

    bool ascend;
    i32 i, ii, j, k, kk, km, nd, nn;
    f64 dij, eps, toldef;

    *info = 0;
    ascend = lsame(*order, 'I');

    if (!ascend && !lsame(*order, 'D')) {
        *info = -1;
    } else if (p < 0) {
        *info = -2;
    } else if (m < 0) {
        *info = -3;
    } else if (md < 1) {
        *info = -4;
    } else if (ldign < imax(1, p)) {
        *info = -6;
    } else if (ldigd < imax(1, p)) {
        *info = -8;
    } else if (ldd < imax(1, p)) {
        *info = -12;
    }

    if (*info != 0) {
        return;
    }

    if (imin(p, m) == 0) {
        return;
    }

    toldef = tol;
    if (toldef <= ZERO) {
        eps = SLC_DLAMCH("Epsilon");
    }

    k = 0;

    if (ascend) {
        for (j = 0; j < m; j++) {
            for (i = 0; i < p; i++) {
                nn = ign[i + j * ldign];
                nd = igd[i + j * ldigd];

                if (nn > nd) {
                    *info = 1;
                    return;
                } else if (nn < nd || (nd == 0 && gn[k] == ZERO)) {
                    d[i + j * ldd] = ZERO;
                } else {
                    kk = k + nn;

                    if (gd[kk] == ZERO) {
                        *info = 2;
                        return;
                    }

                    dij = gn[kk] / gd[kk];
                    d[i + j * ldd] = dij;
                    gn[kk] = ZERO;

                    if (nn > 0) {
                        f64 neg_dij = -dij;
                        SLC_DAXPY(&nn, &neg_dij, &gd[k], &int1, &gn[k], &int1);

                        if (tol <= ZERO) {
                            i32 idx = SLC_IDAMAX(&nn, &gn[k], &int1);
                            toldef = (f64)nn * eps * fabs(gn[k + idx - 1]);
                        }

                        km = nn;
                        for (ii = 0; ii < km; ii++) {
                            kk = kk - 1;
                            nn = nn - 1;
                            if (fabs(gn[kk]) > toldef) {
                                break;
                            }
                        }

                        ign[i + j * ldign] = nn;
                    }
                }
                k = k + md;
            }
        }
    } else {
        for (j = 0; j < m; j++) {
            for (i = 0; i < p; i++) {
                nn = ign[i + j * ldign];
                nd = igd[i + j * ldigd];

                if (nn > nd) {
                    *info = 1;
                    return;
                } else if (nn < nd || (nd == 0 && gn[k] == ZERO)) {
                    d[i + j * ldd] = ZERO;
                } else {
                    kk = k;

                    if (gd[kk] == ZERO) {
                        *info = 2;
                        return;
                    }

                    dij = gn[kk] / gd[kk];
                    d[i + j * ldd] = dij;
                    gn[kk] = ZERO;

                    if (nn > 0) {
                        f64 neg_dij = -dij;
                        SLC_DAXPY(&nn, &neg_dij, &gd[k + 1], &int1, &gn[k + 1], &int1);

                        if (tol <= ZERO) {
                            i32 idx = SLC_IDAMAX(&nn, &gn[k + 1], &int1);
                            toldef = (f64)nn * eps * fabs(gn[k + 1 + idx - 1]);
                        }

                        km = nn;
                        for (ii = 0; ii < km; ii++) {
                            kk = kk + 1;
                            nn = nn - 1;
                            if (fabs(gn[kk]) > toldef) {
                                break;
                            }
                        }

                        ign[i + j * ldign] = nn;

                        for (ii = 0; ii <= nn; ii++) {
                            gn[k + ii] = gn[kk + ii];
                        }
                    }
                }
                k = k + md;
            }
        }
    }
}
