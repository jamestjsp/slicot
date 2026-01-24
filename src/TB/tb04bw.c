/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 2002-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
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

void tb04bw(const char* order, i32 p, i32 m, i32 md,
            i32* ign, i32 ldign, const i32* igd, i32 ldigd,
            f64* gn, const f64* gd, const f64* d, i32 ldd, i32* info)
{
    const f64 ZERO = 0.0;
    const i32 int1 = 1;

    bool ascend;
    i32 i, ii, j, k, kk, km, nd, nn;
    f64 dij;

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

    k = 0;

    if (ascend) {
        for (j = 0; j < m; j++) {
            for (i = 0; i < p; i++) {
                dij = d[i + j * ldd];
                if (dij != ZERO) {
                    nn = ign[i + j * ldign];
                    nd = igd[i + j * ldigd];
                    if (nn == 0 && nd == 0) {
                        if (gn[k] == ZERO) {
                            gn[k] = dij;
                        } else {
                            gn[k] = gn[k] + dij * gd[k];
                        }
                    } else {
                        km = imin(nn, nd) + 1;
                        SLC_DAXPY(&km, &dij, &gd[k], &int1, &gn[k], &int1);
                        if (nn < nd) {
                            for (ii = k + km; ii <= k + nd; ii++) {
                                gn[ii] = dij * gd[ii];
                            }
                            ign[i + j * ldign] = nd;
                        }
                    }
                }
                k = k + md;
            }
        }
    } else {
        for (j = 0; j < m; j++) {
            for (i = 0; i < p; i++) {
                dij = d[i + j * ldd];
                if (dij != ZERO) {
                    nn = ign[i + j * ldign];
                    nd = igd[i + j * ldigd];
                    if (nn == 0 && nd == 0) {
                        if (gn[k] == ZERO) {
                            gn[k] = dij;
                        } else {
                            gn[k] = gn[k] + dij * gd[k];
                        }
                    } else {
                        km = imin(nn, nd) + 1;
                        if (nn < nd) {
                            kk = k + nd - nn;
                            for (ii = k + nn; ii >= k; ii--) {
                                gn[ii + nd - nn] = gn[ii];
                            }
                            for (ii = k; ii < kk; ii++) {
                                gn[ii] = dij * gd[ii];
                            }
                            ign[i + j * ldign] = nd;
                            SLC_DAXPY(&km, &dij, &gd[kk], &int1, &gn[kk], &int1);
                        } else {
                            kk = k + nn - nd;
                            SLC_DAXPY(&km, &dij, &gd[k], &int1, &gn[kk], &int1);
                        }
                    }
                }
                k = k + md;
            }
        }
    }
}
