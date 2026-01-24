/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>

void mb04yw(const bool qrit, const bool updatu, const bool updatv,
            const i32 m, const i32 n, const i32 l, const i32 k,
            const f64 shift, f64 *d, f64 *e, f64 *u, const i32 ldu,
            f64 *v, const i32 ldv, f64 *dwork)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    i32 i, irot, ncv, nm1, nm12, nm13;
    f64 cosl, cosr, cs, f, g, h, oldcs, oldsn, r, sinl, sinr, sn;

    ncv = (m < n) ? m : n;
    if (ncv <= 1 || l == k) {
        return;
    }

    nm1 = ncv - 1;
    nm12 = nm1 + nm1;
    nm13 = nm12 + nm1;
    if (!updatv) {
        nm12 = 0;
        nm13 = nm1;
    }

    if (shift == ZERO) {
        if (qrit) {
            cs = ONE;
            oldcs = ONE;
            f64 temp1 = d[l - 1] * cs;
            f64 temp2 = e[l - 1];
            SLC_DLARTG(&temp1, &temp2, &cs, &sn, &r);
            f64 temp3 = oldcs * r;
            f64 temp4 = d[l] * sn;
            SLC_DLARTG(&temp3, &temp4, &oldcs, &oldsn, &d[l - 1]);

            if (updatv) {
                dwork[0] = cs;
                dwork[nm1] = sn;
            }
            if (updatu) {
                dwork[nm12] = oldcs;
                dwork[nm13] = oldsn;
            }
            irot = 1;

            for (i = l; i <= k - 2; i++) {
                f64 di = d[i] * cs;
                f64 ei = e[i];
                SLC_DLARTG(&di, &ei, &cs, &sn, &r);
                e[i - 1] = oldsn * r;
                f64 temp = oldcs * r;
                f64 di1_sn = d[i + 1] * sn;
                SLC_DLARTG(&temp, &di1_sn, &oldcs, &oldsn, &d[i]);
                irot = irot + 1;
                if (updatv) {
                    dwork[irot - 1] = cs;
                    dwork[irot - 1 + nm1] = sn;
                }
                if (updatu) {
                    dwork[irot - 1 + nm12] = oldcs;
                    dwork[irot - 1 + nm13] = oldsn;
                }
            }

            h = d[k - 1] * cs;
            d[k - 1] = h * oldcs;
            e[k - 2] = h * oldsn;

            if (updatv) {
                i32 ncols = k - l + 1;
                SLC_DLASR("R", "V", "F", &n, &ncols, &dwork[0], &dwork[ncv - 1],
                          &v[(l - 1) * ldv], &ldv);
            }
            if (updatu) {
                i32 ncols = k - l + 1;
                SLC_DLASR("R", "V", "F", &m, &ncols, &dwork[nm12], &dwork[nm13],
                          &u[(l - 1) * ldu], &ldu);
            }
        } else {
            cs = ONE;
            oldcs = ONE;
            f64 temp1 = d[k - 1] * cs;
            f64 temp2 = e[k - 2];
            SLC_DLARTG(&temp1, &temp2, &cs, &sn, &r);
            f64 temp3 = oldcs * r;
            f64 temp4 = d[k - 2] * sn;
            SLC_DLARTG(&temp3, &temp4, &oldcs, &oldsn, &d[k - 1]);

            if (updatv) {
                dwork[k - l - 1] = oldcs;
                dwork[k - l - 1 + nm1] = -oldsn;
            }
            if (updatu) {
                dwork[k - l - 1 + nm12] = cs;
                dwork[k - l - 1 + nm13] = -sn;
            }
            irot = k - l;

            for (i = k - 2; i >= l; i--) {
                f64 di = d[i] * cs;
                f64 ei1 = e[i - 1];
                SLC_DLARTG(&di, &ei1, &cs, &sn, &r);
                e[i] = oldsn * r;
                f64 temp2 = oldcs * r;
                f64 di1_sn = d[i - 1] * sn;
                SLC_DLARTG(&temp2, &di1_sn, &oldcs, &oldsn, &d[i]);
                irot = irot - 1;
                if (updatv) {
                    dwork[irot - 1] = oldcs;
                    dwork[irot - 1 + nm1] = -oldsn;
                }
                if (updatu) {
                    dwork[irot - 1 + nm12] = cs;
                    dwork[irot - 1 + nm13] = -sn;
                }
            }

            h = d[l - 1] * cs;
            d[l - 1] = h * oldcs;
            e[l - 1] = h * oldsn;

            if (updatv) {
                i32 ncols = k - l + 1;
                SLC_DLASR("R", "V", "B", &n, &ncols, &dwork[0], &dwork[ncv - 1],
                          &v[(l - 1) * ldv], &ldv);
            }
            if (updatu) {
                i32 ncols = k - l + 1;
                SLC_DLASR("R", "V", "B", &m, &ncols, &dwork[nm12], &dwork[nm13],
                          &u[(l - 1) * ldu], &ldu);
            }
        }
    } else {
        if (qrit) {
            f64 dl = d[l - 1];
            f64 abs_dl = fabs(dl);
            f64 sign_dl = (dl >= 0.0) ? ONE : -ONE;
            f = (abs_dl - shift) * (sign_dl + shift / dl);
            g = e[l - 1];
            SLC_DLARTG(&f, &g, &cosr, &sinr, &r);
            f = cosr * d[l - 1] + sinr * e[l - 1];
            e[l - 1] = cosr * e[l - 1] - sinr * d[l - 1];
            g = sinr * d[l];
            d[l] = cosr * d[l];
            SLC_DLARTG(&f, &g, &cosl, &sinl, &r);
            d[l - 1] = r;
            f = cosl * e[l - 1] + sinl * d[l];
            d[l] = cosl * d[l] - sinl * e[l - 1];
            g = sinl * e[l];
            e[l] = cosl * e[l];

            if (updatv) {
                dwork[0] = cosr;
                dwork[nm1] = sinr;
            }
            if (updatu) {
                dwork[nm12] = cosl;
                dwork[nm13] = sinl;
            }
            irot = 1;

            for (i = l; i <= k - 3; i++) {
                SLC_DLARTG(&f, &g, &cosr, &sinr, &r);
                e[i - 1] = r;
                f = cosr * d[i] + sinr * e[i];
                e[i] = cosr * e[i] - sinr * d[i];
                g = sinr * d[i + 1];
                d[i + 1] = cosr * d[i + 1];
                SLC_DLARTG(&f, &g, &cosl, &sinl, &r);
                d[i] = r;
                f = cosl * e[i] + sinl * d[i + 1];
                d[i + 1] = cosl * d[i + 1] - sinl * e[i];
                g = sinl * e[i + 1];
                e[i + 1] = cosl * e[i + 1];
                irot = irot + 1;
                if (updatv) {
                    dwork[irot - 1] = cosr;
                    dwork[irot - 1 + nm1] = sinr;
                }
                if (updatu) {
                    dwork[irot - 1 + nm12] = cosl;
                    dwork[irot - 1 + nm13] = sinl;
                }
            }

            if (l < k - 1) {
                SLC_DLARTG(&f, &g, &cosr, &sinr, &r);
                e[k - 3] = r;
                f = cosr * d[k - 2] + sinr * e[k - 2];
                e[k - 2] = cosr * e[k - 2] - sinr * d[k - 2];
                g = sinr * d[k - 1];
                d[k - 1] = cosr * d[k - 1];
                SLC_DLARTG(&f, &g, &cosl, &sinl, &r);
                d[k - 2] = r;
                f = cosl * e[k - 2] + sinl * d[k - 1];
                d[k - 1] = cosl * d[k - 1] - sinl * e[k - 2];
                irot = irot + 1;
                if (updatv) {
                    dwork[irot - 1] = cosr;
                    dwork[irot - 1 + nm1] = sinr;
                }
                if (updatu) {
                    dwork[irot - 1 + nm12] = cosl;
                    dwork[irot - 1 + nm13] = sinl;
                }
            }
            e[k - 2] = f;

            if (updatv) {
                i32 ncols = k - l + 1;
                SLC_DLASR("R", "V", "F", &n, &ncols, &dwork[0], &dwork[ncv - 1],
                          &v[(l - 1) * ldv], &ldv);
            }
            if (updatu) {
                i32 ncols = k - l + 1;
                SLC_DLASR("R", "V", "F", &m, &ncols, &dwork[nm12], &dwork[nm13],
                          &u[(l - 1) * ldu], &ldu);
            }
        } else {
            f64 dk = d[k - 1];
            f64 abs_dk = fabs(dk);
            f64 sign_dk = (dk >= 0.0) ? ONE : -ONE;
            f = (abs_dk - shift) * (sign_dk + shift / dk);
            g = e[k - 2];

            if (l < k - 1) {
                SLC_DLARTG(&f, &g, &cosr, &sinr, &r);
                f = cosr * d[k - 1] + sinr * e[k - 2];
                e[k - 2] = cosr * e[k - 2] - sinr * d[k - 1];
                g = sinr * d[k - 2];
                d[k - 2] = cosr * d[k - 2];
                SLC_DLARTG(&f, &g, &cosl, &sinl, &r);
                d[k - 1] = r;
                f = cosl * e[k - 2] + sinl * d[k - 2];
                d[k - 2] = cosl * d[k - 2] - sinl * e[k - 2];
                g = sinl * e[k - 3];
                e[k - 3] = cosl * e[k - 3];
                if (updatv) {
                    dwork[k - l - 1] = cosl;
                    dwork[k - l - 1 + nm1] = -sinl;
                }
                if (updatu) {
                    dwork[k - l - 1 + nm12] = cosr;
                    dwork[k - l - 1 + nm13] = -sinr;
                }
                irot = k - l;
            } else {
                irot = k - l + 1;
            }

            for (i = k - 2; i >= l + 1; i--) {
                SLC_DLARTG(&f, &g, &cosr, &sinr, &r);
                e[i] = r;
                f = cosr * d[i] + sinr * e[i - 1];
                e[i - 1] = cosr * e[i - 1] - sinr * d[i];
                g = sinr * d[i - 1];
                d[i - 1] = cosr * d[i - 1];
                SLC_DLARTG(&f, &g, &cosl, &sinl, &r);
                d[i] = r;
                f = cosl * e[i - 1] + sinl * d[i - 1];
                d[i - 1] = cosl * d[i - 1] - sinl * e[i - 1];
                g = sinl * e[i - 2];
                e[i - 2] = cosl * e[i - 2];
                irot = irot - 1;
                if (updatv) {
                    dwork[irot - 1] = cosl;
                    dwork[irot - 1 + nm1] = -sinl;
                }
                if (updatu) {
                    dwork[irot - 1 + nm12] = cosr;
                    dwork[irot - 1 + nm13] = -sinr;
                }
            }

            SLC_DLARTG(&f, &g, &cosr, &sinr, &r);
            e[l] = r;
            f = cosr * d[l] + sinr * e[l - 1];
            e[l - 1] = cosr * e[l - 1] - sinr * d[l];
            g = sinr * d[l - 1];
            d[l - 1] = cosr * d[l - 1];
            SLC_DLARTG(&f, &g, &cosl, &sinl, &r);
            d[l] = r;
            f = cosl * e[l - 1] + sinl * d[l - 1];
            d[l - 1] = cosl * d[l - 1] - sinl * e[l - 1];
            irot = irot - 1;
            if (updatv) {
                dwork[irot - 1] = cosl;
                dwork[irot - 1 + nm1] = -sinl;
            }
            if (updatu) {
                dwork[irot - 1 + nm12] = cosr;
                dwork[irot - 1 + nm13] = -sinr;
            }
            e[l - 1] = f;

            if (updatv) {
                i32 ncols = k - l + 1;
                SLC_DLASR("R", "V", "B", &n, &ncols, &dwork[0], &dwork[ncv - 1],
                          &v[(l - 1) * ldv], &ldv);
            }
            if (updatu) {
                i32 ncols = k - l + 1;
                SLC_DLASR("R", "V", "B", &m, &ncols, &dwork[nm12], &dwork[nm13],
                          &u[(l - 1) * ldu], &ldu);
            }
        }
    }
}
