/*
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * IB01AD - System identification driver
 *
 * Preprocesses input-output data for estimating state-space matrices
 * and finds an estimate of the system order using MOESP or N4SID method.
 * This driver calls IB01MD (R factor), IB01ND (SVD), IB01OD (order estimation).
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

void ib01ad(const char *meth_str, const char *alg_str, const char *jobd_str,
            const char *batch_str, const char *conct_str, const char *ctrl_str,
            i32 nobr, i32 m, i32 l, i32 nsmp,
            const f64 *u, i32 ldu, const f64 *y, i32 ldy,
            i32 *n, f64 *r, i32 ldr, f64 *sv, f64 rcond, f64 tol,
            i32 *iwork, f64 *dwork, i32 ldwork,
            i32 *iwarn, i32 *info)
{
    char meth = meth_str[0];
    char alg = alg_str[0];
    char jobd = jobd_str[0];
    char batch = batch_str[0];
    char conct = conct_str[0];
    char ctrl = ctrl_str[0];

    bool moesp = (meth == 'M' || meth == 'm');
    bool n4sid = (meth == 'N' || meth == 'n');
    bool fqralg = (alg == 'F' || alg == 'f');
    bool qralg = (alg == 'Q' || alg == 'q');
    bool chalg = (alg == 'C' || alg == 'c');
    bool jobdm = (jobd == 'M' || jobd == 'm');
    bool onebch = (batch == 'O' || batch == 'o');
    bool first = (batch == 'F' || batch == 'f') || onebch;
    bool interm = (batch == 'I' || batch == 'i');
    bool last = (batch == 'L' || batch == 'l') || onebch;
    bool contrl = (ctrl == 'C' || ctrl == 'c');

    bool connec = false;
    if (!onebch) {
        connec = (conct == 'C' || conct == 'c');
    }

    i32 mnobr = m * nobr;
    i32 lnobr = l * nobr;
    i32 lmnobr = lnobr + mnobr;
    i32 nr = lmnobr + lmnobr;
    i32 nobr21 = 2 * nobr - 1;

    *iwarn = 0;
    *info = 0;

    i32 icycle, maxwrk, nsmpsm;
    if (first) {
        icycle = 1;
        maxwrk = 1;
        nsmpsm = 0;
    } else if (!onebch) {
        icycle = iwork[0];
        maxwrk = iwork[1];
        nsmpsm = iwork[2];
    } else {
        icycle = 1;
        maxwrk = 1;
        nsmpsm = 0;
    }
    nsmpsm = nsmpsm + nsmp;

    if (!(moesp || n4sid)) {
        *info = -1;
    } else if (!(fqralg || qralg || chalg)) {
        *info = -2;
    } else if (moesp && !(jobdm || jobd == 'N' || jobd == 'n')) {
        *info = -3;
    } else if (!(first || interm || last)) {
        *info = -4;
    } else if (!onebch) {
        bool conct_n = (conct == 'N' || conct == 'n');
        if (!(connec || conct_n)) {
            *info = -5;
        }
    }

    if (*info == 0) {
        if (!(contrl || ctrl == 'N' || ctrl == 'n')) {
            *info = -6;
        } else if (nobr <= 0) {
            *info = -7;
        } else if (m < 0) {
            *info = -8;
        } else if (l <= 0) {
            *info = -9;
        } else if (nsmp < 2 * nobr || (last && nsmpsm < nr + nobr21)) {
            *info = -10;
        } else if (ldu < 1 || (m > 0 && ldu < nsmp)) {
            *info = -12;
        } else if (ldy < nsmp) {
            *info = -14;
        } else if (ldr < nr || (moesp && jobdm && ldr < 3 * mnobr)) {
            *info = -17;
        } else {
            i32 ns = nsmp - nobr21;
            i32 minwrk = 1;

            if (chalg) {
                if (!last) {
                    if (connec) {
                        minwrk = 2 * (nr - m - l);
                    } else {
                        minwrk = 1;
                    }
                } else if (moesp) {
                    if (connec && !onebch) {
                        i32 t1 = 2 * (nr - m - l);
                        i32 t2 = 5 * lnobr;
                        minwrk = (t1 > t2) ? t1 : t2;
                    } else {
                        minwrk = 5 * lnobr;
                        if (jobdm) {
                            i32 t1 = 2 * mnobr - nobr;
                            if (t1 < 0) t1 = 1;
                            i32 t2 = lmnobr;
                            i32 t3 = 5 * lnobr;
                            minwrk = t1;
                            minwrk = (t2 > minwrk) ? t2 : minwrk;
                            minwrk = (t3 > minwrk) ? t3 : minwrk;
                        }
                    }
                } else {
                    minwrk = 5 * lmnobr + 1;
                }
            } else if (fqralg) {
                if (!onebch && connec) {
                    minwrk = nr * (m + l + 3);
                } else if ((first || interm) && !onebch) {
                    minwrk = nr * (m + l + 1);
                } else {
                    minwrk = 2 * nr * (m + l + 1) + nr;
                }
            } else {
                minwrk = 2 * nr;
                if (onebch && ldr >= ns) {
                    if (moesp) {
                        i32 t = 5 * lnobr;
                        minwrk = (minwrk > t) ? minwrk : t;
                    } else {
                        minwrk = 5 * lmnobr + 1;
                    }
                }
                if (first) {
                    if (ldr < ns) {
                        minwrk = minwrk + nr;
                    }
                } else {
                    if (connec) {
                        minwrk = minwrk * (nobr + 1);
                    } else {
                        minwrk = minwrk + nr;
                    }
                }
            }

            maxwrk = minwrk;

            if (ldwork < minwrk) {
                *info = -23;
                dwork[0] = (f64)minwrk;
            }
        }
    }

    if (*info != 0) {
        if (!onebch) {
            iwork[0] = 1;
            iwork[1] = maxwrk;
            iwork[2] = 0;
        }
        return;
    }

    i32 iwarnl = 0;

    ib01md(meth_str, alg_str, batch_str, conct_str, nobr, m, l, nsmp,
           u, ldu, y, ldy, r, ldr, iwork, dwork, ldwork, &iwarnl, info);

    if (*info == 1) {
        return;
    }

    maxwrk = (maxwrk > (i32)dwork[0]) ? maxwrk : (i32)dwork[0];

    if (!last) {
        icycle = icycle + 1;
        iwork[0] = icycle;
        iwork[1] = maxwrk;
        iwork[2] = nsmpsm;
        return;
    }

    SLC_IB01ND(meth, jobd, nobr, m, l, r, ldr, sv, rcond,
               iwork, dwork, ldwork, &iwarnl, info);

    *iwarn = (*iwarn > iwarnl) ? *iwarn : iwarnl;

    if (*info == 2) {
        return;
    }

    SLC_IB01OD(ctrl, nobr, l, sv, n, tol, &iwarnl, info);
    *iwarn = (*iwarn > iwarnl) ? *iwarn : iwarnl;

    dwork[0] = (f64)((maxwrk > (i32)dwork[0]) ? maxwrk : (i32)dwork[0]);

    if (!onebch) {
        iwork[0] = icycle;
        iwork[1] = maxwrk;
        iwork[2] = nsmpsm;
    }
}
