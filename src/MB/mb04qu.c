/*
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <stdlib.h>

void mb04qu(
    const char *tranc, const char *trand, const char *tranq,
    const char *storev, const char *storew,
    i32 m, i32 n, i32 k,
    f64 *v, i32 ldv, f64 *w, i32 ldw,
    f64 *c, i32 ldc, f64 *d, i32 ldd,
    const f64 *cs, const f64 *tau,
    f64 *dwork, i32 ldwork, i32 *info)
{
    i32 i;
    i32 int1 = 1;
    f64 dbl1 = 1.0;
    f64 nu;
    bool lcolv, lcolw, ltrc, ltrd, ltrq;
    i32 mi1;

    *info = 0;

    lcolv = (storev[0] == 'C' || storev[0] == 'c');
    lcolw = (storew[0] == 'C' || storew[0] == 'c');
    ltrc = (tranc[0] == 'T' || tranc[0] == 't' || tranc[0] == 'C' || tranc[0] == 'c');
    ltrd = (trand[0] == 'T' || trand[0] == 't' || trand[0] == 'C' || trand[0] == 'c');
    ltrq = (tranq[0] == 'T' || tranq[0] == 't');

    if (!ltrc && !(tranc[0] == 'N' || tranc[0] == 'n')) {
        *info = -1;
    } else if (!ltrd && !(trand[0] == 'N' || trand[0] == 'n')) {
        *info = -2;
    } else if (!ltrq && !(tranq[0] == 'N' || tranq[0] == 'n')) {
        *info = -3;
    } else if (!lcolv && !(storev[0] == 'R' || storev[0] == 'r')) {
        *info = -4;
    } else if (!lcolw && !(storew[0] == 'R' || storew[0] == 'r')) {
        *info = -5;
    } else if (m < 0) {
        *info = -6;
    } else if (n < 0) {
        *info = -7;
    } else if (k < 0 || k > m) {
        *info = -8;
    } else if ((lcolv && ldv < (m > 1 ? m : 1)) ||
               (!lcolv && ldv < (k > 1 ? k : 1))) {
        *info = -10;
    } else if ((lcolw && ldw < (m > 1 ? m : 1)) ||
               (!lcolw && ldw < (k > 1 ? k : 1))) {
        *info = -12;
    } else if ((ltrc && ldc < (n > 1 ? n : 1)) ||
               (!ltrc && ldc < (m > 1 ? m : 1))) {
        *info = -14;
    } else if ((ltrd && ldd < (n > 1 ? n : 1)) ||
               (!ltrd && ldd < (m > 1 ? m : 1))) {
        *info = -16;
    } else if (ldwork < (n > 1 ? n : 1)) {
        dwork[0] = (f64)(n > 1 ? n : 1);
        *info = -20;
    }

    if (*info != 0) {
        return;
    }

    if (k == 0 || m == 0 || n == 0) {
        dwork[0] = dbl1;
        return;
    }

    if (ltrq) {
        for (i = 0; i < k; i++) {
            mi1 = m - i;

            nu = w[i + i * ldw];
            w[i + i * ldw] = dbl1;
            if (lcolw) {
                if (ltrc) {
                    SLC_DLARF("R", &n, &mi1, &w[i + i * ldw], &int1, &nu, &c[0 + i * ldc], &ldc, dwork);
                } else {
                    SLC_DLARF("L", &mi1, &n, &w[i + i * ldw], &int1, &nu, &c[i + 0 * ldc], &ldc, dwork);
                }
                if (ltrd) {
                    SLC_DLARF("R", &n, &mi1, &w[i + i * ldw], &int1, &nu, &d[0 + i * ldd], &ldd, dwork);
                } else {
                    SLC_DLARF("L", &mi1, &n, &w[i + i * ldw], &int1, &nu, &d[i + 0 * ldd], &ldd, dwork);
                }
            } else {
                if (ltrc) {
                    SLC_DLARF("R", &n, &mi1, &w[i + i * ldw], &ldw, &nu, &c[0 + i * ldc], &ldc, dwork);
                } else {
                    SLC_DLARF("L", &mi1, &n, &w[i + i * ldw], &ldw, &nu, &c[i + 0 * ldc], &ldc, dwork);
                }
                if (ltrd) {
                    SLC_DLARF("R", &n, &mi1, &w[i + i * ldw], &ldw, &nu, &d[0 + i * ldd], &ldd, dwork);
                } else {
                    SLC_DLARF("L", &mi1, &n, &w[i + i * ldw], &ldw, &nu, &d[i + 0 * ldd], &ldd, dwork);
                }
            }
            w[i + i * ldw] = nu;

            if (ltrc && ltrd) {
                SLC_DROT(&n, &c[0 + i * ldc], &int1, &d[0 + i * ldd], &int1, &cs[2*i], &cs[2*i + 1]);
            } else if (ltrc) {
                SLC_DROT(&n, &c[0 + i * ldc], &int1, &d[i + 0 * ldd], &ldd, &cs[2*i], &cs[2*i + 1]);
            } else if (ltrd) {
                SLC_DROT(&n, &c[i + 0 * ldc], &ldc, &d[0 + i * ldd], &int1, &cs[2*i], &cs[2*i + 1]);
            } else {
                SLC_DROT(&n, &c[i + 0 * ldc], &ldc, &d[i + 0 * ldd], &ldd, &cs[2*i], &cs[2*i + 1]);
            }

            nu = v[i + i * ldv];
            v[i + i * ldv] = dbl1;
            if (lcolv) {
                if (ltrc) {
                    SLC_DLARF("R", &n, &mi1, &v[i + i * ldv], &int1, &tau[i], &c[0 + i * ldc], &ldc, dwork);
                } else {
                    SLC_DLARF("L", &mi1, &n, &v[i + i * ldv], &int1, &tau[i], &c[i + 0 * ldc], &ldc, dwork);
                }
                if (ltrd) {
                    SLC_DLARF("R", &n, &mi1, &v[i + i * ldv], &int1, &tau[i], &d[0 + i * ldd], &ldd, dwork);
                } else {
                    SLC_DLARF("L", &mi1, &n, &v[i + i * ldv], &int1, &tau[i], &d[i + 0 * ldd], &ldd, dwork);
                }
            } else {
                if (ltrc) {
                    SLC_DLARF("R", &n, &mi1, &v[i + i * ldv], &ldv, &tau[i], &c[0 + i * ldc], &ldc, dwork);
                } else {
                    SLC_DLARF("L", &mi1, &n, &v[i + i * ldv], &ldv, &tau[i], &c[i + 0 * ldc], &ldc, dwork);
                }
                if (ltrd) {
                    SLC_DLARF("R", &n, &mi1, &v[i + i * ldv], &ldv, &tau[i], &d[0 + i * ldd], &ldd, dwork);
                } else {
                    SLC_DLARF("L", &mi1, &n, &v[i + i * ldv], &ldv, &tau[i], &d[i + 0 * ldd], &ldd, dwork);
                }
            }
            v[i + i * ldv] = nu;
        }
    } else {
        for (i = k - 1; i >= 0; i--) {
            mi1 = m - i;

            nu = v[i + i * ldv];
            v[i + i * ldv] = dbl1;
            if (lcolv) {
                if (ltrc) {
                    SLC_DLARF("R", &n, &mi1, &v[i + i * ldv], &int1, &tau[i], &c[0 + i * ldc], &ldc, dwork);
                } else {
                    SLC_DLARF("L", &mi1, &n, &v[i + i * ldv], &int1, &tau[i], &c[i + 0 * ldc], &ldc, dwork);
                }
                if (ltrd) {
                    SLC_DLARF("R", &n, &mi1, &v[i + i * ldv], &int1, &tau[i], &d[0 + i * ldd], &ldd, dwork);
                } else {
                    SLC_DLARF("L", &mi1, &n, &v[i + i * ldv], &int1, &tau[i], &d[i + 0 * ldd], &ldd, dwork);
                }
            } else {
                if (ltrc) {
                    SLC_DLARF("R", &n, &mi1, &v[i + i * ldv], &ldv, &tau[i], &c[0 + i * ldc], &ldc, dwork);
                } else {
                    SLC_DLARF("L", &mi1, &n, &v[i + i * ldv], &ldv, &tau[i], &c[i + 0 * ldc], &ldc, dwork);
                }
                if (ltrd) {
                    SLC_DLARF("R", &n, &mi1, &v[i + i * ldv], &ldv, &tau[i], &d[0 + i * ldd], &ldd, dwork);
                } else {
                    SLC_DLARF("L", &mi1, &n, &v[i + i * ldv], &ldv, &tau[i], &d[i + 0 * ldd], &ldd, dwork);
                }
            }
            v[i + i * ldv] = nu;

            f64 neg_s = -cs[2*i + 1];
            if (ltrc && ltrd) {
                SLC_DROT(&n, &c[0 + i * ldc], &int1, &d[0 + i * ldd], &int1, &cs[2*i], &neg_s);
            } else if (ltrc) {
                SLC_DROT(&n, &c[0 + i * ldc], &int1, &d[i + 0 * ldd], &ldd, &cs[2*i], &neg_s);
            } else if (ltrd) {
                SLC_DROT(&n, &c[i + 0 * ldc], &ldc, &d[0 + i * ldd], &int1, &cs[2*i], &neg_s);
            } else {
                SLC_DROT(&n, &c[i + 0 * ldc], &ldc, &d[i + 0 * ldd], &ldd, &cs[2*i], &neg_s);
            }

            nu = w[i + i * ldw];
            w[i + i * ldw] = dbl1;
            if (lcolw) {
                if (ltrc) {
                    SLC_DLARF("R", &n, &mi1, &w[i + i * ldw], &int1, &nu, &c[0 + i * ldc], &ldc, dwork);
                } else {
                    SLC_DLARF("L", &mi1, &n, &w[i + i * ldw], &int1, &nu, &c[i + 0 * ldc], &ldc, dwork);
                }
                if (ltrd) {
                    SLC_DLARF("R", &n, &mi1, &w[i + i * ldw], &int1, &nu, &d[0 + i * ldd], &ldd, dwork);
                } else {
                    SLC_DLARF("L", &mi1, &n, &w[i + i * ldw], &int1, &nu, &d[i + 0 * ldd], &ldd, dwork);
                }
            } else {
                if (ltrc) {
                    SLC_DLARF("R", &n, &mi1, &w[i + i * ldw], &ldw, &nu, &c[0 + i * ldc], &ldc, dwork);
                } else {
                    SLC_DLARF("L", &mi1, &n, &w[i + i * ldw], &ldw, &nu, &c[i + 0 * ldc], &ldc, dwork);
                }
                if (ltrd) {
                    SLC_DLARF("R", &n, &mi1, &w[i + i * ldw], &ldw, &nu, &d[0 + i * ldd], &ldd, dwork);
                } else {
                    SLC_DLARF("L", &mi1, &n, &w[i + i * ldw], &ldw, &nu, &d[i + 0 * ldd], &ldd, dwork);
                }
            }
            w[i + i * ldw] = nu;
        }
    }

    dwork[0] = (f64)(n > 1 ? n : 1);
}
