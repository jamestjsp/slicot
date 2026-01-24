// SPDX-License-Identifier: BSD-3-Clause

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <string.h>
#include <ctype.h>

void mb04qb(const char *tranc, const char *trand, const char *tranq,
            const char *storev, const char *storew, i32 m, i32 n, i32 k,
            const f64 *v, i32 ldv, const f64 *w, i32 ldw,
            f64 *c, i32 ldc, f64 *d, i32 ldd,
            const f64 *cs, const f64 *tau,
            f64 *dwork, i32 ldwork, i32 *info) {

    const f64 ONE = 1.0;

    bool lcolv = (storev[0] == 'C' || storev[0] == 'c');
    bool lcolw = (storew[0] == 'C' || storew[0] == 'c');
    bool ltrc = (tranc[0] == 'T' || tranc[0] == 't' ||
                 tranc[0] == 'C' || tranc[0] == 'c');
    bool ltrd = (trand[0] == 'T' || trand[0] == 't' ||
                 trand[0] == 'C' || trand[0] == 'c');
    bool ltrq = (tranq[0] == 'T' || tranq[0] == 't');

    *info = 0;

    if (!(ltrc || tranc[0] == 'N' || tranc[0] == 'n')) {
        *info = -1;
    } else if (!(ltrd || trand[0] == 'N' || trand[0] == 'n')) {
        *info = -2;
    } else if (!(ltrq || tranq[0] == 'N' || tranq[0] == 'n')) {
        *info = -3;
    } else if (!(lcolv || storev[0] == 'R' || storev[0] == 'r')) {
        *info = -4;
    } else if (!(lcolw || storew[0] == 'R' || storew[0] == 'r')) {
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
    }

    if (*info != 0) {
        return;
    }

    bool lquery = (ldwork == -1);
    i32 minwrk = (n > 1 ? n : 1);
    i32 wrkopt = 1;
    i32 nb = 1;

    if (n == 0) {
        wrkopt = 1;
    } else {
        i32 ic_dim = ltrc ? n : m;
        i32 jc_dim = ltrc ? m : n;
        const char *side = ltrc ? "R" : "L";
        i32 lwork_query = -1;
        i32 lda_query = (m > n ? m : n);
        if (lda_query < 1) lda_query = 1;

        f64 work_query;
        SLC_DORMQR(side, tranc, &ic_dim, &jc_dim, &k, dwork, &lda_query,
                   dwork, dwork, &lda_query, &work_query, &lwork_query, info);
        wrkopt = (i32)work_query;
        if (wrkopt < minwrk) wrkopt = minwrk;
        nb = wrkopt / n;
        if (nb > n) nb = n;
        i32 blocked_work = 9 * n * nb + 15 * nb * nb;
        if (blocked_work > wrkopt) wrkopt = blocked_work;
    }

    if (ldwork < minwrk && !lquery) {
        dwork[0] = (f64)minwrk;
        *info = -20;
        return;
    }

    if (lquery) {
        dwork[0] = (f64)wrkopt;
        return;
    }

    if (k == 0 || m == 0 || n == 0) {
        dwork[0] = ONE;
        return;
    }

    i32 nbmin = 2;
    i32 nx = 0;

    if (nb > 1 && nb < k) {
        nx = 0;
        if (nx < k) {
            if (ldwork < wrkopt) {
                f64 disc = 81.0 * n * n + 60.0 * ldwork;
                nb = (i32)((sqrt(disc) - 9.0 * n) / 30.0);
                if (nb < 2) nb = 2;
            }
        }
    }

    i32 pdrs = 0;
    i32 pdt = pdrs + 6 * nb * nb;
    i32 pdw = pdt + 9 * nb * nb;

    i32 ic = 0, jc = 0, id = 0, jd = 0;

    if (ltrq) {
        if (nb >= nbmin && nb < k && nx < k) {
            for (i32 i = 0; i < k - nx; i += nb) {
                i32 ib = (k - i < nb) ? (k - i) : nb;

                mb04qf("F", storev, storew, m - i, ib,
                       (f64*)&v[lcolv ? (i + i * ldv) : (i + i * ldv)], ldv,
                       (f64*)&w[lcolw ? (i + i * ldw) : (i + i * ldw)], ldw,
                       &cs[2 * i], &tau[i],
                       &dwork[pdrs], nb, &dwork[pdt], nb, &dwork[pdw]);

                if (ltrc) {
                    jc = i;
                } else {
                    ic = i;
                }
                if (ltrd) {
                    jd = i;
                } else {
                    id = i;
                }

                mb04qc("N", tranc, trand, tranq, "F", storev, storew,
                       m - i, n, ib,
                       &v[lcolv ? (i + i * ldv) : (i + i * ldv)], ldv,
                       &w[lcolw ? (i + i * ldw) : (i + i * ldw)], ldw,
                       &dwork[pdrs], nb, &dwork[pdt], nb,
                       &c[ic + jc * ldc], ldc,
                       &d[id + jd * ldd], ldd, &dwork[pdw]);
            }
        }

        i32 i = (nb >= nbmin && nb < k && nx < k) ? ((k - nx) / nb) * nb : 0;
        if (i < k) {
            if (ltrc) {
                jc = i;
            } else {
                ic = i;
            }
            if (ltrd) {
                jd = i;
            } else {
                id = i;
            }

            i32 ierr;
            mb04qu(tranc, trand, tranq, storev, storew,
                   m - i, n, k - i,
                   (f64*)&v[lcolv ? (i + i * ldv) : (i + i * ldv)], ldv,
                   (f64*)&w[lcolw ? (i + i * ldw) : (i + i * ldw)], ldw,
                   &c[ic + jc * ldc], ldc,
                   &d[id + jd * ldd], ldd,
                   &cs[2 * i], &tau[i], dwork, ldwork, &ierr);
        }
    } else {
        i32 kk = 0;
        i32 ki = 0;

        if (nb >= nbmin && nb < k && nx < k) {
            ki = ((k - nx - 1) / nb) * nb;
            kk = (k < ki + nb) ? k : (ki + nb);
        }

        if (kk < k) {
            if (ltrc) {
                jc = kk;
            } else {
                ic = kk;
            }
            if (ltrd) {
                jd = kk;
            } else {
                id = kk;
            }

            i32 ierr;
            mb04qu(tranc, trand, tranq, storev, storew,
                   m - kk, n, k - kk,
                   (f64*)&v[lcolv ? (kk + kk * ldv) : (kk + kk * ldv)], ldv,
                   (f64*)&w[lcolw ? (kk + kk * ldw) : (kk + kk * ldw)], ldw,
                   &c[ic + jc * ldc], ldc,
                   &d[id + jd * ldd], ldd,
                   &cs[2 * kk], &tau[kk], dwork, ldwork, &ierr);
        }

        if (kk > 0) {
            for (i32 i = ki; i >= 0; i -= nb) {
                i32 ib = (nb < k - i) ? nb : (k - i);

                mb04qf("F", storev, storew, m - i, ib,
                       (f64*)&v[lcolv ? (i + i * ldv) : (i + i * ldv)], ldv,
                       (f64*)&w[lcolw ? (i + i * ldw) : (i + i * ldw)], ldw,
                       &cs[2 * i], &tau[i],
                       &dwork[pdrs], nb, &dwork[pdt], nb, &dwork[pdw]);

                if (ltrc) {
                    jc = i;
                } else {
                    ic = i;
                }
                if (ltrd) {
                    jd = i;
                } else {
                    id = i;
                }

                mb04qc("N", tranc, trand, tranq, "F", storev, storew,
                       m - i, n, ib,
                       &v[lcolv ? (i + i * ldv) : (i + i * ldv)], ldv,
                       &w[lcolw ? (i + i * ldw) : (i + i * ldw)], ldw,
                       &dwork[pdrs], nb, &dwork[pdt], nb,
                       &c[ic + jc * ldc], ldc,
                       &d[id + jd * ldd], ldd, &dwork[pdw]);
            }
        }
    }

    dwork[0] = (f64)wrkopt;
}
