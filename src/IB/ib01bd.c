/*
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * IB01BD - Estimate system matrices (A,C,B,D) and optionally noise covariances
 *          (Q,Ry,S) and Kalman gain K from the triangular factor R computed by
 *          IB01AD. Uses N4SID/MOESP/Combined subspace identification methods.
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

void ib01bd(const char *meth, const char *job, const char *jobck,
            i32 nobr, i32 n, i32 m, i32 l, i32 nsmpl,
            f64 *r, i32 ldr, f64 *a, i32 lda, f64 *c, i32 ldc,
            f64 *b, i32 ldb, f64 *d, i32 ldd, f64 *q, i32 ldq,
            f64 *ry, i32 ldry, f64 *s, i32 lds, f64 *k, i32 ldk,
            f64 tol, i32 *iwork, f64 *dwork, i32 ldwork, i32 *bwork,
            i32 *iwarn, i32 *info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    bool moesp = (meth[0] == 'M' || meth[0] == 'm');
    bool n4sid = (meth[0] == 'N' || meth[0] == 'n');
    bool combin = (meth[0] == 'C' || meth[0] == 'c');
    bool withal = (job[0] == 'A' || job[0] == 'a');
    bool withc  = (job[0] == 'C' || job[0] == 'c') || withal;
    bool withd  = (job[0] == 'D' || job[0] == 'd') || withal;
    bool withb  = (job[0] == 'B' || job[0] == 'b') || withd;
    bool withk  = (jobck[0] == 'K' || jobck[0] == 'k');
    bool withco = (jobck[0] == 'C' || jobck[0] == 'c') || withk;

    i32 mnobr = m * nobr;
    i32 lnobr = l * nobr;
    i32 lmnobr = lnobr + mnobr;
    i32 nr = lmnobr + lmnobr;
    i32 nn = n * n;
    i32 n2 = n + n;
    i32 npl = n + l;
    i32 nl = n * l;
    i32 ll = l * l;

    *iwarn = 0;
    *info = 0;

    if (!(moesp || n4sid || combin)) {
        *info = -1;
    } else if (!(withb || withc)) {
        *info = -2;
    } else if (!(withco || jobck[0] == 'N' || jobck[0] == 'n')) {
        *info = -3;
    } else if (nobr <= 1) {
        *info = -4;
    } else if (n <= 0 || n >= nobr) {
        *info = -5;
    } else if (m < 0) {
        *info = -6;
    } else if (l <= 0) {
        *info = -7;
    } else if (withco && nsmpl < nr) {
        *info = -8;
    } else if (ldr < nr) {
        *info = -10;
    } else if (lda < 1 || ((withc || (withb && !moesp)) && lda < n)) {
        *info = -12;
    } else if (ldc < 1 || ((withc || (withb && !moesp)) && ldc < l)) {
        *info = -14;
    } else if (ldb < 1 || (withb && m > 0 && ldb < n)) {
        *info = -16;
    } else if (ldd < 1 || (withd && m > 0 && ldd < l)) {
        *info = -18;
    } else if (ldq < 1 || (withco && ldq < n)) {
        *info = -20;
    } else if (ldry < 1 || (withco && ldry < l)) {
        *info = -22;
    } else if (lds < 1 || (withco && lds < n)) {
        *info = -24;
    } else if (ldk < 1 || (withk && ldk < n)) {
        *info = -26;
    }

    i32 minwrk;
    i32 ldun2 = lnobr - l;
    i32 ldunn = ldun2 * n;
    i32 mnobrn = mnobr + n;
    i32 lmmnob = lnobr + 2 * mnobr;
    i32 lmmnol = lmmnob + l;

    if (*info == 0) {
        minwrk = ldunn + 4*n;

        if (moesp || combin) {
            if (withc) {
                i32 alt1 = 2*ldunn + n2;
                i32 alt2 = ldunn + nn + 7*n;
                if (alt1 > minwrk) minwrk = alt1;
                if (alt2 > minwrk) minwrk = alt2;
            }
        }

        if ((m > 0 && withb) || n4sid) {
            i32 idn = (n4sid || combin) ? n : 0;
            i32 alt = 2*ldunn + nn + idn + 7*n;
            if (alt > minwrk) minwrk = alt;
            if (moesp || combin) {
                i32 alt1 = ldunn + n + 6*mnobr;
                i32 tmp1 = l + mnobr;
                i32 tmp2 = 3*lnobr + 1;
                if (m > tmp2) tmp2 = m;
                i32 tmp3 = lnobr + tmp2;
                i32 alt2 = ldunn + n + ((tmp1 > tmp3) ? tmp1 : tmp3);
                if (alt1 > minwrk) minwrk = alt1;
                if (alt2 > minwrk) minwrk = alt2;
            }
        }

        if (withco || n4sid || combin) {
            i32 iaw = 0;
            if ((moesp || combin) && !(m > 0 && withb))
                iaw = n + nn;
            i32 tmp1 = 5*n > lmmnol ? 5*n : lmmnol;
            i32 alt1 = ldunn + iaw + n2 + tmp1;
            i32 idn = (n4sid || combin) ? n : 0;
            i32 alt2 = idn + 4*mnobrn + 1;
            i32 alt3 = idn + mnobrn + npl;
            if (alt1 > minwrk) minwrk = alt1;
            if (alt2 > minwrk) minwrk = alt2;
            if (alt3 > minwrk) minwrk = alt3;

            if ((n4sid || combin) && m > 0 && withb) {
                i32 npl2 = npl * npl;
                i32 mnpl = m * npl;
                i32 tmp = 4*mnpl + 1;
                if (npl2 > tmp) tmp = npl2;
                i32 alt = mnobr * npl * (mnpl + 1) + tmp;
                if (alt > minwrk) minwrk = alt;
            }

            if (withk) {
                i32 liwork = (2*n > m) ? 2*n : m;
                if (liwork < n2) liwork = n2;
                i32 tmp = 3*l > nl ? 3*l : nl;
                i32 alt1k = 4*nn + 2*nl + ll + tmp;
                i32 alt2k = 14*nn + 12*n + 5;
                i32 alt = alt1k > alt2k ? alt1k : alt2k;
                if (alt > minwrk) minwrk = alt;
            }
        }

        if (ldwork < minwrk) {
            *info = -30;
            dwork[0] = (f64)minwrk;
        }
    }

    if (*info != 0) {
        return;
    }

    char jobcv[2];
    if (!withk) {
        jobcv[0] = jobck[0];
    } else {
        jobcv[0] = 'C';
    }
    jobcv[1] = '\0';

    i32 io = 0;
    i32 jwork;
    if (!moesp || withco) {
        jwork = io + lnobr * n;
    } else {
        jwork = io;
    }

    i32 iwarnl = 0;
    i32 infol = 0;
    f64 *o = &dwork[io];
    i32 ldo = lnobr;

    f64 rcnd[8];
    i32 maxwrk = minwrk;

    if (!combin) {
        char methpd[2] = {meth[0], '\0'};
        char jobpd[2] = {job[0], '\0'};

        ib01pd(methpd, jobpd, jobcv, nobr, n, m, l, nsmpl,
               r, ldr, a, lda, c, ldc, b, ldb, d, ldd,
               q, ldq, ry, ldry, s, lds, o, ldo,
               tol, iwork, &dwork[jwork], ldwork - jwork, &iwarnl, &infol);
    } else {
        char jobcov[2];
        if (withc) {
            if (withal) {
                jobcov[0] = 'N';
            } else {
                jobcov[0] = jobcv[0];
            }
            jobcov[1] = '\0';

            ib01pd("M", "C", jobcov, nobr, n, m, l, nsmpl,
                   r, ldr, a, lda, c, ldc, b, ldb, d, ldd,
                   q, ldq, ry, ldry, s, lds, o, ldo,
                   tol, iwork, &dwork[jwork], ldwork - jwork, &iwarnl, &infol);

            if (infol != 0) {
                *info = infol;
                return;
            }
            if (iwarnl != 0) *iwarn = iwarnl;
        }

        if (withb) {
            char jobbd[2];
            if (!withal) {
                jobbd[0] = job[0];
            } else {
                jobbd[0] = 'D';
            }
            jobbd[1] = '\0';

            ib01pd("N", jobbd, jobcv, nobr, n, m, l, nsmpl,
                   r, ldr, a, lda, c, ldc, b, ldb, d, ldd,
                   q, ldq, ry, ldry, s, lds, o, ldo,
                   tol, iwork, &dwork[jwork], ldwork - jwork, &iwarnl, &infol);

            if (iwarnl != 0 && *iwarn == 0) *iwarn = iwarnl;
        }
    }

    if (iwarnl != 0 && *iwarn == 0) *iwarn = iwarnl;
    if (infol != 0) {
        *info = infol;
        return;
    }

    if ((i32)dwork[jwork] + jwork - 1 > maxwrk) {
        maxwrk = (i32)dwork[jwork] + jwork - 1;
    }

    for (i32 i = 0; i < 4; i++) {
        rcnd[i] = dwork[jwork + i + 1];
    }

    if (withk) {
        if (*iwarn == 5) {
            SLC_DLASET("Full", &n, &l, &ZERO, &ZERO, k, &ldk);

            for (i32 i = 5; i < 12; i++) {
                dwork[i] = ONE;
            }
            dwork[12] = ZERO;
        } else {
            i32 ix = 0;
            i32 iq = ix + nn;
            i32 ia = iq + nn;
            i32 ig = ia + nn;
            i32 ic = ig + nn;
            i32 ir = ic + nl;
            i32 is = ir + ll;
            jwork = is + nl;

            ma02ad("Full", n, n, a, lda, &dwork[ia], n);
            ma02ad("Full", l, n, c, ldc, &dwork[ic], n);
            SLC_DLACPY("Upper", &n, &n, q, &ldq, &dwork[iq], &n);
            SLC_DLACPY("Upper", &l, &l, ry, &ldry, &dwork[ir], &l);
            SLC_DLACPY("Full", &n, &l, s, &lds, &dwork[is], &n);

            i32 ifact;
            sb02mt("G needed", "Nonzero S", "Not factored",
                   "Upper", n, l, &dwork[ia], n, &dwork[ic], n,
                   &dwork[iq], n, &dwork[ir], l, &dwork[is], n,
                   iwork, &ifact, &dwork[ig], n, &iwork[l],
                   &dwork[jwork], ldwork - jwork, &infol);

            if (infol != 0) {
                *info = 3;
                return;
            }
            if ((i32)dwork[jwork] + jwork - 1 > maxwrk) {
                maxwrk = (i32)dwork[jwork] + jwork - 1;
            }
            f64 rcondr = dwork[jwork + 1];

            i32 it = ic;
            i32 iv = it + nn;
            i32 iwr = iv + nn;
            i32 iwi = iwr + n2;
            is = iwi + n2;
            jwork = is + n2 * n2;

            f64 sep, rcond, ferr;
            sb02rd("All", "Discrete", "Direct", "NoTranspose",
                   "Upper", "General scaling", "Unstable first",
                   "Not factored", "Reduced", n, &dwork[ia], n,
                   &dwork[it], n, &dwork[iv], n, &dwork[ig], n,
                   &dwork[iq], n, &dwork[ix], n, &sep, &rcond, &ferr,
                   &dwork[iwr], &dwork[iwi], &dwork[is], n2, iwork,
                   &dwork[jwork], ldwork - jwork, bwork, &infol);

            if (infol != 0 && infol < 7) {
                *info = infol + 3;
                return;
            }
            if ((i32)dwork[jwork] + jwork - 1 > maxwrk) {
                maxwrk = (i32)dwork[jwork] + jwork - 1;
            }

            for (i32 i = 0; i < 4; i++) {
                rcnd[i + 4] = dwork[jwork + i + 1];
            }

            ia = ix + nn;
            ic = ia + nn;
            ir = ic + nl;
            i32 ik = ir + ll;
            jwork = ik + nl;

            ma02ad("Full", n, n, a, lda, &dwork[ia], n);
            ma02ad("Full", l, n, c, ldc, &dwork[ic], n);
            SLC_DLACPY("Upper", &l, &l, ry, &ldry, &dwork[ir], &l);

            f64 rnorm = SLC_DLANGE("1", &l, &l, ry, &ldry, NULL);
            i32 oufact[2];
            sb02nd("Discrete", "NotFactored", "Upper",
                   "Nonzero S", n, l, 0, &dwork[ia], n, &dwork[ic],
                   n, &dwork[ir], l, iwork, s, lds, &dwork[ix], n,
                   rnorm, &dwork[ik], l, oufact,
                   &dwork[jwork], ldwork - jwork, &infol);

            if (infol != 0) {
                if (infol <= l + 1) {
                    *info = 3;
                } else if (infol == l + 2) {
                    *info = 10;
                }
                return;
            }
            if ((i32)dwork[jwork] + jwork - 1 > maxwrk) {
                maxwrk = (i32)dwork[jwork] + jwork - 1;
            }

            ma02ad("Full", l, n, &dwork[ik], l, k, ldk);

            dwork[10] = dwork[jwork + 1];

            for (i32 i = 5; i < 9; i++) {
                dwork[i] = rcnd[i - 1];
            }
            dwork[9] = rcondr;
            dwork[11] = rcond;
            dwork[12] = ferr;
        }

        dwork[0] = (f64)maxwrk;
        for (i32 i = 0; i < 4; i++) {
            dwork[i + 1] = rcnd[i];
        }
        return;
    }

    dwork[0] = (f64)maxwrk;
    for (i32 i = 0; i < 4; i++) {
        dwork[i + 1] = rcnd[i];
    }
}
