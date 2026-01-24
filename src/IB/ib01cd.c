/*
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * IB01CD - Estimate initial state and system matrices B, D (driver)
 *
 * Given system matrices (A,B,C,D) or (A,C) and input/output trajectories,
 * estimates the initial state x(0) and optionally B and D for discrete-time:
 *   x(k+1) = A*x(k) + B*u(k)
 *   y(k)   = C*x(k) + D*u(k)
 *
 * This is a driver routine that:
 * 1. Transforms A to real Schur form via TB01WD
 * 2. Calls IB01QD (if COMUSE='C') or IB01RD (otherwise) for estimation
 * 3. Back-transforms results to original coordinates
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <ctype.h>

void slicot_ib01cd(
    const char* jobx0, const char* comuse, const char* job,
    i32 n, i32 m, i32 l, i32 nsmp,
    const f64* a, i32 lda,
    f64* b, i32 ldb,
    const f64* c, i32 ldc,
    f64* d, i32 ldd,
    f64* u, i32 ldu,
    const f64* y, i32 ldy,
    f64* x0,
    f64* v, i32 ldv,
    f64 tol,
    i32* iwork, f64* dwork, i32 ldwork,
    i32* iwarn, i32* info)
{
    char jobx0_c = toupper((unsigned char)jobx0[0]);
    char comuse_c = toupper((unsigned char)comuse[0]);
    char job_c = toupper((unsigned char)job[0]);

    bool withx0 = (jobx0_c == 'X');
    bool compbd = (comuse_c == 'C');
    bool usebd = (comuse_c == 'U');
    bool withd = (job_c == 'D');
    bool withb = (job_c == 'B') || withd;
    bool maxdim = (withx0 && usebd) || compbd;
    bool maxdia = withx0 || compbd;

    *iwarn = 0;
    *info = 0;

    i32 ldw = (n > 1) ? n : 1;
    i32 lm = l * m;
    i32 ln = l * n;
    i32 nn = n * n;
    i32 nm = n * m;
    i32 n2m = n * nm;

    i32 ncol, minsmp, iq;
    if (compbd) {
        ncol = nm;
        if (withx0) ncol += n;
        minsmp = ncol;
        if (withd) {
            minsmp += m;
            iq = minsmp;
        } else if (!withx0) {
            iq = minsmp;
            minsmp += 1;
        } else {
            iq = minsmp;
        }
    } else {
        ncol = n;
        if (withx0) {
            minsmp = n;
        } else {
            minsmp = 0;
        }
        iq = minsmp;
    }

    if (jobx0_c != 'X' && jobx0_c != 'N') {
        *info = -1;
        return;
    }
    if (comuse_c != 'C' && comuse_c != 'U' && comuse_c != 'N') {
        *info = -2;
        return;
    }
    if (!withb) {
        *info = -3;
        return;
    }
    if (n < 0) {
        *info = -4;
        return;
    }
    if (m < 0) {
        *info = -5;
        return;
    }
    if (l <= 0) {
        *info = -6;
        return;
    }
    if (nsmp < minsmp) {
        *info = -7;
        return;
    }
    if (lda < 1 || (maxdia && lda < ldw)) {
        *info = -9;
        return;
    }
    if (ldb < 1 || (m > 0 && maxdim && ldb < ldw)) {
        *info = -11;
        return;
    }
    if (ldc < 1 || (n > 0 && maxdia && ldc < l)) {
        *info = -13;
        return;
    }
    if (ldd < 1 || (m > 0 && maxdim && withd && ldd < l)) {
        *info = -15;
        return;
    }
    if (ldu < 1 || (m > 0 && maxdim && ldu < nsmp)) {
        *info = -17;
        return;
    }
    if (ldy < 1 || (maxdia && ldy < nsmp)) {
        *info = -19;
        return;
    }
    if (ldv < 1 || (maxdia && ldv < ldw)) {
        *info = -22;
        return;
    }
    if (tol > 1.0) {
        *info = -23;
        return;
    }

    i32 minwrk, ia_base;
    if (!maxdia || (n == 0 && m == 0) || (n == 0)) {
        minwrk = 2;
        ia_base = 2;
    } else {
        i32 nsmpl = nsmp * l;
        iq = iq * l;
        i32 ncp1 = ncol + 1;
        i32 isize = nsmpl * ncp1;

        i32 ic;
        if (compbd) {
            if (n > 0 && withx0) {
                ic = 2 * nn + n;
            } else {
                ic = 0;
            }
        } else {
            ic = 2 * nn;
        }

        i32 minwls = ncol * ncp1;
        if (compbd && withd) {
            minwls += lm * ncp1;
        }

        i32 ia_calc;
        if (compbd) {
            if (m > 0 && withd) {
                i32 twoncol = 2 * ncol;
                ia_calc = m + ((twoncol > m) ? twoncol : m);
            } else {
                ia_calc = 2 * ncol;
            }
        } else {
            ia_calc = 2 * ncol;
        }

        i32 itau = n2m + ((ic > ia_calc) ? ic : ia_calc);
        if (compbd && withx0) {
            itau += ln;
        } else if (!compbd) {
            itau = ic + ln;
        }

        i32 ldw2, ldw3;
        if (compbd) {
            i32 max_ic_ia = (ic > ia_calc) ? ic : ia_calc;
            i32 t1 = n + max_ic_ia;
            i32 t2 = 6 * ncol;
            ldw2 = isize + ((t1 > t2) ? t1 : t2);
            ldw3 = minwls + ((iq * ncp1 + itau > 6 * ncol) ? (iq * ncp1 + itau) : 6 * ncol);
            if (m > 0 && withd) {
                i32 t3 = isize + 2 * m * m + 6 * m;
                if (t3 > ldw2) ldw2 = t3;
                t3 = minwls + 2 * m * m + 6 * m;
                if (t3 > ldw3) ldw3 = t3;
                ia_base = 3;
            } else {
                ia_base = 2;
            }
        } else {
            ldw2 = isize + 2 * n + ((ic > 4 * n) ? ic : (4 * n));
            ldw3 = minwls + 2 * n + ((iq * ncp1 + itau > 4 * n) ? (iq * ncp1 + itau) : (4 * n));
            ia_base = 2;
        }

        i32 min_ldw = (ldw2 < ldw3) ? ldw2 : ldw3;
        i32 t_5n = 5 * n;
        i32 max_5n_ia = (t_5n > ia_base) ? t_5n : ia_base;
        i32 max_term = (max_5n_ia > min_ldw) ? max_5n_ia : min_ldw;
        minwrk = ia_base + nn + nm + ln + max_term;
    }

    if (ldwork < minwrk) {
        *info = -26;
        dwork[0] = (f64)minwrk;
        return;
    }

    if (!maxdia || (n == 0 && m == 0)) {
        dwork[1] = 1.0;
        if (compbd && m > 0 && withd) {
            dwork[0] = 3.0;
            dwork[2] = 1.0;
        } else {
            dwork[0] = 2.0;
        }
        if (n > 0 && usebd) {
            f64 dum = 0.0;
            i32 int1 = 1, int0 = 0;
            SLC_DCOPY(&n, &dum, &int0, x0, &int1);
        }
        return;
    }

    if (n == 0) {
        dwork[1] = 1.0;
        if (compbd && m > 0 && withd) {
            dwork[0] = 3.0;
            dwork[2] = 1.0;
        } else {
            dwork[0] = 2.0;
        }
        return;
    }

    i32 ia = ia_base;
    i32 ic_off = ia + nn;
    i32 ib = ic_off + ln;

    SLC_DLACPY("F", &n, &n, a, &lda, &dwork[ia], &ldw);
    SLC_DLACPY("F", &l, &n, c, &ldc, &dwork[ic_off], &l);

    i32 mtmp;
    if (usebd) {
        mtmp = m;
        SLC_DLACPY("F", &n, &m, b, &ldb, &dwork[ib], &ldw);
    } else {
        mtmp = 0;
    }

    i32 iwr = ib + nm;
    i32 iwi = iwr + n;
    i32 jwork = iwi + n;

    i32 ldw_tb01 = ldwork - jwork;
    i32 ierr = 0;

    tb01wd(n, mtmp, l, &dwork[ia], ldw, &dwork[ib], ldw, &dwork[ic_off], l,
           v, ldv, &dwork[iwr], &dwork[iwi], &dwork[jwork], ldw_tb01, &ierr);

    if (ierr > 0) {
        *info = 1;
        return;
    }

    i32 maxwrk = (i32)dwork[jwork] + jwork;

    for (i32 i = iwr; i < iwi; i++) {
        f64 re = dwork[i];
        f64 im = dwork[i + n];
        f64 mod = sqrt(re * re + im * im);
        if (mod >= 1.0) {
            *iwarn = 6;
            break;
        }
    }

    jwork = iwr;

    i32 iwarnl = 0;
    f64 rcond = 0.0;
    f64 rcondu = 1.0;

    if (compbd) {
        i32 ldw_ib01qd = ldwork - jwork;
        slicot_ib01qd(jobx0, job, n, m, l, nsmp,
                      &dwork[ia], ldw, &dwork[ic_off], l,
                      u, ldu, y, ldy,
                      x0, &dwork[ib], ldw, d, ldd,
                      tol, iwork, &dwork[jwork], ldw_ib01qd,
                      &iwarnl, info);

        if (*info == 0) {
            if (m > 0 && withd) {
                rcondu = dwork[jwork + 2];
            }

            f64 dbl1 = 1.0, dbl0 = 0.0;
            SLC_DGEMM("N", "N", &n, &m, &n, &dbl1, v, &ldv, &dwork[ib], &ldw, &dbl0, b, &ldb);
        }
    } else {
        const char* jobd_str;
        if (withd) {
            jobd_str = "N";
        } else {
            jobd_str = "Z";
        }

        i32 ldw_ib01rd = ldwork - jwork;
        slicot_ib01rd(jobd_str, n, mtmp, l, nsmp,
                      &dwork[ia], ldw, &dwork[ib], ldw, &dwork[ic_off], l,
                      d, ldd, u, ldu, y, ldy,
                      x0, tol, iwork, &dwork[jwork], ldw_ib01rd,
                      &iwarnl, info);
    }

    if (iwarnl > *iwarn) {
        *iwarn = iwarnl;
    }

    if (*info == 0) {
        rcond = dwork[jwork + 1];
        i32 opt_inner = (i32)dwork[jwork] + jwork;
        if (opt_inner > maxwrk) maxwrk = opt_inner;

        if (withx0) {
            f64 dbl1 = 1.0, dbl0 = 0.0;
            i32 int1 = 1;
            SLC_DGEMV("N", &n, &n, &dbl1, v, &ldv, x0, &int1, &dbl0, &dwork[jwork], &int1);
            SLC_DCOPY(&n, &dwork[jwork], &int1, x0, &int1);
        }

        dwork[0] = (f64)maxwrk;
        dwork[1] = rcond;
        if (compbd && m > 0 && withd) {
            dwork[2] = rcondu;
        }
    }
}
