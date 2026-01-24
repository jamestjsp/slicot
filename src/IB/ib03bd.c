/*
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * IB03BD - Wiener System Identification using Levenberg-Marquardt Algorithm
 *
 * Computes parameters for approximating a Wiener system (linear + nonlinear):
 *   x(t+1) = A*x(t) + B*u(t)       (linear state-space)
 *   z(t)   = C*x(t) + D*u(t)
 *   y(t)   = f(z(t), wb(1:L))      (static nonlinearity via neural network)
 *
 * The parameter vector X = (wb(1),...,wb(L), theta) where:
 * - wb(i): neural network weights for nonlinear part
 * - theta: linear part parameters in output normal form
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <ctype.h>

void ib03bd(
    const char* init,
    i32 nobr, i32 m, i32 l, i32 nsmp,
    i32* n,
    i32 nn, i32 itmax1, i32 itmax2, i32 nprint,
    const f64* u, i32 ldu,
    const f64* y, i32 ldy,
    f64* x, i32* lx,
    f64 tol1, f64 tol2,
    i32* iwork, f64* dwork, i32 ldwork,
    i32* iwarn, i32* info)
{
    char init_c = toupper((unsigned char)init[0]);

    bool init1 = (init_c == 'B') || (init_c == 'L');  // Initialize linear part
    bool init2 = (init_c == 'B') || (init_c == 'S');  // Initialize nonlinear part

    i32 ml = m + l;
    *info = 0;
    *iwarn = 0;

    // Parameter validation
    if (init_c != 'L' && init_c != 'S' && init_c != 'B' && init_c != 'N') {
        *info = -1;
        return;
    }
    if (init1 && nobr <= 0) {
        *info = -2;
        return;
    }
    if (m < 0) {
        *info = -3;
        return;
    }
    if (l < 0 || (init1 && l == 0)) {
        *info = -4;
        return;
    }
    if (nsmp < 0 || (init1 && nsmp < 2 * (ml + 1) * nobr - 1)) {
        *info = -5;
        return;
    }
    if ((*n < 0 && !init1) || ((*n == 0 || *n >= nobr) && init1)) {
        *info = -6;
        return;
    }
    if (nn < 0) {
        *info = -7;
        return;
    }
    if (init2 && itmax1 < 0) {
        *info = -8;
        return;
    }
    if (itmax2 < 0) {
        *info = -9;
        return;
    }
    if (ldu < (nsmp > 1 ? nsmp : 1)) {
        *info = -12;
        return;
    }
    if (ldy < (nsmp > 1 ? nsmp : 1)) {
        *info = -14;
        return;
    }

    // Compute derived parameters
    i32 lnol = l * nobr - l;
    i32 mno = m * nobr;
    i32 bsn = nn * (l + 2) + 1;
    i32 nths = bsn * l;
    i32 nsml = nsmp * l;

    i32 ldac = 0, isad = 0, n2 = 0;
    if (*n > 0) {
        ldac = *n + l;
        isad = ldac * (*n + m);
        n2 = (*n) * (*n);
    }

    // Compute workspace requirements
    i32 jwork = 0;
    i32 iw1, iw2, iw3;

    if (init1) {
        // Workspace for IB01AD
        jwork = 2 * ml * nobr * (2 * ml * (nobr + 1) + 3) + l * nobr;
        if (*n > 0) {
            // Workspace for IB01BD
            i32 t1 = 2 * lnol * (*n) + 2 * (*n);
            i32 t2 = lnol * (*n) + n2 + 7 * (*n);
            i32 t3 = lnol * (*n) + 2 * (*n) + (m + ml) * nobr + l;
            i32 t4 = 2 * lnol * (*n) + n2 + 8 * (*n);
            i32 t5 = (*n) + 4 * (mno + (*n)) + 1;
            i32 t6 = mno + 3 * (*n) + l;
            i32 inner_max = t3;
            if (t4 > inner_max) inner_max = t4;
            if (t5 > inner_max) inner_max = t5;
            if (t6 > inner_max) inner_max = t6;

            iw1 = l * nobr * (*n) + inner_max;
            if (t1 > iw1) iw1 = t1;
            if (t2 > iw1) iw1 = t2;

            if (m > 0) {
                i32 ldac_sq = ldac * ldac;
                i32 t7 = 4 * m * ldac + 1;
                iw2 = l * nobr * (*n) + mno * ldac * (m * ldac + 1) +
                      (ldac_sq > t7 ? ldac_sq : t7);
            } else {
                iw2 = 0;
            }

            i32 two_ml_nobr_sq = (2 * ml * nobr) * (2 * ml * nobr);
            i32 ib01bd_work = two_ml_nobr_sq + isad + (iw1 > iw2 ? iw1 : iw2);
            if (ib01bd_work > jwork) jwork = ib01bd_work;

            // Workspace for IB01CD
            i32 ldw1 = nsml * ((*n) + 1) + 2 * (*n) + (2 * n2 > 4 * (*n) ? 2 * n2 : 4 * (*n));
            i32 inner1 = (*n) * l * ((*n) + 1) + 2 * n2 + l * (*n);
            i32 inner2 = 4 * (*n);
            i32 ldw2 = (*n) * ((*n) + 1) + 2 * (*n) + (inner1 > inner2 ? inner1 : inner2);
            i32 min_ldw = ldw1 < ldw2 ? ldw1 : ldw2;

            i32 max_5n = 5 * (*n);
            i32 ib01cd_work = isad + 2 + (*n) * ((*n) + 1 + ldac + m) +
                             (max_5n > 2 ? (max_5n > min_ldw ? max_5n : min_ldw) :
                              (2 > min_ldw ? 2 : min_ldw));
            if (ib01cd_work > jwork) jwork = ib01cd_work;

            // Workspace for TF01MX
            i32 tf01mx_work = nsml + isad + ldac + 2 * (*n) + m;
            if (tf01mx_work > jwork) jwork = tf01mx_work;

            // Workspace for TB01VD
            i32 nl = (*n) > l ? (*n) : l;
            i32 inner3 = n2 + (*n) * nl + 6 * (*n) + ((*n) < l ? (*n) : l);
            i32 inner4 = (*n) * m;
            i32 inner5 = n2 + (inner3 > inner4 ? inner3 : inner4);
            i32 inner6 = n2 * l + (*n) * l + (*n);
            i32 tb01vd_inner = 1 > inner6 ? 1 : inner6;
            if (inner5 > tb01vd_inner) tb01vd_inner = inner5;
            i32 tb01vd_work = nsml + isad + (*n) + tb01vd_inner;
            if (tb01vd_work > jwork) jwork = tb01vd_work;
        }
    }

    if (init2) {
        // Workspace for MD03BD (initialization of nonlinear part)
        i32 inner1 = nsmp * bsn + (2 * nn > 5 * bsn + 1 ? 2 * nn : 5 * bsn + 1);
        i32 inner2 = bsn * bsn + bsn + (nsmp + 2 * nn > 5 * bsn ? nsmp + 2 * nn : 5 * bsn);
        i32 md03bd_init_work = nsml + bsn + (4 > (nsmp + (inner1 > inner2 ? inner1 : inner2)) ?
                               4 : (nsmp + (inner1 > inner2 ? inner1 : inner2)));
        if (md03bd_init_work > jwork) jwork = md03bd_init_work;

        if (*n > 0 && !init1) {
            // Workspace for TB01VY
            i32 tb01vy_work = nsml + ldac * (2 * (*n) + m) + 2 * (*n);
            if (tb01vy_work > jwork) jwork = tb01vy_work;

            // Workspace for TF01MX
            i32 tf01mx_iw1 = (m > 0) ? (*n + m) : 0;
            i32 tf01mx_work = nsml + isad + tf01mx_iw1 + ldac + (*n);
            if (tf01mx_work > jwork) jwork = tf01mx_work;
        }
    }

    // Compute number of parameters for given N
    i32 lths = 0, nx = 0;
    if (*n >= 0) {
        lths = (*n) * (ml + 1) + l * m;
        nx = nths + lths;

        if (*lx < nx) {
            *info = -16;
            return;
        }

        // Workspace for MD03BD (whole optimization)
        if (m > 0) {
            iw1 = ldac + m;
        } else {
            iw1 = l;
        }
        i32 n_ldac = (*n) * ldac;
        iw1 = nsml + (2 * nn > (isad + 2 * (*n) + (n_ldac > iw1 ? n_ldac : iw1)) ?
              2 * nn : (isad + 2 * (*n) + (n_ldac > iw1 ? n_ldac : iw1)));

        if (l <= 1 || bsn == 0) {
            iw3 = 4 * nx;
            iw2 = iw3 + 1;
        } else {
            iw2 = bsn + (3 * bsn + 1 > lths ? 3 * bsn + 1 : lths);
            if (nsmp > bsn) {
                i32 t = 4 * lths + 1;
                if (t > iw2) iw2 = t;
                if (nsmp < 2 * bsn) {
                    i32 t2 = (nsmp - bsn) * (l - 1);
                    if (t2 > iw2) iw2 = t2;
                }
            }
            iw3 = lths * bsn + 2 * nx + 2 * (bsn > lths ? bsn : lths);
        }

        i32 inner1 = nsml * (bsn + lths) + (nsml + iw1 > iw2 + nx ? nsml + iw1 : iw2 + nx);
        i32 inner2 = nx * (bsn + lths) + nx + (nsml + iw1 > nx + iw3 ? nsml + iw1 : nx + iw3);
        i32 main_work = nsml + nx + (4 > (nsml + (inner1 > inner2 ? inner1 : inner2)) ?
                        4 : (nsml + (inner1 > inner2 ? inner1 : inner2)));
        if (main_work > jwork) jwork = main_work;
    }

    if (ldwork < jwork) {
        *info = -21;
        dwork[0] = (f64)jwork;
        return;
    }

    // Initialize pointers and save seed
    i32 z_idx = 0;
    i32 ac = z_idx + nsml;
    f64 seed[4];
    SLC_DCOPY(&(i32){4}, dwork, &(i32){1}, seed, &(i32){1});

    i32 wrkopt = 1;
    i32 nfev = 0, njev = 0;
    i32 iw1_total = 0, iw2_total = 0;
    i32 iwarnl = 0, infol = 0;
    f64 rcnd[16];
    i32 ircnd = 0;

    // Fixed parameters for IB01AD, IB01BD, IB01CD
    static const char* meth = "M";    // MOESP method
    static const char* alg = "F";     // Fast QR
    static const char* jobd = "N";    // Not MOESP D computation
    static const char* batch = "O";   // One batch
    static const char* conct = "N";   // Not connected
    static const char* ctrl = "N";    // Not confirmed
    static const char* methb = "C";   // Combined MOESP+N4SID
    static const char* job = "A";     // All matrices
    static const char* jobck = "N";   // No Kalman gain
    static const char* comuse = "U";  // Use B, D
    static const char* jobxd = "D";   // D also
    f64 rcond = -1.0;
    f64 toln = -1.0;
    static const f64 gtol = 0.0;
    static const f64 tol = 0.0;
    static const f64 factor = 100.0;

    f64 work[4] = {0.0, 0.0, 0.0, 0.0};
    i32 ipar[7];
    i32 lipar = 7;
    i32 bwork[1] = {0};

    if (init1) {
        // Initialize linear part using MOESP/N4SID
        i32 ns = *n;
        i32 ir = 0;
        i32 isv = 2 * ml * nobr;
        i32 ldr = isv;
        isv = ir + ldr * isv;
        jwork = isv + l * nobr;

        // Call IB01AD for order estimation and R factor
        i32 ldwork_local = ldwork - jwork;
        ib01ad(meth, alg, jobd, batch, conct, ctrl,
               nobr, m, l, nsmp,
               u, ldu, y, ldy,
               n, &dwork[ir], ldr, &dwork[isv],
               rcond, toln,
               iwork, &dwork[jwork], ldwork_local,
               &iwarnl, &infol);

        if (infol != 0) {
            *info = 100 * infol;
            return;
        }
        if (iwarnl != 0) {
            *iwarn = 100 * iwarnl;
        }
        i32 opt = (i32)dwork[jwork] + jwork;
        if (opt > wrkopt) wrkopt = opt;

        if (meth[0] == 'N' || meth[0] == 'n') {
            ircnd = 2;
            SLC_DCOPY(&ircnd, &dwork[jwork + 1], &(i32){1}, rcnd, &(i32){1});
        }

        if (ns >= 0) {
            *n = ns;
        } else {
            // N was auto-detected, recompute derived values
            ldac = *n + l;
            isad = ldac * (*n + m);
            n2 = (*n) * (*n);
            lths = (*n) * (ml + 1) + l * m;
            nx = nths + lths;

            if (*lx < nx) {
                *lx = nx;
                *info = -16;
                return;
            }
        }

        // Set up pointers for IB01BD
        i32 bd = ac + ldac * (*n);
        i32 ix = bd + ldac * m;
        i32 ia = isv;
        i32 ib = ia + ldac * (*n);
        i32 iq, iry, is_idx, ik;
        iq = ib + ldac * m;
        iry = iq;
        is_idx = iq;
        ik = iq;
        jwork = iq;

        // Call IB01BD for state-space matrices estimation
        ldwork_local = ldwork - jwork;
        ib01bd(methb, job, jobck,
               nobr, *n, m, l, nsmp,
               &dwork[ir], ldr,
               &dwork[ia], ldac, &dwork[ia + (*n)], ldac,
               &dwork[ib], ldac, &dwork[ib + (*n)], ldac,
               &dwork[iq], *n, &dwork[iry], l, &dwork[is_idx], *n, &dwork[ik], *n,
               rcond, iwork, &dwork[jwork], ldwork_local, bwork,
               &iwarnl, &infol);

        if (infol == -30) {
            *info = -21;
            dwork[0] = dwork[jwork];
            return;
        }
        if (infol != 0) {
            *info = 100 * infol;
            return;
        }
        if (iwarnl != 0) {
            *iwarn = 100 * iwarnl;
        }
        opt = (i32)dwork[jwork] + jwork;
        if (opt > wrkopt) wrkopt = opt;

        i32 ircndb = 4;
        SLC_DCOPY(&ircndb, &dwork[jwork + 1], &(i32){1}, &rcnd[ircnd], &(i32){1});
        ircnd += ircndb;

        // Copy system matrices to beginning of DWORK
        SLC_DCOPY(&isad, &dwork[ia], &(i32){1}, dwork, &(i32){1});
        ia = 0;
        ib = ia + ldac * (*n);
        i32 ix0 = ib + ldac * m;
        i32 iv = ix0 + (*n);

        // Call IB01CD for initial state estimation
        jwork = iv + n2;
        ldwork_local = ldwork - jwork;

        slicot_ib01cd("X", comuse, jobxd,
                      *n, m, l, nsmp,
                      &dwork[ia], ldac,
                      &dwork[ib], ldac,
                      &dwork[ia + (*n)], ldac,
                      &dwork[ib + (*n)], ldac,
                      (f64*)u, ldu, y, ldy,
                      &dwork[ix0], &dwork[iv], *n,
                      rcond, iwork, &dwork[jwork], ldwork_local,
                      &iwarnl, &infol);

        if (infol == -26) {
            *info = -21;
            dwork[0] = dwork[jwork];
            return;
        }
        if (infol == 1) {
            infol = 10;
        }
        if (infol != 0) {
            *info = 100 * infol;
            return;
        }
        if (iwarnl != 0) {
            *iwarn = 100 * iwarnl;
        }
        opt = (i32)dwork[jwork] + jwork;
        if (opt > wrkopt) wrkopt = opt;
        ircnd++;
        rcnd[ircnd - 1] = dwork[jwork + 1];

        // Copy system matrices and x0 to final location
        i32 copy_len = isad + (*n);
        if (iv < ac) {
            SLC_DCOPY(&copy_len, &dwork[ia], &(i32){1}, &dwork[ac], &(i32){1});
        } else {
            for (i32 j = ac + copy_len - 1; j >= ac; j--) {
                dwork[j] = dwork[ia + j - ac];
            }
        }

        // Compute output of linear part using TF01MX
        ix = ac + isad;
        jwork = ix + (*n);
        SLC_DCOPY(n, &dwork[ix], &(i32){1}, &x[nths], &(i32){1});
        tf01mx(*n, m, l, nsmp, &dwork[ac], ldac, u, ldu, &x[nths],
               &dwork[z_idx], nsmp, &dwork[jwork], ldwork - jwork, &infol);

        // Convert state-space to output normal form using TB01VD
        bd = ac + ldac * (*n);
        f64 scale_dummy;
        tb01vd("A", *n, m, l, &dwork[ac], ldac, &dwork[bd], ldac,
               &dwork[ac + (*n)], ldac, &dwork[bd + (*n)], ldac,
               &dwork[ix], &x[nths], lths, &scale_dummy,
               &dwork[jwork], ldwork - jwork, &infol);

        if (infol > 0) {
            *info = infol + 4;
            return;
        }
        opt = (i32)dwork[jwork] + jwork;
        if (opt > wrkopt) wrkopt = opt;
    }

    // Re-compute ldac after potential N update
    if (*n > 0) {
        ldac = *n + l;
        isad = ldac * (*n + m);
        n2 = (*n) * (*n);
    }
    lths = (*n) * (ml + 1) + l * m;
    nx = nths + lths;

    i32 idiag = ac;

    if (init2) {
        // Initialize nonlinear part
        i32 bd = ac + ldac * (*n);
        i32 ix = bd + ldac * m;

        if (!init1) {
            // Convert output normal form to state-space model
            jwork = ix + (*n);
            tb01vy("A", *n, m, l, &x[nths], lths, &dwork[ac], ldac, &dwork[bd], ldac,
                   &dwork[ac + (*n)], ldac, &dwork[bd + (*n)], ldac,
                   &dwork[ix], &dwork[jwork], ldwork - jwork, &infol);

            // Compute output of linear part
            tf01mx(*n, m, l, nsmp, &dwork[ac], ldac, u, ldu,
                   &dwork[ix], &dwork[z_idx], nsmp, &dwork[jwork], ldwork - jwork, &infol);
        }

        // Optimize nonlinear part parameters for each output
        work[0] = 0.0;
        work[1] = 0.0;
        work[2] = 0.0;
        work[3] = 0.0;

        ipar[0] = nsmp;
        ipar[1] = l;
        ipar[2] = nn;
        jwork = idiag + bsn;

        // FCN callback type for md03bd (matches expected const signature)
        typedef void (*fcn_t)(i32*, i32, i32, i32*, i32, const f64*, i32, const f64*, i32,
                              const f64*, i32*, f64*, f64*, i32*, f64*, i32, i32*);

        for (i32 i = 0; i < l; i++) {
            SLC_DCOPY(&(i32){4}, seed, &(i32){1}, &dwork[jwork], &(i32){1});

            i32 ldwork_local = ldwork - jwork;
            md03bd("R", "I", "E", (fcn_t)nf01be, md03ba, md03bb,
                   nsmp, bsn, itmax1, factor, nprint,
                   ipar, lipar, &dwork[z_idx], nsmp, &y[i * ldy], ldy,
                   &x[i * bsn], &dwork[idiag], &nfev, &njev,
                   tol1, tol1, gtol, tol, iwork, &dwork[jwork], ldwork_local,
                   &iwarnl, &infol);

            if (infol != 0) {
                *info = 10 * infol;
                return;
            }
            if (iwarnl < 0) {
                *info = infol;
                *iwarn = iwarnl;
                goto finish;
            } else if (iwarnl > 0) {
                if (*iwarn > 100) {
                    i32 new_warn = (*iwarn / 100) * 100 + 10 * iwarnl;
                    if (new_warn > *iwarn) *iwarn = new_warn;
                } else {
                    i32 new_warn = 10 * iwarnl;
                    if (new_warn > *iwarn) *iwarn = new_warn;
                }
            }

            if (dwork[jwork] > work[0]) work[0] = dwork[jwork];
            if (dwork[jwork + 1] > work[1]) work[1] = dwork[jwork + 1];
            if (dwork[jwork + 3] > work[3]) work[3] = dwork[jwork + 3];
            work[2] += dwork[jwork + 2];
            iw1_total += nfev;
            iw2_total += njev;
        }
    }

    // Main optimization of both linear and nonlinear parts
    ipar[0] = lths;
    ipar[1] = l;
    ipar[2] = nsmp;
    ipar[3] = bsn;
    ipar[4] = m;
    ipar[5] = *n;
    ipar[6] = nn;
    jwork = idiag + nx;

    // FCN callback type for md03bd (matches expected const signature)
    typedef void (*fcn_t)(i32*, i32, i32, i32*, i32, const f64*, i32, const f64*, i32,
                          const f64*, i32*, f64*, f64*, i32*, f64*, i32, i32*);

    i32 ldwork_local = ldwork - jwork;
    md03bd("G", "I", "E", (fcn_t)nf01bf, nf01bs, nf01bp,
           nsml, nx, itmax2, factor, nprint,
           ipar, lipar, u, ldu, y, ldy,
           x, &dwork[idiag], &nfev, &njev,
           tol2, tol2, gtol, tol, iwork, &dwork[jwork], ldwork_local,
           &iwarnl, info);

    if (*info != 0) {
        return;
    }

    // Shift IWORK data
    for (i32 i = nx + l; i >= 1; i--) {
        iwork[i + 3 - 1] = iwork[i - 1];
    }

finish:
    iwork[0] = iw1_total + nfev;
    iwork[1] = iw2_total + njev;

    if (iwarnl < 0) {
        *iwarn = iwarnl;
    } else {
        *iwarn = *iwarn + iwarnl;
    }

    // Copy final results to DWORK
    SLC_DCOPY(&(i32){4}, &dwork[jwork], &(i32){1}, dwork, &(i32){1});
    if (init2) {
        SLC_DCOPY(&(i32){4}, work, &(i32){1}, &dwork[4], &(i32){1});
    }
    if (init1) {
        iwork[2] = ircnd;
        SLC_DCOPY(&ircnd, rcnd, &(i32){1}, &dwork[8], &(i32){1});
    } else {
        iwork[2] = 0;
    }

    dwork[0] = (f64)wrkopt;
}
