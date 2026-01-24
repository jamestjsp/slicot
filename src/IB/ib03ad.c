/*
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * IB03AD - Wiener system identification using neural networks and
 *          Levenberg-Marquardt algorithm.
 *
 * Computes parameters for approximating a Wiener system in a least-squares
 * sense. The Wiener system is:
 *   x(t+1) = A*x(t) + B*u(t)
 *   z(t)   = C*x(t) + D*u(t)
 *   y(t)   = f(z(t), wb(1:L))
 * where f is a nonlinear function modeled by neural networks.
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <ctype.h>
#include <stdlib.h>
#include <math.h>

static void nf01ba_fcn_wrapper(i32 *iflag, i32 m, i32 n, i32 *ipar, i32 lipar,
                               f64 *dpar1, i32 ldpar1, f64 *dpar2, i32 ldpar2,
                               const f64 *x_const, i32 *nfevl, f64 *e, f64 *j,
                               i32 *ldj, f64 *jte, f64 *dwork, i32 ldwork, i32 *info)
{
    f64 *x = (f64 *)x_const;
    nf01ba(iflag, m, n, ipar, lipar, dpar1, ldpar1, dpar2, ldpar2,
           x, nfevl, e, j, ldj, jte, dwork, ldwork, info);
}

static void nf01bb_fcn_wrapper(i32 *iflag, i32 m, i32 n, i32 *ipar, i32 lipar,
                               f64 *dpar1, i32 ldpar1, f64 *dpar2, i32 ldpar2,
                               const f64 *x_const, i32 *nfevl, f64 *e, f64 *j,
                               i32 *ldj, f64 *jte, f64 *dwork, i32 ldwork, i32 *info)
{
    f64 *x = (f64 *)x_const;
    nf01bb(iflag, m, n, ipar, lipar, dpar1, ldpar1, dpar2, ldpar2,
           x, nfevl, e, j, ldj, jte, dwork, ldwork, info);
}

void ib03ad(const char *init, const char *alg, const char *stor,
            i32 nobr, i32 m, i32 l, i32 nsmp, i32 *n, i32 nn,
            i32 itmax1, i32 itmax2, i32 nprint,
            const f64 *u, i32 ldu, const f64 *y, i32 ldy,
            f64 *x, i32 *lx, f64 tol1, f64 tol2,
            i32 *iwork, f64 *dwork, i32 ldwork,
            i32 *iwarn, i32 *info)
{
    const f64 ZERO = 0.0;
    const char UPLO[] = "U";
    const char IALG[] = "F";    /* Fast QR */
    const char BATCH[] = "O";   /* One batch */
    const char CONCT[] = "N";   /* Not connect */
    const char CTRL[] = "N";    /* Not confirm */
    const char JOBD[] = "N";    /* Not MOESP */
    const char METH[] = "M";    /* MOESP */
    const char JOB[] = "A";     /* All matrices */
    const char JOBCK[] = "N";   /* No Kalman gain */
    const char METHB[] = "C";   /* Combined MOESP+N4SID */
    const char COMUSE[] = "U";  /* Use B, D */
    const char JOBXD[] = "D";   /* D also */
    const f64 TOLN = -1.0;
    const f64 RCOND = -1.0;

    i32 int1 = 1;

    char init_c = toupper((unsigned char)init[0]);
    char alg_c = toupper((unsigned char)alg[0]);
    char stor_c = toupper((unsigned char)stor[0]);

    bool chol = (alg_c == 'D');
    bool full = (stor_c == 'F');
    bool init1 = (init_c == 'B' || init_c == 'L');  /* Initialize linear part */
    bool init2 = (init_c == 'B' || init_c == 'S');  /* Initialize nonlinear part */

    i32 ml = m + l;
    *info = 0;
    *iwarn = 0;

    /* Parameter validation */
    if (!init1 && !init2 && init_c != 'N') {
        *info = -1;
    } else if (!chol && alg_c != 'I') {
        *info = -2;
    } else if (chol && !full && stor_c != 'P') {
        *info = -3;
    } else if (init1 && nobr <= 0) {
        *info = -4;
    } else if (m < 0) {
        *info = -5;
    } else if (l < 0 || (init1 && l == 0)) {
        *info = -6;
    } else if (nsmp < 0 || (init1 && nsmp < 2 * (ml + 1) * nobr - 1)) {
        *info = -7;
    } else if ((*n < 0 && !init1) || ((*n == 0 || *n >= nobr) && init1)) {
        *info = -8;
    } else if (nn < 0) {
        *info = -9;
    } else if (init2 && itmax1 < 0) {
        *info = -10;
    } else if (itmax2 < 0) {
        *info = -11;
    } else if (ldu < 1 || (nsmp > 0 && ldu < nsmp)) {
        *info = -14;
    } else if (ldy < 1 || (nsmp > 0 && ldy < nsmp)) {
        *info = -16;
    }

    if (*info != 0) {
        return;
    }

    /* Compute parameters */
    i32 bsn = nn * (l + 2) + 1;     /* Nonlinear part parameter count per output */
    i32 nths = bsn * l;              /* Total nonlinear parameters */
    i32 nsml = nsmp * l;             /* Total samples * outputs */
    i32 ldac = 0, isad = 0, n2 = 0, lths = 0, nx = 0;

    if (*n > 0) {
        ldac = *n + l;
        isad = ldac * (*n + m);
        n2 = (*n) * (*n);
    }

    /* Check workspace - simplified */
    i32 jwork = 5;
    if (ldwork < jwork) {
        *info = -23;
        dwork[0] = (f64)jwork;
        return;
    }

    /* Initialize pointers */
    i32 z_idx = 0;
    i32 ac = z_idx + nsml;

    /* Save seed for random numbers */
    f64 seed[4];
    SLC_DCOPY(&(i32){4}, dwork, &int1, seed, &int1);

    i32 wrkopt = 1;
    i32 ircnd = 0;
    f64 rcnd[16];

    i32 nfev = 0, njev = 0;
    i32 infol = 0, iwarnl = 0;
    i32 bwork[1] = {0};

    if (init1) {
        /* Initialize linear part using IB01AD/IB01BD/slicot_ib01cd */
        i32 ns = *n;
        i32 ir = 0;
        i32 isv = 2 * ml * nobr;
        i32 ldr = isv;

        isv = ir + ldr * isv;
        jwork = isv + l * nobr;

        ib01ad(METH, IALG, JOBD, BATCH, CONCT, CTRL, nobr, m, l, nsmp,
               u, ldu, y, ldy, n, &dwork[ir], ldr, &dwork[isv], RCOND, TOLN,
               iwork, &dwork[jwork], ldwork - jwork, &iwarnl, &infol);

        if (infol != 0) {
            *info = 100 * infol;
            return;
        }
        if (iwarnl != 0) {
            *iwarn = 100 * iwarnl;
        }
        wrkopt = wrkopt > (i32)dwork[jwork] + jwork ? wrkopt : (i32)dwork[jwork] + jwork;

        if (ns >= 0) {
            *n = ns;
        } else {
            ldac = *n + l;
            isad = ldac * (*n + m);
            n2 = (*n) * (*n);
            lths = (*n) * (ml + 1) + l * m;
            nx = nths + lths;

            if (*lx < nx) {
                *lx = nx;
                *info = -18;
                return;
            }
        }

        i32 bd = ac + ldac * (*n);
        i32 ix = bd + ldac * m;
        i32 ia = isv;
        i32 ib = ia + ldac * (*n);
        i32 iq = ib + ldac * m;
        i32 iry, is_, ik;

        if (JOBCK[0] == 'N') {
            iry = iq;
            is_ = iq;
            ik = iq;
            jwork = iq;
        } else {
            iry = iq + n2;
            is_ = iry + l * l;
            ik = is_ + (*n) * l;
            jwork = ik + (*n) * l;
        }

        ib01bd(METHB, JOB, JOBCK, nobr, *n, m, l, nsmp, &dwork[ir], ldr,
               &dwork[ia], ldac, &dwork[ia + (*n)], ldac, &dwork[ib], ldac,
               &dwork[ib + (*n)], ldac, &dwork[iq], *n, &dwork[iry], l,
               &dwork[is_], *n, &dwork[ik], *n, RCOND, iwork,
               &dwork[jwork], ldwork - jwork, bwork, &iwarnl, &infol);

        if (infol == -30) {
            *info = -23;
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
        wrkopt = wrkopt > (i32)dwork[jwork] + jwork ? wrkopt : (i32)dwork[jwork] + jwork;
        ircnd = 4;
        SLC_DCOPY(&ircnd, &dwork[jwork + 1], &int1, rcnd, &int1);

        /* Copy system matrices to beginning of DWORK */
        SLC_DCOPY(&isad, &dwork[ia], &int1, dwork, &int1);
        ia = 0;
        ib = ia + ldac * (*n);
        i32 ix0 = ib + ldac * m;
        i32 iv = ix0 + (*n);

        jwork = iv + n2;
        slicot_ib01cd("X", COMUSE, JOBXD, *n, m, l, nsmp,
                      &dwork[ia], ldac, &dwork[ib], ldac, &dwork[ia + (*n)], ldac,
                      &dwork[ib + (*n)], ldac, (f64*)u, ldu, y, ldy,
                      &dwork[ix0], &dwork[iv], *n, RCOND,
                      iwork, &dwork[jwork], ldwork - jwork, &iwarnl, &infol);

        if (infol == -26) {
            *info = -23;
            dwork[0] = dwork[jwork];
            return;
        }
        if (infol == 1) infol = 10;
        if (infol != 0) {
            *info = 100 * infol;
            return;
        }
        if (iwarnl != 0) {
            *iwarn = 100 * iwarnl;
        }
        wrkopt = wrkopt > (i32)dwork[jwork] + jwork ? wrkopt : (i32)dwork[jwork] + jwork;
        ircnd++;
        rcnd[ircnd - 1] = dwork[jwork + 1];

        /* Save system matrices and x0 in final location */
        i32 len = isad + (*n);
        if (iv < ac) {
            SLC_DCOPY(&len, &dwork[ia], &int1, &dwork[ac], &int1);
        } else {
            for (i32 j = ac + len - 1; j >= ac; j--) {
                dwork[j] = dwork[ia + j - ac];
            }
        }

        /* Compute output of linear part */
        jwork = ix + (*n);
        SLC_DCOPY(n, &dwork[ix], &int1, &x[nths], &int1);
        tf01mx(*n, m, l, nsmp, &dwork[ac], ldac, u, ldu, &x[nths],
               &dwork[z_idx], nsmp, &dwork[jwork], ldwork - jwork, &infol);

        /* Convert state-space to output normal form */
        f64 scale;
        tb01vd("A", *n, m, l, &dwork[ac], ldac, &dwork[bd], ldac,
               &dwork[ac + (*n)], ldac, &dwork[bd + (*n)], ldac,
               &dwork[ix], &x[nths], lths, &scale, &dwork[jwork], ldwork - jwork, &infol);

        if (infol > 0) {
            *info = infol + 3;
            return;
        }
        wrkopt = wrkopt > (i32)dwork[jwork] + jwork ? wrkopt : (i32)dwork[jwork] + jwork;
    }

    i32 lipar = 7;
    i32 nfev_total = 0, njev_total = 0;
    f64 work[5] = {ZERO, ZERO, ZERO, ZERO, ZERO};

    if (init2) {
        /* Initialize nonlinear part */
        i32 bd_nl, ix_nl;

        if (!init1) {
            ldac = *n + l;
            isad = ldac * (*n + m);
            bd_nl = ac + ldac * (*n);
            ix_nl = bd_nl + ldac * m;

            /* Convert output normal form to state-space model */
            jwork = ix_nl + (*n);
            tb01vy("A", *n, m, l, &x[nths], lths, &dwork[ac], ldac,
                   &dwork[bd_nl], ldac, &dwork[ac + (*n)], ldac,
                   &dwork[bd_nl + (*n)], ldac, &dwork[ix_nl],
                   &dwork[jwork], ldwork - jwork, &infol);

            /* Compute output of linear part */
            tf01mx(*n, m, l, nsmp, &dwork[ac], ldac, u, ldu,
                   &dwork[ix_nl], &dwork[z_idx], nsmp, &dwork[jwork],
                   ldwork - jwork, &infol);
        }

        /* Optimize parameters of nonlinear part */
        jwork = ac;
        i32 ipar[7];
        ipar[0] = nsmp;
        ipar[1] = l;
        ipar[2] = nn;

        for (i32 i = 0; i < l; i++) {
            SLC_DCOPY(&(i32){4}, seed, &int1, &dwork[jwork], &int1);

            if (chol) {
                md03ad("R", alg, stor, UPLO, nf01ba_fcn_wrapper,
                       (md03ad_jpj_direct)nf01bv,
                       nsmp, bsn, itmax1, nprint, ipar, lipar,
                       &dwork[z_idx], nsmp, (f64 *)&y[i * ldy], ldy,
                       &x[i * bsn], &nfev, &njev, tol1, tol1,
                       &dwork[jwork], ldwork - jwork, &iwarnl, &infol);
            } else {
                md03ad("R", alg, stor, UPLO, nf01ba_fcn_wrapper,
                       (md03ad_jpj_direct)nf01bx,
                       nsmp, bsn, itmax1, nprint, ipar, lipar,
                       &dwork[z_idx], nsmp, (f64 *)&y[i * ldy], ldy,
                       &x[i * bsn], &nfev, &njev, tol1, tol1,
                       &dwork[jwork], ldwork - jwork, &iwarnl, &infol);
            }

            if (infol != 0) {
                *info = 10 * infol;
                return;
            }
            if (iwarnl < 0) {
                *info = infol;
                *iwarn = iwarnl;
                goto finalize;
            } else if (iwarnl > 0) {
                if (*iwarn > 100) {
                    i32 t = (*iwarn / 100) * 100 + 10 * iwarnl;
                    *iwarn = *iwarn > t ? *iwarn : t;
                } else {
                    *iwarn = *iwarn > 10 * iwarnl ? *iwarn : 10 * iwarnl;
                }
            }
            work[0] = work[0] > dwork[jwork] ? work[0] : dwork[jwork];
            work[1] = work[1] > dwork[jwork + 1] ? work[1] : dwork[jwork + 1];
            work[4] = work[4] > dwork[jwork + 4] ? work[4] : dwork[jwork + 4];
            work[2] += dwork[jwork + 2];
            work[3] += dwork[jwork + 3];
            nfev_total += nfev;
            njev_total += njev;
        }
    }

    /* Main iteration - optimize both linear and nonlinear parts */
    lths = (*n) * (ml + 1) + l * m;
    nx = nths + lths;

    i32 ipar[7];
    ipar[0] = lths;
    ipar[1] = l;
    ipar[2] = nsmp;
    ipar[3] = bsn;
    ipar[4] = m;
    ipar[5] = *n;
    ipar[6] = nn;

    if (chol) {
        md03ad("G", alg, stor, UPLO, nf01bb_fcn_wrapper,
               (md03ad_jpj_direct)nf01bu,
               nsml, nx, itmax2, nprint, ipar, lipar,
               (f64 *)u, ldu, (f64 *)y, ldy, x,
               &nfev, &njev, tol2, tol2,
               dwork, ldwork, &iwarnl, info);
    } else {
        md03ad("G", alg, stor, UPLO, nf01bb_fcn_wrapper,
               (md03ad_jpj_direct)nf01bw,
               nsml, nx, itmax2, nprint, ipar, lipar,
               (f64 *)u, ldu, (f64 *)y, ldy, x,
               &nfev, &njev, tol2, tol2,
               dwork, ldwork, &iwarnl, info);
    }

    if (*info != 0) {
        return;
    }

finalize:
    iwork[0] = nfev_total + nfev;
    iwork[1] = njev_total + njev;
    if (iwarnl < 0) {
        *iwarn = iwarnl;
    } else {
        *iwarn = *iwarn + iwarnl;
    }
    if (init2) {
        SLC_DCOPY(&(i32){5}, work, &int1, &dwork[5], &int1);
    }
    if (init1) {
        iwork[2] = ircnd;
        SLC_DCOPY(&ircnd, rcnd, &int1, &dwork[10], &int1);
    } else {
        iwork[2] = 0;
    }
}
