/*
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * MD03AD - Levenberg-Marquardt optimizer for nonlinear least squares
 *
 * Minimizes sum of squares of m nonlinear functions in n variables
 * using Levenberg-Marquardt algorithm with either Cholesky-based (ALG='D')
 * or conjugate gradients (ALG='I') solver.
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <ctype.h>

void md03ad(
    const char* xinit,
    const char* alg,
    const char* stor,
    const char* uplo,
    md03ad_fcn fcn,
    md03ad_jpj_direct jpj,
    i32 m,
    i32 n,
    i32 itmax,
    i32 nprint,
    i32* ipar,
    i32 lipar,
    f64* dpar1,
    i32 ldpar1,
    f64* dpar2,
    i32 ldpar2,
    f64* x,
    i32* nfev,
    i32* njev,
    f64 tol,
    f64 cgtol,
    f64* dwork,
    i32 ldwork,
    i32* iwarn,
    i32* info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const f64 FIVE = 5.0;
    const f64 FACTOR = 100.0;
    const f64 MARQF = 4.0;      /* 2^2 */
    const f64 MINIMP = 0.125;   /* 2^(-3) */
    const f64 PARMAX = 1.0e20;

    char xinit_c = toupper((unsigned char)xinit[0]);
    char alg_c = toupper((unsigned char)alg[0]);
    char stor_c = toupper((unsigned char)stor[0]);
    char uplo_c = toupper((unsigned char)uplo[0]);

    bool init = (xinit_c == 'R');
    bool chol = (alg_c == 'D');
    bool full = (stor_c == 'F');
    bool upper = (uplo_c == 'U');

    *iwarn = 0;
    *info = 0;

    /* Parameter validation */
    if (!init && xinit_c != 'G') {
        *info = -1;
    } else if (!chol && alg_c != 'I') {
        *info = -2;
    } else if (chol && !full && stor_c != 'P') {
        *info = -3;
    } else if (chol && !upper && uplo_c != 'L') {
        *info = -4;
    } else if (m < 0) {
        *info = -7;
    } else if (n < 0 || n > m) {
        *info = -8;
    } else if (itmax < 0) {
        *info = -9;
    } else if (lipar < 5) {
        *info = -12;
    } else if (ldpar1 < 0) {
        *info = -14;
    } else if (ldpar2 < 0) {
        *info = -16;
    } else if (ldwork < 5) {
        *info = -23;
    }

    if (*info != 0) {
        return;
    }

    /* Quick return */
    *nfev = 0;
    *njev = 0;

    i32 min_n_itmax = (n < itmax) ? n : itmax;
    if (min_n_itmax == 0) {
        dwork[0] = FIVE;
        dwork[1] = ZERO;
        dwork[2] = ZERO;
        dwork[3] = ZERO;
        dwork[4] = ZERO;
        return;
    }

    /* Call FCN with IFLAG=3 to get workspace requirements */
    i32 iflag = 3;
    i32 iw1_save = ipar[0];
    i32 iw2_save = ipar[1];
    i32 jw1_save = ipar[2];
    i32 jw2_save = ipar[3];
    i32 ljtj_save = ipar[4];

    i32 nfevl = 0;
    i32 ldj = 0;
    i32 infol = 0;

    fcn(&iflag, m, n, ipar, lipar, dpar1, ldpar1, dpar2, ldpar2,
        x, &nfevl, dwork, dwork, &ldj, dwork, dwork, ldwork, &infol);

    i32 sizej = ipar[0];
    i32 lfcn1 = ipar[1];
    i32 lfcn2 = ipar[2];
    i32 ljtjd = ipar[3];
    i32 ljtji = ipar[4];

    /* Restore IPAR */
    ipar[0] = iw1_save;
    ipar[1] = iw2_save;
    ipar[2] = jw1_save;
    ipar[3] = jw2_save;
    ipar[4] = ljtj_save;

    /* Define workspace pointers */
    i32 jac = 0;
    i32 e = jac + sizej;
    i32 jte = e + m;
    i32 iw1 = jte + n;
    i32 iw2 = iw1 + n;
    i32 jw1 = iw2;
    i32 jw2 = iw2 + n;

    /* Compute workspace for solver */
    i32 jwork = jw1;
    i32 ldw;
    i32 dwjtj;
    i32 ljtj;

    if (chol) {
        if (full) {
            ldw = n * n;
        } else {
            ldw = (n * (n + 1)) / 2;
        }
        dwjtj = jwork;
        jwork = dwjtj + ldw;
        ljtj = ljtjd;
    } else {
        ldw = 3 * n;
        ljtj = ljtji;
    }

    /* Check workspace length */
    i32 max_fcn = (lfcn1 + n > lfcn2) ? (lfcn1 + n) : lfcn2;
    max_fcn = (max_fcn > ldw + ljtj) ? max_fcn : (ldw + ljtj);
    i32 min_ldwork = sizej + m + 2*n + max_fcn;
    min_ldwork = (min_ldwork > 5) ? min_ldwork : 5;

    if (ldwork < min_ldwork) {
        *info = -23;
        return;
    }

    /* Set default tolerances */
    f64 epsmch = SLC_DLAMCH("Epsilon");
    f64 sqreps = sqrt(epsmch);
    f64 toldef = tol;
    f64 cgtdef = cgtol;

    if (toldef < ZERO) {
        toldef = sqreps;
    }
    if (cgtdef <= ZERO) {
        cgtdef = sqreps;
    }

    f64 gsmin = FACTOR * epsmch;
    i32 wrkopt = 5;

    f64 smlnum = SLC_DLAMCH("Safe minimum") / SLC_DLAMCH("Precision");
    f64 bignum = ONE / smlnum;
    SLC_DLABAD(&smlnum, &bignum);

    /* Initialize X if requested */
    if (init) {
        i32 seed[4];
        seed[0] = ((i32)dwork[0]) % 4096;
        seed[1] = ((i32)dwork[1]) % 4096;
        seed[2] = ((i32)dwork[2]) % 4096;
        seed[3] = (2 * ((i32)dwork[3]) + 1) % 4096;
        i32 idist = 2;
        SLC_DLARNV(&idist, seed, &n, x);
    }

    /* Evaluate function at starting point */
    iflag = 1;
    fcn(&iflag, m, n, ipar, lipar, dpar1, ldpar1, dpar2, ldpar2,
        x, &nfevl, &dwork[e], &dwork[jac], &ldj, &dwork[jte],
        &dwork[jw1], ldwork - jw1, &infol);

    if (infol != 0) {
        *info = 1;
        return;
    }

    i32 temp_wrkopt = (i32)dwork[jw1] + jw1;
    wrkopt = (wrkopt > temp_wrkopt) ? wrkopt : temp_wrkopt;
    *nfev = 1;

    i32 int1 = 1;
    f64 fnorm = SLC_DNRM2(&m, &dwork[e], &int1);

    f64 actred = ZERO;
    i32 itercg = 0;
    i32 iter = 0;
    i32 iwarnl = 0;
    f64 par = ZERO;

    if (iflag < 0 || fnorm == ZERO) {
        goto terminate;
    }

    /* Initialize search direction to zero */
    dwork[iw1] = ZERO;
    SLC_DCOPY(&n, &dwork[iw1], &int1, &dwork[iw1], &int1);

    /* Main iteration loop */
    while (1) {
        iter++;

        /* Calculate Jacobian */
        iflag = 2;
        fcn(&iflag, m, n, ipar, lipar, dpar1, ldpar1, dpar2, ldpar2,
            x, &nfevl, &dwork[e], &dwork[jac], &ldj, &dwork[jte],
            &dwork[jw1], ldwork - jw1, &infol);

        if (infol != 0) {
            *info = 2;
            return;
        }

        /* Compute gradient norm */
        f64 gnorm = SLC_DNRM2(&n, &dwork[jte], &int1);

        if (nfevl > 0) {
            *nfev += nfevl;
        }
        *njev += 1;

        if (gnorm <= gsmin) {
            *iwarn = 3;
        }

        if (*iwarn != 0) {
            goto terminate;
        }

        if (iter == 1) {
            temp_wrkopt = (i32)dwork[jw1] + jw1;
            wrkopt = (wrkopt > temp_wrkopt) ? wrkopt : temp_wrkopt;
            f64 sqrt_parmax = sqrt(PARMAX);
            par = (gnorm < sqrt_parmax) ? gnorm : sqrt_parmax;
        }

        if (iflag < 0) {
            goto terminate;
        }

        /* Print callback if requested */
        if (nprint > 0 && ((iter - 1) % nprint) == 0) {
            iflag = 0;
            fcn(&iflag, m, n, ipar, lipar, dpar1, ldpar1, dpar2, ldpar2,
                x, nfev, &dwork[e], &dwork[jac], &ldj, &dwork[jte],
                &dwork[jw1], ldwork - jw1, &infol);
            if (iflag < 0) {
                goto terminate;
            }
        }

        /* Inner loop: try to find step that reduces error */
        while (1) {
            /* Store Levenberg factor in dwork[e] for JPJ */
            dwork[e] = par;

            /* Solve (J'*J + par*I)*p = J'*e */
            if (chol) {
                /* Direct method using MB02XD */
                SLC_DCOPY(&n, &dwork[jte], &int1, &dwork[iw1], &int1);
                i32 nrhs = 1;
                i32 ldpar_one = 1;
                mb02xd("Function", stor, uplo, jpj,
                       &m, &n, &nrhs, ipar, &lipar,
                       &dwork[e], &ldpar_one, &dwork[jac], &ldj,
                       &dwork[iw1], &n, &dwork[dwjtj], &n,
                       &dwork[jwork], &(i32){ldwork - jwork}, &infol);
            } else {
                /* Iterative method using MB02WD */
                i32 cg_itmax = 3 * n;
                f64 cg_tol = cgtdef * gnorm;
                mb02wd("Function", (mb02wd_func)jpj,
                       n, ipar, lipar, &dwork[e], 1, cg_itmax,
                       &dwork[jac], ldj, &dwork[jte], 1,
                       &dwork[iw1], 1, cg_tol,
                       &dwork[jwork], ldwork - jwork, &iwarnl, &infol);
                itercg += (i32)dwork[jwork];
                iwarnl = (2 * iwarnl > iwarnl) ? (2 * iwarnl) : iwarnl;
            }

            if (infol != 0) {
                *info = 3;
                return;
            }

            /* Compute x_new = x - p */
            for (i32 i = 0; i < n; i++) {
                dwork[iw2 + i] = x[i] - dwork[iw1 + i];
            }

            /* Evaluate function at new point */
            iflag = 1;
            fcn(&iflag, m, n, ipar, lipar, dpar1, ldpar1, dpar2, ldpar2,
                &dwork[iw2], &nfevl, &dwork[e], &dwork[jac], &ldj,
                &dwork[jte], &dwork[jw2], ldwork - jw2, &infol);

            if (infol != 0) {
                *info = 1;
                return;
            }

            *nfev += 1;

            if (iflag < 0) {
                goto terminate;
            }

            f64 fnorm1 = SLC_DNRM2(&m, &dwork[e], &int1);

            /* Check if step was successful */
            if (fnorm < fnorm1) {
                /* Unsuccessful step: increase PAR */
                actred = ONE;
                if (par > PARMAX) {
                    if (par / MARQF <= bignum) {
                        par = par * MARQF;
                    }
                } else {
                    par = par * MARQF;
                }
            } else {
                /* Successful step: update PAR, X, and fnorm */
                actred = ONE - (fnorm1 / fnorm) * (fnorm1 / fnorm);

                f64 ddot_val = SLC_DDOT(&n, &dwork[iw1], &int1, &dwork[jte], &int1);

                if ((fnorm - fnorm1) * (fnorm + fnorm1) < MINIMP * ddot_val) {
                    if (par > PARMAX) {
                        if (par / MARQF <= bignum) {
                            par = par * MARQF;
                        }
                    } else {
                        par = par * MARQF;
                    }
                } else {
                    f64 new_par = par / MARQF;
                    par = (new_par > smlnum) ? new_par : smlnum;
                }

                SLC_DCOPY(&n, &dwork[iw2], &int1, x, &int1);
                fnorm = fnorm1;
            }

            /* Check termination conditions */
            if (actred <= toldef || iter > itmax || par > PARMAX) {
                goto terminate;
            }

            if (actred <= epsmch) {
                *iwarn = 4;
                goto terminate;
            }

            /* Break inner loop if successful step */
            if (fnorm >= fnorm1) {
                break;
            }
        }
    }

terminate:
    if (actred > toldef) {
        *iwarn = 1;
    }
    if (iwarnl != 0) {
        *iwarn = 2;
    }
    if (iflag < 0) {
        *iwarn = iflag;
    }

    /* Final print callback */
    if (nprint > 0) {
        iflag = 0;
        fcn(&iflag, m, n, ipar, lipar, dpar1, ldpar1, dpar2, ldpar2,
            x, nfev, &dwork[e], &dwork[jac], &ldj, &dwork[jte],
            &dwork[jw1], ldwork - jw1, &infol);
        if (iflag < 0) {
            *iwarn = iflag;
        }
    }

    /* Set output values */
    dwork[0] = (f64)wrkopt;
    dwork[1] = fnorm;
    dwork[2] = (f64)iter;
    dwork[3] = (f64)itercg;
    dwork[4] = par;
}
