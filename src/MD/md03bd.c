/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>

static bool lsame(char ca, char cb) {
    if (ca >= 'a' && ca <= 'z') ca -= 32;
    if (cb >= 'a' && cb <= 'z') cb -= 32;
    return ca == cb;
}

void md03bd(
    const char* xinit,
    const char* scale,
    const char* cond,
    void (*fcn)(i32*, i32, i32, i32*, i32, const f64*, i32, const f64*, i32,
                const f64*, i32*, f64*, f64*, i32*, f64*, i32, i32*),
    void (*qrfact)(i32, const i32*, i32, f64, f64*, i32*, f64*, f64*, f64*,
                   i32*, f64*, i32, i32*),
    void (*lmparm)(const char*, i32, const i32*, i32, f64*, i32, const i32*,
                   const f64*, const f64*, f64, f64*, i32*, f64*, f64*, f64,
                   f64*, i32, i32*),
    i32 m,
    i32 n,
    i32 itmax,
    f64 factor,
    i32 nprint,
    i32* ipar,
    i32 lipar,
    const f64* dpar1,
    i32 ldpar1,
    const f64* dpar2,
    i32 ldpar2,
    f64* x,
    f64* diag,
    i32* nfev,
    i32* njev,
    f64 ftol,
    f64 xtol,
    f64 gtol,
    f64 tol,
    i32* iwork,
    f64* dwork,
    i32 ldwork,
    i32* iwarn,
    i32* info
)
{
    const f64 zero = 0.0;
    const f64 one = 1.0;
    const f64 four = 4.0;
    const f64 p1 = 0.1;
    const f64 p5 = 0.5;
    const f64 p25 = 0.25;
    const f64 p75 = 0.75;
    const f64 p0001 = 0.0001;

    bool init, iscal, sscal, badscl;
    i32 e, iflag, infol, iter, iw1, iw2, iw3, j, jac, jw1, jw2, jwork, l;
    i32 ldj, ldjsav, lfcn1, lfcn2, llmp, lqrf, nc, nfevl, sizej, wrkopt;
    f64 actred, delta, dirder, epsmch, fnorm, fnorm1, ftdef, gnorm, gtdef;
    f64 par, pnorm, prered, ratio, temp, temp1, temp2, toldef, xnorm, xtdef;

    init = lsame(*xinit, 'R');
    iscal = lsame(*scale, 'I');
    sscal = lsame(*scale, 'S');
    *info = 0;
    *iwarn = 0;

    if (!init && !lsame(*xinit, 'G')) {
        *info = -1;
    } else if (!iscal && !sscal) {
        *info = -2;
    } else if (!lsame(*cond, 'E') && !lsame(*cond, 'N')) {
        *info = -3;
    } else if (m < 0) {
        *info = -7;
    } else if (n < 0 || n > m) {
        *info = -8;
    } else if (itmax < 0) {
        *info = -9;
    } else if (factor <= zero) {
        *info = -10;
    } else if (lipar < 5) {
        *info = -13;
    } else if (ldpar1 < 0) {
        *info = -15;
    } else if (ldpar2 < 0) {
        *info = -17;
    } else if (ldwork < 4) {
        *info = -28;
    } else if (sscal) {
        badscl = false;
        for (j = 0; j < n; j++) {
            if (diag[j] <= zero) {
                badscl = true;
                break;
            }
        }
        if (badscl) {
            *info = -19;
        }
    }

    if (*info != 0) {
        return;
    }

    *nfev = 0;
    *njev = 0;

    if (n == 0) {
        dwork[0] = four;
        dwork[1] = zero;
        dwork[2] = zero;
        dwork[3] = zero;
        return;
    }

    iflag = 3;
    i32 iw1_save = ipar[0];
    i32 iw2_save = ipar[1];
    i32 iw3_save = ipar[2];
    i32 jw1_save = ipar[3];
    i32 jw2_save = ipar[4];

    fcn(&iflag, m, n, ipar, lipar, dpar1, ldpar1, dpar2, ldpar2,
        x, &nfevl, dwork, dwork, &ldjsav, dwork, ldwork, &infol);

    sizej = ipar[0];
    lfcn1 = ipar[1];
    lfcn2 = ipar[2];
    lqrf = ipar[3];
    llmp = ipar[4];

    if (ldjsav > 0) {
        nc = sizej / ldjsav;
    } else {
        nc = sizej;
    }

    ipar[0] = iw1_save;
    ipar[1] = iw2_save;
    ipar[2] = iw3_save;
    ipar[3] = jw1_save;
    ipar[4] = jw2_save;

    e = 0;
    jac = e + m;
    jw1 = jac + sizej;
    jw2 = jw1 + n;
    iw1 = jac + n*nc;
    iw2 = iw1 + n;
    iw3 = iw2 + n;
    jwork = iw2 + m;

    l = 4;
    i32 temp_max1 = (lfcn1 > lfcn2) ? lfcn1 : lfcn2;
    temp_max1 = (temp_max1 > (n + lqrf)) ? temp_max1 : (n + lqrf);
    i32 temp_max2 = (m + lfcn1 > n + llmp) ? (m + lfcn1) : (n + llmp);
    i32 temp_max3 = (n*nc + n + temp_max2 > sizej + temp_max1) ?
                    (n*nc + n + temp_max2) : (sizej + temp_max1);
    l = (m + temp_max3 > l) ? (m + temp_max3) : l;

    if (ldwork < l) {
        *info = -28;
        return;
    }

    epsmch = SLC_DLAMCH("Epsilon");
    ftdef = ftol;
    xtdef = xtol;
    gtdef = gtol;
    toldef = tol;

    f64 min_tol = (ftdef < xtdef) ? ftdef : xtdef;
    min_tol = (min_tol < gtdef) ? min_tol : gtdef;
    min_tol = (min_tol < toldef) ? min_tol : toldef;

    if (min_tol <= zero) {
        if (ftdef < zero) ftdef = sqrt(epsmch);
        if (xtdef < zero) xtdef = sqrt(epsmch);
        if (gtdef < zero) gtdef = epsmch;
        if (toldef <= zero) toldef = (f64)n * epsmch;
    }

    wrkopt = 1;

    if (init) {
        i32 seed[4];
        seed[0] = ((i32)dwork[0]) % 4096;
        seed[1] = ((i32)dwork[1]) % 4096;
        seed[2] = ((i32)dwork[2]) % 4096;
        seed[3] = (2 * ((i32)dwork[3]) + 1) % 4096;
        i32 idist = 2;  // Uniform distribution on (-1, 1)
        SLC_DLARNV(&idist, seed, &n, x);
    }

    par = zero;
    iter = 1;

    iflag = 1;
    fcn(&iflag, m, n, ipar, lipar, dpar1, ldpar1, dpar2, ldpar2,
        x, &nfevl, &dwork[e], &dwork[jac], &ldj, &dwork[jw1],
        ldwork - jw1, &infol);

    if (infol != 0) {
        *info = 1;
        return;
    }

    wrkopt = ((i32)dwork[jw1] + jw1 > wrkopt) ? ((i32)dwork[jw1] + jw1) : wrkopt;
    *nfev = 1;

    i32 m_int = m;
    i32 int1 = 1;
    fnorm = SLC_DNRM2(&m_int, &dwork[e], &int1);

    if (iflag < 0 || fnorm == zero) {
        goto L90;
    }

    while (true) {
        ldj = ldjsav;
        iflag = 2;
        fcn(&iflag, m, n, ipar, lipar, dpar1, ldpar1, dpar2, ldpar2,
            x, &nfevl, &dwork[e], &dwork[jac], &ldj, &dwork[jw1],
            ldwork - jw1, &infol);

        if (infol != 0) {
            *info = 2;
            return;
        }

        if (iter == 1) {
            wrkopt = ((i32)dwork[jw1] + jw1 > wrkopt) ?
                     ((i32)dwork[jw1] + jw1) : wrkopt;
        }

        if (nfevl > 0) {
            *nfev += nfevl;
        }

        *njev += 1;

        if (iflag < 0) {
            goto L90;
        }

        if (nprint > 0) {
            if ((iter - 1) % nprint == 0) {
                iflag = 0;
                fcn(&iflag, m, n, ipar, lipar, dpar1, ldpar1, dpar2, ldpar2,
                    x, nfev, &dwork[e], &dwork[jac], &ldj, &dwork[jw1],
                    ldwork - jw1, &infol);
                if (iflag < 0) {
                    goto L90;
                }
            }
        }

        qrfact(n, ipar, lipar, fnorm, &dwork[jac], &ldj, &dwork[e],
               &dwork[jw1], &gnorm, iwork, &dwork[jw2], ldwork - jw2, &infol);

        if (infol != 0) {
            *info = 3;
            return;
        }

        if (iter == 1) {
            wrkopt = ((i32)dwork[jw2] + jw2 > wrkopt) ?
                     ((i32)dwork[jw2] + jw2) : wrkopt;

            if (iscal) {
                for (j = 0; j < n; j++) {
                    diag[j] = dwork[jw1 + j];
                    if (diag[j] == zero) {
                        diag[j] = one;
                    }
                }
            }

            for (j = 0; j < n; j++) {
                dwork[iw1 + j] = diag[j] * x[j];
            }

            i32 n_int = n;
            xnorm = SLC_DNRM2(&n_int, &dwork[iw1], &int1);
            delta = factor * xnorm;
            if (delta == zero) {
                delta = factor;
            }
        } else {
            if (iscal) {
                for (j = 0; j < n; j++) {
                    diag[j] = (diag[j] > dwork[jw1 + j]) ? diag[j] : dwork[jw1 + j];
                }
            }
        }

        if (gnorm <= gtdef) {
            *iwarn = 4;
        }

        if (*iwarn != 0) {
            goto L90;
        }

        while (true) {
            lmparm(cond, n, ipar, lipar, &dwork[jac], ldj, iwork, diag,
                   &dwork[e], delta, &par, &iwork[n], &dwork[iw1], &dwork[iw2],
                   toldef, &dwork[iw3], ldwork - iw3, &infol);

            if (infol != 0) {
                *info = 4;
                return;
            }

            if (iter == 1) {
                wrkopt = ((i32)dwork[iw3] + iw3 > wrkopt) ?
                         ((i32)dwork[iw3] + iw3) : wrkopt;
            }

            i32 n_int = n;
            temp1 = SLC_DNRM2(&n_int, &dwork[iw2], &int1) / fnorm;

            for (j = 0; j < n; j++) {
                dwork[iw2 + j] = diag[j] * dwork[iw1 + j];
                dwork[iw1 + j] = x[j] - dwork[iw1 + j];
            }

            pnorm = SLC_DNRM2(&n_int, &dwork[iw2], &int1);
            temp2 = (sqrt(par) * pnorm) / fnorm;
            prered = temp1*temp1 + temp2*temp2 / p5;
            dirder = -(temp1*temp1 + temp2*temp2);

            if (iter == 1) {
                delta = (delta < pnorm) ? delta : pnorm;
            }

            iflag = 1;
            fcn(&iflag, m, n, ipar, lipar, dpar1, ldpar1, dpar2, ldpar2,
                &dwork[iw1], &nfevl, &dwork[iw2], &dwork[jac], &ldj,
                &dwork[jwork], ldwork - jwork, &infol);

            if (infol != 0) {
                *info = 1;
                return;
            }

            *nfev += 1;

            if (iflag < 0) {
                goto L90;
            }

            fnorm1 = SLC_DNRM2(&m_int, &dwork[iw2], &int1);

            actred = -one;
            if (p1 * fnorm1 < fnorm) {
                actred = one - (fnorm1 / fnorm) * (fnorm1 / fnorm);
            }

            ratio = zero;
            if (prered != zero) {
                ratio = actred / prered;
            }

            if (ratio <= p25) {
                if (actred >= zero) {
                    temp = p5;
                } else {
                    temp = p5 * dirder / (dirder + p5 * actred);
                }

                if (p1 * fnorm1 >= fnorm || temp < p1) {
                    temp = p1;
                }

                delta = temp * ((delta < pnorm / p1) ? delta : (pnorm / p1));
                par = par / temp;
            } else {
                if (par == zero || ratio >= p75) {
                    delta = pnorm / p5;
                    par = p5 * par;
                }
            }

            if (ratio >= p0001) {
                for (j = 0; j < n; j++) {
                    x[j] = dwork[iw1 + j];
                    dwork[iw1 + j] = diag[j] * x[j];
                }

                SLC_DCOPY(&m_int, &dwork[iw2], &int1, &dwork[e], &int1);
                xnorm = SLC_DNRM2(&n_int, &dwork[iw1], &int1);
                fnorm = fnorm1;
                iter++;
            }

            if (fabs(actred) <= ftdef && prered <= ftdef && p5 * ratio <= one) {
                *iwarn = 1;
            }

            if (delta <= xtdef * xnorm) {
                *iwarn = 2;
            }

            if (fabs(actred) <= ftdef && prered <= ftdef &&
                p5 * ratio <= one && *iwarn == 2) {
                *iwarn = 3;
            }

            if (*iwarn != 0) {
                goto L90;
            }

            if (iter >= itmax) {
                *iwarn = 5;
            }

            if (fabs(actred) <= epsmch && prered <= epsmch && p5 * ratio <= one) {
                *iwarn = 6;
            }

            if (delta <= epsmch * xnorm) {
                *iwarn = 7;
            }

            if (gnorm <= epsmch) {
                *iwarn = 8;
            }

            if (*iwarn != 0) {
                goto L90;
            }

            if (ratio >= p0001) {
                break;
            }
        }
    }

L90:
    if (iflag < 0) {
        *iwarn = iflag;
    }

    if (nprint > 0) {
        iflag = 0;
        fcn(&iflag, m, n, ipar, lipar, dpar1, ldpar1, dpar2, ldpar2,
            x, nfev, &dwork[e], &dwork[jac], &ldj, &dwork[jwork],
            ldwork - jwork, &infol);
        if (iflag < 0) {
            *iwarn = iflag;
        }
    }

    if (*iwarn >= 0) {
        for (j = m + n*nc - 1; j >= 0; j--) {
            dwork[4 + j] = dwork[j];
        }
    }

    dwork[0] = (f64)wrkopt;
    dwork[1] = fnorm;
    dwork[2] = (f64)iter;
    dwork[3] = par;
}
