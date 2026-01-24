/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

/*
 * AB09JD - Frequency-weighted Hankel-norm approximation with invertible weights
 *
 * Purpose:
 *   To compute a reduced order model (Ar,Br,Cr,Dr) for an original
 *   state-space representation (A,B,C,D) by using the frequency
 *   weighted optimal Hankel-norm approximation method.
 *   The Hankel norm of the weighted error
 *
 *         op(V)*(G-Gr)*op(W)
 *
 *   is minimized, where G and Gr are the transfer-function matrices
 *   of the original and reduced systems, respectively, V and W are
 *   invertible transfer-function matrices representing the left and
 *   right frequency weights, and op(X) denotes X, inv(X), conj(X) or
 *   conj(inv(X)).
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <stdlib.h>

#define max2(a, b) ((a) > (b) ? (a) : (b))
#define max3(a, b, c) max2(max2(a, b), c)
#define min2(a, b) ((a) < (b) ? (a) : (b))

void ab09jd(
    const char* jobv,
    const char* jobw,
    const char* jobinv,
    const char* dico,
    const char* equil,
    const char* ordsel,
    i32 n,
    i32 nv,
    i32 nw,
    i32 m,
    i32 p,
    i32* nr,
    f64 alpha,
    f64* a,
    i32 lda,
    f64* b,
    i32 ldb,
    f64* c,
    i32 ldc,
    f64* d,
    i32 ldd,
    f64* av,
    i32 ldav,
    f64* bv,
    i32 ldbv,
    f64* cv,
    i32 ldcv,
    f64* dv,
    i32 lddv,
    f64* aw,
    i32 ldaw,
    f64* bw,
    i32 ldbw,
    f64* cw,
    i32 ldcw,
    f64* dw,
    i32 lddw,
    i32* ns,
    f64* hsv,
    f64 tol1,
    f64 tol2,
    i32* iwork,
    f64* dwork,
    i32 ldwork,
    i32* iwarn,
    i32* info)
{
    const f64 C100 = 100.0;
    const f64 ONE = 1.0;
    const f64 P0001 = 0.0001;
    const f64 ZERO = 0.0;

    bool autom, conjv, conjw, discr, fixord, invfr, lefti, leftw, righti, rightw;
    char jobvl, jobwl;
    i32 ierr, iwarnl, kav, kaw, kbv, kbw, kcv, kcw, kdv, kdw, kev, kew;
    i32 ki, kl, ku, kw, ldabv, ldabw, ldcdv, ldcdw, lw, nra, nu, nu1, nvp, nwm, rank;
    f64 alpwrk, maxred, rcond, sqreps, tol, wrkopt;
    f64 temp[1];

    char jobv_c = jobv[0];
    char jobw_c = jobw[0];
    char jobinv_c = jobinv[0];
    char dico_c = dico[0];
    char equil_c = equil[0];
    char ordsel_c = ordsel[0];

    *info = 0;
    *iwarn = 0;

    discr = (dico_c == 'D' || dico_c == 'd');
    fixord = (ordsel_c == 'F' || ordsel_c == 'f');
    lefti = (jobv_c == 'I' || jobv_c == 'i' || jobv_c == 'R' || jobv_c == 'r');
    leftw = (jobv_c == 'V' || jobv_c == 'v' || jobv_c == 'C' || jobv_c == 'c' || lefti);
    conjv = (jobv_c == 'C' || jobv_c == 'c' || jobv_c == 'R' || jobv_c == 'r');
    righti = (jobw_c == 'I' || jobw_c == 'i' || jobw_c == 'R' || jobw_c == 'r');
    rightw = (jobw_c == 'W' || jobw_c == 'w' || jobw_c == 'C' || jobw_c == 'c' || righti);
    conjw = (jobw_c == 'C' || jobw_c == 'c' || jobw_c == 'R' || jobw_c == 'r');
    invfr = (jobinv_c == 'N' || jobinv_c == 'n');
    autom = (jobinv_c == 'A' || jobinv_c == 'a');

    nvp = nv + p;
    nwm = nw + m;

    lw = 1;
    if (leftw) {
        i32 tmp1 = 2 * nvp * (nvp + p) + p * p;
        i32 tmp2a = 2 * nvp * nvp + max2(11 * nvp + 16, p * nvp);
        i32 tmp2b = nvp * n + max3(nvp * n + n * n, p * n, p * m);
        lw = max2(lw, tmp1 + max2(tmp2a, tmp2b));
    }
    if (rightw) {
        i32 tmp1 = 2 * nwm * (nwm + m) + m * m;
        i32 tmp2a = 2 * nwm * nwm + max2(11 * nwm + 16, m * nwm);
        i32 tmp2b = nwm * n + max3(nwm * n + n * n, m * n, p * m);
        lw = max2(lw, tmp1 + max2(tmp2a, tmp2b));
    }
    lw = max2(lw, n * (2 * n + max3(n, m, p) + 5) + (n * (n + 1)) / 2);
    lw = max2(lw, n * (m + p + 2) + 2 * m * p + min2(n, m) + max2(3 * m + 1, min2(n, m) + p));

    bool jobv_n = (jobv_c == 'N' || jobv_c == 'n');
    bool jobw_n = (jobw_c == 'N' || jobw_c == 'n');
    bool dico_c_valid = (dico_c == 'C' || dico_c == 'c');
    bool equil_s = (equil_c == 'S' || equil_c == 's');
    bool equil_n = (equil_c == 'N' || equil_c == 'n');
    bool ordsel_a = (ordsel_c == 'A' || ordsel_c == 'a');
    bool jobinv_i = (jobinv_c == 'I' || jobinv_c == 'i');

    if (!jobv_n && !leftw) {
        *info = -1;
    } else if (!jobw_n && !rightw) {
        *info = -2;
    } else if (!invfr && !autom && !jobinv_i) {
        *info = -3;
    } else if (!dico_c_valid && !discr) {
        *info = -4;
    } else if (!equil_s && !equil_n) {
        *info = -5;
    } else if (!fixord && !ordsel_a) {
        *info = -6;
    } else if (n < 0) {
        *info = -7;
    } else if (nv < 0) {
        *info = -8;
    } else if (nw < 0) {
        *info = -9;
    } else if (m < 0) {
        *info = -10;
    } else if (p < 0) {
        *info = -11;
    } else if (fixord && (*nr < 0 || *nr > n)) {
        *info = -12;
    } else if ((discr && (alpha < ZERO || alpha > ONE)) ||
               (!discr && alpha > ZERO)) {
        *info = -13;
    } else if (lda < max2(1, n)) {
        *info = -15;
    } else if (ldb < max2(1, n)) {
        *info = -17;
    } else if (ldc < max2(1, p)) {
        *info = -19;
    } else if (ldd < max2(1, p)) {
        *info = -21;
    } else if (ldav < 1 || (leftw && ldav < nv)) {
        *info = -23;
    } else if (ldbv < 1 || (leftw && ldbv < nv)) {
        *info = -25;
    } else if (ldcv < 1 || (leftw && ldcv < p)) {
        *info = -27;
    } else if (lddv < 1 || (leftw && lddv < p)) {
        *info = -29;
    } else if (ldaw < 1 || (rightw && ldaw < nw)) {
        *info = -31;
    } else if (ldbw < 1 || (rightw && ldbw < nw)) {
        *info = -33;
    } else if (ldcw < 1 || (rightw && ldcw < m)) {
        *info = -35;
    } else if (lddw < 1 || (rightw && lddw < m)) {
        *info = -37;
    } else if (tol1 >= ONE) {
        *info = -40;
    } else if ((tol2 > ZERO && !fixord && tol2 > tol1) || tol2 >= ONE) {
        *info = -41;
    } else if (ldwork < lw) {
        *info = -44;
    }

    if (*info != 0) {
        i32 neginfo = -(*info);
        SLC_XERBLA("AB09JD", &neginfo);
        return;
    }

    if (min2(n, min2(m, p)) == 0) {
        *nr = 0;
        *ns = 0;
        dwork[0] = ONE;
        return;
    }

    if (equil_s) {
        maxred = C100;
        tb01id("All", n, m, p, &maxred, a, lda, b, ldb, c, ldc, dwork, info);
    }

    alpwrk = alpha;
    sqreps = sqrt(SLC_DLAMCH("E"));
    if (discr) {
        if (alpha == ONE) alpwrk = ONE - sqreps;
    } else {
        if (alpha == ZERO) alpwrk = -sqreps;
    }

    ku = 0;
    kl = ku + n * n;
    ki = kl + n;
    kw = ki + n;

    tb01kd(dico, "Unstable", "General", n, m, p, alpwrk, a, lda,
           b, ldb, c, ldc, &nu, &dwork[ku], n, &dwork[kl],
           &dwork[ki], &dwork[kw], ldwork - kw, &ierr);

    if (ierr != 0) {
        if (ierr != 3) {
            *info = 1;
        } else {
            *info = 2;
        }
        return;
    }

    wrkopt = dwork[kw] + (f64)(kw);
    iwarnl = 0;

    *ns = n - nu;
    if (fixord) {
        nra = max2(0, *nr - nu);
        if (*nr < nu)
            iwarnl = 2;
    } else {
        nra = 0;
    }

    if (*ns == 0) {
        *nr = nu;
        dwork[0] = wrkopt;
        return;
    }

    nu1 = nu;
    if (conjv) {
        jobvl = 'C';
    } else {
        jobvl = 'V';
    }
    if (conjw) {
        jobwl = 'C';
    } else {
        jobwl = 'W';
    }

    if (leftw) {
        tol = ZERO;
        ab08md("S", nv, p, p, av, ldav, bv, ldbv, cv, ldcv,
               dv, lddv, &rank, tol, iwork, dwork, ldwork, &ierr);
        if (rank != p) {
            *info = 20;
            return;
        }
        wrkopt = max2(wrkopt, dwork[0]);

        if (lefti) {
            if (invfr) {
                ierr = 1;
            } else {
                kav = 0;
                kbv = kav + nv * nv;
                kcv = kbv + nv * p;
                kdv = kcv + p * nv;
                kw = kdv + p * p;

                ldabv = max2(nv, 1);
                ldcdv = p;
                SLC_DLACPY("Full", &nv, &nv, av, &ldav, &dwork[kav], &ldabv);
                SLC_DLACPY("Full", &nv, &p, bv, &ldbv, &dwork[kbv], &ldabv);
                SLC_DLACPY("Full", &p, &nv, cv, &ldcv, &dwork[kcv], &ldcdv);
                SLC_DLACPY("Full", &p, &p, dv, &lddv, &dwork[kdv], &ldcdv);

                ierr = ab07nd(nv, p, &dwork[kav], ldabv, &dwork[kbv], ldabv,
                              &dwork[kcv], ldcdv, &dwork[kdv], ldcdv,
                              &rcond, iwork, &dwork[kw], ldwork - kw);
                wrkopt = max2(wrkopt, dwork[kw] + (f64)(kw));

                if (autom) {
                    if (ierr == 0 && rcond <= P0001) ierr = 1;
                } else {
                    if (ierr == 0 && rcond <= sqreps) ierr = 1;
                }
                if (ierr != 0 && nv == 0) {
                    *info = 20;
                    return;
                }
            }

            if (ierr != 0) {
                kav = 0;
                kev = kav + nvp * nvp;
                kbv = kev + nvp * nvp;
                kcv = kbv + nvp * p;
                kdv = kcv + p * nvp;
                kw = kdv + p * p;

                ldabv = max2(nvp, 1);
                ldcdv = p;

                ag07bd("I", nv, p, av, ldav, temp, 1, bv, ldbv,
                       cv, ldcv, dv, lddv, &dwork[kav], ldabv,
                       &dwork[kev], ldabv, &dwork[kbv], ldabv,
                       &dwork[kcv], ldcdv, &dwork[kdv], ldcdv, &ierr);

                ab09jv(&jobvl, dico, "G", "C", *ns, m, p, nvp, p,
                       &a[nu1 + nu1 * lda], lda, &b[nu1], ldb,
                       &c[nu1 * ldc], ldc, d, ldd,
                       &dwork[kav], ldabv, &dwork[kev], ldabv,
                       &dwork[kbv], ldabv, &dwork[kcv], ldcdv,
                       &dwork[kdv], ldcdv, iwork, &dwork[kw],
                       ldwork - kw, &ierr);
                if (ierr != 0) {
                    if (ierr == 1) {
                        *info = 5;
                    } else if (ierr == 2) {
                        *info = 16;
                    } else if (ierr == 4) {
                        *info = 18;
                    }
                    return;
                }
            } else {
                ab09jv(&jobvl, dico, "I", "C", *ns, m, p, nv, p,
                       &a[nu1 + nu1 * lda], lda, &b[nu1], ldb,
                       &c[nu1 * ldc], ldc, d, ldd, &dwork[kav], ldabv,
                       temp, 1, &dwork[kbv], ldabv,
                       &dwork[kcv], ldcdv, &dwork[kdv], ldcdv, iwork,
                       &dwork[kw], ldwork - kw, &ierr);
                if (ierr != 0) {
                    if (ierr == 1) {
                        *info = 10;
                    } else if (ierr == 3) {
                        *info = 14;
                    } else if (ierr == 4) {
                        *info = 18;
                    }
                    return;
                }
            }

            wrkopt = max2(wrkopt, dwork[kw] + (f64)(kw - 1));
        } else {
            ab09jv(&jobvl, dico, "I", "C", *ns, m, p, nv, p,
                   &a[nu1 + nu1 * lda], lda, &b[nu1], ldb,
                   &c[nu1 * ldc], ldc, d, ldd, av, ldav,
                   temp, 1, bv, ldbv, cv, ldcv, dv, lddv, iwork,
                   dwork, ldwork, &ierr);
            if (ierr != 0) {
                if (ierr == 1) {
                    *info = 3;
                } else if (ierr == 3) {
                    *info = 12;
                } else if (ierr == 4) {
                    *info = 18;
                }
                return;
            }

            wrkopt = max2(wrkopt, dwork[0]);
        }
    }

    if (rightw) {
        tol = ZERO;
        ab08md("S", nw, m, m, aw, ldaw, bw, ldbw, cw, ldcw,
               dw, lddw, &rank, tol, iwork, dwork, ldwork, &ierr);
        if (rank != m) {
            *info = 21;
            return;
        }
        wrkopt = max2(wrkopt, dwork[0]);

        if (righti) {
            if (invfr) {
                ierr = 1;
            } else {
                kaw = 0;
                kbw = kaw + nw * nw;
                kcw = kbw + nw * m;
                kdw = kcw + m * nw;
                kw = kdw + m * m;

                ldabw = max2(nw, 1);
                ldcdw = m;
                SLC_DLACPY("Full", &nw, &nw, aw, &ldaw, &dwork[kaw], &ldabw);
                SLC_DLACPY("Full", &nw, &m, bw, &ldbw, &dwork[kbw], &ldabw);
                SLC_DLACPY("Full", &m, &nw, cw, &ldcw, &dwork[kcw], &ldcdw);
                SLC_DLACPY("Full", &m, &m, dw, &lddw, &dwork[kdw], &ldcdw);

                ierr = ab07nd(nw, m, &dwork[kaw], ldabw, &dwork[kbw], ldabw,
                              &dwork[kcw], ldcdw, &dwork[kdw], ldcdw,
                              &rcond, iwork, &dwork[kw], ldwork - kw);
                wrkopt = max2(wrkopt, dwork[kw] + (f64)(kw));

                if (autom) {
                    if (ierr == 0 && rcond <= P0001) ierr = 1;
                } else {
                    if (ierr == 0 && rcond <= sqreps) ierr = 1;
                }
                if (ierr != 0 && nw == 0) {
                    *info = 21;
                    return;
                }
            }

            if (ierr != 0) {
                kaw = 0;
                kew = kaw + nwm * nwm;
                kbw = kew + nwm * nwm;
                kcw = kbw + nwm * m;
                kdw = kcw + m * nwm;
                kw = kdw + m * m;

                ldabw = max2(nwm, 1);
                ldcdw = m;

                ag07bd("I", nw, m, aw, ldaw, temp, 1, bw, ldbw,
                       cw, ldcw, dw, lddw, &dwork[kaw], ldabw,
                       &dwork[kew], ldabw, &dwork[kbw], ldabw,
                       &dwork[kcw], ldcdw, &dwork[kdw], ldcdw, &ierr);

                ab09jw(&jobwl, dico, "G", "C", *ns, m, p, nwm, m,
                       &a[nu1 + nu1 * lda], lda, &b[nu1], ldb,
                       &c[nu1 * ldc], ldc, d, ldd, &dwork[kaw], ldabw,
                       &dwork[kew], ldabw, &dwork[kbw], ldabw,
                       &dwork[kcw], ldcdw, &dwork[kdw], ldcdw,
                       iwork, &dwork[kw], ldwork - kw, &ierr);
                if (ierr != 0) {
                    if (ierr == 1) {
                        *info = 6;
                    } else if (ierr == 2) {
                        *info = 17;
                    } else if (ierr == 4) {
                        *info = 19;
                    }
                    return;
                }
            } else {
                ab09jw(&jobwl, dico, "I", "C", *ns, m, p, nw, m,
                       &a[nu1 + nu1 * lda], lda, &b[nu1], ldb,
                       &c[nu1 * ldc], ldc, d, ldd, &dwork[kaw], ldabw,
                       temp, 1, &dwork[kbw], ldabw,
                       &dwork[kcw], ldcdw, &dwork[kdw], ldcdw,
                       iwork, &dwork[kw], ldwork - kw, &ierr);
                if (ierr != 0) {
                    if (ierr == 1) {
                        *info = 11;
                    } else if (ierr == 3) {
                        *info = 15;
                    } else if (ierr == 4) {
                        *info = 19;
                    }
                    return;
                }
            }

            wrkopt = max2(wrkopt, dwork[kw] + (f64)(kw - 1));
        } else {
            ab09jw(&jobwl, dico, "I", "C", *ns, m, p, nw, m,
                   &a[nu1 + nu1 * lda], lda, &b[nu1], ldb, &c[nu1 * ldc], ldc,
                   d, ldd, aw, ldaw, temp, 1, bw, ldbw, cw, ldcw,
                   dw, lddw, iwork, dwork, ldwork, &ierr);
            if (ierr != 0) {
                if (ierr == 1) {
                    *info = 4;
                } else if (ierr == 3) {
                    *info = 13;
                } else if (ierr == 4) {
                    *info = 19;
                }
                return;
            }

            wrkopt = max2(wrkopt, dwork[0]);
        }
    }

    ab09cx(dico, ordsel, *ns, m, p, &nra, &a[nu1 + nu1 * lda], lda,
           &b[nu1], ldb, &c[nu1 * ldc], ldc, d, ldd, hsv, tol1,
           tol2, iwork, dwork, ldwork, iwarn, &ierr);

    if (ierr != 0) {
        *info = ierr + 5;
        return;
    }

    *iwarn = max2(iwarnl, *iwarn);
    wrkopt = max2(wrkopt, dwork[0]);

    if (leftw) {
        if (!lefti) {
            if (invfr) {
                ierr = 1;
            } else {
                kav = 0;
                kbv = kav + nv * nv;
                kcv = kbv + nv * p;
                kdv = kcv + p * nv;
                kw = kdv + p * p;

                ldabv = max2(nv, 1);
                ldcdv = p;
                SLC_DLACPY("Full", &nv, &nv, av, &ldav, &dwork[kav], &ldabv);
                SLC_DLACPY("Full", &nv, &p, bv, &ldbv, &dwork[kbv], &ldabv);
                SLC_DLACPY("Full", &p, &nv, cv, &ldcv, &dwork[kcv], &ldcdv);
                SLC_DLACPY("Full", &p, &p, dv, &lddv, &dwork[kdv], &ldcdv);

                ierr = ab07nd(nv, p, &dwork[kav], ldabv, &dwork[kbv], ldabv,
                              &dwork[kcv], ldcdv, &dwork[kdv], ldcdv,
                              &rcond, iwork, &dwork[kw], ldwork - kw);
                wrkopt = max2(wrkopt, dwork[kw] + (f64)(kw));

                if (autom) {
                    if (ierr == 0 && rcond <= P0001) ierr = 1;
                } else {
                    if (ierr == 0 && rcond <= sqreps) ierr = 1;
                }
                if (ierr != 0 && nv == 0) {
                    *info = 20;
                    return;
                }
            }

            if (ierr != 0) {
                kav = 0;
                kev = kav + nvp * nvp;
                kbv = kev + nvp * nvp;
                kcv = kbv + nvp * p;
                kdv = kcv + p * nvp;
                kw = kdv + p * p;

                ldabv = max2(nvp, 1);
                ldcdv = p;

                ag07bd("I", nv, p, av, ldav, temp, 1, bv, ldbv,
                       cv, ldcv, dv, lddv, &dwork[kav], ldabv,
                       &dwork[kev], ldabv, &dwork[kbv], ldabv,
                       &dwork[kcv], ldcdv, &dwork[kdv], ldcdv, &ierr);

                ab09jv(&jobvl, dico, "G", "N", nra, m, p, nvp, p,
                       &a[nu1 + nu1 * lda], lda, &b[nu1], ldb,
                       &c[nu1 * ldc], ldc, d, ldd,
                       &dwork[kav], ldabv, &dwork[kev], ldabv,
                       &dwork[kbv], ldabv, &dwork[kcv], ldcdv,
                       &dwork[kdv], ldcdv, iwork, &dwork[kw],
                       ldwork - kw, &ierr);
                if (ierr != 0) {
                    if (ierr == 1) {
                        *info = 5;
                    } else if (ierr == 2) {
                        *info = 16;
                    }
                    return;
                }
            } else {
                ab09jv(&jobvl, dico, "I", "N", nra, m, p, nv, p,
                       &a[nu1 + nu1 * lda], lda, &b[nu1], ldb,
                       &c[nu1 * ldc], ldc, d, ldd, &dwork[kav], ldabv,
                       temp, 1, &dwork[kbv], ldabv,
                       &dwork[kcv], ldcdv, &dwork[kdv], ldcdv, iwork,
                       &dwork[kw], ldwork - kw, &ierr);
                if (ierr != 0) {
                    if (ierr == 1) {
                        *info = 10;
                    } else if (ierr == 3) {
                        *info = 14;
                    }
                    return;
                }
            }

            wrkopt = max2(wrkopt, dwork[kw] + (f64)(kw - 1));
        } else {
            ab09jv(&jobvl, dico, "I", "N", nra, m, p, nv, p,
                   &a[nu1 + nu1 * lda], lda, &b[nu1], ldb,
                   &c[nu1 * ldc], ldc, d, ldd, av, ldav,
                   temp, 1, bv, ldbv, cv, ldcv, dv, lddv, iwork,
                   dwork, ldwork, &ierr);
            if (ierr != 0) {
                if (ierr == 1) {
                    *info = 3;
                } else if (ierr == 3) {
                    *info = 12;
                }
                return;
            }

            wrkopt = max2(wrkopt, dwork[0]);
        }
    }

    if (rightw) {
        if (!righti) {
            if (invfr) {
                ierr = 1;
            } else {
                kaw = 0;
                kbw = kaw + nw * nw;
                kcw = kbw + nw * m;
                kdw = kcw + m * nw;
                kw = kdw + m * m;

                ldabw = max2(nw, 1);
                ldcdw = m;
                SLC_DLACPY("Full", &nw, &nw, aw, &ldaw, &dwork[kaw], &ldabw);
                SLC_DLACPY("Full", &nw, &m, bw, &ldbw, &dwork[kbw], &ldabw);
                SLC_DLACPY("Full", &m, &nw, cw, &ldcw, &dwork[kcw], &ldcdw);
                SLC_DLACPY("Full", &m, &m, dw, &lddw, &dwork[kdw], &ldcdw);

                ierr = ab07nd(nw, m, &dwork[kaw], ldabw, &dwork[kbw], ldabw,
                              &dwork[kcw], ldcdw, &dwork[kdw], ldcdw,
                              &rcond, iwork, &dwork[kw], ldwork - kw);
                wrkopt = max2(wrkopt, dwork[kw] + (f64)(kw));

                if (autom) {
                    if (ierr == 0 && rcond <= P0001) ierr = 1;
                } else {
                    if (ierr == 0 && rcond <= sqreps) ierr = 1;
                }
                if (ierr != 0 && nw == 0) {
                    *info = 21;
                    return;
                }
            }

            if (ierr != 0) {
                kaw = 0;
                kew = kaw + nwm * nwm;
                kbw = kew + nwm * nwm;
                kcw = kbw + nwm * m;
                kdw = kcw + m * nwm;
                kw = kdw + m * m;

                ldabw = max2(nwm, 1);
                ldcdw = m;

                ag07bd("I", nw, m, aw, ldaw, temp, 1, bw, ldbw,
                       cw, ldcw, dw, lddw, &dwork[kaw], ldabw,
                       &dwork[kew], ldabw, &dwork[kbw], ldabw,
                       &dwork[kcw], ldcdw, &dwork[kdw], ldcdw, &ierr);

                ab09jw(&jobwl, dico, "G", "N", nra, m, p, nwm, m,
                       &a[nu1 + nu1 * lda], lda, &b[nu1], ldb,
                       &c[nu1 * ldc], ldc, d, ldd, &dwork[kaw], ldabw,
                       &dwork[kew], ldabw, &dwork[kbw], ldabw,
                       &dwork[kcw], ldcdw, &dwork[kdw], ldcdw,
                       iwork, &dwork[kw], ldwork - kw, &ierr);
                if (ierr != 0) {
                    if (ierr == 1) {
                        *info = 6;
                    } else if (ierr == 2) {
                        *info = 17;
                    }
                    return;
                }
            } else {
                ab09jw(&jobwl, dico, "I", "N", nra, m, p, nw, m,
                       &a[nu1 + nu1 * lda], lda, &b[nu1], ldb,
                       &c[nu1 * ldc], ldc, d, ldd, &dwork[kaw], ldabw,
                       temp, 1, &dwork[kbw], ldabw,
                       &dwork[kcw], ldcdw, &dwork[kdw], ldcdw,
                       iwork, &dwork[kw], ldwork - kw, &ierr);
                if (ierr != 0) {
                    if (ierr == 1) {
                        *info = 11;
                    } else if (ierr == 3) {
                        *info = 15;
                    }
                    return;
                }
            }

            wrkopt = max2(wrkopt, dwork[kw] + (f64)(kw - 1));
        } else {
            ab09jw(&jobwl, dico, "I", "N", nra, m, p, nw, m,
                   &a[nu1 + nu1 * lda], lda, &b[nu1], ldb, &c[nu1 * ldc], ldc,
                   d, ldd, aw, ldaw, temp, 1, bw, ldbw, cw, ldcw,
                   dw, lddw, iwork, dwork, ldwork, &ierr);

            if (ierr != 0) {
                if (ierr == 1) {
                    *info = 4;
                } else if (ierr == 3) {
                    *info = 13;
                }
                return;
            }

            wrkopt = max2(wrkopt, dwork[0]);
        }
    }

    *nr = nra + nu;
    dwork[0] = wrkopt;
}

#undef max2
#undef max3
#undef min2
