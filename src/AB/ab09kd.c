/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

/*
 * AB09KD - Frequency-weighted Hankel-norm approximation model reduction
 *
 * Purpose:
 *   To compute a reduced order model (Ar,Br,Cr,Dr) for an original
 *   state-space representation (A,B,C,D) by using the frequency
 *   weighted optimal Hankel-norm approximation method.
 *   The Hankel norm of the weighted error
 *
 *         V*(G-Gr)*W    or    conj(V)*(G-Gr)*conj(W)
 *
 *   is minimized, where G and Gr are the transfer-function matrices
 *   of the original and reduced systems, respectively.
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <stdlib.h>

#define max2(a, b) ((a) > (b) ? (a) : (b))
#define max3(a, b, c) max2(max2(a, b), c)
#define min2(a, b) ((a) < (b) ? (a) : (b))

void ab09kd(
    const char* job,
    const char* dico,
    const char* weight,
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
    const f64 ZERO = 0.0;

    bool conjs, discr, fixord, frwght, leftw, rightw;
    i32 ia, ib, ierr, iwarnl, ki, kl, ku, kw, lw, nmin, nra, nu, nu1;
    f64 alpwrk, maxred, rcond, wrkopt;

    char job_c = job[0];
    char dico_c = dico[0];
    char weight_c = weight[0];
    char equil_c = equil[0];
    char ordsel_c = ordsel[0];

    conjs = (job_c == 'C' || job_c == 'c');
    discr = (dico_c == 'D' || dico_c == 'd');
    fixord = (ordsel_c == 'F' || ordsel_c == 'f');
    leftw = (weight_c == 'L' || weight_c == 'l' || weight_c == 'B' || weight_c == 'b');
    rightw = (weight_c == 'R' || weight_c == 'r' || weight_c == 'B' || weight_c == 'b');
    frwght = leftw || rightw;

    if (discr && conjs) {
        ia = 2 * nv;
        ib = 2 * nw;
    } else {
        ia = 0;
        ib = 0;
    }

    lw = 1;
    if (leftw) {
        i32 tmp1 = nv * (nv + 5);
        i32 tmp2 = nv * n + max3(ia, p * n, p * m);
        lw = max2(lw, max2(tmp1, tmp2));
    }
    if (rightw) {
        i32 tmp1 = nw * (nw + 5);
        i32 tmp2 = nw * n + max3(ib, m * n, p * m);
        lw = max2(lw, max2(tmp1, tmp2));
    }
    lw = max2(lw, n * (2 * n + max3(n, m, p) + 5) + (n * (n + 1)) / 2);
    lw = max2(lw, n * (m + p + 2) + 2 * m * p + min2(n, m) +
                  max2(3 * m + 1, min2(n, m) + p));

    *info = 0;
    *iwarn = 0;

    bool job_n = (job_c == 'N' || job_c == 'n');
    bool dico_c_valid = (dico_c == 'C' || dico_c == 'c');
    bool weight_n = (weight_c == 'N' || weight_c == 'n');
    bool equil_s = (equil_c == 'S' || equil_c == 's');
    bool equil_n = (equil_c == 'N' || equil_c == 'n');
    bool ordsel_a = (ordsel_c == 'A' || ordsel_c == 'a');

    if (!job_n && !conjs) {
        *info = -1;
    } else if (!dico_c_valid && !discr) {
        *info = -2;
    } else if (!frwght && !weight_n) {
        *info = -3;
    } else if (!equil_s && !equil_n) {
        *info = -4;
    } else if (!fixord && !ordsel_a) {
        *info = -5;
    } else if (n < 0) {
        *info = -6;
    } else if (nv < 0) {
        *info = -7;
    } else if (nw < 0) {
        *info = -8;
    } else if (m < 0) {
        *info = -9;
    } else if (p < 0) {
        *info = -10;
    } else if (fixord && (*nr < 0 || *nr > n)) {
        *info = -11;
    } else if ((discr && (alpha < ZERO || alpha > ONE)) ||
               (!discr && alpha > ZERO)) {
        *info = -12;
    } else if (lda < max2(1, n)) {
        *info = -14;
    } else if (ldb < max2(1, n)) {
        *info = -16;
    } else if (ldc < max2(1, p)) {
        *info = -18;
    } else if (ldd < max2(1, p)) {
        *info = -20;
    } else if (ldav < 1 || (leftw && ldav < nv)) {
        *info = -22;
    } else if (ldbv < 1 || (leftw && ldbv < nv)) {
        *info = -24;
    } else if (ldcv < 1 || (leftw && ldcv < p)) {
        *info = -26;
    } else if (lddv < 1 || (leftw && lddv < p)) {
        *info = -28;
    } else if (ldaw < 1 || (rightw && ldaw < nw)) {
        *info = -30;
    } else if (ldbw < 1 || (rightw && ldbw < nw)) {
        *info = -32;
    } else if (ldcw < 1 || (rightw && ldcw < m)) {
        *info = -34;
    } else if (lddw < 1 || (rightw && lddw < m)) {
        *info = -36;
    } else if (tol2 > ZERO && !fixord && tol2 > tol1) {
        *info = -40;
    } else if (ldwork < lw) {
        *info = -43;
    }

    if (*info != 0) {
        i32 neginfo = -(*info);
        SLC_XERBLA("AB09KD", &neginfo);
        return;
    }

    if (min2(n, min2(m, p)) == 0) {
        *nr = 0;
        *ns = 0;
        iwork[0] = 0;
        dwork[0] = ONE;
        return;
    }

    if (equil_s) {
        maxred = C100;
        tb01id("All", n, m, p, &maxred, a, lda, b, ldb, c, ldc, dwork, info);
    }

    alpwrk = alpha;
    if (discr) {
        if (alpha == ONE) {
            alpwrk = ONE - sqrt(SLC_DLAMCH("E"));
        }
    } else {
        if (alpha == ZERO) {
            alpwrk = -sqrt(SLC_DLAMCH("E"));
        }
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

    *ns = n - nu;

    if (*ns == 0) {
        *nr = nu;
        iwork[0] = 0;
        dwork[0] = wrkopt;
        return;
    }

    nu1 = nu;

    if (frwght) {
        ab09kx(job, dico, weight, *ns, nv, nw, m, p,
               &a[nu1 + nu1 * lda], lda, &b[nu1], ldb, &c[nu1 * ldc], ldc, d, ldd,
               av, ldav, bv, ldbv, cv, ldcv, dv, lddv,
               aw, ldaw, bw, ldbw, cw, ldcw, dw, lddw,
               dwork, ldwork, &iwarnl, &ierr);

        if (ierr != 0) {
            *info = ierr + 2;
            return;
        }

        if (iwarnl != 0) {
            if (iwarnl == 1 || iwarnl == 3) {
                *info = 5;
            } else {
                *info = 6;
            }
            return;
        }

        if (dwork[0] > wrkopt) {
            wrkopt = dwork[0];
        }
    }

    iwarnl = 0;
    if (fixord) {
        nra = (*nr > nu) ? (*nr - nu) : 0;
        if (nra == 0) {
            iwarnl = 2;
        }
    } else {
        nra = 0;
    }

    ab09cx(dico, ordsel, *ns, m, p, &nra, &a[nu1 + nu1 * lda], lda,
           &b[nu1], ldb, &c[nu1 * ldc], ldc, d, ldd, hsv, tol1,
           tol2, iwork, dwork, ldwork, iwarn, &ierr);

    if (*iwarn < iwarnl) {
        *iwarn = iwarnl;
    }

    if (ierr != 0) {
        *info = ierr + 5;
        return;
    }

    if (dwork[0] > wrkopt) {
        wrkopt = dwork[0];
    }
    nmin = iwork[0];

    if (leftw) {
        ierr = ab07nd(nv, p, av, ldav, bv, ldbv, cv, ldcv, dv, lddv,
                      &rcond, iwork, dwork, ldwork);
        if (ierr != 0) {
            *info = 10;
            return;
        }
    }

    if (rightw) {
        ierr = ab07nd(nw, m, aw, ldaw, bw, ldbw, cw, ldcw, dw, lddw,
                      &rcond, iwork, dwork, ldwork);
        if (ierr != 0) {
            *info = 11;
            return;
        }
    }

    if (dwork[0] > wrkopt) {
        wrkopt = dwork[0];
    }

    if (frwght) {
        ab09kx(job, dico, weight, nra, nv, nw, m, p,
               &a[nu1 + nu1 * lda], lda, &b[nu1], ldb, &c[nu1 * ldc], ldc, d, ldd,
               av, ldav, bv, ldbv, cv, ldcv, dv, lddv,
               aw, ldaw, bw, ldbw, cw, ldcw, dw, lddw,
               dwork, ldwork, &iwarnl, &ierr);

        if (ierr != 0) {
            if (ierr <= 2) {
                *info = ierr + 2;
            } else {
                *info = ierr + 9;
            }
            return;
        }
    }

    *nr = nra + nu;
    iwork[0] = nmin;
    dwork[0] = wrkopt;
}

#undef max2
#undef max3
#undef min2
