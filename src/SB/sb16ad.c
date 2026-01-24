/**
 * @file sb16ad.c
 * @brief Frequency-weighted controller reduction via balancing.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 2025, slicot.c contributors
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <string.h>
#include <stdbool.h>

void sb16ad(
    const char* dico,
    const char* jobc,
    const char* jobo,
    const char* jobmr,
    const char* weight,
    const char* equil,
    const char* ordsel,
    const i32 n,
    const i32 m,
    const i32 p,
    const i32 nc,
    i32* ncr,
    const f64 alpha,
    f64* a,
    const i32 lda,
    f64* b,
    const i32 ldb,
    f64* c,
    const i32 ldc,
    const f64* d,
    const i32 ldd,
    f64* ac,
    const i32 ldac,
    f64* bc,
    const i32 ldbc,
    f64* cc,
    const i32 ldcc,
    f64* dc,
    const i32 lddc,
    i32* ncs,
    f64* hsvc,
    const f64 tol1,
    const f64 tol2,
    i32* iwork,
    f64* dwork,
    const i32 ldwork,
    i32* iwarn,
    i32* info)
{
    const f64 C100 = 100.0;
    const f64 ONE = 1.0;

    *info = 0;
    *iwarn = 0;

    bool discr = (*dico == 'D' || *dico == 'd');
    bool conti = (*dico == 'C' || *dico == 'c');
    bool bta = (*jobmr == 'B' || *jobmr == 'b' || *jobmr == 'F' || *jobmr == 'f');
    bool spa = (*jobmr == 'S' || *jobmr == 's' || *jobmr == 'P' || *jobmr == 'p');
    bool fixord = (*ordsel == 'F' || *ordsel == 'f');
    bool istab = (*weight == 'I' || *weight == 'i');
    bool ostab = (*weight == 'O' || *weight == 'o');
    bool perf = (*weight == 'P' || *weight == 'p');
    bool leftw = ostab || perf;
    bool rightw = istab || perf;
    bool frwght = leftw || rightw;

    i32 nnc = n + nc;
    i32 mp = m + p;
    i32 lw;

    if (frwght) {
        i32 max_nnc_m_p = nnc;
        if (m > max_nnc_m_p) max_nnc_m_p = m;
        if (p > max_nnc_m_p) max_nnc_m_p = p;
        i32 lw1 = nnc * (nnc + max_nnc_m_p + 7);
        i32 lw2 = mp * (mp + 4);
        lw = nnc * (nnc + 2 * mp) + (lw1 > lw2 ? lw1 : lw2);
    } else {
        i32 max_m_p = m > p ? m : p;
        lw = nc * (max_m_p + 5);
        if (*equil == 'S' || *equil == 's') {
            if (n > lw) lw = n;
        }
    }
    i32 lsqred = 2 * nc * nc + 5 * nc;
    if (lsqred < 1) lsqred = 1;
    i32 lfreq = lw;
    lw = 2 * nc * nc + (lfreq > lsqred ? lfreq : lsqred);
    if (lw < 1) lw = 1;

    if (!conti && !discr) {
        *info = -1;
    } else if (!(*jobc == 'S' || *jobc == 's' || *jobc == 'E' || *jobc == 'e')) {
        *info = -2;
    } else if (!(*jobo == 'S' || *jobo == 's' || *jobo == 'E' || *jobo == 'e')) {
        *info = -3;
    } else if (!bta && !spa) {
        *info = -4;
    } else if (!(frwght || *weight == 'N' || *weight == 'n')) {
        *info = -5;
    } else if (!(*equil == 'S' || *equil == 's' || *equil == 'N' || *equil == 'n')) {
        *info = -6;
    } else if (!(fixord || *ordsel == 'A' || *ordsel == 'a')) {
        *info = -7;
    } else if (n < 0) {
        *info = -8;
    } else if (m < 0) {
        *info = -9;
    } else if (p < 0) {
        *info = -10;
    } else if (nc < 0) {
        *info = -11;
    } else if (fixord && (*ncr < 0 || *ncr > nc)) {
        *info = -12;
    } else if ((discr && (alpha < 0.0 || alpha > 1.0)) ||
               (!discr && alpha > 0.0)) {
        *info = -13;
    } else if (lda < (1 > n ? 1 : n)) {
        *info = -15;
    } else if (ldb < (1 > n ? 1 : n)) {
        *info = -17;
    } else if (ldc < (1 > p ? 1 : p)) {
        *info = -19;
    } else if (ldd < (1 > p ? 1 : p)) {
        *info = -21;
    } else if (ldac < (1 > nc ? 1 : nc)) {
        *info = -23;
    } else if (ldbc < (1 > nc ? 1 : nc)) {
        *info = -25;
    } else if (ldcc < (1 > m ? 1 : m)) {
        *info = -27;
    } else if (lddc < (1 > m ? 1 : m)) {
        *info = -29;
    } else if (tol2 > 0.0 && !fixord && tol2 > tol1) {
        *info = -33;
    } else if (ldwork < lw) {
        *info = -36;
    }

    if (*info != 0) {
        return;
    }

    i32 min_val = nc;
    if (m < min_val) min_val = m;
    if (p < min_val) min_val = p;
    if (min_val == 0) {
        *ncr = 0;
        *ncs = 0;
        iwork[0] = 0;
        dwork[0] = ONE;
        return;
    }

    if (*equil == 'S' || *equil == 's') {
        f64 maxred = C100;
        i32 tb01id_info;
        tb01id("All", n, m, p, &maxred, a, lda, b, ldb, c, ldc, dwork, &tb01id_info);
        maxred = C100;
        tb01id("All", nc, p, m, &maxred, ac, ldac, bc, ldbc, cc, ldcc, dwork, &tb01id_info);
    }

    f64 alpwrk = alpha;
    f64 eps_sqrt = sqrt(SLC_DLAMCH("E"));
    if (discr) {
        if (alpha == 1.0) alpwrk = 1.0 - eps_sqrt;
    } else {
        if (alpha == 0.0) alpwrk = -eps_sqrt;
    }

    i32 wrkopt = 1;
    i32 ku = 0;
    i32 kr = ku + nc * nc;
    i32 ki = kr + nc;
    i32 kw = ki + nc;

    i32 ncu;
    i32 tb01kd_info;
    tb01kd(dico, "Unstable", "General", nc, p, m, alpwrk,
           ac, ldac, bc, ldbc, cc, ldcc, &ncu, &dwork[ku], nc,
           &dwork[kr], &dwork[ki], &dwork[kw], ldwork - kw, &tb01kd_info);

    if (tb01kd_info != 0) {
        if (tb01kd_info != 3) {
            *info = 5;
        } else {
            *info = 6;
        }
        return;
    }
    i32 opt_tb01kd = (i32)dwork[kw] + kw;
    if (opt_tb01kd > wrkopt) wrkopt = opt_tb01kd;

    i32 iwarnl = 0;
    *ncs = nc - ncu;
    i32 nra;
    if (fixord) {
        nra = *ncr - ncu;
        if (nra < 0) nra = 0;
        if (*ncr < ncu) iwarnl = 3;
    } else {
        nra = 0;
    }

    if (*ncs == 0) {
        *ncr = ncu;
        iwork[0] = 0;
        dwork[0] = (f64)wrkopt;
        return;
    }

    i32 kt = 0;
    i32 kti = kt + nc * nc;
    kw = kti + nc * nc;

    f64 scalec, scaleo;
    i32 sb16ay_info;
    sb16ay(dico, jobc, jobo, weight, n, m, p, nc, *ncs,
           a, lda, b, ldb, c, ldc, d, ldd,
           ac, ldac, bc, ldbc, cc, ldcc, dc, lddc,
           &scalec, &scaleo, &dwork[kti], nc, &dwork[kt], nc,
           iwork, &dwork[kw], ldwork - kw, &sb16ay_info);
    if (sb16ay_info != 0) {
        *info = sb16ay_info;
        return;
    }
    i32 opt_sb16ay = (i32)dwork[kw] + kw;
    if (opt_sb16ay > wrkopt) wrkopt = opt_sb16ay;

    i32 ncu1 = ncu;
    i32 nmr;
    i32 ab09ix_iwarn;
    i32 ab09ix_info = ab09ix(dico, jobmr, "Schur", ordsel, *ncs, p, m, &nra, scalec,
                             scaleo, &ac[ncu1 + ncu1 * ldac], ldac, &bc[ncu1], ldbc,
                             &cc[ncu1 * ldcc], ldcc, dc, lddc, &dwork[kti], nc,
                             &dwork[kt], nc, &nmr, hsvc, tol1, tol2,
                             iwork, &dwork[kw], ldwork - kw, &ab09ix_iwarn);

    if (ab09ix_iwarn > *iwarn) *iwarn = ab09ix_iwarn;
    if (iwarnl > *iwarn) *iwarn = iwarnl;

    if (ab09ix_info != 0) {
        *info = 7;
        return;
    }
    *ncr = nra + ncu;
    iwork[0] = nmr;

    i32 opt_ab09ix = (i32)dwork[kw] + kw;
    if (opt_ab09ix > wrkopt) wrkopt = opt_ab09ix;
    dwork[0] = (f64)wrkopt;
}
