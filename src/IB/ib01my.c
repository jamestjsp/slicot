// SPDX-License-Identifier: BSD-3-Clause
//
// IB01MY - Fast QR factorization for system identification
//
// Constructs the upper triangular factor R of concatenated block Hankel
// matrices using input-output data via fast QR based on displacement rank.

#include "slicot.h"
#include "slicot_blas.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>

void ib01my(const char *meth_str, const char *batch_str, const char *conct_str,
            i32 nobr, i32 m, i32 l, i32 nsmp,
            const f64 *u, i32 ldu, const f64 *y, i32 ldy,
            f64 *r, i32 ldr, i32 *iwork, f64 *dwork, i32 ldwork,
            i32 *iwarn, i32 *info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const i32 MAXCYC = 100;
    const i32 int1 = 1;
    const i32 int0 = 0;

    char meth = toupper((unsigned char)meth_str[0]);
    char batch = toupper((unsigned char)batch_str[0]);
    char conct = toupper((unsigned char)conct_str[0]);

    bool moesp = (meth == 'M');
    bool n4sid = (meth == 'N');
    bool onebch = (batch == 'O');
    bool first = (batch == 'F') || onebch;
    bool interm = (batch == 'I');
    bool last = (batch == 'L') || onebch;
    bool connec = false;
    if (!onebch) {
        connec = (conct == 'C');
    }

    i32 mnobr = m * nobr;
    i32 lnobr = l * nobr;
    i32 mmnobr = mnobr + mnobr;
    i32 llnobr = lnobr + lnobr;
    i32 nobr2 = 2 * nobr;
    i32 nobr21 = nobr2 - 1;
    i32 nr = mmnobr + llnobr;
    i32 nrg = m + l + 1;
    i32 mnrg = m * nrg;
    i32 lnrg = l * nrg;

    *iwarn = 0;
    *info = 0;

    i32 icycle, maxwrk, nsmpsm;
    if (first) {
        icycle = 1;
        maxwrk = 1;
        nsmpsm = 0;
    } else if (!onebch) {
        icycle = iwork[0];
        maxwrk = iwork[1];
        nsmpsm = iwork[2];
    } else {
        icycle = 1;
        maxwrk = 1;
        nsmpsm = 0;
    }
    nsmpsm = nsmpsm + nsmp;

    // Parameter validation
    if (!(moesp || n4sid)) {
        *info = -1;
    } else if (!(first || interm || last)) {
        *info = -2;
    } else if (!onebch) {
        if (!(connec || conct == 'N')) {
            *info = -3;
        }
    }

    if (*info == 0) {
        if (nobr <= 0) {
            *info = -4;
        } else if (m < 0) {
            *info = -5;
        } else if (l <= 0) {
            *info = -6;
        } else if (nsmp < nobr2 || (last && nsmpsm < nr + nobr21)) {
            *info = -7;
        } else if (ldu < 1 || (m > 0 && ldu < nsmp)) {
            *info = -9;
        } else if (ldy < nsmp) {
            *info = -11;
        } else if (ldr < nr) {
            *info = -13;
        } else {
            i32 minwrk;
            if (!onebch && connec) {
                minwrk = nr * (nrg + 2);
            } else if (first || interm) {
                minwrk = nr * nrg;
            } else {
                minwrk = 2 * nr * nrg + nr;
            }
            maxwrk = (minwrk > maxwrk) ? minwrk : maxwrk;

            bool lquery = (ldwork == -1);
            if (lquery) {
                i32 ii = nrg * 2 * nr;
                i32 ierr;
                if (m > 0) {
                    ii = ii + m;
                    f64 work_tmp;
                    SLC_DGEQRF(&nrg, &m, dwork, &nrg, dwork, &work_tmp, &(i32){-1}, &ierr);
                    i32 opt = ii + (i32)work_tmp;
                    if (opt > maxwrk) maxwrk = opt;
                    i32 nrm = nr - m;
                    SLC_DORMQR("Left", "Transpose", &nrg, &nrm, &m, dwork,
                               &nrg, dwork, dwork, &nrg, &work_tmp, &(i32){-1}, &ierr);
                    opt = ii + (i32)work_tmp;
                    if (opt > maxwrk) maxwrk = opt;
                    ii = ii - m;
                }
                ii = ii + l;
                f64 work_tmp;
                SLC_DGEQRF(&nrg, &l, dwork, &nrg, dwork, &work_tmp, &(i32){-1}, &ierr);
                i32 opt = ii + (i32)work_tmp;
                if (opt > maxwrk) maxwrk = opt;
                i32 llnobrl = llnobr - l;
                SLC_DORMQR("Left", "Transpose", &nrg, &llnobrl, &l, dwork,
                           &nrg, dwork, dwork, &nrg, &work_tmp, &(i32){-1}, &ierr);
                opt = ii + (i32)work_tmp;
                if (opt > maxwrk) maxwrk = opt;
                if (moesp && m > 0) {
                    i32 mnobr1 = mnobr - 1;
                    mb04id(mmnobr, mmnobr, mnobr1, llnobr, r, ldr, r, ldr, dwork,
                           &work_tmp, -1, &ierr);
                    opt = mmnobr + (i32)work_tmp;
                    if (opt > maxwrk) maxwrk = opt;
                }
                dwork[0] = (f64)maxwrk;
                return;
            } else if (ldwork < minwrk) {
                *info = -16;
            }
        }
    }

    if (*info != 0) {
        nsmpsm = 0;
        if (!onebch) {
            iwork[0] = 1;
            iwork[1] = maxwrk;
            iwork[2] = nsmpsm;
        }
        if (*info == -16) {
            i32 minwrk;
            if (!onebch && connec) {
                minwrk = nr * (nrg + 2);
            } else if (first || interm) {
                minwrk = nr * nrg;
            } else {
                minwrk = 2 * nr * nrg + nr;
            }
            dwork[0] = (f64)minwrk;
        }
        return;
    }

    // Main algorithm
    i32 ns = nsmp - nobr21;
    i32 nsm = ns - 1;
    i32 ldrwrk = 2 * nobr2;
    f64 upd = first ? ZERO : ONE;
    f64 dum = ZERO;

    i32 ipg = 0;
    i32 ipy = ipg + m;
    i32 ing = ipg + nrg * nr;
    i32 iconn = ing;

    if (!first && connec) {
        i32 irev = iconn + nr;
        i32 icol = iconn + 2 * nr;

        for (i32 i = 1; i < m + l; i++) {
            irev -= nobr2;
            icol -= ldrwrk;
            SLC_DCOPY(&nobr2, &dwork[irev], &int1, &dwork[icol], &int1);
        }

        if (m > 0) {
            SLC_DLACPY("Full", &nobr2, &m, u, &ldu, &dwork[iconn + nobr2], &ldrwrk);
        }
        SLC_DLACPY("Full", &nobr2, &l, y, &ldy, &dwork[iconn + ldrwrk * m + nobr2], &ldrwrk);
    }

    if (m > 0) {
        // Compute Guu correlations
        if (!first && connec) {
            SLC_DSYRK("Upper", "Transpose", &m, &nobr2, &ONE,
                      &dwork[iconn], &ldrwrk, &upd, &dwork[ipg], &nrg);
        }
        SLC_DSYRK("Upper", "Transpose", &m, &nsm, &ONE, u, &ldu, &upd, &dwork[ipg], &nrg);
        ma02ed('U', m, &dwork[ipg], nrg);

        i32 jd = 0;

        if (first || !connec) {
            for (i32 j = 1; j < nobr2; j++) {
                jd += m;
                SLC_DGEMM("Transpose", "NoTranspose", &m, &m, &nsm, &ONE,
                          u, &ldu, &u[j], &ldu, &upd, &dwork[ipg + jd * nrg], &nrg);
            }
        } else {
            for (i32 j = 1; j < nobr2; j++) {
                jd += m;
                SLC_DGEMM("Transpose", "NoTranspose", &m, &m, &nobr2,
                          &ONE, &dwork[iconn], &ldrwrk, &dwork[iconn + j], &ldrwrk,
                          &upd, &dwork[ipg + jd * nrg], &nrg);
                SLC_DGEMM("Transpose", "NoTranspose", &m, &m, &nsm, &ONE,
                          u, &ldu, &u[j], &ldu, &ONE, &dwork[ipg + jd * nrg], &nrg);
            }
        }

        // Compute Guy correlations
        jd = mmnobr;

        if (first || !connec) {
            for (i32 j = 0; j < nobr2; j++) {
                SLC_DGEMM("Transpose", "NoTranspose", &m, &l, &nsm, &ONE,
                          u, &ldu, &y[j], &ldy, &upd, &dwork[ipg + jd * nrg], &nrg);
                jd += l;
            }
        } else {
            for (i32 j = 0; j < nobr2; j++) {
                SLC_DGEMM("Transpose", "NoTranspose", &m, &l, &nobr2,
                          &ONE, &dwork[iconn], &ldrwrk,
                          &dwork[iconn + ldrwrk * m + j], &ldrwrk, &upd,
                          &dwork[ipg + jd * nrg], &nrg);
                SLC_DGEMM("Transpose", "NoTranspose", &m, &l, &nsm, &ONE,
                          u, &ldu, &y[j], &ldy, &ONE, &dwork[ipg + jd * nrg], &nrg);
                jd += l;
            }
        }

        // Transpose Guy(1,1) to Gyu position
        for (i32 j = 0; j < l; j++) {
            SLC_DCOPY(&m, &dwork[ipg + (mmnobr + j) * nrg], &int1, &dwork[ipy + j], &nrg);
        }

        // Compute Gyu correlations
        jd = 0;

        if (first || !connec) {
            for (i32 j = 1; j < nobr2; j++) {
                jd += m;
                SLC_DGEMM("Transpose", "NoTranspose", &l, &m, &nsm, &ONE,
                          y, &ldy, &u[j], &ldu, &upd, &dwork[ipy + jd * nrg], &nrg);
            }
        } else {
            for (i32 j = 1; j < nobr2; j++) {
                jd += m;
                SLC_DGEMM("Transpose", "NoTranspose", &l, &m, &nobr2, &ONE,
                          &dwork[iconn + ldrwrk * m], &ldrwrk, &dwork[iconn + j], &ldrwrk,
                          &upd, &dwork[ipy + jd * nrg], &nrg);
                SLC_DGEMM("Transpose", "NoTranspose", &l, &m, &nsm, &ONE,
                          y, &ldy, &u[j], &ldu, &ONE, &dwork[ipy + jd * nrg], &nrg);
            }
        }
    }

    // Compute Gyy correlations
    i32 jd = mmnobr;

    if (!first && connec) {
        SLC_DSYRK("Upper", "Transpose", &l, &nobr2, &ONE,
                  &dwork[iconn + ldrwrk * m], &ldrwrk, &upd, &dwork[ipy + mmnobr * nrg], &nrg);
    }
    SLC_DSYRK("Upper", "Transpose", &l, &nsm, &ONE, y, &ldy, &upd, &dwork[ipy + mmnobr * nrg], &nrg);
    ma02ed('U', l, &dwork[ipy + mmnobr * nrg], nrg);

    if (first || !connec) {
        for (i32 j = 1; j < nobr2; j++) {
            jd += l;
            SLC_DGEMM("Transpose", "NoTranspose", &l, &l, &nsm, &ONE,
                      y, &ldy, &y[j], &ldy, &upd, &dwork[ipy + jd * nrg], &nrg);
        }
    } else {
        for (i32 j = 1; j < nobr2; j++) {
            jd += l;
            SLC_DGEMM("Transpose", "NoTranspose", &l, &l, &nobr2, &ONE,
                      &dwork[iconn + ldrwrk * m], &ldrwrk,
                      &dwork[iconn + ldrwrk * m + j], &ldrwrk, &upd,
                      &dwork[ipy + jd * nrg], &nrg);
            SLC_DGEMM("Transpose", "NoTranspose", &l, &l, &nsm, &ONE,
                      y, &ldy, &y[j], &ldy, &ONE, &dwork[ipy + jd * nrg], &nrg);
        }
    }

    if (!last) {
        if (first) {
            // Save initial data for last negative generator
            jd = nrg - 1;
            if (m > 0) {
                SLC_DCOPY(&m, &dum, &int0, &dwork[jd], &nrg);
                for (i32 j = 0; j < nobr21; j++) {
                    jd += mnrg;
                    SLC_DCOPY(&m, &u[j], &ldu, &dwork[jd], &nrg);
                }
                jd += mnrg;
            }
            SLC_DCOPY(&l, &dum, &int0, &dwork[jd], &nrg);
            for (i32 j = 0; j < nobr21; j++) {
                jd += lnrg;
                SLC_DCOPY(&l, &y[j], &ldy, &dwork[jd], &nrg);
            }
        }

        if (connec) {
            // Save connection elements (ns is 1-based index)
            if (m > 0) {
                SLC_DLACPY("Full", &nobr2, &m, &u[ns - 1], &ldu, &dwork[iconn], &nobr2);
            }
            SLC_DLACPY("Full", &nobr2, &l, &y[ns - 1], &ldy, &dwork[iconn + mmnobr], &nobr2);
        }

        icycle++;
        iwork[0] = icycle;
        iwork[1] = maxwrk;
        iwork[2] = nsmpsm;
        if (icycle > MAXCYC) {
            *iwarn = 1;
        }
        return;
    }

    // Last batch - compute R factor
    i32 jwork = nrg * 2 * nr;

    // Extract diagonal elements for scaling
    i32 nrgp1 = nrg + 1;
    SLC_DCOPY(&m, &dwork[ipg], &nrgp1, &dwork[jwork], &int1);
    SLC_DCOPY(&l, &dwork[ipy + mmnobr * nrg], &nrgp1, &dwork[jwork + m], &int1);

    // Find scaling order using IDAMAX
    i32 ml = m + l;
    for (i32 i = 0; i < ml; i++) {
        iwork[i] = SLC_IDAMAX(&ml, &dwork[jwork], &int1);
        dwork[jwork + iwork[i] - 1] = ZERO;
    }

    // Scale generators
    for (i32 i = 0; i < ml; i++) {
        i32 imax = iwork[i];
        i32 icol;
        if (imax <= m) {
            icol = imax;
        } else {
            icol = mmnobr - m + imax;
        }
        f64 beta = sqrt(fabs(dwork[ipg + (imax - 1) + (icol - 1) * nrg]));
        if (beta == ZERO) {
            *info = 1;
            return;
        }
        f64 scale = ONE / beta;
        SLC_DSCAL(&nr, &scale, &dwork[ipg + (imax - 1)], &nrg);
        SLC_DCOPY(&nr, &dwork[ipg + (imax - 1)], &nrg, &dwork[ing + (imax - 1)], &nrg);
        dwork[ipg + (imax - 1) + (icol - 1) * nrg] = beta;
        dwork[ing + (imax - 1) + (icol - 1) * nrg] = ZERO;

        for (i32 j = i + 1; j < ml; j++) {
            dwork[ipg + (iwork[j] - 1) + (icol - 1) * nrg] = ZERO;
        }
    }

    // Compute last two generators
    if (!first) {
        SLC_DCOPY(&nr, &dwork[nrg - 1], &nrg, &dwork[ing + nrg - 1], &nrg);
    }

    jd = nrg - 1;
    if (m > 0) {
        for (i32 j = ns - 1; j < nsmp; j++) {
            SLC_DCOPY(&m, &u[j], &ldu, &dwork[jd], &nrg);
            jd += mnrg;
        }
    }
    for (i32 j = ns - 1; j < nsmp; j++) {
        SLC_DCOPY(&l, &y[j], &ldy, &dwork[jd], &nrg);
        jd += lnrg;
    }

    if (first) {
        if (m > 0) {
            SLC_DCOPY(&m, &dum, &int0, &dwork[jd], &nrg);
            for (i32 j = 0; j < nobr21; j++) {
                jd += mnrg;
                SLC_DCOPY(&m, &u[j], &ldu, &dwork[jd], &nrg);
            }
            jd += mnrg;
        }
        SLC_DCOPY(&l, &dum, &int0, &dwork[jd], &nrg);
        for (i32 j = 0; j < nobr21; j++) {
            jd += lnrg;
            SLC_DCOPY(&l, &y[j], &ldy, &dwork[jd], &nrg);
        }
    }

    i32 itau = jwork;
    i32 ipgc = ipg + mmnobr * nrg;
    i32 ierr;

    if (m > 0) {
        // Process input part of generators
        jwork = itau + m;
        i32 ingc = ing;

        // QR factorization of first M columns of positive generators
        i32 ldwork_rem = ldwork - jwork;
        SLC_DGEQRF(&nrg, &m, &dwork[ipg], &nrg, &dwork[itau],
                   &dwork[jwork], &ldwork_rem, &ierr);
        i32 opt = (i32)dwork[jwork] + jwork;
        if (opt > maxwrk) maxwrk = opt;

        // Apply Q^T to remaining columns
        i32 ncols = nr - m;
        SLC_DORMQR("Left", "Transpose", &nrg, &ncols, &m, &dwork[ipg],
                   &nrg, &dwork[itau], &dwork[ipg + mnrg], &nrg,
                   &dwork[jwork], &ldwork_rem, &ierr);
        opt = (i32)dwork[jwork] + jwork;
        if (opt > maxwrk) maxwrk = opt;

        // Annihilate first M columns of negative generators
        // Fortran: DO 210 J = 1, M
        for (i32 j = 0; j < m; j++) {
            f64 tau_val;
            SLC_DLARFG(&nrg, &dwork[ingc], &dwork[ingc + 1], &int1, &tau_val);
            f64 beta = dwork[ingc];
            dwork[ingc] = ONE;
            i32 ingp = ingc + nrg;
            i32 ncols_j = nr - j - 1;
            SLC_DLARF("Left", &nrg, &ncols_j, &dwork[ingc], &int1, &tau_val,
                      &dwork[ingp], &nrg, &dwork[itau]);
            dwork[ingc] = beta;

            // Modified hyperbolic rotation
            f64 cs, sn;
            ma02fd(&dwork[ipg + j * nrgp1], dwork[ingc], &cs, &sn, &ierr);
            if (ierr != 0) {
                *info = 1;
                return;
            }

            for (i32 ii = (j + 1) * nrg; ii <= (nr - 1) * nrg; ii += nrg) {
                f64 temp = (dwork[ipg + j + ii] - sn * dwork[ing + ii]) / cs;
                dwork[ing + ii] = -sn * temp + cs * dwork[ing + ii];
                dwork[ipg + j + ii] = temp;
            }

            ingc = ingp;
        }

        // Save block row of R
        SLC_DLACPY("Upper", &m, &nr, &dwork[ipg], &nrg, r, &ldr);

        // Shift generators for next rows
        for (i32 ii = (mmnobr - m) * nrg; ii >= mnrg; ii -= mnrg) {
            SLC_DLACPY("Full", &m, &m, &dwork[ipg + ii - mnrg], &nrg,
                       &dwork[ipg + ii], &nrg);
        }
        for (i32 ii = (nr - l) * nrg; ii >= (mmnobr + l) * nrg; ii -= lnrg) {
            SLC_DLACPY("Full", &m, &l, &dwork[ipg + ii - lnrg], &nrg,
                       &dwork[ipg + ii], &nrg);
        }
        SLC_DLASET("Full", &m, &l, &ZERO, &ZERO, &dwork[ipgc], &nrg);

        // Update input part using Schur algorithm
        i32 jds = mnrg;
        i32 icol = m;

        for (i32 k = 1; k < nobr2; k++) {
            i32 ncols_k = nr - icol - m;
            i32 lp1 = l + 1;
            mb04od("Full", m, ncols_k, lp1, &dwork[ipg + jds], nrg,
                   &dwork[ipy + jds], nrg, &dwork[ipg + jds + mnrg], nrg,
                   &dwork[ipy + jds + mnrg], nrg, &dwork[itau], &dwork[jwork]);

            for (i32 j = 0; j < m; j++) {
                i32 icj = icol + j;
                f64 tau_val;
                SLC_DLARFG(&nrg, &dwork[ingc], &dwork[ingc + 1], &int1, &tau_val);
                f64 beta = dwork[ingc];
                dwork[ingc] = ONE;
                i32 ingp = ingc + nrg;
                i32 ncols_j = nr - icj - 1;
                SLC_DLARF("Left", &nrg, &ncols_j, &dwork[ingc], &int1, &tau_val,
                          &dwork[ingp], &nrg, &dwork[itau]);
                dwork[ingc] = beta;

                f64 cs, sn;
                ma02fd(&dwork[ipg + j + icj * nrg], dwork[ingc], &cs, &sn, &ierr);
                if (ierr != 0) {
                    *info = 1;
                    return;
                }

                for (i32 ii = (icj + 1) * nrg; ii <= (nr - 1) * nrg; ii += nrg) {
                    f64 temp = (dwork[ipg + j + ii] - sn * dwork[ing + ii]) / cs;
                    dwork[ing + ii] = -sn * temp + cs * dwork[ing + ii];
                    dwork[ipg + j + ii] = temp;
                }

                ingc = ingp;
            }

            // Save block row of R
            i32 ncopy = nr - icol;
            SLC_DLACPY("Upper", &m, &ncopy, &dwork[ipg + jds], &nrg,
                       &r[icol + icol * ldr], &ldr);
            icol += m;

            // Shift generators
            for (i32 ii = (mmnobr - m) * nrg; ii >= icol * nrg; ii -= mnrg) {
                SLC_DLACPY("Full", &m, &m, &dwork[ipg + ii - mnrg], &nrg,
                           &dwork[ipg + ii], &nrg);
            }
            for (i32 ii = (nr - l) * nrg; ii >= (mmnobr + l) * nrg; ii -= lnrg) {
                SLC_DLACPY("Full", &m, &l, &dwork[ipg + ii - lnrg], &nrg,
                           &dwork[ipg + ii], &nrg);
            }
            SLC_DLASET("Full", &m, &l, &ZERO, &ZERO, &dwork[ipgc], &nrg);
            jds += mnrg;
        }
    }

    // Process output part of generators
    jwork = itau + l;
    i32 ingc = ing + mmnobr * nrg;

    // QR factorization of first L columns of output generators
    i32 ldwork_rem = ldwork - jwork;
    SLC_DGEQRF(&nrg, &l, &dwork[ipgc], &nrg, &dwork[itau],
               &dwork[jwork], &ldwork_rem, &ierr);
    i32 opt = (i32)dwork[jwork] + jwork;
    if (opt > maxwrk) maxwrk = opt;

    // Apply Q^T to remaining columns
    i32 ncols = llnobr - l;
    SLC_DORMQR("Left", "Transpose", &nrg, &ncols, &l, &dwork[ipgc],
               &nrg, &dwork[itau], &dwork[ipgc + lnrg], &nrg,
               &dwork[jwork], &ldwork_rem, &ierr);
    opt = (i32)dwork[jwork] + jwork;
    if (opt > maxwrk) maxwrk = opt;

    // Annihilate first L columns of output negative generators
    for (i32 j = 0; j < l; j++) {
        f64 tau_val;
        SLC_DLARFG(&nrg, &dwork[ingc], &dwork[ingc + 1], &int1, &tau_val);
        f64 beta = dwork[ingc];
        dwork[ingc] = ONE;
        i32 ingp = ingc + nrg;
        i32 ncols_j = llnobr - j - 1;
        SLC_DLARF("Left", &nrg, &ncols_j, &dwork[ingc], &int1, &tau_val,
                  &dwork[ingp], &nrg, &dwork[itau]);
        dwork[ingc] = beta;

        f64 cs, sn;
        ma02fd(&dwork[ipgc + j * nrgp1], dwork[ingc], &cs, &sn, &ierr);
        if (ierr != 0) {
            *info = 1;
            return;
        }

        // Fortran: DO 290 I = (J+MMNOBR)*NRG, (NR-1)*NRG, NRG
        for (i32 ii = (j + 1 + mmnobr) * nrg; ii <= (nr - 1) * nrg; ii += nrg) {
            f64 temp = (dwork[ipg + j + ii] - sn * dwork[ing + ii]) / cs;
            dwork[ing + ii] = -sn * temp + cs * dwork[ing + ii];
            dwork[ipg + j + ii] = temp;
        }

        ingc = ingp;
    }

    // Save block row of R
    SLC_DLACPY("Upper", &l, &llnobr, &dwork[ipgc], &nrg, &r[mmnobr + mmnobr * ldr], &ldr);

    // Shift generators
    for (i32 ii = (nr - l) * nrg; ii >= (mmnobr + l) * nrg; ii -= lnrg) {
        SLC_DLACPY("Full", &l, &l, &dwork[ipg + ii - lnrg], &nrg,
                   &dwork[ipg + ii], &nrg);
    }

    // Update output part using Schur algorithm
    i32 jds = lnrg;
    i32 icol = l;

    for (i32 k = 1; k < nobr2; k++) {
        i32 ncols_k = llnobr - icol - l;
        i32 mp1 = m + 1;
        mb04od("Full", l, ncols_k, mp1, &dwork[ipgc + jds], nrg,
               &dwork[ipgc + l + jds], nrg, &dwork[ipgc + jds + lnrg], nrg,
               &dwork[ipgc + l + jds + lnrg], nrg, &dwork[itau], &dwork[jwork]);

        for (i32 j = 0; j < l; j++) {
            i32 icj = icol + j;
            f64 tau_val;
            SLC_DLARFG(&nrg, &dwork[ingc], &dwork[ingc + 1], &int1, &tau_val);
            f64 beta = dwork[ingc];
            dwork[ingc] = ONE;
            i32 ingp = ingc + nrg;
            i32 ncols_j = llnobr - icj - 1;
            SLC_DLARF("Left", &nrg, &ncols_j, &dwork[ingc], &int1, &tau_val,
                      &dwork[ingp], &nrg, &dwork[itau]);
            dwork[ingc] = beta;

            f64 cs, sn;
            ma02fd(&dwork[ipgc + j + icj * nrg], dwork[ingc], &cs, &sn, &ierr);
            if (ierr != 0) {
                *info = 1;
                return;
            }

            // Fortran: DO 320 I = (ICJ+MMNOBR)*NRG, (NR-1)*NRG, NRG
            for (i32 ii = (icj + 1 + mmnobr) * nrg; ii <= (nr - 1) * nrg; ii += nrg) {
                f64 temp = (dwork[ipg + j + ii] - sn * dwork[ing + ii]) / cs;
                dwork[ing + ii] = -sn * temp + cs * dwork[ing + ii];
                dwork[ipg + j + ii] = temp;
            }

            ingc = ingp;
        }

        // Save block row of R
        i32 ncopy = llnobr - icol;
        SLC_DLACPY("Upper", &l, &ncopy, &dwork[ipgc + jds], &nrg,
                   &r[mmnobr + icol + (mmnobr + icol) * ldr], &ldr);

        // Shift generators
        for (i32 ii = (nr - l) * nrg; ii >= (mmnobr + icol) * nrg; ii -= lnrg) {
            SLC_DLACPY("Full", &l, &l, &dwork[ipg + ii - lnrg], &nrg,
                       &dwork[ipg + ii], &nrg);
        }

        icol += l;
        jds += lnrg;
    }

    // MOESP: Interchange past/future input parts and retriangularize
    if (moesp && m > 0) {
        for (i32 j = 0; j < mnobr; j++) {
            i32 len = j + 1;
            SLC_DSWAP(&len, &r[j * ldr], &int1, &r[(mnobr + j) * ldr], &int1);
            SLC_DCOPY(&mnobr, &r[(j + 1) + (mnobr + j) * ldr], &int1, &r[(j + 1) + j * ldr], &int1);
            i32 zerolen = mmnobr - j - 1;
            SLC_DCOPY(&zerolen, &dum, &int0, &r[(j + 1) + (mnobr + j) * ldr], &int1);
        }

        i32 itau_mb = 0;
        i32 jwork_mb = itau_mb + mmnobr;
        i32 mnobr1 = mnobr - 1;
        i32 ldwork_mb = ldwork - jwork_mb;

        mb04id(mmnobr, mmnobr, mnobr1, llnobr, r, ldr, &r[mmnobr * ldr], ldr,
               &dwork[itau_mb], &dwork[jwork_mb], ldwork_mb, &ierr);

        i32 opt_mb = (i32)dwork[jwork_mb] + jwork_mb;
        if (opt_mb > maxwrk) maxwrk = opt_mb;
    }

    nsmpsm = 0;
    icycle = 1;

    dwork[0] = (f64)maxwrk;
    maxwrk = 1;
    if (!onebch) {
        iwork[0] = icycle;
        iwork[1] = maxwrk;
        iwork[2] = nsmpsm;
    }
}
