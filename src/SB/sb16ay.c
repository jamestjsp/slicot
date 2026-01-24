/**
 * @file sb16ay.c
 * @brief Cholesky factors of frequency-weighted Grammians for controller reduction.
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>

void sb16ay(
    const char* dico,
    const char* jobc,
    const char* jobo,
    const char* weight,
    const i32 n,
    const i32 m,
    const i32 p,
    const i32 nc,
    const i32 ncs,
    const f64* a,
    const i32 lda,
    const f64* b,
    const i32 ldb,
    const f64* c,
    const i32 ldc,
    const f64* d,
    const i32 ldd,
    const f64* ac,
    const i32 ldac,
    const f64* bc,
    const i32 ldbc,
    const f64* cc,
    const i32 ldcc,
    const f64* dc,
    const i32 lddc,
    f64* scalec,
    f64* scaleo,
    f64* s,
    const i32 lds,
    f64* r,
    const i32 ldr,
    i32* iwork,
    f64* dwork,
    const i32 ldwork,
    i32* info)
{
    const f64 zero = 0.0;
    const f64 one = 1.0;
    const f64 negone = -1.0;
    const i32 int1 = 1;

    bool discr = (*dico == 'D' || *dico == 'd');
    bool leftw = (*weight == 'O' || *weight == 'o');
    bool rightw = (*weight == 'I' || *weight == 'i');
    bool perf = (*weight == 'P' || *weight == 'p');
    bool frwght = leftw || rightw || perf;

    *info = 0;
    i32 nnc = n + nc;
    i32 mp = m + p;
    i32 lw;

    if (frwght) {
        i32 max_nnc_m_p = nnc > m ? (nnc > p ? nnc : p) : (m > p ? m : p);
        lw = nnc * (nnc + 2 * mp) +
             (nnc * (nnc + max_nnc_m_p + 7) > mp * (mp + 4) ?
              nnc * (nnc + max_nnc_m_p + 7) : mp * (mp + 4));
    } else {
        lw = ncs * ((m > p ? m : p) + 5);
    }
    lw = lw > 1 ? lw : 1;

    if (!(*dico == 'C' || *dico == 'c' || discr)) {
        *info = -1;
    } else if (!(*jobc == 'S' || *jobc == 's' || *jobc == 'E' || *jobc == 'e')) {
        *info = -2;
    } else if (!(*jobo == 'S' || *jobo == 's' || *jobo == 'E' || *jobo == 'e')) {
        *info = -3;
    } else if (!(frwght || *weight == 'N' || *weight == 'n')) {
        *info = -4;
    } else if (n < 0) {
        *info = -5;
    } else if (m < 0) {
        *info = -6;
    } else if (p < 0) {
        *info = -7;
    } else if (nc < 0) {
        *info = -8;
    } else if (ncs < 0 || ncs > nc) {
        *info = -9;
    } else if (lda < (1 > n ? 1 : n)) {
        *info = -11;
    } else if (ldb < (1 > n ? 1 : n)) {
        *info = -13;
    } else if (ldc < (1 > p ? 1 : p)) {
        *info = -15;
    } else if (ldd < (1 > p ? 1 : p)) {
        *info = -17;
    } else if (ldac < (1 > nc ? 1 : nc)) {
        *info = -19;
    } else if (ldbc < (1 > nc ? 1 : nc)) {
        *info = -21;
    } else if (ldcc < (1 > m ? 1 : m)) {
        *info = -23;
    } else if (lddc < (1 > m ? 1 : m)) {
        *info = -25;
    } else if (lds < (1 > ncs ? 1 : ncs)) {
        *info = -29;
    } else if (ldr < (1 > ncs ? 1 : ncs)) {
        *info = -31;
    } else if (ldwork < lw) {
        *info = -34;
    }

    if (*info != 0) {
        return;
    }

    *scalec = one;
    *scaleo = one;
    i32 min_val = ncs < m ? ncs : m;
    min_val = min_val < p ? min_val : p;
    if (min_val == 0) {
        dwork[0] = one;
        return;
    }

    i32 wrkopt = 1;
    i32 ncu = nc - ncs;
    i32 ncu1 = ncu;  // 0-based index (Fortran NCU1 = NCU + 1)

    i32 ku, ktau, kw, kwa, kwb, kwc, kwd, kl;
    i32 kq, kr, ki, ldu;
    i32 ierr;
    i32 ne, me, pe;
    f64 rcond;
    char jobfac;
    f64 tol, t;

    if (!perf) {
        if (leftw || (*weight == 'N' || *weight == 'n')) {
            ku = 0;
            ktau = ku + ncs * p;
            kw = ktau + ncs;

            SLC_DLACPY("Full", &ncs, &p, &bc[ncu1 + 0 * ldbc], &ldbc,
                       &dwork[ku], &ncs);
            sb03ou(discr, true, ncs, p, &ac[ncu1 + ncu1 * ldac], ldac,
                   &dwork[ku], ncs, &dwork[ktau], s, lds, scalec,
                   &dwork[kw], ldwork - kw, &ierr);
            if (ierr != 0) {
                *info = 5;
                return;
            }
            wrkopt = wrkopt > ((i32)dwork[kw] + kw) ? wrkopt : ((i32)dwork[kw] + kw);
        }

        if (rightw || (*weight == 'N' || *weight == 'n')) {
            ku = 0;
            ktau = ku + m * ncs;
            kw = ktau + ncs;

            SLC_DLACPY("Full", &m, &ncs, &cc[0 + ncu1 * ldcc], &ldcc,
                       &dwork[ku], &m);
            sb03ou(discr, false, ncs, m, &ac[ncu1 + ncu1 * ldac], ldac,
                   &dwork[ku], m, &dwork[ktau], r, ldr, scaleo,
                   &dwork[kw], ldwork - kw, &ierr);
            if (ierr != 0) {
                *info = 5;
                return;
            }
            wrkopt = wrkopt > ((i32)dwork[kw] + kw) ? wrkopt : ((i32)dwork[kw] + kw);
        }

        if (*weight == 'N' || *weight == 'n') {
            dwork[0] = (f64)wrkopt;
            return;
        }
    }

    if (frwght) {
        kwa = 0;
        kwb = kwa + nnc * nnc;
        kwc = kwb + nnc * mp;
        kwd = kwc + nnc * mp;
        kw = kwd + mp * mp;
        kl = kwd;

        if (leftw) {
            SLC_DLASET("Full", &m, &p, &zero, &zero, &dwork[kwd], &mp);
            ne = ab05pd('N', ncs, p, m, ncu, one,
                        &ac[ncu1 + ncu1 * ldac], ldac, &bc[ncu1 + 0 * ldbc], ldbc,
                        &cc[0 + ncu1 * ldcc], ldcc, &dwork[kwd], mp,
                        ac, ldac, bc, ldbc, cc, ldcc, dc, lddc,
                        &ne, &dwork[kwa], nnc, &dwork[kwb], nnc,
                        &dwork[kwc], mp, &dwork[kwd], mp);

            (void)ab05qd('O', nc, p, m, n, m, p,
                         &dwork[kwa], nnc, &dwork[kwb], nnc, &dwork[kwc], mp, &dwork[kwd], mp,
                         a, lda, b, ldb, c, ldc, d, ldd,
                         &ne, &me, &pe,
                         &dwork[kwa], nnc, &dwork[kwb], nnc,
                         &dwork[kwc], mp, &dwork[kwd], mp);

            SLC_DLASET("Full", &m, &m, &zero, &negone, &dwork[kwd + mp * p], &mp);
            SLC_DLASET("Full", &p, &p, &zero, &negone, &dwork[kwd + m], &mp);
        } else {
            (void)ab05qd('N', n, m, p, nc, p, m,
                         a, lda, b, ldb, c, ldc, d, ldd,
                         ac, ldac, bc, ldbc, cc, ldcc, dc, lddc,
                         &ne, &me, &pe,
                         &dwork[kwa], nnc, &dwork[kwb], nnc,
                         &dwork[kwc], mp, &dwork[kwd], mp);

            SLC_DLASET("Full", &p, &p, &zero, &negone, &dwork[kwd + mp * m], &mp);
            SLC_DLASET("Full", &m, &m, &zero, &negone, &dwork[kwd + p], &mp);
        }

        ierr = ab07nd(nnc, mp, &dwork[kwa], nnc, &dwork[kwb], nnc,
                      &dwork[kwc], mp, &dwork[kwd], mp, &rcond,
                      iwork, &dwork[kw], ldwork - kw);
        if (ierr != 0) {
            *info = 1;
            return;
        }
        wrkopt = wrkopt > ((i32)dwork[kw] + kw) ? wrkopt : ((i32)dwork[kw] + kw);

        if (rightw) {
            me = m;
            kwb = kwb + nnc * p;
        } else if (perf) {
            me = p;
            kwc = kwc + m;
        }
    }

    if (leftw || perf) {
        ldu = nnc > p ? nnc : p;
        ku = kl;
        kq = ku + nnc * ldu;
        kr = kq + nnc * nnc;
        ki = kr + nnc;
        kw = ki + nnc;

        jobfac = 'N';
        SLC_DLACPY("Full", &p, &nnc, &dwork[kwc], &mp, &dwork[ku], &ldu);
        sb03od(dico, &jobfac, "N", nnc, p,
               &dwork[kwa], nnc, &dwork[kq], nnc, &dwork[ku], ldu,
               scaleo, &dwork[kr], &dwork[ki], &dwork[kw],
               ldwork - kw, &ierr);
        if (ierr != 0) {
            if (ierr == 6) {
                *info = 2;
            } else {
                *info = 3;
            }
            return;
        }
        wrkopt = wrkopt > ((i32)dwork[kw] + kw) ? wrkopt : ((i32)dwork[kw] + kw);

        if (leftw) {
            SLC_DLACPY("Upper", &ncs, &ncs, &dwork[ku], &ldu, r, &ldr);
        } else {
            i32 nncu = n + ncu;
            SLC_DLACPY("Upper", &ncs, &ncs, &dwork[ku + (ldu + 1) * nncu], &ldu,
                       r, &ldr);
            ktau = ku;
            f64 dum = 0.0;
            i32 int0 = 0;
            mb04od("Full", ncs, int0, nncu, r, ldr,
                   &dwork[ku + ldu * nncu], ldu, &dum, int1,
                   &dum, int1, &dwork[ktau], &dwork[kw]);

            for (i32 j = 0; j < ncs; j++) {
                if (r[j + j * ldr] < zero) {
                    i32 len = ncs - j;
                    SLC_DSCAL(&len, &negone, &r[j + j * ldr], &ldr);
                }
            }
        }
    }

    if (rightw || perf) {
        ku = kl;
        i32 max_nnc_me = nnc > me ? nnc : me;
        kq = ku + nnc * max_nnc_me;
        kr = kq + nnc * nnc;
        ki = kr + nnc;
        kw = ki + nnc;

        SLC_DLACPY("Full", &nnc, &me, &dwork[kwb], &nnc, &dwork[ku], &nnc);
        jobfac = 'F';
        if (rightw) jobfac = 'N';
        sb03od(dico, &jobfac, "T", nnc, me,
               &dwork[kwa], nnc, &dwork[kq], nnc, &dwork[ku], nnc,
               scalec, &dwork[kr], &dwork[ki], &dwork[kw],
               ldwork - kw, &ierr);
        if (ierr != 0) {
            if (ierr == 6) {
                *info = 2;
            } else {
                *info = 3;
            }
            return;
        }
        wrkopt = wrkopt > ((i32)dwork[kw] + kw) ? wrkopt : ((i32)dwork[kw] + kw);

        i32 nncu = n + ncu;
        SLC_DLACPY("Upper", &ncs, &ncs, &dwork[ku + (nnc + 1) * nncu], &nnc,
                   s, &lds);
    }

    ku = 0;
    if (leftw || perf) {
        if (*jobo == 'E' || *jobo == 'e') {
            SLC_DLACPY("Upper", &ncs, &ncs, r, &ldr, &dwork[ku], &ncs);
            SLC_DLACPY("Full", &ncs, &ncs, &ac[ncu1 + ncu1 * ldac], &ldac,
                       &dwork[ku + ncs * ncs], &ncs);
            mb01wd(dico, "U", "N", "H", ncs, negone, zero, r, ldr,
                   &dwork[ku + ncs * ncs], ncs, &dwork[ku], ncs, &ierr);

            kw = ku + ncs;
            SLC_DSYEV("V", "U", &ncs, r, &ldr, &dwork[ku],
                      &dwork[kw], &(i32){ldwork - kw}, &ierr);
            if (ierr > 0) {
                *info = 4;
                return;
            }
            wrkopt = wrkopt > ((i32)dwork[kw] + kw) ? wrkopt : ((i32)dwork[kw] + kw);

            f64 abs_min = fabs(dwork[ku]);
            f64 abs_max = fabs(dwork[ku + ncs - 1]);
            tol = (abs_min > abs_max ? abs_min : abs_max) * SLC_DLAMCH("E");

            i32 pcbar = 0;
            i32 jj = ku;
            for (i32 j = 0; j < ncs; j++) {
                if (dwork[jj] > tol) {
                    f64 sqrt_val = sqrt(dwork[jj]);
                    SLC_DSCAL(&ncs, &sqrt_val, &r[0 + j * ldr], &int1);
                    SLC_DCOPY(&ncs, &r[0 + j * ldr], &int1, &dwork[kw + pcbar], &ncs);
                    pcbar++;
                }
                jj++;
            }

            ku = kw;
            ktau = ku + ncs * ncs;
            kw = ktau + ncs;

            sb03ou(discr, false, ncs, pcbar, &ac[ncu1 + ncu1 * ldac], ldac,
                   &dwork[ku], ncs, &dwork[ktau], r, ldr, &t,
                   &dwork[kw], ldwork - kw, &ierr);
            if (ierr != 0) {
                *info = 5;
                return;
            }
            *scaleo = (*scaleo) * t;
            wrkopt = wrkopt > ((i32)dwork[kw] + kw) ? wrkopt : ((i32)dwork[kw] + kw);
        }
    }

    if (rightw || perf) {
        if (*jobc == 'E' || *jobc == 'e') {
            SLC_DLACPY("Upper", &ncs, &ncs, s, &lds, &dwork[ku], &ncs);
            SLC_DLACPY("Full", &ncs, &ncs, &ac[ncu1 + ncu1 * ldac], &ldac,
                       &dwork[ku + ncs * ncs], &ncs);
            mb01wd(dico, "U", "T", "H", ncs, negone, zero, s, lds,
                   &dwork[ku + ncs * ncs], ncs, &dwork[ku], ncs, &ierr);

            kw = ku + ncs;
            SLC_DSYEV("V", "U", &ncs, s, &lds, &dwork[ku],
                      &dwork[kw], &(i32){ldwork - kw}, &ierr);
            if (ierr > 0) {
                *info = 4;
                return;
            }
            wrkopt = wrkopt > ((i32)dwork[kw] + kw) ? wrkopt : ((i32)dwork[kw] + kw);

            f64 abs_min = fabs(dwork[ku]);
            f64 abs_max = fabs(dwork[ku + ncs - 1]);
            tol = (abs_min > abs_max ? abs_min : abs_max) * SLC_DLAMCH("E");

            i32 mbbar = 0;
            i32 i_idx = kw;
            i32 jj = ku;
            for (i32 j = 0; j < ncs; j++) {
                if (dwork[jj] > tol) {
                    mbbar++;
                    f64 sqrt_val = sqrt(dwork[jj]);
                    SLC_DSCAL(&ncs, &sqrt_val, &s[0 + j * lds], &int1);
                    SLC_DCOPY(&ncs, &s[0 + j * lds], &int1, &dwork[i_idx], &int1);
                    i_idx += ncs;
                }
                jj++;
            }

            ku = kw;
            ktau = ku + mbbar * ncs;
            kw = ktau + ncs;

            sb03ou(discr, true, ncs, mbbar, &ac[ncu1 + ncu1 * ldac], ldac,
                   &dwork[ku], ncs, &dwork[ktau], s, lds, &t,
                   &dwork[kw], ldwork - kw, &ierr);
            if (ierr != 0) {
                *info = 5;
                return;
            }
            *scalec = (*scalec) * t;
            wrkopt = wrkopt > ((i32)dwork[kw] + kw) ? wrkopt : ((i32)dwork[kw] + kw);
        }
    }

    dwork[0] = (f64)wrkopt;
}
