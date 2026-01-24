/**
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * SB02RD - Riccati Equation Solver
 *
 * Solves continuous-time algebraic Riccati equation:
 *     Q + op(A)'*X + X*op(A) - X*G*X = 0                   (DICO='C')
 *
 * or discrete-time algebraic Riccati equation:
 *     Q + op(A)'*X*(I_n + G*X)^(-1)*op(A) - X = 0         (DICO='D')
 *
 * Using the Schur vector method with optional scaling.
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <stdlib.h>
#include <ctype.h>
#include <math.h>


void sb02rd(
    const char* job_str,
    const char* dico_str,
    const char* hinv_str,
    const char* trana_str,
    const char* uplo_str,
    const char* scal_str,
    const char* sort_str,
    const char* fact_str,
    const char* lyapun_str,
    const i32 n,
    f64* a,
    const i32 lda,
    f64* t,
    const i32 ldt,
    f64* v,
    const i32 ldv,
    f64* g,
    const i32 ldg,
    f64* q,
    const i32 ldq,
    f64* x,
    const i32 ldx,
    f64* sep,
    f64* rcond,
    f64* ferr,
    f64* wr,
    f64* wi,
    f64* s,
    const i32 lds,
    i32* iwork,
    f64* dwork,
    const i32 ldwork,
    i32* bwork,
    i32* info)
{
    const f64 zero = 0.0;
    const f64 half = 0.5;
    const f64 one = 1.0;
    const i32 int1 = 1;

    char job = toupper((unsigned char)job_str[0]);
    char dico = toupper((unsigned char)dico_str[0]);
    char hinv = toupper((unsigned char)hinv_str[0]);
    char trana = toupper((unsigned char)trana_str[0]);
    char uplo = toupper((unsigned char)uplo_str[0]);
    char scal = toupper((unsigned char)scal_str[0]);
    char sort = toupper((unsigned char)sort_str[0]);
    char fact = toupper((unsigned char)fact_str[0]);
    char lyapun = toupper((unsigned char)lyapun_str[0]);

    i32 n2 = n + n;
    i32 nn = n * n;
    i32 np1 = n + 1;

    bool joba = (job == 'A');
    bool jobc = (job == 'C');
    bool jobe = (job == 'E');
    bool jobx = (job == 'X');
    bool discr = (dico == 'D');
    bool lhinv = (discr && (jobx || joba)) ? (hinv == 'D') : false;
    bool notrna = (trana == 'N');
    bool luplo = (uplo == 'U');
    bool lscal = (scal == 'G');
    bool lsort = (sort == 'S');
    bool nofact = (fact == 'N');
    bool update = (lyapun == 'O');
    bool jbxa = jobx || joba;

    *info = 0;

    if (!jbxa && !jobc && !jobe) {
        *info = -1;
    } else if (!discr && dico != 'C') {
        *info = -2;
    } else if (discr && jbxa && !lhinv && hinv != 'I') {
        *info = -3;
    } else if (!notrna && trana != 'T' && trana != 'C') {
        *info = -4;
    } else if (!luplo && uplo != 'L') {
        *info = -5;
    } else if (jbxa && !lscal && scal != 'N') {
        *info = -6;
    } else if (jbxa && !lsort && sort != 'U') {
        *info = -7;
    } else if (!jobx && !nofact && fact != 'F') {
        *info = -8;
    } else if (!jobx && !update && lyapun != 'R') {
        *info = -9;
    } else if (n < 0) {
        *info = -10;
    } else if (lda < 1 || ((jbxa || nofact || update) && lda < n)) {
        *info = -12;
    } else if (ldt < 1 || (!jobx && ldt < n)) {
        *info = -14;
    } else if (ldv < 1 || (!jobx && ldv < n)) {
        *info = -16;
    } else if (ldg < (n > 1 ? n : 1)) {
        *info = -18;
    } else if (ldq < (n > 1 ? n : 1)) {
        *info = -20;
    } else if (ldx < (n > 1 ? n : 1)) {
        *info = -22;
    } else if (lds < 1 || (jbxa && lds < n2)) {
        *info = -29;
    } else if (jbxa && ldwork < 5 + (4 * nn + 8 * n > 1 ? 4 * nn + 8 * n : 1)) {
        *info = -32;
    }

    if (*info != 0) {
        return;
    }

    if (n == 0) {
        if (jobx) {
            *sep = one;
        }
        if (jobc || joba) {
            *rcond = one;
        }
        if (jobe || joba) {
            *ferr = zero;
        }
        dwork[0] = one;
        dwork[1] = one;
        dwork[2] = one;
        if (discr) {
            dwork[3] = one;
            dwork[4] = one;
        }
        return;
    }

    f64 wrkopt = 0.0;
    f64 rconda = 0.0;
    f64 pivota = 0.0;
    f64 qnorm = 0.0;
    f64 gnorm = 0.0;
    bool lscl = false;

    if (jbxa) {
        i32 ierr;
        i32 ldwork_ru = discr ? 6 * n : 0;

        sb02ru(dico_str, hinv_str, trana_str, uplo_str, n,
               a, lda, g, ldg, q, ldq, s, lds, iwork, dwork, ldwork_ru, &ierr);

        if (ierr != 0) {
            *info = 1;
            if (discr) {
                dwork[3] = dwork[0];
                dwork[4] = dwork[1];
            }
            return;
        }

        if (discr) {
            wrkopt = (f64)(6 * n);
            rconda = dwork[0];
            pivota = dwork[1];
        } else {
            wrkopt = 0.0;
        }

        if (lscal) {
            qnorm = sqrt(SLC_DLANSY("1", &uplo, &n, q, &ldq, dwork));
            gnorm = sqrt(SLC_DLANSY("1", &uplo, &n, g, &ldg, dwork));

            lscl = (qnorm > gnorm) && (gnorm > zero);
            if (lscl) {
                i32 kl = 0, ku = 0;
                SLC_DLASCL("G", &kl, &ku, &qnorm, &gnorm, &n, &n, &s[np1 - 1], &lds, &ierr);
                SLC_DLASCL("G", &kl, &ku, &gnorm, &qnorm, &n, &n, &s[(np1 - 1) * lds], &lds, &ierr);
            }
        }

        i32 iu = 5;
        i32 iw = iu + 4 * nn;
        i32 ldw = ldwork - iw + 1;
        i32 nrot;

        int (*select_func)(const f64*, const f64*);
        if (!discr) {
            select_func = lsort ? sb02mv : sb02mr;
        } else {
            select_func = lsort ? sb02mw : sb02ms;
        }

        SLC_DGEES("V", "S", select_func, &n2, s, &lds, &nrot, wr, wi,
                  &dwork[iu], &n2, &dwork[iw], &ldw, bwork, &ierr);

        if (discr && lhinv) {
            SLC_DSWAP(&n, wr, &int1, &wr[np1 - 1], &int1);
            SLC_DSWAP(&n, wi, &int1, &wi[np1 - 1], &int1);
        }

        if (ierr > n2) {
            *info = 3;
        } else if (ierr > 0) {
            *info = 2;
        } else if (nrot != n) {
            *info = 4;
        }

        if (*info != 0) {
            if (discr) {
                dwork[3] = rconda;
                dwork[4] = pivota;
            }
            return;
        }

        wrkopt = fmax(wrkopt, dwork[iw] + (f64)(iw - 1));

        for (i32 i = 0; i < n - 1; i++) {
            SLC_DSWAP(&(i32){n - i - 1}, &dwork[iu + n + (i + 1) * (n2 + 1) - 1], &n2,
                      &dwork[iu + n + i * (n2 + 1) + 1], &int1);
        }

        i32 iwr = iw;
        i32 iwc = iwr + n;
        i32 iwf = iwc + n;
        i32 iwb = iwf + n;
        iw = iwb + n;

        f64 rcondu;
        char equed = 'N';

        mb02pd("E", "T", n, n, &dwork[iu], n2, &s[np1 - 1], lds, iwork, &equed,
               &dwork[iwr], &dwork[iwc], &dwork[iu + n], n2, x, ldx, &rcondu,
               &dwork[iwf], &dwork[iwb], &iwork[np1 - 1], &dwork[iw], &ierr);

        if (jobx) {
            for (i32 i = 0; i < n - 1; i++) {
                SLC_DSWAP(&(i32){n - i - 1}, &dwork[iu + n + (i + 1) * (n2 + 1) - 1], &n2,
                          &dwork[iu + n + i * (n2 + 1) + 1], &int1);
            }

            if (equed != 'N') {
                bool rowequ = (equed == 'R') || (equed == 'B');
                bool colequ = (equed == 'C') || (equed == 'B');

                if (rowequ) {
                    for (i32 i = 0; i < n; i++) {
                        dwork[iwr + i] = one / dwork[iwr + i];
                    }
                    mb01sd('R', n, n, &dwork[iu], n2, &dwork[iwr], &dwork[iwc]);
                }

                if (colequ) {
                    for (i32 i = 0; i < n; i++) {
                        dwork[iwc + i] = one / dwork[iwc + i];
                    }
                    mb01sd('C', n, n, &dwork[iu], n2, &dwork[iwr], &dwork[iwc]);
                    mb01sd('C', n, n, &dwork[iu + n], n2, &dwork[iwr], &dwork[iwc]);
                }
            }

            SLC_DLASET("F", &n, &n, &zero, &zero, &s[np1 - 1], &lds);
        }

        f64 pivotu = dwork[iw];

        if (ierr > 0) {
            *info = 5;
            dwork[1] = rcondu;
            dwork[2] = pivotu;
            if (discr) {
                dwork[3] = rconda;
                dwork[4] = pivota;
            }
            return;
        }

        for (i32 i = 0; i < n - 1; i++) {
            SLC_DAXPY(&(i32){n - i - 1}, &one, &x[i + (i + 1) * ldx], &ldx, &x[i + 1 + i * ldx], &int1);
            SLC_DSCAL(&(i32){n - i - 1}, &half, &x[i + 1 + i * ldx], &int1);
            SLC_DCOPY(&(i32){n - i - 1}, &x[i + 1 + i * ldx], &int1, &x[i + (i + 1) * ldx], &ldx);
        }

        if (lscal && lscl) {
            i32 kl = 0, ku = 0;
            SLC_DLASCL("G", &kl, &ku, &gnorm, &qnorm, &n, &n, x, &ldx, &ierr);
        }

        dwork[0] = wrkopt;
        dwork[1] = rcondu;
        dwork[2] = pivotu;
        if (discr) {
            dwork[3] = rconda;
            dwork[4] = pivota;
        }

        if (jobx) {
            if (lscl) {
                *sep = qnorm / gnorm;
            } else {
                *sep = one;
            }
        } else if (joba) {
            // JOB='A' - should call SB02QD/SB02SD for proper conditioning
            // As workaround, use rcondu as approximation for rcond
            // TODO: Implement SB02QD/SB02SD for proper condition estimation
            f64 rcondu = dwork[1];
            *rcond = rcondu;
            if (lscl) {
                *sep = qnorm / gnorm;
            } else {
                *sep = one;
            }
            *ferr = zero;
        }
    }

    if (!jbxa) {
        // JOB='C' or JOB='E' - conditioning/error only, no solution computed
        // TODO: Implement SB02QD/SB02SD for proper condition estimation
        *info = 0;
        *sep = zero;
        *rcond = zero;
        *ferr = zero;
    }
}
