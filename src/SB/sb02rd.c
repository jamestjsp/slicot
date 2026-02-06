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
    f64 rcondu = 0.0;
    f64 pivotu = 0.0;
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

        pivotu = dwork[iw];

        if (ierr > 0) {
            *info = 5;
            goto done;
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
    }

    if (!jobx) {
        // Estimate conditioning and/or error bound using SB02QD/SB02SD.
        if (!joba)
            wrkopt = 0;

        i32 iw = 5;
        char lofact = fact;
        i32 ierr;

        if (nofact && !update) {
            // Compute Ac and its Schur factorization.
            if (discr) {
                // Ac = inv(I_n + G*X)*A (TRANA='N') or A*inv(I_n + X*G) (TRANA='T')
                SLC_DLASET("F", &n, &n, &zero, &one, &dwork[iw], &n);
                SLC_DSYMM("L", &uplo, &n, &n, &one, g, &ldg, x, &ldx,
                          &one, &dwork[iw], &n);
                if (notrna) {
                    SLC_DLACPY("F", &n, &n, a, &lda, t, &ldt);
                    SLC_DGESV(&n, &n, &dwork[iw], &n, iwork, t, &ldt, &ierr);
                } else {
                    ma02ad("F", n, n, a, lda, t, ldt);
                    SLC_DGESV(&n, &n, &dwork[iw], &n, iwork, t, &ldt, &ierr);
                    for (i32 i = 1; i < n; i++) {
                        SLC_DSWAP(&(i32){i}, &t[i * ldt], &int1, &t[i], &ldt);
                    }
                }
                if (ierr != 0) {
                    *info = 6;
                    goto done;
                }
            } else {
                // Ac = A - G*X (TRANA='N') or A - X*G (TRANA='T')
                SLC_DLACPY("F", &n, &n, a, &lda, t, &ldt);
                if (notrna) {
                    f64 neg1 = -one;
                    SLC_DSYMM("L", &uplo, &n, &n, &neg1, g, &ldg, x, &ldx,
                              &one, t, &ldt);
                } else {
                    f64 neg1 = -one;
                    SLC_DSYMM("R", &uplo, &n, &n, &neg1, g, &ldg, x, &ldx,
                              &one, t, &ldt);
                }
            }

            // Schur factorization of Ac: Ac = V*T*V'
            i32 iwr = iw;
            i32 iwi = iwr + n;
            i32 iw2 = iwi + n;
            i32 ldw = ldwork - iw2;
            i32 nrot;

            SLC_DGEES("V", "N", sb02ms, &n, t, &ldt, &nrot,
                      &dwork[iwr], &dwork[iwi], v, &ldv,
                      &dwork[iw2], &ldw, bwork, &ierr);

            if (ierr != 0) {
                *info = 6;
                goto done;
            }

            wrkopt = fmax(wrkopt, dwork[iw2] + (f64)iw2);
            lofact = 'F';
            iw = 5;
        }

        if (!update) {
            char tranat = notrna ? 'T' : 'N';

            // Save diagonal elements of G and Q at dwork[iw..iw+2n-1].
            SLC_DCOPY(&n, g, &(i32){ldg + 1}, &dwork[iw], &int1);
            SLC_DCOPY(&n, q, &(i32){ldq + 1}, &dwork[iw + n], &int1);
            iw += n2;

            // Save X in S(NP1,1) if JOB='A' (will restore later).
            if (joba) {
                SLC_DLACPY("F", &n, &n, x, &ldx, &s[np1 - 1], &lds);
            }

            // Transform X <- V'*X*V
            char tranat_str[2] = {tranat, '\0'};
            mb01ru(&uplo, tranat_str, n, n, zero, one, x, ldx, v, ldv,
                   x, ldx, &dwork[iw], nn, &ierr);
            SLC_DSCAL(&n, &half, x, &(i32){ldx + 1});
            ma02ed(uplo, n, x, ldx);

            if (!discr) {
                ma02ed(uplo, n, g, ldg);
                ma02ed(uplo, n, q, ldq);
            }

            // Transform G <- V'*G*V
            mb01ru(&uplo, tranat_str, n, n, zero, one, g, ldg, v, ldv,
                   g, ldg, &dwork[iw], nn, &ierr);
            SLC_DSCAL(&n, &half, g, &(i32){ldg + 1});

            // Transform Q <- V'*Q*V
            mb01ru(&uplo, tranat_str, n, n, zero, one, q, ldq, v, ldv,
                   q, ldq, &dwork[iw], nn, &ierr);
            SLC_DSCAL(&n, &half, q, &(i32){ldq + 1});
        }

        // Call SB02QD (continuous) or SB02SD (discrete) for conditioning/error.
        char jobs[2];
        if (joba) {
            jobs[0] = 'B'; jobs[1] = '\0';
        } else {
            jobs[0] = job; jobs[1] = '\0';
        }
        char lofact_str[2] = {lofact, '\0'};
        i32 ldw_qd = ldwork - iw;

        if (discr) {
            sb02sd(jobs, lofact_str, trana_str, uplo_str, lyapun_str,
                   n, a, lda, t, ldt, v, ldv, g, ldg, q, ldq, x, ldx,
                   sep, rcond, ferr, iwork, &dwork[iw], ldw_qd, &ierr);
        } else {
            sb02qd(jobs, lofact_str, trana_str, uplo_str, lyapun_str,
                   n, a, lda, t, ldt, v, ldv, g, ldg, q, ldq, x, ldx,
                   sep, rcond, ferr, iwork, &dwork[iw], ldw_qd, &ierr);
        }

        wrkopt = fmax(wrkopt, dwork[iw] + (f64)iw);

        if (ierr == np1) {
            *info = 7;
        } else if (ierr > 0) {
            *info = 6;
            goto done;
        }

        if (!update) {
            // Restore X, G, and Q, and set S(2,1) to zero if needed.
            if (joba) {
                SLC_DLACPY("F", &n, &n, &s[np1 - 1], &lds, x, &ldx);
                SLC_DLASET("F", &n, &n, &zero, &zero, &s[np1 - 1], &lds);
            } else {
                // Undo V transformation on X: X <- V*X*V'
                mb01ru(&uplo, trana_str, n, n, zero, one, x, ldx, v, ldv,
                       x, ldx, &dwork[iw], nn, &ierr);
                SLC_DSCAL(&n, &half, x, &(i32){ldx + 1});
                ma02ed(uplo, n, x, ldx);
            }

            // Restore G and Q diagonal elements and mirror.
            char loup = luplo ? 'L' : 'U';
            i32 iw_save = 5;
            SLC_DCOPY(&n, &dwork[iw_save], &int1, g, &(i32){ldg + 1});
            ma02ed(loup, n, g, ldg);
            SLC_DCOPY(&n, &dwork[iw_save + n], &int1, q, &(i32){ldq + 1});
            ma02ed(loup, n, q, ldq);
        }
    }

done:
    // Set optimal workspace and output details.
    dwork[0] = wrkopt;
    if (jbxa) {
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
        }
    }
}
