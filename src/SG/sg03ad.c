/**
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * SG03AD - Generalized Lyapunov/Stein equation solver
 *
 * Solves for X either the generalized continuous-time Lyapunov equation:
 *   op(A)' * X * op(E) + op(E)' * X * op(A) = SCALE * Y
 *
 * or the generalized discrete-time Lyapunov equation:
 *   op(A)' * X * op(A) - op(E)' * X * op(E) = SCALE * Y
 *
 * Also provides estimates of separation and forward error.
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <stdlib.h>
#include <math.h>
#include <ctype.h>

void sg03ad(
    const char* dico_str,
    const char* job_str,
    const char* fact_str,
    const char* trans_str,
    const char* uplo_str,
    const i32 n,
    f64* a,
    const i32 lda,
    f64* e,
    const i32 lde,
    f64* q,
    const i32 ldq,
    f64* z,
    const i32 ldz,
    f64* x,
    const i32 ldx,
    f64* scale,
    f64* sep,
    f64* ferr,
    f64* alphar,
    f64* alphai,
    f64* beta,
    i32* iwork,
    f64* dwork,
    const i32 ldwork,
    i32* info)
{
    const f64 ONE = 1.0;
    const f64 TWO = 2.0;
    const f64 ZERO = 0.0;
    const i32 int1 = 1;

    char dico = toupper((unsigned char)dico_str[0]);
    char job = toupper((unsigned char)job_str[0]);
    char fact = toupper((unsigned char)fact_str[0]);
    char trans = toupper((unsigned char)trans_str[0]);
    char uplo = toupper((unsigned char)uplo_str[0]);

    bool isdisc = (dico == 'D');
    bool wantx = (job == 'X');
    bool wantsp = (job == 'S');
    bool wantbh = (job == 'B');
    bool isfact = (fact == 'F');
    bool istran = (trans == 'T');
    bool isuppr = (uplo == 'U');
    bool lquery = (ldwork == -1);

    *info = 0;

    if (!isdisc && dico != 'C') {
        *info = -1;
    } else if (!wantx && !wantsp && !wantbh) {
        *info = -2;
    } else if (!isfact && fact != 'N') {
        *info = -3;
    } else if (!istran && trans != 'N') {
        *info = -4;
    } else if (!isuppr && uplo != 'L') {
        *info = -5;
    } else if (n < 0) {
        *info = -6;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -8;
    } else if (lde < (n > 1 ? n : 1)) {
        *info = -10;
    } else if (ldq < (n > 1 ? n : 1)) {
        *info = -12;
    } else if (ldz < (n > 1 ? n : 1)) {
        *info = -14;
    } else if (ldx < (n > 1 ? n : 1)) {
        *info = -16;
    } else {
        i32 minwrk;
        if (wantx) {
            if (isfact) {
                minwrk = (n > 1 ? n : 1);
            } else {
                minwrk = (4*n > 1 ? 4*n : 1);
            }
        } else {
            if (isfact) {
                minwrk = (2*n*n > 1 ? 2*n*n : 1);
            } else {
                i32 tmp = 2*n*n > 4*n ? 2*n*n : 4*n;
                minwrk = tmp > 1 ? tmp : 1;
            }
        }
        i32 mingg = minwrk > 8*n + 16 ? minwrk : 8*n + 16;

        if (lquery) {
            i32 optwrk;
            if (isfact) {
                optwrk = mingg;
            } else {
                i32 info1;
                i32 dummy;
                f64 dwork_query;
                SLC_DGGES("V", "V", "N", delctg, &n, a, &lda, e, &lde,
                          &dummy, alphar, alphai, beta, q, &ldq, z, &ldz,
                          &dwork_query, &int1, iwork, &info1);
                i32 nn = n*n;
                optwrk = (i32)dwork_query;
                optwrk = optwrk > mingg ? optwrk : mingg;
                optwrk = optwrk > nn ? optwrk : nn;
            }
            dwork[0] = (f64)optwrk;
            return;
        } else if (minwrk > ldwork) {
            *info = -25;
        }
    }

    if (*info != 0) {
        return;
    }

    if (n == 0) {
        *scale = ONE;
        if (!wantx) *sep = ZERO;
        if (wantbh) *ferr = ZERO;
        dwork[0] = ONE;
        return;
    }

    if (isfact) {
        for (i32 i = 0; i < n - 2; i++) {
            if (a[(i+1) + i*lda] != ZERO && a[(i+2) + (i+1)*lda] != ZERO) {
                *info = 1;
                return;
            }
        }
    }

    i32 optwrk = 0;
    if (!isfact) {
        i32 info1;
        i32 mingg = ldwork > 8*n + 16 ? ldwork : 8*n + 16;

        if (ldwork < mingg) {
            SLC_DGEGS("V", "V", &n, a, &lda, e, &lde, alphar, alphai, beta,
                      q, &ldq, z, &ldz, dwork, &ldwork, &info1);
        } else {
            i32 dummy;
            SLC_DGGES("V", "V", "N", delctg, &n, a, &lda, e, &lde,
                      &dummy, alphar, alphai, beta, q, &ldq, z, &ldz,
                      dwork, &ldwork, iwork, &info1);
        }
        if (info1 != 0) {
            *info = 2;
            return;
        }
        optwrk = (i32)dwork[0];
    }

    if (wantbh || wantx) {
        i32 nn = n * n;
        i32 info1;

        if (ldwork < nn) {
            if (istran) {
                mb01rw(uplo_str, "T", n, n, x, ldx, q, ldq, dwork, &info1);
            } else {
                mb01rw(uplo_str, "T", n, n, x, ldx, z, ldz, dwork, &info1);
            }
        } else {
            if (istran) {
                mb01rd(uplo_str, "T", n, n, ZERO, ONE, x, ldx, q, ldq, x, ldx, dwork, ldwork, &info1);
            } else {
                mb01rd(uplo_str, "T", n, n, ZERO, ONE, x, ldx, z, ldz, x, ldx, dwork, ldwork, &info1);
            }
        }

        if (!isuppr) {
            for (i32 i = 0; i < n - 1; i++) {
                i32 len = n - i - 1;
                SLC_DCOPY(&len, &x[(i+1) + i*ldx], &int1, &x[i + (i+1)*ldx], &ldx);
            }
        }
        optwrk = optwrk > nn ? optwrk : nn;

        if (isdisc) {
            sg03ax(trans_str, n, a, lda, e, lde, x, ldx, scale, &info1);
            if (info1 != 0) {
                *info = 3;
            }
        } else {
            sg03ay(trans_str, n, a, lda, e, lde, x, ldx, scale, &info1);
            if (info1 != 0) {
                *info = 4;
            }
        }

        if (ldwork < nn) {
            if (istran) {
                mb01rw("U", "N", n, n, x, ldx, z, ldz, dwork, &info1);
            } else {
                mb01rw("U", "N", n, n, x, ldx, q, ldq, dwork, &info1);
            }
        } else {
            if (istran) {
                mb01rd("U", "N", n, n, ZERO, ONE, x, ldx, z, ldz, x, ldx, dwork, ldwork, &info1);
            } else {
                mb01rd("U", "N", n, n, ZERO, ONE, x, ldx, q, ldq, x, ldx, dwork, ldwork, &info1);
            }
        }

        for (i32 i = 0; i < n - 1; i++) {
            i32 len = n - i - 1;
            SLC_DCOPY(&len, &x[i + (i+1)*ldx], &ldx, &x[(i+1) + i*ldx], &int1);
        }
    }

    if (wantbh || wantsp) {
        f64 est = ZERO;
        i32 kase = 0;
        i32 isave[3] = {0, 0, 0};
        i32 nn = n * n;
        f64 scale1;
        i32 info1;

        while (1) {
            SLC_DLACN2(&nn, &dwork[nn], dwork, iwork, &est, &kase, isave);
            if (kase == 0) break;

            char etrans;
            if ((kase == 1 && !istran) || (kase != 1 && istran)) {
                etrans = 'N';
            } else {
                etrans = 'T';
            }

            if (isdisc) {
                sg03ax(&etrans, n, a, lda, e, lde, dwork, n, &scale1, &info1);
                if (info1 != 0) {
                    *info = 3;
                }
            } else {
                sg03ay(&etrans, n, a, lda, e, lde, dwork, n, &scale1, &info1);
                if (info1 != 0) {
                    *info = 4;
                }
            }
        }
        *sep = scale1 / est;
    }

    if (wantbh) {
        f64 eps = SLC_DLAMCH("P");

        for (i32 i = 0; i < n; i++) {
            i32 len = (i + 2 < n) ? i + 2 : n;
            dwork[i] = SLC_DNRM2(&len, &a[i*lda], &int1);
            i32 ei_len = i + 1;
            dwork[n + i] = SLC_DNRM2(&ei_len, &e[i*lde], &int1);
        }

        f64 norma = SLC_DNRM2(&n, dwork, &int1);
        f64 norme = SLC_DNRM2(&n, &dwork[n], &int1);

        if (isdisc) {
            *ferr = (norma * norma + norme * norme) * eps / (*sep);
        } else {
            *ferr = TWO * norma * norme * eps / (*sep);
        }
    }

    i32 minwrk;
    if (wantx) {
        if (isfact) {
            minwrk = (n > 1 ? n : 1);
        } else {
            minwrk = (4*n > 1 ? 4*n : 1);
        }
    } else {
        if (isfact) {
            minwrk = (2*n*n > 1 ? 2*n*n : 1);
        } else {
            i32 tmp = 2*n*n > 4*n ? 2*n*n : 4*n;
            minwrk = tmp > 1 ? tmp : 1;
        }
    }

    dwork[0] = (f64)(optwrk > minwrk ? optwrk : minwrk);
}
