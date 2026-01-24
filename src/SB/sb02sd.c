// SPDX-License-Identifier: BSD-3-Clause

#include "slicot.h"
#include "slicot_blas.h"
#include <ctype.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>

static int select_all(const f64* wr, const f64* wi) {
    (void)wr;
    (void)wi;
    return 1;
}

void sb02sd(const char* job, const char* fact, const char* trana, const char* uplo,
            const char* lyapun, i32 n, const f64* a, i32 lda, f64* t, i32 ldt,
            f64* u, i32 ldu, const f64* g, i32 ldg, const f64* q, i32 ldq,
            const f64* x, i32 ldx, f64* sepd, f64* rcond, f64* ferr,
            i32* iwork, f64* dwork, i32 ldwork, i32* info)
{
    const f64 zero = 0.0;
    const f64 one = 1.0;
    const f64 four = 4.0;
    const f64 half = 0.5;
    i32 int1 = 1;

    char job_c = (char)toupper((unsigned char)job[0]);
    char fact_c = (char)toupper((unsigned char)fact[0]);
    char trana_c = (char)toupper((unsigned char)trana[0]);
    char uplo_c = (char)toupper((unsigned char)uplo[0]);
    char lyapun_c = (char)toupper((unsigned char)lyapun[0]);

    bool jobc = (job_c == 'C');
    bool jobe = (job_c == 'E');
    bool jobb = (job_c == 'B');
    bool nofact = (fact_c == 'N');
    bool notrna = (trana_c == 'N');
    bool lower = (uplo_c == 'L');
    bool update = (lyapun_c == 'O');

    bool needac = update && !jobc;

    i32 nn = n * n;
    i32 lwa = update ? nn : 0;

    i32 ldw;
    if (jobc) {
        ldw = (2 * nn > 3 ? 2 * nn : 3) + nn;
    } else {
        ldw = (2 * nn > 3 ? 2 * nn : 3) + 2 * nn;
        if (!update) ldw += n;
    }
    if (nofact) {
        i32 t1 = lwa + 5 * n;
        ldw = (t1 > ldw) ? t1 : ldw;
    }

    *info = 0;

    if (!jobb && !jobc && !jobe) {
        *info = -1;
    } else if (!nofact && fact_c != 'F') {
        *info = -2;
    } else if (!notrna && trana_c != 'T' && trana_c != 'C') {
        *info = -3;
    } else if (!lower && uplo_c != 'U') {
        *info = -4;
    } else if (!update && lyapun_c != 'R') {
        *info = -5;
    } else if (n < 0) {
        *info = -6;
    } else if (lda < 1 || (lda < n && (update || nofact))) {
        *info = -8;
    } else if (ldt < (n > 1 ? n : 1)) {
        *info = -10;
    } else if (ldu < 1 || (ldu < n && update)) {
        *info = -12;
    } else if (ldg < (n > 1 ? n : 1)) {
        *info = -14;
    } else if (ldq < (n > 1 ? n : 1)) {
        *info = -16;
    } else if (ldx < (n > 1 ? n : 1)) {
        *info = -18;
    } else if (ldwork < ldw && ldwork != -1) {
        *info = -24;
    }

    if (*info != 0) {
        return;
    }

    if (ldwork == -1) {
        dwork[0] = (f64)ldw;
        return;
    }

    if (n == 0) {
        if (!jobe) *rcond = one;
        if (!jobc) *ferr = zero;
        dwork[0] = one;
        return;
    }

    f64 xnorm = SLC_DLANSY("1", &uplo_c, &n, x, &ldx, dwork);
    if (xnorm == zero) {
        if (!jobe) *rcond = zero;
        if (!jobc) *ferr = zero;
        dwork[0] = (f64)n;
        return;
    }

    i32 ires = 0;
    i32 ixbs = ires + nn;
    i32 ixma = (2 * nn > 3) ? 2 * nn : 3;
    i32 iabs = ixma + nn;
    i32 iwrk = iabs + nn;

    i32 wrkopt = 0;

    if (update || nofact) {
        SLC_DLASET("F", &n, &n, &zero, &one, &dwork[ixbs], &n);
        SLC_DSYMM("L", &uplo_c, &n, &n, &one, g, &ldg, x, &ldx, &one, &dwork[ixbs], &n);

        if (notrna) {
            SLC_DLACPY("F", &n, &n, a, &lda, dwork, &n);
            SLC_DGESV(&n, &n, &dwork[ixbs], &n, iwork, dwork, &n, info);
        } else {
            for (i32 j = 0; j < n; j++) {
                SLC_DCOPY(&n, &a[j * lda], &int1, &dwork[j], &n);
            }
            SLC_DGESV(&n, &n, &dwork[ixbs], &n, iwork, dwork, &n, info);
            for (i32 j = 1; j < n; j++) {
                i32 jm1 = j;
                SLC_DSWAP(&jm1, &dwork[j * n], &int1, &dwork[j], &n);
            }
        }

        wrkopt = 2 * nn;
        if (nofact) {
            SLC_DLACPY("F", &n, &n, dwork, &n, t, &ldt);
        }
    } else {
        wrkopt = n;
    }

    if (nofact) {
        char sjob = update ? 'V' : 'N';
        i32 sdim;
        i32 bwork_val = 0;

        SLC_DGEES(&sjob, "N", select_all, &n, t, &ldt, &sdim,
                  &dwork[lwa], &dwork[lwa + n], u, &ldu,
                  &dwork[lwa + 2 * n], &ldwork, &bwork_val, info);
        if (*info > 0) {
            if (lwa > 0) {
                i32 two_n = 2 * n;
                SLC_DCOPY(&two_n, &dwork[lwa], &int1, dwork, &int1);
            }
            return;
        }

        i32 dgees_opt = (i32)dwork[lwa + 2 * n] + lwa + 2 * n;
        wrkopt = (wrkopt > dgees_opt) ? wrkopt : dgees_opt;
    }

    i32 lwr;
    if (needac) {
        SLC_DLACPY("F", &n, &n, dwork, &n, &dwork[iabs], &n);
        lwr = nn;
    } else {
        lwr = 0;
    }

    char tranat = notrna ? 'T' : 'N';

    if (update) {
        SLC_DGEMM("N", &trana_c, &n, &n, &n, &one, x, &ldx, dwork, &n,
                  &zero, &dwork[ixma], &n);
    } else {
        i32 info2;
        mb01ud("R", &trana_c, n, n, one, t, ldt, x, ldx, &dwork[ixma], n, &info2);
    }

    if (!jobe) {
        f64 thnorm;
        sb03sy("B", &trana_c, &lyapun_c, n, t, ldt, u, ldu, &dwork[ixma], n,
               sepd, &thnorm, iwork, dwork, ixma, info);

        i32 t1 = lwr + ((2 * nn > 3) ? 2 * nn : 3) + nn;
        wrkopt = (wrkopt > t1) ? wrkopt : t1;

        if (*sepd == zero) {
            *rcond = zero;
            if (jobb) *ferr = one;
            dwork[0] = (f64)wrkopt;
            return;
        }

        i32 kase = 0;
        i32 isave[3] = {0, 0, 0};
        f64 est = zero;
        f64 scale = one;

        while (1) {
            SLC_DLACN2(&nn, &dwork[ixbs], dwork, iwork, &est, &kase, isave);
            if (kase == 0) break;

            f64 upper_norm = SLC_DLANSY("1", "U", &n, dwork, &n, &dwork[ixbs]);
            f64 lower_norm = SLC_DLANSY("1", "L", &n, dwork, &n, &dwork[ixbs]);

            char loup = (upper_norm >= lower_norm) ? 'U' : 'L';

            i32 info2;
            mb01ru(&loup, &tranat, n, n, zero, one, dwork, n,
                   &dwork[ixma], n, dwork, n, &dwork[ixbs], nn, &info2);
            i32 n1 = n + 1;
            SLC_DSCAL(&n, &half, dwork, &n1);

            if (update) {
                mb01ru(&loup, "T", n, n, zero, one, dwork, n,
                       u, ldu, dwork, n, &dwork[ixbs], nn, &info2);
                SLC_DSCAL(&n, &half, dwork, &n1);
            }

            ma02ed(loup, n, dwork, n);

            if (kase == 1) {
                sb03mx(&trana_c, n, t, ldt, dwork, n, &scale, &dwork[ixbs], &info2);
            } else {
                sb03mx(&tranat, n, t, ldt, dwork, n, &scale, &dwork[ixbs], &info2);
            }

            if (update) {
                mb01ru(&loup, "N", n, n, zero, one, dwork, n,
                       u, ldu, dwork, n, &dwork[ixbs], nn, &info2);
                SLC_DSCAL(&n, &half, dwork, &n1);
                ma02ed(loup, n, dwork, n);
            }
        }

        f64 pinorm;
        f64 bignum = one / SLC_DLAMCH("S");
        if (est < scale) {
            pinorm = est / scale;
        } else {
            if (est < scale * bignum) {
                pinorm = est / scale;
            } else {
                pinorm = bignum;
            }
        }

        f64 anorm;
        if (update) {
            anorm = SLC_DLANGE("1", &n, &n, a, &lda, dwork);
        } else {
            anorm = SLC_DLANHS("1", &n, t, &ldt, dwork);
        }

        f64 qnorm = SLC_DLANSY("1", &uplo_c, &n, q, &ldq, dwork);
        f64 gnorm = SLC_DLANSY("1", &uplo_c, &n, g, &ldg, dwork);

        f64 tmax = *sepd;
        if (xnorm > tmax) tmax = xnorm;
        if (anorm > tmax) tmax = anorm;
        if (gnorm > tmax) tmax = gnorm;

        f64 temp, denom;
        if (tmax <= one) {
            temp = (*sepd) * xnorm;
            denom = qnorm + (*sepd * anorm) * thnorm + (*sepd * gnorm) * pinorm;
        } else {
            temp = (*sepd / tmax) * (xnorm / tmax);
            denom = ((one / tmax) * (qnorm / tmax)) +
                    ((*sepd / tmax) * (anorm / tmax)) * thnorm +
                    ((*sepd / tmax) * (gnorm / tmax)) * pinorm;
        }

        if (temp >= denom) {
            *rcond = one;
        } else {
            *rcond = temp / denom;
        }
    }

    if (!jobc) {
        SLC_DLACPY(&uplo_c, &n, &n, q, &ldq, &dwork[ires], &n);
        i32 jj = ires;
        f64 mone = -one;
        if (lower) {
            for (i32 j = 0; j < n; j++) {
                i32 len = n - j;
                SLC_DAXPY(&len, &mone, &x[j + j * ldx], &int1, &dwork[jj], &int1);
                jj += n + 1;
            }
        } else {
            for (i32 j = 0; j < n; j++) {
                i32 len = j + 1;
                SLC_DAXPY(&len, &mone, &x[j * ldx], &int1, &dwork[jj], &int1);
                jj += n;
            }
        }

        i32 info2;
        if (update) {
            slicot_mb01rx('L', uplo_c, tranat, n, n, one, one, &dwork[ires], n,
                   a, lda, &dwork[ixma], n);
        } else {
            mb01ry("L", &uplo_c, &tranat, n, one, one, &dwork[ires], n,
                   t, ldt, &dwork[ixma], n, &dwork[iwrk], &info2);
            SLC_DSYMM("L", &uplo_c, &n, &n, &one, g, &ldg, &dwork[ixma], &n,
                      &zero, &dwork[ixbs], &n);
            slicot_mb01rx('L', uplo_c, 'T', n, n, one, one, &dwork[ires], n,
                   &dwork[ixma], n, &dwork[ixbs], n);
        }

        f64 eps = SLC_DLAMCH("E");
        f64 epsn = eps * (f64)(n + 4);
        f64 epst = eps * (f64)(2 * (n + 1));
        f64 temp_eps = eps * four;

        for (i32 j = 0; j < n; j++) {
            for (i32 i = 0; i < n; i++) {
                dwork[ixbs + j * n + i] = fabs(x[i + j * ldx]);
            }
        }

        if (lower) {
            for (i32 j = 0; j < n; j++) {
                for (i32 i = j; i < n; i++) {
                    dwork[ires + j * n + i] = temp_eps * (fabs(q[i + j * ldq]) +
                        fabs(x[i + j * ldx])) + fabs(dwork[ires + j * n + i]);
                }
            }
        } else {
            for (i32 j = 0; j < n; j++) {
                for (i32 i = 0; i <= j; i++) {
                    dwork[ires + j * n + i] = temp_eps * (fabs(q[i + j * ldq]) +
                        fabs(x[i + j * ldx])) + fabs(dwork[ires + j * n + i]);
                }
            }
        }

        if (update) {
            for (i32 j = 0; j < n; j++) {
                for (i32 i = 0; i < n; i++) {
                    dwork[iabs + j * n + i] = fabs(dwork[iabs + j * n + i]);
                }
            }
            SLC_DGEMM("N", &trana_c, &n, &n, &n, &one, &dwork[ixbs], &n,
                      &dwork[iabs], &n, &zero, &dwork[ixma], &n);
            slicot_mb01rx('L', uplo_c, tranat, n, n, one, epsn, &dwork[ires], n,
                   &dwork[iabs], n, &dwork[ixma], n);
        } else {
            for (i32 j = 0; j < n; j++) {
                i32 imax = (j + 1 < n) ? j + 1 : n - 1;
                for (i32 i = 0; i <= imax; i++) {
                    dwork[iabs + j * n + i] = fabs(t[i + j * ldt]);
                }
            }
            mb01ud("R", &trana_c, n, n, one, &dwork[iabs], n,
                   &dwork[ixbs], n, &dwork[ixma], n, &info2);
            mb01ry("L", &uplo_c, &tranat, n, one, epsn, &dwork[ires], n,
                   &dwork[iabs], n, &dwork[ixma], n, &dwork[iwrk], &info2);
        }

        if (lower) {
            for (i32 j = 0; j < n; j++) {
                for (i32 i = j; i < n; i++) {
                    dwork[iabs + j * n + i] = fabs(g[i + j * ldg]);
                }
            }
        } else {
            for (i32 j = 0; j < n; j++) {
                for (i32 i = 0; i <= j; i++) {
                    dwork[iabs + j * n + i] = fabs(g[i + j * ldg]);
                }
            }
        }

        if (update) {
            mb01ru(&uplo_c, &tranat, n, n, one, epst, &dwork[ires], n,
                   &dwork[ixma], n, &dwork[iabs], n, &dwork[ixbs], nn, &info2);
            i32 t1 = ((2 * nn > 3) ? 2 * nn : 3) + 2 * nn;
            wrkopt = (wrkopt > t1) ? wrkopt : t1;
        } else {
            SLC_DSYMM("L", &uplo_c, &n, &n, &one, &dwork[iabs], &n,
                      &dwork[ixma], &n, &zero, &dwork[ixbs], &n);
            mb01ry("L", &uplo_c, &tranat, n, one, epst, &dwork[ires], n,
                   &dwork[ixma], n, &dwork[ixbs], n, &dwork[iwrk], &info2);
            i32 t1 = ((2 * nn > 3) ? 2 * nn : 3) + 2 * nn + n;
            wrkopt = (wrkopt > t1) ? wrkopt : t1;
        }

        f64 xanorm = SLC_DLANSY("M", &uplo_c, &n, x, &ldx, dwork);

        sb03sx(&trana_c, &uplo_c, &lyapun_c, n, xanorm, t, ldt, u, ldu,
               &dwork[ires], n, ferr, iwork, &dwork[ixbs], ixma, info);
    }

    dwork[0] = (f64)wrkopt;
}
