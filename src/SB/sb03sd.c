// SPDX-License-Identifier: BSD-3-Clause

#include "slicot.h"
#include "slicot_blas.h"
#include <ctype.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>

static int select_none(const f64* wr, const f64* wi) {
    (void)wr;
    (void)wi;
    return 0;  // Not ordered
}

void sb03sd(const char* job, const char* fact, const char* trana,
            const char* uplo, const char* lyapun, i32 n, f64 scale,
            const f64* a, i32 lda, f64* t, i32 ldt, f64* u, i32 ldu,
            const f64* c, i32 ldc, f64* x, i32 ldx, f64* sepd,
            f64* rcond, f64* ferr, i32* iwork, f64* dwork, i32 ldwork,
            i32* info)
{
    const f64 zero = 0.0;
    const f64 one = 1.0;
    const f64 three = 3.0;
    const f64 minusone = -1.0;
    i32 int1 = 1;
    i32 int0 = 0;

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

    i32 nn = n * n;
    i32 ldw = (3 > 2 * nn) ? 3 : 2 * nn;
    ldw += nn;

    *info = 0;

    // Validate parameters
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
    } else if (scale < zero || scale > one) {
        *info = -7;
    } else if (lda < 1 || (lda < n && (update || nofact))) {
        *info = -9;
    } else if (ldt < (1 > n ? 1 : n)) {
        *info = -11;
    } else if (ldu < 1 || (ldu < n && update)) {
        *info = -13;
    } else if (ldc < (1 > n ? 1 : n)) {
        *info = -15;
    } else if (ldx < (1 > n ? 1 : n)) {
        *info = -17;
    } else {
        i32 iwrk;
        if (jobc) {
            if (nofact) {
                iwrk = (ldw > 5 * n) ? ldw : 5 * n;
            } else {
                iwrk = ldw;
            }
        } else {
            iwrk = ldw + 2 * n;
        }
        iwrk = (1 > iwrk) ? 1 : iwrk;

        bool lquery = (ldwork == -1);

        char sjob_c = 'N';
        if (nofact) {
            if (update) {
                sjob_c = 'V';
            } else {
                sjob_c = 'N';
            }
        }

        if (lquery) {
            i32 wrkopt;
            if (nofact) {
                i32 sdim_dummy = 0;
                i32 bwork_dummy[1];
                i32 minusone = -1;
                SLC_DGEES(&sjob_c, "N", select_none, &n, t, &ldt, &sdim_dummy,
                         dwork, dwork, u, &ldu, dwork, &minusone, bwork_dummy, info);
                wrkopt = (iwrk > (i32)dwork[0] + 2 * n) ? iwrk : (i32)dwork[0] + 2 * n;
            } else {
                wrkopt = iwrk;
            }
            dwork[0] = (f64)wrkopt;
            return;
        }

        if (ldwork < iwrk) {
            *info = -23;
        }
    }

    if (*info != 0) {
        return;
    }

    // Quick return if n=0
    if (n == 0) {
        if (!jobe)
            *rcond = one;
        if (!jobc)
            *ferr = zero;
        dwork[0] = one;
        return;
    }

    // Compute the 1-norm of matrix X
    f64 xnorm = SLC_DLANSY("1", uplo, &n, x, &ldx, dwork);
    if (xnorm == zero) {
        if (!jobe)
            *rcond = zero;
        if (!jobc)
            *ferr = zero;
        dwork[0] = (f64)n;
        return;
    }

    // Compute the 1-norm of A or T
    f64 anorm;
    if (nofact || update) {
        anorm = SLC_DLANGE("1", &n, &n, a, &lda, dwork);
    } else {
        anorm = SLC_DLANHS("1", &n, t, &ldt, dwork);
    }

    // Special case A = I: set SEPD and RCOND to 0
    // Special case A = 0: set SEPD and RCOND to 1
    if (anorm == one) {
        if (nofact || update) {
            SLC_DLACPY("F", &n, &n, a, &lda, dwork, &n);
        } else {
            SLC_DLACPY("F", &n, &n, t, &ldt, dwork, &n);
            if (n > 2) {
                i32 nm2 = n - 2;
                SLC_DLASET("L", &nm2, &nm2, &zero, &zero, &dwork[2], &n);
            }
        }
        dwork[nn] = one;
        // dwork[nn] = 1, stride 0, subtract from diagonal with stride n+1
        SLC_DAXPY(&n, &minusone, &dwork[nn], &int0, dwork, &(i32){n + 1});
        if (SLC_DLANGE("M", &n, &n, dwork, &n, dwork) == zero) {
            if (!jobe) {
                *sepd = zero;
                *rcond = zero;
            }
            if (!jobc)
                *ferr = one;
            dwork[0] = (f64)(nn + 1);
            return;
        }
    } else if (anorm == zero) {
        if (!jobe) {
            *sepd = one;
            *rcond = one;
        }
        if (jobc) {
            dwork[0] = (f64)n;
            return;
        } else {
            // Set FERR for A = 0
            SLC_DLACPY(uplo, &n, &n, x, &ldx, dwork, &n);
            if (lower) {
                for (i32 j = 0; j < n; j++) {
                    i32 len = n - j;
                    SLC_DAXPY(&len, &scale, &c[j + j * ldc], &int1, &dwork[j + j * n], &int1);
                }
            } else {
                for (i32 j = 0; j < n; j++) {
                    i32 len = j + 1;
                    SLC_DAXPY(&len, &scale, &c[j * ldc], &int1, &dwork[j * n], &int1);
                }
            }
            f64 ferr_norm = SLC_DLANSY("1", uplo, &n, dwork, &n, &dwork[nn]);
            f64 temp = ferr_norm / xnorm;
            *ferr = (temp < one) ? temp : one;
            dwork[0] = (f64)(nn + n);
            return;
        }
    }

    // General case
    f64 cnorm = SLC_DLANSY("1", uplo, &n, c, &ldc, dwork);

    // Workspace usage
    i32 iabs = nn;
    i32 ixma = (3 > 2 * nn) ? 3 : 2 * nn;
    i32 ires = ixma;
    i32 iwrk = ixma + nn;
    i32 wrkopt = 0;

    // Compute Schur factorization if needed
    if (nofact) {
        SLC_DLACPY("F", &n, &n, a, &lda, t, &ldt);
        char sjob_c = update ? 'V' : 'N';
        i32 sdim_dummy = 0;
        i32* bwork = (i32*)malloc(n * sizeof(i32));
        if (!bwork) {
            *info = -22;  // workspace error
            return;
        }
        i32 ldwork_dgees = ldwork - 2 * n;
        SLC_DGEES(&sjob_c, "N", select_none, &n, t, &ldt, &sdim_dummy,
                 dwork, &dwork[n], u, &ldu, &dwork[2 * n], &ldwork_dgees,
                 bwork, info);
        free(bwork);
        if (*info > 0) {
            return;
        }
        wrkopt = (i32)dwork[2 * n] + 2 * n;
    }

    // Compute X*op(A) or X*op(T)
    i32 info_dummy = 0;
    if (update) {
        SLC_DGEMM("N", trana, &n, &n, &n, &one, x, &ldx, a, &lda, &zero, &dwork[ixma], &n);
    } else {
        mb01ud("R", trana, n, n, one, t, ldt, x, ldx, &dwork[ixma], n, &info_dummy);
    }

    if (!jobe) {
        // Estimate sepd and thnorm
        f64 thnorm;
        sb03sy("B", trana, lyapun, n, t, ldt, u, ldu, &dwork[ixma], n, sepd, &thnorm,
               iwork, dwork, ixma, info);

        i32 max_ws = (3 > 2 * nn) ? 3 : 2 * nn;
        max_ws += nn;
        wrkopt = (wrkopt > max_ws) ? wrkopt : max_ws;

        if (*sepd == zero) {
            *rcond = zero;
            if (jobb)
                *ferr = one;
            dwork[0] = (f64)wrkopt;
            return;
        }

        // Estimate reciprocal condition number
        f64 tmax = *sepd;
        if (xnorm > tmax) tmax = xnorm;
        if (anorm > tmax) tmax = anorm;

        f64 temp, denom;
        if (tmax <= one) {
            temp = (*sepd) * xnorm;
            denom = scale * cnorm + (*sepd) * anorm * thnorm;
        } else {
            temp = ((*sepd) / tmax) * (xnorm / tmax);
            denom = ((scale / tmax) * (cnorm / tmax)) +
                    (((*sepd) / tmax) * (anorm / tmax)) * thnorm;
        }
        if (temp >= denom) {
            *rcond = one;
        } else {
            *rcond = temp / denom;
        }
    }

    if (!jobc) {
        // Form residual R = scale*C + X - op(A)'*X*op(A) or R = scale*C + X - op(T)'*X*op(T)
        char tranat = notrna ? 'T' : 'N';

        SLC_DLACPY(uplo, &n, &n, c, &ldc, dwork, &n);

        if (update) {
            i32 mb01rx_info;
            mb01rx("L", uplo, &tranat, n, n, scale, minusone, dwork, n,
                   a, lda, &dwork[ixma], n, &mb01rx_info);
        } else {
            mb01ry("L", uplo, &tranat, n, scale, minusone, dwork, n,
                   t, ldt, &dwork[ixma], n, &dwork[iwrk], &info_dummy);
        }

        // Add X
        if (lower) {
            for (i32 j = 0; j < n; j++) {
                i32 len = n - j;
                SLC_DAXPY(&len, &one, &x[j + j * ldx], &int1, &dwork[j + j * n], &int1);
            }
        } else {
            for (i32 j = 0; j < n; j++) {
                i32 len = j + 1;
                SLC_DAXPY(&len, &one, &x[j * ldx], &int1, &dwork[j * n], &int1);
            }
        }

        SLC_DLACPY(uplo, &n, &n, dwork, &n, &dwork[ires], &n);

        // Get machine precision
        f64 eps = SLC_DLAMCH("E");
        f64 epsn = eps * (f64)(2 * n + 2);

        // Store abs(A) or abs(T)
        if (update) {
            for (i32 j = 0; j < n; j++) {
                for (i32 i = 0; i < n; i++) {
                    dwork[iabs + j * n + i] = fabs(a[i + j * lda]);
                }
            }
        } else {
            for (i32 j = 0; j < n; j++) {
                i32 imax = (j + 1 < n) ? j + 1 : n - 1;
                for (i32 i = 0; i <= imax; i++) {
                    dwork[iabs + j * n + i] = fabs(t[i + j * ldt]);
                }
            }
        }

        // Save diagonal of X
        i32 stride_x = ldx + 1;
        SLC_DCOPY(&n, x, &stride_x, &dwork[iwrk], &int1);

        // Store abs(X) in X and update residual
        if (lower) {
            for (i32 j = 0; j < n; j++) {
                for (i32 i = j; i < n; i++) {
                    f64 temp = fabs(x[i + j * ldx]);
                    x[i + j * ldx] = temp;
                    dwork[ires + j * n + i] =
                        fabs(dwork[ires + j * n + i]) +
                        eps * three * (scale * fabs(c[i + j * ldc]) + temp);
                }
            }
        } else {
            for (i32 j = 0; j < n; j++) {
                for (i32 i = 0; i <= j; i++) {
                    f64 temp = fabs(x[i + j * ldx]);
                    x[i + j * ldx] = temp;
                    dwork[ires + j * n + i] =
                        fabs(dwork[ires + j * n + i]) +
                        eps * three * (scale * fabs(c[i + j * ldc]) + temp);
                }
            }
        }

        if (update) {
            mb01ru(uplo, &tranat, n, n, one, epsn, &dwork[ires], n, &dwork[iabs], n,
                   x, ldx, dwork, nn, &info_dummy);
        } else {
            // Compute W = abs(X)*abs(op(T))
            mb01ud("R", trana, n, n, one, &dwork[iabs], n, x, ldx, dwork, n, &info_dummy);
            mb01ry("L", uplo, &tranat, n, one, epsn, &dwork[ires], n,
                   &dwork[iabs], n, dwork, n, &dwork[iwrk + n], &info_dummy);
        }

        i32 max_ws2 = (3 > 2 * nn) ? 3 : 2 * nn;
        max_ws2 += nn + 2 * n;
        wrkopt = (wrkopt > max_ws2) ? wrkopt : max_ws2;

        // Restore X diagonal
        i32 ldx1 = ldx + 1;
        SLC_DCOPY(&n, &dwork[iwrk], &int1, x, &ldx1);
        if (lower) {
            ma02ed('U', n, x, ldx);
        } else {
            ma02ed('L', n, x, ldx);
        }

        // Compute forward error bound
        f64 xanorm = SLC_DLANSY("M", uplo, &n, x, &ldx, dwork);

        sb03sx(trana, uplo, lyapun, n, xanorm, t, ldt, u, ldu,
               &dwork[ires], n, ferr, iwork, dwork, ires, info);
    }

    dwork[0] = (f64)wrkopt;
}
