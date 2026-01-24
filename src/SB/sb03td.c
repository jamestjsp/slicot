/**
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * SB03TD - Solution of continuous-time Lyapunov equations and condition/error estimation
 *
 * Solves the real continuous-time Lyapunov matrix equation
 *     op(A)' * X + X * op(A) = scale * C
 * estimates the conditioning, and computes an error bound on the solution X.
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <ctype.h>
#include <math.h>

static int select_func(const f64* wr, const f64* wi) {
    (void)wr;
    (void)wi;
    return 0;
}

void sb03td(
    const char* job,
    const char* fact,
    const char* trana,
    const char* uplo,
    const char* lyapun,
    const i32 n,
    f64* scale,
    const f64* a,
    const i32 lda,
    f64* t,
    const i32 ldt,
    f64* u,
    const i32 ldu,
    f64* c,
    const i32 ldc,
    f64* x,
    const i32 ldx,
    f64* sep,
    f64* rcond,
    f64* ferr,
    f64* wr,
    f64* wi,
    i32* iwork,
    f64* dwork,
    const i32 ldwork,
    i32* info
)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const f64 HALF = 0.5;

    char job_c = (char)toupper((unsigned char)job[0]);
    char fact_c = (char)toupper((unsigned char)fact[0]);
    char trana_c = (char)toupper((unsigned char)trana[0]);
    char uplo_c = (char)toupper((unsigned char)uplo[0]);
    char lyapun_c = (char)toupper((unsigned char)lyapun[0]);

    bool jobx = (job_c == 'X');
    bool jobs = (job_c == 'S');
    bool jobc = (job_c == 'C');
    bool jobe = (job_c == 'E');
    bool joba = (job_c == 'A');
    bool nofact = (fact_c == 'N');
    bool notrna = (trana_c == 'N');
    bool lower = (uplo_c == 'L');
    bool update = (lyapun_c == 'O');

    i32 nn = n * n;
    i32 ldw;
    if (jobx) {
        ldw = nn;
    } else if (jobs || jobc) {
        ldw = 2 * nn;
    } else {
        ldw = 3 * nn;
    }
    if ((jobe || joba) && !update) {
        ldw = ldw + n - 1;
    }
    if (nofact) {
        ldw = (ldw > 3 * n) ? ldw : 3 * n;
    }

    *info = 0;

    if (!jobx && !jobs && !jobc && !jobe && !joba) {
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
    } else if ((jobc || jobe) && (*scale < ZERO || *scale > ONE)) {
        *info = -7;
    } else if (lda < 1 || (lda < n && ((update && !jobx) || nofact))) {
        *info = -9;
    } else if (ldt < (n > 1 ? n : 1)) {
        *info = -11;
    } else if (ldu < 1 || (ldu < n && update)) {
        *info = -13;
    } else if (ldc < 1 || (!jobs && ldc < n)) {
        *info = -15;
    } else if (ldx < 1 || (!jobs && ldx < n)) {
        *info = -17;
    } else if (ldwork < 1 || ldwork < ldw) {
        *info = -25;
    }

    if (*info != 0) {
        i32 neginfo = -(*info);
        SLC_XERBLA("SB03TD", &neginfo);
        return;
    }

    if (n == 0) {
        if (jobx || joba) {
            *scale = ONE;
        }
        if (jobc || joba) {
            *rcond = ONE;
        }
        if (jobe || joba) {
            *ferr = ZERO;
        }
        dwork[0] = ONE;
        return;
    }

    char cfact;
    i32 sdim;
    sl_int bwork[1];

    if (nofact) {
        SLC_DLACPY("Full", &n, &n, a, &lda, t, &ldt);
        const char* sjob = update ? "V" : "N";
        SLC_DGEES(sjob, "N", select_func, &n, t, &ldt, &sdim, wr, wi, u, &ldu, dwork, &ldwork, bwork, info);
        if (*info > 0) {
            return;
        }
        cfact = 'F';
    } else {
        cfact = fact_c;
    }

    if (jobx || joba) {
        SLC_DLACPY(uplo, &n, &n, c, &ldc, x, &ldx);

        if (update) {
            i32 info_local = 0;
            mb01ru(uplo, "T", n, n, ZERO, ONE, x, ldx, u, ldu, x, ldx, dwork, ldwork, &info_local);
            i32 inc = ldx + 1;
            SLC_DSCAL(&n, &HALF, x, &inc);
        }

        ma02ed(uplo_c, n, x, ldx);

        sb03my(trana, n, t, ldt, x, ldx, scale, info);
        if (*info > 0) {
            *info = n + 1;
        }

        if (update) {
            i32 info_local = 0;
            mb01ru(uplo, "N", n, n, ZERO, ONE, x, ldx, u, ldu, x, ldx, dwork, ldwork, &info_local);
            i32 inc = ldx + 1;
            SLC_DSCAL(&n, &HALF, x, &inc);
            ma02ed(uplo_c, n, x, ldx);
        }
    }

    char jobl;
    f64 thnorm;

    if (jobs) {
        sb03qy("Separation", trana, lyapun, n, t, ldt, u, ldu, x, ldx, sep, &thnorm, iwork, dwork, ldwork, info);
    } else if (!jobx) {
        if (joba) {
            jobl = 'B';
        } else {
            jobl = job_c;
        }
        char cfact_str[2] = {cfact, '\0'};
        char jobl_str[2] = {jobl, '\0'};
        sb03qd(jobl_str, cfact_str, trana, uplo, lyapun, n, *scale, a, lda, t, ldt, u, ldu, c, ldc, x, ldx, sep, rcond, ferr, iwork, dwork, ldwork, info);
        i32 ldw_tmp = (i32)dwork[0];
        ldw = (ldw > ldw_tmp) ? ldw : ldw_tmp;
    }

    dwork[0] = (f64)ldw;
}
