/**
 * @file mb04rb.c
 * @brief Reduction of skew-Hamiltonian matrix to PVL form (blocked).
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "slicot.h"
#include "slicot_blas.h"

void mb04rb(
    const i32 n,
    const i32 ilo,
    f64* a, const i32 lda,
    f64* qg, const i32 ldqg,
    f64* cs,
    f64* tau,
    f64* dwork, const i32 ldwork,
    i32* info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    i32 i, ierr, minwrk, wrkopt;
    bool lquery;

    *info = 0;
    minwrk = (n > 1) ? n - 1 : 1;

    if (n < 0) {
        *info = -1;
    } else if (ilo < 1 || ilo > n + 1) {
        *info = -2;
    } else if (lda < ((n > 1) ? n : 1)) {
        *info = -4;
    } else if (ldqg < ((n > 1) ? n : 1)) {
        *info = -6;
    } else {
        lquery = (ldwork == -1);
        if (ldwork < minwrk && !lquery) {
            dwork[0] = (f64)minwrk;
            *info = -10;
        } else {
            if (n <= ilo) {
                wrkopt = 1;
            } else {
                i32 lwork_query = -1;
                SLC_DGEHRD(&n, &(i32){1}, &n, dwork, &n, dwork, dwork, &lwork_query, &ierr);
                wrkopt = (minwrk > (i32)dwork[0]) ? minwrk : (i32)dwork[0];
                i32 nb = (i32)wrkopt / n;
                if (nb > n) nb = n;
                i32 opt_blocked = 8 * n * nb + 3 * nb;
                wrkopt = (wrkopt > opt_blocked) ? wrkopt : opt_blocked;
            }
            if (lquery) {
                dwork[0] = (f64)wrkopt;
                return;
            }
        }
    }

    if (*info != 0) {
        return;
    }

    if (n <= ilo) {
        dwork[0] = ONE;
        return;
    }

    for (i = 0; i < ilo - 1; i++) {
        tau[i] = ZERO;
        cs[2 * i] = ONE;
        cs[2 * i + 1] = ZERO;
    }

    mb04ru(n, ilo, a, lda, qg, ldqg, cs, tau, dwork, ldwork, &ierr);

    dwork[0] = (f64)wrkopt;
}
