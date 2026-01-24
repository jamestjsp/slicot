// SPDX-License-Identifier: BSD-3-Clause

#include "slicot.h"
#include "slicot_blas.h"

void tg01dd(
    const char* compz,
    const i32 l, const i32 n, const i32 p,
    f64* a, const i32 lda,
    f64* e, const i32 lde,
    f64* c, const i32 ldc,
    f64* z, const i32 ldz,
    f64* dwork, const i32 ldwork,
    i32* info
) {
    const f64 ONE = 1.0;
    const f64 ZERO = 0.0;

    bool ilz;
    i32 icompz;
    i32 ln, wrkopt;

    if (*compz == 'N' || *compz == 'n') {
        ilz = false;
        icompz = 1;
    } else if (*compz == 'U' || *compz == 'u') {
        ilz = true;
        icompz = 2;
    } else if (*compz == 'I' || *compz == 'i') {
        ilz = true;
        icompz = 3;
    } else {
        icompz = 0;
    }

    *info = 0;
    i32 max_lnp = l > n ? l : n;
    max_lnp = max_lnp > p ? max_lnp : p;
    ln = l < n ? l : n;
    wrkopt = ln + max_lnp;
    if (wrkopt < 1) wrkopt = 1;

    if (icompz == 0) {
        *info = -1;
    } else if (l < 0) {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    } else if (p < 0) {
        *info = -4;
    } else if (lda < (l > 1 ? l : 1)) {
        *info = -6;
    } else if (lde < (l > 1 ? l : 1)) {
        *info = -8;
    } else if (ldc < (p > 1 ? p : 1)) {
        *info = -10;
    } else if ((ilz && ldz < n) || ldz < 1) {
        *info = -12;
    } else if (ldwork < wrkopt) {
        *info = -14;
    }

    if (*info != 0) {
        return;
    }

    if (icompz == 3) {
        SLC_DLASET("F", &n, &n, &ZERO, &ONE, z, &ldz);
    }

    if (l == 0 || n == 0) {
        dwork[0] = ONE;
        return;
    }

    ln = l < n ? l : n;

    i32 lwork_rq = ldwork - ln;
    i32 ierr;
    SLC_DGERQF(&l, &n, e, &lde, dwork, &dwork[ln], &lwork_rq, &ierr);
    wrkopt = (i32)dwork[ln] + ln;
    if (wrkopt < 1) wrkopt = 1;

    i32 e_row = l - ln;
    SLC_DORMRQ("R", "T", &l, &n, &ln, &e[e_row], &lde,
               dwork, a, &lda, &dwork[ln], &lwork_rq, &ierr);
    i32 opt = (i32)dwork[ln] + ln;
    if (opt > wrkopt) wrkopt = opt;

    SLC_DORMRQ("R", "T", &p, &n, &ln, &e[e_row], &lde,
               dwork, c, &ldc, &dwork[ln], &lwork_rq, &ierr);
    opt = (i32)dwork[ln] + ln;
    if (opt > wrkopt) wrkopt = opt;

    if (ilz) {
        SLC_DORMRQ("R", "T", &n, &n, &ln, &e[e_row], &lde,
                   dwork, z, &ldz, &dwork[ln], &lwork_rq, &ierr);
        opt = (i32)dwork[ln] + ln;
        if (opt > wrkopt) wrkopt = opt;
    }

    if (l < n) {
        i32 n_minus_l = n - l;
        SLC_DLASET("F", &l, &n_minus_l, &ZERO, &ZERO, e, &lde);
        if (l >= 2) {
            i32 l_minus_1 = l - 1;
            SLC_DLASET("L", &l_minus_1, &l, &ZERO, &ZERO, &e[1 + (n - l) * lde], &lde);
        }
    } else {
        if (n >= 2) {
            i32 n_minus_1 = n - 1;
            SLC_DLASET("L", &n_minus_1, &n, &ZERO, &ZERO, &e[(l - n + 1)], &lde);
        }
    }

    dwork[0] = (f64)wrkopt;
}
