// SPDX-License-Identifier: BSD-3-Clause

#include "slicot.h"
#include "slicot_blas.h"

void mb03vy(i32 n, i32 p, i32 ilo, i32 ihi, f64* a, i32 lda1, i32 lda2,
            f64* tau, i32 ldtau, f64* dwork, i32 ldwork, i32* info) {
    const f64 zero = 0.0;
    const f64 one = 1.0;

    *info = 0;

    i32 max_1_n = (1 > n) ? 1 : n;
    i32 min_ilo_n = (ilo < n) ? ilo : n;
    i32 nh = ihi - ilo + 1;
    bool lquery = (ldwork == -1);

    if (n < 0) {
        *info = -1;
    } else if (p < 1) {
        *info = -2;
    } else if (ilo < 1 || ilo > max_1_n) {
        *info = -3;
    } else if (ihi < min_ilo_n || ihi > n) {
        *info = -4;
    } else if (lda1 < max_1_n) {
        *info = -6;
    } else if (lda2 < max_1_n) {
        *info = -7;
    } else if (ldtau < (n > 1 ? n - 1 : 1)) {
        *info = -9;
    } else {
        if (lquery) {
            i32 wrkopt = (1 > n) ? 1 : n;
            i32 lw_info = 0;
            i32 lw_query = -1;

            SLC_DORGHR(&n, &ilo, &ihi, a, &lda1, tau, dwork, &lw_query, &lw_info);
            i32 opt1 = (i32)dwork[0];
            wrkopt = (wrkopt > opt1) ? wrkopt : opt1;

            if (nh > 1) {
                i32 nh_m1 = nh - 1;
                SLC_DORGQR(&nh, &nh, &nh_m1, a, &lda1, tau, dwork, &lw_query, &lw_info);
                i32 opt2 = (i32)dwork[0];
                wrkopt = (wrkopt > opt2) ? wrkopt : opt2;
            }
            dwork[0] = (f64)wrkopt;
            return;
        }
        if (ldwork < ((1 > n) ? 1 : n)) {
            *info = -11;
        }
    }

    if (*info != 0) {
        return;
    }

    if (n == 0) {
        dwork[0] = one;
        return;
    }

    i32 lda12 = lda1 * lda2;

    SLC_DORGHR(&n, &ilo, &ihi, a, &lda1, tau, dwork, &ldwork, info);
    i32 wrkopt = (i32)dwork[0];

    for (i32 j = 1; j < p; j++) {
        f64* aj = a + j * lda12;
        f64* tau_j = tau + j * ldtau;

        i32 ilo_m1 = ilo - 1;
        SLC_DLASET("Full", &n, &ilo_m1, &zero, &one, aj, &lda1);

        i32 ilo_m1_rows = ilo - 1;
        SLC_DLASET("Full", &ilo_m1_rows, &nh, &zero, &zero, &aj[0 + (ilo - 1) * lda1], &lda1);

        if (nh > 1) {
            i32 nh_m1 = nh - 1;
            f64* aj_ilo_ilo = &aj[(ilo - 1) + (ilo - 1) * lda1];
            SLC_DORGQR(&nh, &nh, &nh_m1, aj_ilo_ilo, &lda1, &tau_j[ilo - 1], dwork, &ldwork, info);
        } else {
            aj[(ilo - 1) + (ilo - 1) * lda1] = one;
        }

        if (ihi < n) {
            i32 n_ihi = n - ihi;
            SLC_DLASET("Full", &n_ihi, &nh, &zero, &zero, &aj[ihi + (ilo - 1) * lda1], &lda1);

            SLC_DLASET("Full", &ihi, &n_ihi, &zero, &zero, &aj[0 + ihi * lda1], &lda1);

            SLC_DLASET("Full", &n_ihi, &n_ihi, &zero, &one, &aj[ihi + ihi * lda1], &lda1);
        }
    }

    i32 opt_final = (i32)dwork[0];
    dwork[0] = (f64)((wrkopt > opt_final) ? wrkopt : opt_final);
}
