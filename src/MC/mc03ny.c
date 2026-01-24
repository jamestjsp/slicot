/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"

void mc03ny(i32 nblcks, i32 nra, i32 nca, f64 *a, i32 lda,
            f64 *e, i32 lde, i32 *imuk, const i32 *inuk,
            f64 *veps, i32 ldveps, i32 *info)
{
    *info = 0;

    if (nblcks < 0) {
        *info = -1;
    } else if (nra < 0) {
        *info = -2;
    } else if (nca < 0) {
        *info = -3;
    } else if (lda < (nra > 1 ? nra : 1)) {
        *info = -5;
    } else if (lde < (nra > 1 ? nra : 1)) {
        *info = -7;
    } else if (ldveps < (nca > 1 ? nca : 1)) {
        *info = -11;
    }

    if (*info != 0) {
        i32 neg_info = -(*info);
        SLC_XERBLA("MC03NY", &neg_info);
        return;
    }

    if (nblcks == 0 || nra == 0 || nca == 0) {
        return;
    }

    f64 zero = 0.0;
    f64 one = 1.0;
    f64 minusone = -1.0;
    i32 int1 = 1;

    i32 ec1 = 0;
    i32 ar1 = 0;

    for (i32 i = 0; i < nblcks - 1; i++) {
        i32 nui = inuk[i];
        if (nui == 0) {
            goto label60;
        }
        i32 mui = imuk[i];
        ec1 += mui;
        i32 ac1 = ec1 - nui;

        i32 ncols = nca - ec1;
        SLC_DTRTRS("Upper", "No transpose", "Non-unit", &nui, &ncols,
                   &a[ar1 + ac1 * lda], &lda, &e[ar1 + ec1 * lde], &lde, info);
        if (*info > 0) {
            *info = i + 1;
            return;
        }

        for (i32 j = 0; j < nui; j++) {
            i32 len = j + 1;
            SLC_DSCAL(&len, &minusone, &a[ar1 + (ac1 + j) * lda], &int1);
        }

        SLC_DTRTRS("Upper", "No transpose", "Non-unit", &nui, &ncols,
                   &a[ar1 + ac1 * lda], &lda, &a[ar1 + ec1 * lda], &lda, info);

        ar1 += nui;
    }

label60:;

    i32 smui = 0;
    i32 ncv = 0;

    for (i32 i = 0; i < nblcks; i++) {
        i32 mui = imuk[i];
        smui += mui;
        imuk[i] = smui;
        ncv += (i + 1) * (mui - inuk[i]);
    }

    i32 nrv = nca;

    SLC_DLASET("Full", &nrv, &ncv, &zero, &zero, veps, &ldveps);

    i32 nui = imuk[0] - inuk[0];
    for (i32 k = 0; k < nui; k++) {
        veps[k + k * ldveps] = one;
    }

    i32 wr1 = imuk[0];
    i32 wc1 = nui;

    for (i32 i = 1; i < nblcks; i++) {
        nui = imuk[i] - imuk[i - 1] - inuk[i];
        for (i32 k = 0; k < nui; k++) {
            veps[(wr1 + k) + (wc1 + k) * ldveps] = one;
        }
        wr1 = imuk[i];
        wc1 += (i + 1) * nui;
    }

    i32 vc1 = imuk[0] - inuk[0];
    i32 ari = 0;

    for (i32 j = 1; j < nblcks; j++) {
        i32 dif = imuk[j] - imuk[j - 1] - inuk[j];
        ari += inuk[j - 1];
        i32 ark = ari;

        for (i32 k = 0; k <= j - 1; k++) {
            i32 vc2 = vc1 + dif - 1;
            i32 ac2 = imuk[j - k] - 1;
            ar1 = ark;
            if (j - k - 2 >= 0) {
                ark -= inuk[j - k - 1];
            }

            for (i32 i = j - k - 2; i >= 0; i--) {
                i32 vr2 = imuk[i] - 1;
                i32 ac1_loop = vr2 + 1;
                i32 vr1 = ac1_loop - inuk[i];
                ar1 -= inuk[i];

                i32 m_rows = inuk[i];
                i32 n_cols = dif;
                i32 k_inner = ac2 - vr2;
                if (m_rows > 0 && n_cols > 0 && k_inner > 0) {
                    SLC_DGEMM("No transpose", "No transpose", &m_rows, &n_cols, &k_inner,
                              &one, &a[ar1 + ac1_loop * lda], &lda,
                              &veps[ac1_loop + vc1 * ldveps], &ldveps,
                              &one, &veps[vr1 + vc1 * ldveps], &ldveps);
                }
            }

            i32 er1 = 0;

            for (i32 i = 0; i < j - k; i++) {
                i32 vr2_e = imuk[i] - 1;
                i32 ec1_loop = vr2_e + 1;
                i32 vr1_e = ec1_loop - inuk[i];

                i32 m_rows = inuk[i];
                i32 n_cols = dif;
                i32 k_inner = ac2 - vr2_e;
                if (m_rows > 0 && n_cols > 0 && k_inner > 0) {
                    SLC_DGEMM("No transpose", "No transpose", &m_rows, &n_cols, &k_inner,
                              &one, &e[er1 + ec1_loop * lde], &lde,
                              &veps[ec1_loop + vc1 * ldveps], &ldveps,
                              &zero, &veps[vr1_e + (vc2 + 1) * ldveps], &ldveps);
                }
                er1 += inuk[i];
            }

            vc1 = vc2 + 1;
        }

        vc1 += dif;
    }

    i32 smui1 = 0;

    for (i32 i = 0; i < nblcks; i++) {
        smui = imuk[i];
        imuk[i] = smui - smui1;
        smui1 = smui;
    }
}
