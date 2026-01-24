/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>

void mb03rx(
    const char* jobv,
    const i32 n,
    const i32 kl,
    i32* ku,
    f64* a,
    const i32 lda,
    f64* x,
    const i32 ldx,
    f64* wr,
    f64* wi,
    f64* dwork
)
{
    const f64 ZERO = 0.0;

    if (*ku <= kl) {
        return;
    }

    i32 ifst = *ku;
    i32 ilst;
    i32 ierr;

    do {
        ilst = kl;
        SLC_DTREXC(jobv, &n, a, &lda, x, &ldx, &ifst, &ilst, dwork, &ierr);

        if (ierr != 0) {
            ifst = ilst - 1;
            if (ifst > 1) {
                if (a[(ifst - 1) + (ifst - 2) * lda] != ZERO) {
                    ifst = ilst - 2;
                }
            }
            if (ilst <= kl) {
                break;
            }
        }
    } while (ierr != 0 && ilst > kl);

    if (wi[*ku - 1] != ZERO) {
        if (a[*ku + (*ku - 1) * lda] == ZERO) {
            *ku = *ku + 1;
        }
    }

    i32 l = kl;
    while (l < *ku || (l == *ku && l < n)) {
        if (a[l + (l - 1) * lda] != ZERO) {
            wr[l - 1] = a[(l - 1) + (l - 1) * lda];
            wr[l] = wr[l - 1];
            wi[l - 1] = sqrt(fabs(a[(l - 1) + l * lda])) *
                        sqrt(fabs(a[l + (l - 1) * lda]));
            wi[l] = -wi[l - 1];
            l = l + 2;
        } else {
            wr[l - 1] = a[(l - 1) + (l - 1) * lda];
            wi[l - 1] = ZERO;
            l = l + 1;
        }
    }
    if (l == n) {
        wr[l - 1] = a[(l - 1) + (l - 1) * lda];
        wi[l - 1] = ZERO;
    }
}
