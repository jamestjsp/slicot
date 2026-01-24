/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <stdbool.h>

void mb03qd(
    const char* dico,
    const char* stdom,
    const char* jobu,
    const i32 n,
    const i32 nlow,
    const i32 nsup,
    const f64 alpha,
    f64* a,
    const i32 lda,
    f64* u,
    const i32 ldu,
    i32* ndim,
    f64* dwork,
    i32* info
)
{
    const f64 ONE = 1.0;
    const f64 ZERO = 0.0;

    bool discr = (*dico == 'D' || *dico == 'd');
    bool cont = (*dico == 'C' || *dico == 'c');
    bool lstdom = (*stdom == 'S' || *stdom == 's');
    bool unstdom = (*stdom == 'U' || *stdom == 'u');
    bool initu = (*jobu == 'I' || *jobu == 'i');
    bool updateu = (*jobu == 'U' || *jobu == 'u');

    *info = 0;

    if (!cont && !discr) {
        *info = -1;
    } else if (!lstdom && !unstdom) {
        *info = -2;
    } else if (!initu && !updateu) {
        *info = -3;
    } else if (n < 1) {
        *info = -4;
    } else if (nlow < 1) {
        *info = -5;
    } else if (nlow > nsup || nsup > n) {
        *info = -6;
    } else if (discr && alpha < ZERO) {
        *info = -7;
    } else if (lda < n) {
        *info = -9;
    } else if (ldu < n) {
        *info = -11;
    }

    if (*info != 0) {
        i32 neginfo = -(*info);
        SLC_XERBLA("MB03QD", &neginfo);
        return;
    }

    i32 nlow0 = nlow - 1;
    i32 nsup0 = nsup - 1;

    if (nlow0 > 0) {
        if (a[nlow0 + (nlow0-1)*lda] != ZERO) {
            *info = 1;
        }
    }
    if (nsup0 < n - 1) {
        if (a[(nsup0+1) + nsup0*lda] != ZERO) {
            *info = 1;
        }
    }
    if (*info != 0) {
        return;
    }

    if (initu) {
        i32 nn = n;
        f64 one = ONE;
        f64 zero = ZERO;
        SLC_DLASET("Full", &nn, &nn, &zero, &one, u, &ldu);
    }

    *ndim = 0;
    i32 l = nsup0;
    i32 nup = nsup0;

    while (l >= nlow0) {
        i32 ib = 1;
        f64 e1 = 0.0, e2 = 0.0;
        if (l > nlow0) {
            i32 lm1 = l - 1;
            if (a[l + lm1*lda] != ZERO) {
                i32 lfort = lm1 + 1;
                mb03qy(n, lfort, a, lda, u, ldu, &e1, &e2, info);
                if (a[l + lm1*lda] != ZERO) {
                    ib = 2;
                }
            }
        }

        f64 tlambd;
        if (discr) {
            if (ib == 1) {
                tlambd = fabs(a[l + l*lda]);
            } else {
                tlambd = sqrt(e1*e1 + e2*e2);
            }
        } else {
            if (ib == 1) {
                tlambd = a[l + l*lda];
            } else {
                tlambd = e1;
            }
        }

        bool in_domain;
        if (lstdom) {
            in_domain = (tlambd < alpha);
        } else {
            in_domain = (tlambd > alpha);
        }

        if (in_domain) {
            *ndim = *ndim + ib;
            l = l - ib;
        } else {
            if (*ndim != 0) {
                i32 lfort = l + 1;
                i32 nupfort = nup + 1;
                SLC_DTREXC("V", &n, a, &lda, u, &ldu, &lfort, &nupfort, dwork, info);
                if (*info != 0) {
                    *info = 2;
                    return;
                }
                // DTREXC returns updated positions in Fortran 1-based indices
                // Convert to C 0-based and apply the Fortran's "- 1" adjustment
                nup = nupfort - 2;  // nupfort - 1 (to 0-based) - 1 (Fortran logic)
                l = lfort - 2;       // lfort - 1 (to 0-based) - 1 (Fortran logic)
            } else {
                nup = nup - ib;
                l = l - ib;
            }
        }
    }
}
