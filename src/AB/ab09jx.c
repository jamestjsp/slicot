/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 2025, slicot.c contributors
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <stdbool.h>

i32 ab09jx(const char* dico, const char* stdom, const char* evtype,
           i32 n, f64 alpha, f64* er, f64* ei, f64* ed, f64 tolinf) {

    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    i32 info = 0;

    bool discr = (dico[0] == 'D' || dico[0] == 'd');
    bool conti = (dico[0] == 'C' || dico[0] == 'c');
    bool stab = (stdom[0] == 'S' || stdom[0] == 's');
    bool unstab = (stdom[0] == 'U' || stdom[0] == 'u');
    bool stdevp = (evtype[0] == 'S' || evtype[0] == 's');
    bool genevp = (evtype[0] == 'G' || evtype[0] == 'g');
    bool recevp = (evtype[0] == 'R' || evtype[0] == 'r');

    if (!conti && !discr) {
        info = -1;
        i32 neginfo = 1;
        SLC_XERBLA("AB09JX", &neginfo);
        return info;
    }
    if (!stab && !unstab) {
        info = -2;
        i32 neginfo = 2;
        SLC_XERBLA("AB09JX", &neginfo);
        return info;
    }
    if (!stdevp && !genevp && !recevp) {
        info = -3;
        i32 neginfo = 3;
        SLC_XERBLA("AB09JX", &neginfo);
        return info;
    }
    if (n < 0) {
        info = -4;
        i32 neginfo = 4;
        SLC_XERBLA("AB09JX", &neginfo);
        return info;
    }
    if (discr && alpha < ZERO) {
        info = -5;
        i32 neginfo = 5;
        SLC_XERBLA("AB09JX", &neginfo);
        return info;
    }
    if (tolinf < ZERO || tolinf >= ONE) {
        info = -9;
        i32 neginfo = 9;
        SLC_XERBLA("AB09JX", &neginfo);
        return info;
    }

    if (n == 0) {
        return 0;
    }

    f64 scale = ONE;

    if (stab) {
        if (discr) {
            for (i32 i = 0; i < n; i++) {
                f64 absev = SLC_DLAPY2(&er[i], &ei[i]);
                if (recevp) {
                    scale = absev;
                    absev = fabs(ed[i]);
                } else if (!stdevp) {
                    scale = ed[i];
                }
                if (fabs(scale) > tolinf && absev >= alpha * scale) {
                    return 1;
                }
            }
        } else {
            for (i32 i = 0; i < n; i++) {
                f64 rpev = er[i];
                if (recevp) {
                    scale = rpev;
                    rpev = ed[i];
                } else if (!stdevp) {
                    scale = ed[i];
                }
                if (fabs(scale) > tolinf && rpev >= alpha * scale) {
                    return 1;
                }
            }
        }
    } else {
        if (discr) {
            for (i32 i = 0; i < n; i++) {
                f64 absev = SLC_DLAPY2(&er[i], &ei[i]);
                if (recevp) {
                    scale = absev;
                    absev = fabs(ed[i]);
                } else if (!stdevp) {
                    scale = ed[i];
                }
                if (fabs(scale) > tolinf && absev <= alpha * scale) {
                    return 1;
                }
            }
        } else {
            for (i32 i = 0; i < n; i++) {
                f64 rpev = er[i];
                if (recevp) {
                    scale = rpev;
                    rpev = ed[i];
                } else if (!stdevp) {
                    scale = ed[i];
                }
                if (fabs(scale) > tolinf && rpev <= alpha * scale) {
                    return 1;
                }
            }
        }
    }

    return 0;
}
