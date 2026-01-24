/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include <math.h>
#include <stdbool.h>
#include <float.h>
#include <stddef.h>

static bool lsame(char ca, char cb) {
    if (ca >= 'a' && ca <= 'z') ca -= 32;
    if (cb >= 'a' && cb <= 'z') cb -= 32;
    return ca == cb;
}

static i32 imin(i32 a, i32 b) {
    return a < b ? a : b;
}

static i32 imax(i32 a, i32 b) {
    return a > b ? a : b;
}

void mb01qd(char type, i32 m, i32 n, i32 kl, i32 ku,
            f64 cfrom, f64 cto, i32 nbl, const i32* nrows,
            f64* a, i32 lda, i32* info) {

    const f64 zero = 0.0, one = 1.0;
    i32 itype;
    bool done, noblc;
    f64 smlnum, bignum, cfromc, ctoc, cfrom1, cto1, mul;
    i32 i, j, jini, jfin, ifin = 0, k, k1, k2, k3, k4;

    *info = 0;

    if (lsame(type, 'G')) {
        itype = 0;
    } else if (lsame(type, 'L')) {
        itype = 1;
    } else if (lsame(type, 'U')) {
        itype = 2;
    } else if (lsame(type, 'H')) {
        itype = 3;
    } else if (lsame(type, 'B')) {
        itype = 4;
    } else if (lsame(type, 'Q')) {
        itype = 5;
    } else {
        itype = 6;
    }

    if (imin(m, n) == 0)
        return;

    smlnum = DBL_MIN;
    bignum = one / smlnum;

    cfromc = cfrom;
    ctoc = cto;

    do {
        cfrom1 = cfromc * smlnum;
        cto1 = ctoc / bignum;

        if (fabs(cfrom1) > fabs(ctoc) && ctoc != zero) {
            mul = smlnum;
            done = false;
            cfromc = cfrom1;
        } else if (fabs(cto1) > fabs(cfromc)) {
            mul = bignum;
            done = false;
            ctoc = cto1;
        } else {
            mul = ctoc / cfromc;
            done = true;
        }

        noblc = (nbl == 0);

        if (itype == 0) {
            for (j = 0; j < n; j++) {
                for (i = 0; i < m; i++) {
                    a[j * lda + i] *= mul;
                }
            }
        } else if (itype == 1) {
            if (noblc || nrows == NULL) {
                for (j = 0; j < n; j++) {
                    for (i = j; i < m; i++) {
                        a[i + j * lda] *= mul;
                    }
                }
            } else {
                jfin = 0;
                for (k = 0; k < nbl; k++) {
                    jini = jfin;
                    jfin = jfin + nrows[k];
                    for (j = jini; j < jfin; j++) {
                        for (i = jini; i < m; i++) {
                            a[j * lda + i] *= mul;
                        }
                    }
                }
            }
        } else if (itype == 2) {
            if (noblc || nrows == NULL) {
                for (j = 0; j < n; j++) {
                    for (i = 0; i < imin(j + 1, m); i++) {
                        a[j * lda + i] *= mul;
                    }
                }
            } else {
                jfin = 0;
                for (k = 0; k < nbl; k++) {
                    jini = jfin;
                    jfin = jfin + nrows[k];
                    if (k == nbl - 1) jfin = n;
                    for (j = jini; j < jfin; j++) {
                        for (i = 0; i < imin(jfin, m); i++) {
                            a[j * lda + i] *= mul;
                        }
                    }
                }
            }
        } else if (itype == 3) {
            if (noblc || nrows == NULL) {
                for (j = 0; j < n; j++) {
                    for (i = 0; i < imin(j + 2, m); i++) {
                        a[j * lda + i] *= mul;
                    }
                }
            } else {
                jfin = 0;
                for (k = 0; k < nbl; k++) {
                    jini = jfin;
                    jfin = jfin + nrows[k];

                    if (k == nbl - 1) {
                        jfin = n;
                        ifin = n;
                    } else {
                        ifin = jfin + nrows[k + 1];
                    }

                    for (j = jini; j < jfin; j++) {
                        for (i = 0; i < imin(ifin, m); i++) {
                            a[j * lda + i] *= mul;
                        }
                    }
                }
            }
        } else if (itype == 4) {
            k3 = kl + 1;
            k4 = n + 1;
            for (j = 0; j < n; j++) {
                for (i = 0; i < imin(k3, k4 - j - 1); i++) {
                    a[j * lda + i] *= mul;
                }
            }
        } else if (itype == 5) {
            k1 = ku + 2;
            k3 = ku + 1;
            for (j = 0; j < n; j++) {
                for (i = imax(k1 - j - 2, 0); i < k3; i++) {
                    a[j * lda + i] *= mul;
                }
            }
        } else if (itype == 6) {
            k1 = kl + ku + 2;
            k2 = kl + 1;
            k3 = 2 * kl + ku + 1;
            k4 = kl + ku + 1 + m;
            for (j = 0; j < n; j++) {
                for (i = imax(k1 - j - 2, k2 - 1); i < imin(k3, k4 - j - 1); i++) {
                    a[j * lda + i] *= mul;
                }
            }
        }
    } while (!done);
}
