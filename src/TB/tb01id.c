/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <stdbool.h>

static bool lsame(char ca, char cb) {
    if (ca >= 'a' && ca <= 'z') ca -= 32;
    if (cb >= 'a' && cb <= 'z') cb -= 32;
    return ca == cb;
}

static i32 imax(i32 a, i32 b) {
    return a > b ? a : b;
}

static f64 fmax2(f64 a, f64 b) {
    return a > b ? a : b;
}

static f64 fmin2(f64 a, f64 b) {
    return a < b ? a : b;
}

static f64 fmax3(f64 a, f64 b, f64 c) {
    return fmax2(a, fmax2(b, c));
}

static f64 fmin3(f64 a, f64 b, f64 c) {
    return fmin2(a, fmin2(b, c));
}

void tb01id(const char* job, i32 n, i32 m, i32 p, f64* maxred,
            f64* a, i32 lda, f64* b, i32 ldb, f64* c, i32 ldc,
            f64* scale, i32* info) {

    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const f64 SCLFAC = 10.0;
    const f64 FACTOR = 0.95;
    const f64 MAXR = 10.0;

    bool withb, withc, noconv;
    i32 i, j, ica, ira;
    f64 ca, co, f, g, ra, ro, s, snorm, maxnrm, sred;
    f64 sfmin1, sfmax1, sfmin2, sfmax2;

    const i32 one_i = 1;

    *info = 0;

    withb = lsame(*job, 'A') || lsame(*job, 'B');
    withc = lsame(*job, 'A') || lsame(*job, 'C');

    if (!withb && !withc && !lsame(*job, 'N')) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (m < 0) {
        *info = -3;
    } else if (p < 0) {
        *info = -4;
    } else if (*maxred > ZERO && *maxred < ONE) {
        *info = -5;
    } else if (lda < imax(1, n)) {
        *info = -7;
    } else if ((m > 0 && ldb < imax(1, n)) || (m == 0 && ldb < 1)) {
        *info = -9;
    } else if (ldc < imax(1, p)) {
        *info = -11;
    }

    if (*info != 0) {
        return;
    }

    if (n == 0) {
        return;
    }

    snorm = ZERO;
    for (j = 0; j < n; j++) {
        scale[j] = ONE;
        co = SLC_DASUM(&n, &a[j * lda], &one_i);
        if (withc && p > 0) {
            co += SLC_DASUM(&p, &c[j * ldc], &one_i);
        }
        snorm = fmax2(snorm, co);
    }

    if (withb) {
        for (j = 0; j < m; j++) {
            snorm = fmax2(snorm, SLC_DASUM(&n, &b[j * ldb], &one_i));
        }
    }

    if (snorm == ZERO) {
        return;
    }

    sfmin1 = SLC_DLAMCH("S") / SLC_DLAMCH("P");
    sfmax1 = ONE / sfmin1;
    sfmin2 = sfmin1 * SCLFAC;
    sfmax2 = ONE / sfmin2;

    sred = *maxred;
    if (sred <= ZERO) sred = MAXR;

    maxnrm = fmax2(snorm / sred, sfmin1);

    do {
        noconv = false;

        for (i = 0; i < n; i++) {
            co = ZERO;
            ro = ZERO;

            for (j = 0; j < n; j++) {
                if (j != i) {
                    co += fabs(a[j + i * lda]);
                    ro += fabs(a[i + j * lda]);
                }
            }

            ica = SLC_IDAMAX(&n, &a[i * lda], &one_i) - 1;
            if (ica < 0 || ica >= n) ica = 0;
            ca = fabs(a[ica + i * lda]);

            ira = SLC_IDAMAX(&n, &a[i], &lda) - 1;
            if (ira < 0 || ira >= n) ira = 0;
            ra = fabs(a[i + ira * lda]);

            if (withc && p > 0) {
                co += SLC_DASUM(&p, &c[i * ldc], &one_i);
                i32 ica_c = SLC_IDAMAX(&p, &c[i * ldc], &one_i) - 1;
                if (ica_c >= 0 && ica_c < p) {
                    ca = fmax2(ca, fabs(c[ica_c + i * ldc]));
                }
            }

            if (withb && m > 0) {
                ro += SLC_DASUM(&m, &b[i], &ldb);
                i32 ira_b = SLC_IDAMAX(&m, &b[i], &ldb) - 1;
                if (ira_b >= 0 && ira_b < m) {
                    ra = fmax2(ra, fabs(b[i + ira_b * ldb]));
                }
            }

            if (co == ZERO && ro == ZERO) {
                continue;
            }
            if (co == ZERO) {
                if (ro <= maxnrm) continue;
                co = maxnrm;
            }
            if (ro == ZERO) {
                if (co <= maxnrm) continue;
                ro = maxnrm;
            }

            g = ro / SCLFAC;
            f = ONE;
            s = co + ro;

            while (co < g && fmax3(f, co, ca) < sfmax2 && fmin3(ro, g, ra) > sfmin2) {
                f *= SCLFAC;
                co *= SCLFAC;
                ca *= SCLFAC;
                g /= SCLFAC;
                ro /= SCLFAC;
                ra /= SCLFAC;
            }

            g = co / SCLFAC;
            while (g >= ro && fmax2(ro, ra) < sfmax2 && fmin2(fmin2(f, co), fmin2(g, ca)) > sfmin2) {
                f /= SCLFAC;
                co /= SCLFAC;
                ca /= SCLFAC;
                g /= SCLFAC;
                ro *= SCLFAC;
                ra *= SCLFAC;
            }

            if ((co + ro) >= FACTOR * s) {
                continue;
            }
            if (f < ONE && scale[i] < ONE) {
                if (f * scale[i] <= sfmin1) continue;
            }
            if (f > ONE && scale[i] > ONE) {
                if (scale[i] >= sfmax1 / f) continue;
            }

            g = ONE / f;
            scale[i] *= f;
            noconv = true;

            SLC_DSCAL(&n, &g, &a[i], &lda);
            SLC_DSCAL(&n, &f, &a[i * lda], &one_i);
            if (m > 0) SLC_DSCAL(&m, &g, &b[i], &ldb);
            if (p > 0) SLC_DSCAL(&p, &f, &c[i * ldc], &one_i);
        }
    } while (noconv);

    f64 orig_snorm = snorm;
    snorm = ZERO;

    for (j = 0; j < n; j++) {
        co = SLC_DASUM(&n, &a[j * lda], &one_i);
        if (withc && p > 0) {
            co += SLC_DASUM(&p, &c[j * ldc], &one_i);
        }
        snorm = fmax2(snorm, co);
    }

    if (withb) {
        for (j = 0; j < m; j++) {
            snorm = fmax2(snorm, SLC_DASUM(&n, &b[j * ldb], &one_i));
        }
    }

    *maxred = orig_snorm / snorm;
}
