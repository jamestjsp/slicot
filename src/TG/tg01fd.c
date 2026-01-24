/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <stdbool.h>
#include <string.h>

void tg01fd(
    const char* compq, const char* compz, const char* joba,
    const i32 l, const i32 n, const i32 m, const i32 p,
    f64* a, const i32 lda,
    f64* e, const i32 lde,
    f64* b, const i32 ldb,
    f64* c, const i32 ldc,
    f64* q, const i32 ldq,
    f64* z, const i32 ldz,
    i32* ranke, i32* rnka22,
    const f64 tol,
    i32* iwork, f64* dwork, const i32 ldwork,
    i32* info
)
{
    const f64 one = 1.0, zero = 0.0;

    bool ilq, ilz, lquery, reda, redtr, withb, withc;
    i32 i, icompq, icompz, ir1, ire1, j, k, kw, la22;
    i32 lh, ln, lwr, na22, wrkopt;
    f64 svlmax, toldef;
    f64 sval[3];
    i32 int1 = 1;

    *info = 0;

    if (compq[0] == 'N' || compq[0] == 'n') {
        ilq = false;
        icompq = 1;
    } else if (compq[0] == 'U' || compq[0] == 'u') {
        ilq = true;
        icompq = 2;
    } else if (compq[0] == 'I' || compq[0] == 'i') {
        ilq = true;
        icompq = 3;
    } else {
        icompq = 0;
    }

    if (compz[0] == 'N' || compz[0] == 'n') {
        ilz = false;
        icompz = 1;
    } else if (compz[0] == 'U' || compz[0] == 'u') {
        ilz = true;
        icompz = 2;
    } else if (compz[0] == 'I' || compz[0] == 'i') {
        ilz = true;
        icompz = 3;
    } else {
        icompz = 0;
    }

    reda  = (joba[0] == 'R' || joba[0] == 'r');
    redtr = (joba[0] == 'T' || joba[0] == 't');
    withb = (m > 0);
    withc = (p > 0);
    lquery = (ldwork == -1);

    ln = (l < n) ? l : n;
    wrkopt = 1;
    i32 temp1 = n + p;
    i32 temp2 = (3 * n - 1 > m) ? 3 * n - 1 : m;
    temp2 = (temp2 > l) ? temp2 : l;
    temp2 = ln + temp2;
    wrkopt = (temp1 > temp2) ? temp1 : temp2;
    wrkopt = (wrkopt > 1) ? wrkopt : 1;

    if (icompq <= 0) {
        *info = -1;
    } else if (icompz <= 0) {
        *info = -2;
    } else if (joba[0] != 'N' && joba[0] != 'n' && !reda && !redtr) {
        *info = -3;
    } else if (l < 0) {
        *info = -4;
    } else if (n < 0) {
        *info = -5;
    } else if (m < 0) {
        *info = -6;
    } else if (p < 0) {
        *info = -7;
    } else if (lda < ((1 > l) ? 1 : l)) {
        *info = -9;
    } else if (lde < ((1 > l) ? 1 : l)) {
        *info = -11;
    } else if (ldb < 1 || (withb && ldb < l)) {
        *info = -13;
    } else if (ldc < ((1 > p) ? 1 : p)) {
        *info = -15;
    } else if ((ilq && ldq < l) || ldq < 1) {
        *info = -17;
    } else if ((ilz && ldz < n) || ldz < 1) {
        *info = -19;
    } else if (tol >= one) {
        *info = -22;
    } else {
        if (lquery) {
            i32 dummy_info;
            SLC_DORMQR("L", "T", &l, &n, &ln, e, &lde, dwork, a, &lda, dwork, &int1, &dummy_info);
            wrkopt = (wrkopt > ln + (i32)dwork[0]) ? wrkopt : ln + (i32)dwork[0];

            if (withb) {
                SLC_DORMQR("L", "T", &l, &m, &ln, e, &lde, dwork, b, &ldb, dwork, &int1, &dummy_info);
                wrkopt = (wrkopt > ln + (i32)dwork[0]) ? wrkopt : ln + (i32)dwork[0];
            }
            if (ilq) {
                SLC_DORMQR("R", "N", &l, &l, &ln, e, &lde, dwork, q, &ldq, dwork, &int1, &dummy_info);
                wrkopt = (wrkopt > ln + (i32)dwork[0]) ? wrkopt : ln + (i32)dwork[0];
            }
            SLC_DTZRZF(&ln, &n, e, &lde, dwork, dwork, &int1, &dummy_info);
            wrkopt = (wrkopt > ln + (i32)dwork[0]) ? wrkopt : ln + (i32)dwork[0];

            SLC_DORMRZ("R", "T", &l, &n, &ln, &n, e, &lde, dwork, a, &lda, dwork, &int1, &dummy_info);
            wrkopt = (wrkopt > n + (i32)dwork[0]) ? wrkopt : n + (i32)dwork[0];

            if (withc) {
                SLC_DORMRZ("R", "T", &p, &n, &ln, &n, e, &lde, dwork, c, &ldc, dwork, &int1, &dummy_info);
                wrkopt = (wrkopt > n + (i32)dwork[0]) ? wrkopt : n + (i32)dwork[0];
            }
            if (ilz) {
                SLC_DORMRZ("R", "T", &n, &n, &ln, &n, e, &lde, dwork, z, &ldz, dwork, &int1, &dummy_info);
                wrkopt = (wrkopt > n + (i32)dwork[0]) ? wrkopt : n + (i32)dwork[0];
            }
        } else if (ldwork < wrkopt) {
            *info = -25;
        }
    }

    if (*info != 0) {
        return;
    } else if (lquery) {
        dwork[0] = (f64)wrkopt;
        return;
    }

    if (icompq == 3) {
        SLC_DLASET("F", &l, &l, &zero, &one, q, &ldq);
    }
    if (icompz == 3) {
        SLC_DLASET("F", &n, &n, &zero, &one, z, &ldz);
    }

    if (l == 0 || n == 0) {
        dwork[0] = one;
        *ranke = 0;
        if (reda || redtr) {
            *rnka22 = 0;
        }
        return;
    }

    toldef = tol;
    if (toldef <= zero) {
        toldef = (f64)(l * n) * SLC_DLAMCH("E");
    }

    svlmax = zero;

    lwr = ldwork - ln;
    kw = ln + 1;

    mb03oy(l, n, e, lde, toldef, svlmax, ranke, sval, iwork, dwork, &dwork[kw - 1], info);

    if (*ranke > 0) {
        SLC_DORMQR("L", "T", &l, &n, ranke, e, &lde, dwork, a, &lda, &dwork[kw - 1], &lwr, info);
        wrkopt = (wrkopt > ln + (i32)dwork[kw - 1]) ? wrkopt : ln + (i32)dwork[kw - 1];

        if (withb) {
            SLC_DORMQR("L", "T", &l, &m, ranke, e, &lde, dwork, b, &ldb, &dwork[kw - 1], &lwr, info);
            wrkopt = (wrkopt > ln + (i32)dwork[kw - 1]) ? wrkopt : ln + (i32)dwork[kw - 1];
        }

        if (ilq) {
            SLC_DORMQR("R", "N", &l, &l, ranke, e, &lde, dwork, q, &ldq, &dwork[kw - 1], &lwr, info);
            wrkopt = (wrkopt > ln + (i32)dwork[kw - 1]) ? wrkopt : ln + (i32)dwork[kw - 1];
        }

        if (l >= 2) {
            i32 lm1 = l - 1;
            SLC_DLASET("L", &lm1, ranke, &zero, &zero, &e[1], &lde);
        }

        for (j = 0; j < n; j++) {
            iwork[j] = -iwork[j];
        }

        for (i = 0; i < n; i++) {
            if (iwork[i] < 0) {
                j = i;
                iwork[j] = -iwork[j];
                while (true) {
                    k = iwork[j] - 1;
                    if (k < 0 || k >= n) {
                        break;
                    }
                    if (iwork[k] < 0) {
                        SLC_DSWAP(&l, &a[j * lda], &int1, &a[k * lda], &int1);
                        if (withc) {
                            SLC_DSWAP(&p, &c[j * ldc], &int1, &c[k * ldc], &int1);
                        }
                        if (ilz) {
                            SLC_DSWAP(&n, &z[j * ldz], &int1, &z[k * ldz], &int1);
                        }
                        iwork[k] = -iwork[k];
                        j = k;
                    } else {
                        break;
                    }
                }
            }
        }

        if (*ranke < n) {
            kw = *ranke + 1;
            i32 ldwork_minus_kw_plus1 = ldwork - kw + 1;
            SLC_DTZRZF(ranke, &n, e, &lde, dwork, &dwork[kw - 1], &ldwork_minus_kw_plus1, info);
            wrkopt = (wrkopt > (i32)dwork[kw - 1] + kw - 1) ? wrkopt : (i32)dwork[kw - 1] + kw - 1;

            lh = n - *ranke;
            SLC_DORMRZ("R", "T", &l, &n, ranke, &lh, e, &lde, dwork, a, &lda, &dwork[kw - 1], &ldwork_minus_kw_plus1, info);
            wrkopt = (wrkopt > (i32)dwork[kw - 1] + kw - 1) ? wrkopt : (i32)dwork[kw - 1] + kw - 1;

            if (withc) {
                SLC_DORMRZ("R", "T", &p, &n, ranke, &lh, e, &lde, dwork, c, &ldc, &dwork[kw - 1], &ldwork_minus_kw_plus1, info);
                wrkopt = (wrkopt > (i32)dwork[kw - 1] + kw - 1) ? wrkopt : (i32)dwork[kw - 1] + kw - 1;
            }
            if (ilz) {
                SLC_DORMRZ("R", "T", &n, &n, ranke, &lh, e, &lde, dwork, z, &ldz, &dwork[kw - 1], &ldwork_minus_kw_plus1, info);
                wrkopt = (wrkopt > (i32)dwork[kw - 1] + kw - 1) ? wrkopt : (i32)dwork[kw - 1] + kw - 1;
            }

            SLC_DLASET("F", &l, &lh, &zero, &zero, &e[(kw - 1) * lde], &lde);
        }
    } else {
        SLC_DLASET("F", &l, &n, &zero, &zero, e, &lde);
    }

    if (reda || redtr) {
        la22 = l - *ranke;
        na22 = n - *ranke;
        i32 min_la22_na22 = (la22 < na22) ? la22 : na22;

        if (min_la22_na22 == 0) {
            *rnka22 = 0;
        } else {
            svlmax = SLC_DLANGE("F", &l, &n, a, &lda, dwork);
            ir1 = *ranke + 1;
            kw = min_la22_na22 + 1;

            mb03oy(la22, na22, &a[(ir1 - 1) + (ir1 - 1) * lda], lda, toldef, svlmax, rnka22, sval, iwork, dwork, &dwork[kw - 1], info);

            if (*rnka22 > 0) {
                SLC_DORMQR("L", "T", &la22, ranke, rnka22, &a[(ir1 - 1) + (ir1 - 1) * lda], &lda, dwork, &a[(ir1 - 1)], &lda, &dwork[kw - 1], &lwr, info);

                if (withb) {
                    SLC_DORMQR("L", "T", &la22, &m, rnka22, &a[(ir1 - 1) + (ir1 - 1) * lda], &lda, dwork, &b[ir1 - 1], &ldb, &dwork[kw - 1], &lwr, info);
                }

                if (ilq) {
                    SLC_DORMQR("R", "N", &l, &la22, rnka22, &a[(ir1 - 1) + (ir1 - 1) * lda], &lda, dwork, &q[(ir1 - 1) * ldq], &ldq, &dwork[kw - 1], &lwr, info);
                }

                if (la22 >= 2) {
                    i32 la22m1 = la22 - 1;
                    SLC_DLASET("L", &la22m1, rnka22, &zero, &zero, &a[ir1 + (ir1 - 1) * lda], &lda);
                }

                for (j = 0; j < na22; j++) {
                    iwork[j] = -iwork[j];
                }

                for (i = 0; i < na22; i++) {
                    if (iwork[i] < 0) {
                        j = i;
                        iwork[j] = -iwork[j];
                        while (true) {
                            k = iwork[j] - 1;
                            if (k < 0 || k >= na22) {
                                break;
                            }
                            if (iwork[k] < 0) {
                                SLC_DSWAP(ranke, &a[(*ranke + j) * lda], &int1, &a[(*ranke + k) * lda], &int1);
                                if (withc) {
                                    SLC_DSWAP(&p, &c[(*ranke + j) * ldc], &int1, &c[(*ranke + k) * ldc], &int1);
                                }
                                if (ilz) {
                                    SLC_DSWAP(&n, &z[(*ranke + j) * ldz], &int1, &z[(*ranke + k) * ldz], &int1);
                                }
                                iwork[k] = -iwork[k];
                                j = k;
                            } else {
                                break;
                            }
                        }
                    }
                }

                if (reda && *rnka22 < na22) {
                    kw = *ranke + 1;
                    i32 ldwork_minus_kw_plus1 = ldwork - kw + 1;
                    SLC_DTZRZF(rnka22, &na22, &a[(ir1 - 1) + (ir1 - 1) * lda], &lda, dwork, &dwork[kw - 1], &ldwork_minus_kw_plus1, info);
                    wrkopt = (wrkopt > (i32)dwork[kw - 1] + kw - 1) ? wrkopt : (i32)dwork[kw - 1] + kw - 1;

                    lh = na22 - *rnka22;
                    SLC_DORMRZ("R", "T", ranke, &na22, rnka22, &lh, &a[(ir1 - 1) + (ir1 - 1) * lda], &lda, dwork, &a[(ir1 - 1) * lda], &lda, &dwork[kw - 1], &ldwork_minus_kw_plus1, info);

                    if (withc) {
                        SLC_DORMRZ("R", "T", &p, &na22, rnka22, &lh, &a[(ir1 - 1) + (ir1 - 1) * lda], &lda, dwork, &c[(ir1 - 1) * ldc], &ldc, &dwork[kw - 1], &ldwork_minus_kw_plus1, info);
                        wrkopt = (wrkopt > (i32)dwork[kw - 1] + kw - 1) ? wrkopt : (i32)dwork[kw - 1] + kw - 1;
                    }
                    if (ilz) {
                        SLC_DORMRZ("R", "T", &n, &na22, rnka22, &lh, &a[(ir1 - 1) + (ir1 - 1) * lda], &lda, dwork, &z[(ir1 - 1) * ldz], &ldz, &dwork[kw - 1], &ldwork_minus_kw_plus1, info);
                        wrkopt = (wrkopt > (i32)dwork[kw - 1] + kw - 1) ? wrkopt : (i32)dwork[kw - 1] + kw - 1;
                    }

                    ire1 = *ranke + *rnka22 + 1;
                    SLC_DLASET("F", &la22, &lh, &zero, &zero, &a[(ir1 - 1) + (ire1 - 1) * lda], &lda);
                }
            } else {
                SLC_DLASET("F", &la22, &na22, &zero, &zero, &a[(ir1 - 1) + (ir1 - 1) * lda], &lda);
            }
        }
    }

    dwork[0] = (f64)wrkopt;
}
