/*
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * MB03KD - Reorder diagonal blocks in generalized periodic Schur form
 *
 * Reorders the diagonal blocks of a formal matrix product
 * T22_K^S(K) * T22_K-1^S(K-1) * ... * T22_1^S(1) of length K
 * such that M selected eigenvalues end up in the leading part
 * of the matrix sequence T22_k.
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

void mb03kd(const char* compq, const i32* whichq, const char* strong,
            const i32 k, const i32 nc, const i32 kschur,
            const i32* n, const i32* ni, const i32* s,
            const bool* select, f64* t, const i32* ldt, const i32* ixt,
            f64* q, const i32* ldq, const i32* ixq,
            i32* m, const f64 tol, i32* iwork, f64* dwork,
            const i32 ldwork, i32* info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    bool initq, wantq, specq, ws, wantql, pair, swap;
    char compqc;
    i32 i, ip1, it, l, ll, ls, maxn, minn, minsum, mnwork, nkp1, sumd;
    f64 tola[3];
    i32 whichq_local[32];

    *info = 0;

    bool compq_is_I = (compq[0] == 'I' || compq[0] == 'i');
    bool compq_is_U = (compq[0] == 'U' || compq[0] == 'u');
    bool compq_is_N = (compq[0] == 'N' || compq[0] == 'n');
    bool compq_is_W = (compq[0] == 'W' || compq[0] == 'w');

    initq = compq_is_I;
    wantq = compq_is_U || initq;
    specq = compq_is_W;
    ws = (strong[0] == 'S' || strong[0] == 's');

    if (k < 2) {
        *info = -4;
    } else if (!compq_is_N && !wantq && !specq) {
        *info = -1;
    } else if (!(strong[0] == 'N' || strong[0] == 'n' || ws)) {
        *info = -3;
    } else if (tol <= ZERO) {
        *info = -18;
    }

    if (*info == 0 && specq) {
        for (l = 0; l < k; l++) {
            if (whichq[l] < 0 || whichq[l] > 2) {
                *info = -2;
                break;
            }
        }
    }

    sumd = 0;
    if (*info == 0) {
        minsum = 0;
        maxn = 0;
        minn = n[k - 1];

        for (l = 0; l < k; l++) {
            if (l < k - 1 && n[l] < minn) {
                minn = n[l];
            }
            nkp1 = n[(l + 1) % k];
            if (n[l] < 0) {
                *info = -7;
                break;
            }
            if (s[l] == -1) {
                sumd = sumd + (nkp1 - n[l]);
            }
            if (sumd < minsum) {
                minsum = sumd;
            }
            if (n[l] > maxn) {
                maxn = n[l];
            }

            if (*info == 0 && (n[l] < ni[l] + nc || ni[l] < 0)) {
                *info = -8;
                break;
            }
        }
    }

    if (*info == 0 && (nc < 0 || nc > minn)) {
        *info = -5;
    }

    if (*info == 0 && (kschur < 1 || kschur > k)) {
        *info = -6;
    }

    if (*info == 0 && sumd != 0) {
        *info = -7;
    }

    if (*info == 0) {
        for (l = 0; l < k; l++) {
            if (abs(s[l]) != 1) {
                *info = -9;
                break;
            }
        }
    }

    if (*info == 0) {
        for (l = 0; l < k; l++) {
            nkp1 = n[(l + 1) % k];
            if (s[l] == 1) {
                if (ldt[l] < (nkp1 > 1 ? nkp1 : 1)) {
                    *info = -12;
                    break;
                }
            } else {
                if (ldt[l] < (n[l] > 1 ? n[l] : 1)) {
                    *info = -12;
                    break;
                }
            }
        }
    }

    if (*info == 0 && (wantq || specq)) {
        for (l = 0; l < k; l++) {
            wantql = wantq;
            if (specq) {
                wantql = (whichq[l] != 0);
            }
            if (wantql) {
                if (ldq[l] < (n[l] > 1 ? n[l] : 1)) {
                    *info = -15;
                    break;
                }
            }
        }
    }

    *m = 0;
    i = kschur - 1;
    pair = false;
    ip1 = (i + 1) % k;

    for (l = 0; l < nc; l++) {
        if (pair) {
            pair = false;
        } else {
            if (l < nc - 1) {
                if (s[i] == 1) {
                    it = ixt[i] - 1 + (ni[i] + l) * ldt[i] + ni[ip1] + l + 1;
                } else {
                    it = ixt[i] - 1 + (ni[ip1] + l) * ldt[i] + ni[i] + l + 1;
                }
                if (t[it] == ZERO) {
                    if (select[l]) {
                        (*m)++;
                    }
                } else {
                    pair = true;
                    if (select[l] || select[l + 1]) {
                        *m += 2;
                    }
                }
            } else {
                if (select[nc - 1]) {
                    (*m)++;
                }
            }
        }
    }

    if (initq) {
        compqc = 'U';
    } else {
        compqc = compq[0];
    }

    if (specq) {
        for (l = 0; l < k && l < 32; l++) {
            whichq_local[l] = whichq[l];
        }
    }

    if (*info == 0) {
        i32 ifst_dummy = 1;
        i32 ilst_dummy = 1;
        i32 info_local = 0;
        mb03ka(&compqc, specq ? whichq_local : NULL, ws, k, nc, kschur,
               &ifst_dummy, &ilst_dummy, n, ni, s, t, ldt, ixt, q, ldq, ixq,
               tola, iwork, dwork, -1, &info_local);
        mnwork = (i32)dwork[0];
        if (mnwork < 1) mnwork = 1;
        if (ldwork != -1 && ldwork < mnwork) {
            *info = -21;
        }
    }

    if (*info < 0) {
        i32 neg_info = -(*info);
        SLC_XERBLA("MB03KD", &neg_info);
        return;
    } else if (ldwork == -1) {
        dwork[0] = (f64)mnwork;
        return;
    }

    tola[0] = tol;
    tola[1] = SLC_DLAMCH("P");
    tola[2] = SLC_DLAMCH("S") / tola[1];

    for (l = 0; l < k; l++) {
        if (specq) {
            initq = (whichq[l] == 1);
        }
        if (initq) {
            i32 nl = n[l];
            i32 ldql = ldq[l];
            SLC_DLASET("A", &nl, &nl, &ZERO, &ONE, &q[ixq[l] - 1], &ldql);
        }
    }

    ls = 0;
    pair = false;
    i = kschur - 1;
    ip1 = (i + 1) % k;

    for (l = 0; l < nc; l++) {
        if (pair) {
            pair = false;
        } else {
            swap = select[l];
            if (l < nc - 1) {
                if (s[i] == 1) {
                    it = ixt[i] - 1 + (ni[i] + l) * ldt[i] + ni[ip1] + l + 1;
                } else {
                    it = ixt[i] - 1 + (ni[ip1] + l) * ldt[i] + ni[i] + l + 1;
                }
                if (t[it] != ZERO) {
                    pair = true;
                    swap = swap || select[l + 1];
                }
            }
            if (swap) {
                ls++;

                ll = l + 1;
                if (ll != ls) {
                    i32 ifst_val = ll;
                    i32 ilst_val = ls;
                    mb03ka(&compqc, specq ? whichq_local : NULL, ws, k, nc, kschur,
                           &ifst_val, &ilst_val, n, ni, s, t, ldt, ixt,
                           q, ldq, ixq, tola, iwork, dwork, ldwork, info);
                    if (*info != 0) {
                        dwork[0] = (f64)mnwork;
                        return;
                    }
                }
                if (pair) {
                    ls++;
                }
            }
        }
    }

    dwork[0] = (f64)mnwork;
}
