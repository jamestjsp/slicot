/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <stdbool.h>
#include <string.h>

void tg01kd(
    const char* jobe, const char* compc, const char* compq, const char* compz,
    const i32 n,
    f64* a, const i32 lda,
    f64* e, const i32 lde,
    f64* b,
    f64* c, const i32 incc,
    f64* q, const i32 ldq,
    f64* z, const i32 ldz,
    i32* info
)
{
    const f64 one = 1.0, zero = 0.0;
    bool liniq, liniz, lupdq, lupdz, unite, withc, withq, withz;
    i32 ic, k;
    f64 cs, sn, temp;
    i32 int1 = 1;

    unite = (jobe[0] == 'I' || jobe[0] == 'i');
    withc = (compc[0] == 'C' || compc[0] == 'c');
    liniq = (compq[0] == 'I' || compq[0] == 'i');
    lupdq = (compq[0] == 'U' || compq[0] == 'u');
    liniz = (compz[0] == 'I' || compz[0] == 'i');
    lupdz = (compz[0] == 'U' || compz[0] == 'u');
    withq = liniq || lupdq;
    withz = liniz || lupdz;
    *info = 0;

    i32 max1n = (1 > n) ? 1 : n;

    if (!unite && !(jobe[0] == 'U' || jobe[0] == 'u')) {
        *info = -1;
    } else if (!withc && !(compc[0] == 'N' || compc[0] == 'n')) {
        *info = -2;
    } else if (!withq && !(compq[0] == 'N' || compq[0] == 'n')) {
        *info = -3;
    } else if (!withz && !(compz[0] == 'N' || compz[0] == 'n')) {
        *info = -4;
    } else if (n < 0) {
        *info = -5;
    } else if (lda < max1n) {
        *info = -7;
    } else if (lde < 1 || (!unite && lde < max1n)) {
        *info = -9;
    } else if (withc && incc <= 0) {
        *info = -12;
    } else if (ldq < 1 || (withq && ldq < max1n)) {
        *info = -14;
    } else if (ldz < 1 || (withz && ldz < max1n)) {
        *info = -16;
    }

    if (*info != 0) {
        return;
    }

    if (n == 0) {
        return;
    }

    if (liniq || (n == 1 && !lupdq)) {
        SLC_DLASET("Full", &n, &n, &zero, &one, q, &ldq);
    }
    if (liniz || (n == 1 && !lupdq)) {
        SLC_DLASET("Full", &n, &n, &zero, &one, z, &ldz);
    }
    if (n == 1) {
        return;
    }

    if (withc) {
        ic = (n - 1) * incc;
    }

    for (k = n - 1; k >= 1; k--) {
        if (b[k] != zero) {
            SLC_DLARTG(&b[k - 1], &b[k], &cs, &sn, &temp);
            b[k - 1] = temp;
            b[k] = zero;

            SLC_DROT(&n, &a[k - 1], &lda, &a[k], &lda, &cs, &sn);

            if (withq) {
                SLC_DROT(&n, &q[(k - 1) * ldq], &int1, &q[k * ldq], &int1, &cs, &sn);
            }

            if (unite) {
                SLC_DROT(&n, &a[(k - 1) * lda], &int1, &a[k * lda], &int1, &cs, &sn);

                if (withc) {
                    temp = c[ic] * sn + c[ic - incc] * cs;
                    c[ic] = c[ic] * cs - c[ic - incc] * sn;
                    ic = ic - incc;
                    c[ic] = temp;
                }

                if (withz) {
                    if (withq && (liniq == liniz || lupdq == lupdz)) {
                        i32 two = 2;
                        SLC_DLACPY("Full", &n, &two, &q[(k - 1) * ldq], &ldq,
                                   &z[(k - 1) * ldz], &ldz);
                    } else {
                        SLC_DROT(&n, &z[(k - 1) * ldz], &int1, &z[k * ldz], &int1, &cs, &sn);
                    }
                }
            } else {
                e[k + (k - 1) * lde] = sn * e[(k - 1) + (k - 1) * lde];
                e[(k - 1) + (k - 1) * lde] = cs * e[(k - 1) + (k - 1) * lde];

                i32 nmk = n - k;
                SLC_DROT(&nmk, &e[(k - 1) + k * lde], &lde,
                         &e[k + k * lde], &lde, &cs, &sn);

                if (e[k + (k - 1) * lde] != zero) {
                    SLC_DLARTG(&e[k + k * lde], &e[k + (k - 1) * lde], &cs, &sn, &temp);
                    e[k + k * lde] = temp;
                    e[k + (k - 1) * lde] = zero;

                    temp = e[(k - 1) + k * lde] * sn + e[(k - 1) + (k - 1) * lde] * cs;
                    e[(k - 1) + k * lde] = e[(k - 1) + k * lde] * cs - e[(k - 1) + (k - 1) * lde] * sn;
                    e[(k - 1) + (k - 1) * lde] = temp;

                    i32 km1 = k - 1;
                    if (km1 > 0) {
                        SLC_DROT(&km1, &e[(k - 1) * lde], &int1, &e[k * lde], &int1, &cs, &sn);
                    }

                    SLC_DROT(&n, &a[(k - 1) * lda], &int1, &a[k * lda], &int1, &cs, &sn);

                    if (withc) {
                        temp = c[ic] * sn + c[ic - incc] * cs;
                        c[ic] = c[ic] * cs - c[ic - incc] * sn;
                        ic = ic - incc;
                        c[ic] = temp;
                    }

                    if (withz) {
                        SLC_DROT(&n, &z[(k - 1) * ldz], &int1, &z[k * ldz], &int1, &cs, &sn);
                    }
                }
            }
        }
    }
}
