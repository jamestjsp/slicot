/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <complex.h>
#include <stdbool.h>

void tg01kz(
    const char* jobe, const char* compc, const char* compq, const char* compz,
    const i32 n,
    c128* a, const i32 lda,
    c128* e, const i32 lde,
    c128* b,
    c128* c, const i32 incc,
    c128* q, const i32 ldq,
    c128* z, const i32 ldz,
    i32* info
)
{
    const c128 one  = 1.0 + 0.0 * I;
    const c128 zero = 0.0 + 0.0 * I;

    bool unite, withc, liniq, lupdq, liniz, lupdz, withq, withz;
    i32 ic, k;
    f64 cs;
    c128 sn, temp;
    i32 int1 = 1;
    i32 int2 = 2;

    unite = (jobe[0] == 'I' || jobe[0] == 'i');
    withc = (compc[0] == 'C' || compc[0] == 'c');
    liniq = (compq[0] == 'I' || compq[0] == 'i');
    lupdq = (compq[0] == 'U' || compq[0] == 'u');
    liniz = (compz[0] == 'I' || compz[0] == 'i');
    lupdz = (compz[0] == 'U' || compz[0] == 'u');
    withq = liniq || lupdq;
    withz = liniz || lupdz;
    *info = 0;

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
    } else if (lda < (1 > n ? 1 : n)) {
        *info = -7;
    } else if (lde < 1 || (!unite && lde < (1 > n ? 1 : n))) {
        *info = -9;
    } else if (withc && incc <= 0) {
        *info = -12;
    } else if (ldq < 1 || (withq && ldq < (1 > n ? 1 : n))) {
        *info = -14;
    } else if (ldz < 1 || (withz && ldz < (1 > n ? 1 : n))) {
        *info = -16;
    }

    if (*info != 0) {
        return;
    }

    if (n == 0) {
        return;
    }

    if (liniq || (n == 1 && !lupdq)) {
        SLC_ZLASET("Full", &n, &n, &zero, &one, q, &ldq);
    }
    if (liniz || (n == 1 && !lupdz)) {
        SLC_ZLASET("Full", &n, &n, &zero, &one, z, &ldz);
    }
    if (n == 1) {
        return;
    }

    if (withc) {
        ic = (n - 1) * incc;
    }

    for (k = n - 1; k >= 1; k--) {
        if (b[k] != zero) {
            SLC_ZLARTG(&b[k - 1], &b[k], &cs, &sn, &temp);
            b[k - 1] = temp;
            b[k] = zero;

            SLC_ZROT(&n, &a[(k - 1)], &lda, &a[k], &lda, &cs, &sn);

            if (withq) {
                c128 sn_conj = conj(sn);
                SLC_ZROT(&n, &q[(k - 1) * ldq], &int1, &q[k * ldq], &int1, &cs, &sn_conj);
            }

            if (unite) {
                c128 sn_conj = conj(sn);
                SLC_ZROT(&n, &a[(k - 1) * lda], &int1, &a[k * lda], &int1, &cs, &sn_conj);

                if (withc) {
                    temp = c[ic] * sn_conj + c[ic - incc] * cs;
                    c[ic] = c[ic] * cs - c[ic - incc] * sn;
                    ic = ic - incc;
                    c[ic] = temp;
                }

                if (withz) {
                    if (withq && (liniq == liniz || lupdq == lupdz)) {
                        SLC_ZLACPY("Full", &n, &int2, &q[(k - 1) * ldq], &ldq, &z[(k - 1) * ldz], &ldz);
                    } else {
                        SLC_ZROT(&n, &z[(k - 1) * ldz], &int1, &z[k * ldz], &int1, &cs, &sn_conj);
                    }
                }
            } else {
                c128 sn_conj = conj(sn);
                e[k + (k - 1) * lde] = sn_conj * e[(k - 1) + (k - 1) * lde];
                e[(k - 1) + (k - 1) * lde] = cs * e[(k - 1) + (k - 1) * lde];

                i32 nmk1 = n - k;
                SLC_ZROT(&nmk1, &e[(k - 1) + k * lde], &lde, &e[k + k * lde], &lde, &cs, &sn);

                if (e[k + (k - 1) * lde] != zero) {
                    SLC_ZLARTG(&e[k + k * lde], &e[k + (k - 1) * lde], &cs, &sn, &temp);
                    e[k + k * lde] = temp;
                    e[k + (k - 1) * lde] = zero;

                    sn_conj = conj(sn);
                    temp = e[(k - 1) + k * lde] * sn_conj + e[(k - 1) + (k - 1) * lde] * cs;
                    e[(k - 1) + k * lde] = e[(k - 1) + k * lde] * cs - e[(k - 1) + (k - 1) * lde] * sn;
                    e[(k - 1) + (k - 1) * lde] = temp;

                    i32 km2 = k - 1;
                    SLC_ZROT(&km2, &e[(k - 1) * lde], &int1, &e[k * lde], &int1, &cs, &sn_conj);

                    SLC_ZROT(&n, &a[(k - 1) * lda], &int1, &a[k * lda], &int1, &cs, &sn_conj);

                    if (withc) {
                        temp = c[ic] * sn_conj + c[ic - incc] * cs;
                        c[ic] = c[ic] * cs - c[ic - incc] * sn;
                        ic = ic - incc;
                        c[ic] = temp;
                    }

                    if (withz) {
                        SLC_ZROT(&n, &z[(k - 1) * ldz], &int1, &z[k * ldz], &int1, &cs, &sn_conj);
                    }
                }
            }
        }
    }
}
