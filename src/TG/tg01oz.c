/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <complex.h>
#include <stdbool.h>

void tg01oz(
    const char* jobe,
    const i32 n,
    c128* dcba, const i32 lddcba,
    c128* e, const i32 lde,
    i32* nz,
    c128* g,
    const f64 tol,
    c128* zwork, const i32 lzwork,
    i32* info
)
{
    const f64 one = 1.0, three = 3.0, four = 4.0, zero = 0.0;
    const c128 cone = 1.0 + 0.0 * I, czero = 0.0 + 0.0 * I;

    bool descr, lquery;
    i32 i, imax, itau, iwrk, j, jf, maxwrk, minwrk, n1, nc;
    f64 absd, maxa, nrmb, nrmc, toldef;
    c128 tau;
    f64 dwork[1];
    i32 int1 = 1;
    i32 nm1;

    descr = (jobe[0] == 'G' || jobe[0] == 'g');
    *info = 0;
    n1 = n + 1;

    if (!descr && !(jobe[0] == 'I' || jobe[0] == 'i')) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (lddcba < n1) {
        *info = -4;
    } else if (lde < 1 || (descr && lde < (1 > n ? 1 : n))) {
        *info = -6;
    } else {
        if (descr) {
            minwrk = 2 * n + 1;
        } else if (n == 0) {
            minwrk = 1;
        } else {
            minwrk = n1;
        }
        maxwrk = minwrk;
        lquery = (lzwork == -1);

        if (lquery) {
            if (descr) {
                i32 neg1 = -1;
                SLC_ZGEQRF(&n, &n, e, &lde, dcba, zwork, &neg1, info);
                i32 opt1 = (i32)creal(zwork[0]);
                maxwrk = maxwrk > opt1 ? maxwrk : opt1;
                SLC_ZUNMQR("L", "C", &n, &n1, &n, e, &lde, dcba,
                           dcba, &lddcba, zwork, &neg1, info);
                i32 opt2 = (i32)creal(zwork[0]);
                maxwrk = maxwrk > opt2 ? maxwrk : opt2;
                zwork[0] = (f64)maxwrk + 0.0 * I;
            } else {
                zwork[0] = (f64)maxwrk + 0.0 * I;
            }
            return;
        } else if (lzwork < minwrk) {
            zwork[0] = (f64)minwrk + 0.0 * I;
            *info = -11;
        }
    }

    if (*info != 0) {
        return;
    }

    *nz = n;
    if (n == 0) {
        *g = dcba[0];
        zwork[0] = cone;
        return;
    }

    toldef = tol;
    if (toldef <= zero) {
        f64 eps = SLC_DLAMCH("P");
        toldef = pow(eps, three / four);
    }

    *g = cone;
    maxa = SLC_ZLANGE("M", &n, &n, &dcba[1 + 1 * lddcba], &lddcba, dwork);
    nrmb = SLC_DZNRM2(&n, &dcba[1], &int1);
    nrmc = SLC_DZNRM2(&n, &dcba[n1], &n1);

    if (cabs(dcba[0]) * (one + maxa) <= toldef * nrmb * nrmc) {
        const char* jobt;

        if (descr) {
            itau = 0;
            iwrk = itau + n;
            i32 lwork_rem = lzwork - n;
            SLC_ZGEQRF(&n, &n, e, &lde, &zwork[itau], &zwork[iwrk],
                       &lwork_rem, info);
            i32 opt1 = (i32)creal(zwork[iwrk]);
            maxwrk = maxwrk > opt1 ? maxwrk : opt1;

            SLC_ZUNMQR("L", "C", &n, &n1, &n, e, &lde,
                       &zwork[itau], &dcba[1], &lddcba, &zwork[iwrk],
                       &lwork_rem, info);
            i32 opt2 = (i32)creal(zwork[iwrk]);
            maxwrk = maxwrk > opt2 ? maxwrk : opt2;

            if (n > 1) {
                nm1 = n - 1;
                SLC_ZLASET("L", &nm1, &nm1, &czero, &czero, &e[1], &lde);
            }
            jobt = "U";
        } else {
            jobt = jobe;
        }

        for (i = 0; i < n; i++) {
            nc = *nz + 1;

            if (!descr) {
                if (dcba[(i + 1) + i * lddcba] == czero) {
                    imax = SLC_IZAMAX(nz, &dcba[(i + 1) + i * lddcba], &int1) - 1 + (i + 1);
                    if (imax < 0 || imax >= n1) imax = i + 1;
                    SLC_ZSWAP(&nc, &dcba[(i + 1) + i * lddcba], &lddcba,
                              &dcba[imax + i * lddcba], &lddcba);
                    SLC_ZSWAP(&nc, &dcba[i + (i + 1) * lddcba], &int1,
                              &dcba[i + imax * lddcba], &int1);
                }

                i32 ip2 = (i + 2 < n1) ? (i + 2) : n1;
                SLC_ZLARFG(nz, &dcba[(i + 1) + i * lddcba],
                           &dcba[ip2 + i * lddcba], &int1, &tau);
                *g = *g * dcba[(i + 1) + i * lddcba];
                dcba[(i + 1) + i * lddcba] = cone;

                c128 tau_conj = conj(tau);
                SLC_ZLARF("L", nz, nz, &dcba[(i + 1) + i * lddcba], &int1,
                          &tau_conj, &dcba[(i + 1) + (i + 1) * lddcba], &lddcba,
                          zwork);
                SLC_ZLARF("R", &nc, nz, &dcba[(i + 1) + i * lddcba], &int1,
                          &tau, &dcba[i + (i + 1) * lddcba], &lddcba, zwork);
            } else {
                tg01ob(jobt, *nz, &dcba[i + i * lddcba], lddcba,
                       &e[i + i * lde], lde, info);
                c128 num = *g * dcba[(i + 1) + i * lddcba];
                c128 den = e[i + i * lde];
                *g = SLC_ZLADIV(&num, &den);
            }

            SLC_ZCOPY(nz, &dcba[i + (i + 1) * lddcba], &lddcba,
                      &dcba[(i + 1) + (i + 1) * lddcba], &lddcba);

            (*nz)--;
            absd = cabs(dcba[(i + 1) + (i + 1) * lddcba]);
            nrmb = SLC_DZNRM2(nz, &dcba[(i + 2) + (i + 1) * lddcba], &int1);
            nrmc = SLC_DZNRM2(nz, &dcba[(i + 1) + (i + 2) * lddcba], &n1);

            if (absd == zero && (nrmb == zero || nrmc == zero)) {
                *nz = 0;
                goto label_20;
            }
            maxa = SLC_ZLANGE("M", nz, nz, &dcba[(i + 2) + (i + 2) * lddcba],
                              &lddcba, dwork);
            if (absd * (one + maxa) > toldef * nrmb * nrmc) {
                goto label_20;
            }
        }

        i = n - 1;

label_20:
        jf = 0;

        for (j = i + 1; j < n1; j++) {
            i32 nz1 = *nz + 1;
            SLC_ZCOPY(&nz1, &dcba[(i + 1) + j * lddcba], &int1,
                      &dcba[0 + jf * lddcba], &int1);
            jf++;
        }

        if (descr) {
            jf = 0;

            for (j = i + 1; j < n; j++) {
                SLC_ZCOPY(nz, &e[(i + 1) + j * lde], &int1,
                          &e[0 + jf * lde], &int1);
                jf++;
            }
        }
    }

    *g = *g * dcba[0];
    zwork[0] = (f64)maxwrk + 0.0 * I;
}
