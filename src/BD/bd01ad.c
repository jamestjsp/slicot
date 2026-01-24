/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

/*
 * BD01AD - Benchmark examples for continuous-time dynamical systems
 *
 * Purpose:
 *   Generate benchmark examples for time-invariant, continuous-time
 *   dynamical systems (E, A, B, C, D):
 *       E x'(t) = A x(t) + B u(t)
 *         y(t)  = C x(t) + D u(t)
 *
 *   This implements the CTDSX benchmark library from SLICOT Working Note 1998-9.
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <string.h>
#include <ctype.h>

#define ZERO 0.0
#define ONE 1.0
#define TWO 2.0
#define THREE 3.0
#define FOUR 4.0
#define PI 3.141592653589793

static bool lsame_local(char a, char b) {
    return toupper((unsigned char)a) == toupper((unsigned char)b);
}

void bd01ad(const char* def, const i32* nr, f64* dpar, i32* ipar,
            bool* vec, i32* n, i32* m, i32* p,
            f64* e, const i32 lde, f64* a, const i32 lda,
            f64* b, const i32 ldb, f64* c, const i32 ldc,
            f64* d, const i32 ldd, char* note,
            f64* dwork, const i32 ldwork, i32* info)
{
    i32 i, l;
    f64 temp, ttemp, appind;
    f64 b1, b2, c1, c2;
    i32 int1 = 1;
    f64 dbl0 = ZERO, dbl1 = ONE;
    (void)int1;

    *info = 0;

    vec[0] = true;
    vec[1] = true;
    vec[2] = true;
    vec[3] = false;
    vec[4] = true;
    vec[5] = true;
    vec[6] = true;
    vec[7] = false;

    i32 nr1 = nr[0];
    i32 nr2 = nr[1];

    if (nr1 == 1) {
        if (nr2 == 1) {
            strncpy(note, "Laub 1979, Ex.1", 70);
            *n = 2;
            *m = 1;
            *p = 2;
            if (lde < *n) { *info = -10; }
            if (lda < *n) { *info = -12; }
            if (ldb < *n) { *info = -14; }
            if (ldc < *p) { *info = -16; }
            if (ldd < *p) { *info = -18; }
            if (*info != 0) return;

            SLC_DLASET("A", n, n, &dbl0, &dbl1, e, &lde);
            SLC_DLASET("A", n, n, &dbl0, &dbl0, a, &lda);
            a[0 + 1 * lda] = ONE;
            b[0 + 0 * ldb] = ZERO;
            b[1 + 0 * ldb] = ONE;
            SLC_DLASET("A", p, n, &dbl0, &dbl1, c, &ldc);
            SLC_DLASET("A", p, m, &dbl0, &dbl0, d, &ldd);

        } else if (nr2 == 2) {
            strncpy(note, "Laub 1979, Ex.2: uncontrollable-unobservable data", 70);
            *n = 2;
            *m = 1;
            *p = 1;
            if (lde < *n) { *info = -10; }
            if (lda < *n) { *info = -12; }
            if (ldb < *n) { *info = -14; }
            if (ldc < *p) { *info = -16; }
            if (ldd < *p) { *info = -18; }
            if (*info != 0) return;

            SLC_DLASET("A", n, n, &dbl0, &dbl1, e, &lde);
            a[0 + 0 * lda] = FOUR;
            a[1 + 0 * lda] = -4.5;
            a[0 + 1 * lda] = THREE;
            a[1 + 1 * lda] = -3.5;
            b[0 + 0 * ldb] = ONE;
            b[1 + 0 * ldb] = -ONE;
            c[0 + 0 * ldc] = THREE;
            c[0 + 1 * ldc] = TWO;
            SLC_DLASET("A", p, m, &dbl0, &dbl0, d, &ldd);

        } else if (nr2 >= 3 && nr2 <= 10) {
            if (nr2 == 3) {
                strncpy(note, "Beale/Shafai 1989: model of L-1011 aircraft", 70);
                *n = 4; *m = 2; *p = 4;
            } else if (nr2 == 4) {
                strncpy(note, "Bhattacharyya et al. 1983: binary distillation column", 70);
                *n = 8; *m = 2; *p = 8;
            } else if (nr2 == 5) {
                strncpy(note, "Patnaik et al. 1980: tubular ammonia reactor", 70);
                *n = 9; *m = 3; *p = 9;
            } else if (nr2 == 6) {
                strncpy(note, "Davison/Gesing 1978: J-100 jet engine", 70);
                *n = 30; *m = 3; *p = 5;
            } else if (nr2 == 7) {
                strncpy(note, "Davison 1967: binary distillation column", 70);
                *n = 11; *m = 3; *p = 3;
            } else if (nr2 == 8) {
                strncpy(note, "Chien/Ergin/Ling/Lee 1958: drum boiler", 70);
                *n = 9; *m = 3; *p = 2;
            } else if (nr2 == 9) {
                strncpy(note, "Ly, Gangsaas 1981: B-767 airplane", 70);
                *n = 55; *m = 2; *p = 2;
            } else if (nr2 == 10) {
                strncpy(note, "control surface servo for an underwater vehicle", 70);
                *n = 8; *m = 2; *p = 1;
            }

            if (lde < *n) { *info = -10; }
            if (lda < *n) { *info = -12; }
            if (ldb < *n) { *info = -14; }
            if (ldc < *p) { *info = -16; }
            if (ldd < *p) { *info = -18; }
            if (*info != 0) return;

            *info = 1;
            return;

        } else {
            *info = -2;
        }

    } else if (nr1 == 2) {
        if (!(lsame_local(def[0], 'D') || lsame_local(def[0], 'N'))) {
            *info = -1;
            return;
        }

        if (nr2 == 1) {
            strncpy(note, "Chow/Kokotovic 1976: magnetic tape control system", 70);
            if (lsame_local(def[0], 'D')) dpar[0] = 1e-6;
            if (dpar[0] == ZERO) { *info = -3; }
            *n = 4;
            *m = 1;
            *p = 2;
            if (lde < *n) { *info = -10; }
            if (lda < *n) { *info = -12; }
            if (ldb < *n) { *info = -14; }
            if (ldc < *p) { *info = -16; }
            if (ldd < *p) { *info = -18; }
            if (*info != 0) return;

            SLC_DLASET("A", n, n, &dbl0, &dbl1, e, &lde);
            SLC_DLASET("A", n, n, &dbl0, &dbl0, a, &lda);
            a[0 + 1 * lda] = 0.4;
            a[1 + 2 * lda] = 0.345;
            a[2 + 1 * lda] = -0.524 / dpar[0];
            a[2 + 2 * lda] = -0.465 / dpar[0];
            a[2 + 3 * lda] = 0.262 / dpar[0];
            a[3 + 3 * lda] = -ONE / dpar[0];
            SLC_DLASET("A", n, m, &dbl0, &dbl0, b, &ldb);
            b[3 + 0 * ldb] = ONE / dpar[0];
            SLC_DLASET("A", p, n, &dbl0, &dbl0, c, &ldc);
            c[0 + 0 * ldc] = ONE;
            c[1 + 2 * ldc] = ONE;
            SLC_DLASET("A", p, m, &dbl0, &dbl0, d, &ldd);

        } else if (nr2 == 2) {
            strncpy(note, "Arnold/Laub 1984", 70);
            if (lsame_local(def[0], 'D')) dpar[0] = 1e-6;
            *n = 4;
            *m = 1;
            *p = 1;
            if (lde < *n) { *info = -10; }
            if (lda < *n) { *info = -12; }
            if (ldb < *n) { *info = -14; }
            if (ldc < *p) { *info = -16; }
            if (ldd < *p) { *info = -18; }
            if (*info != 0) return;

            SLC_DLASET("A", n, n, &dbl0, &dbl1, e, &lde);
            SLC_DLASET("A", n, n, &dbl0, &dpar[0], a, &lda);
            a[0 + 0 * lda] = -dpar[0];
            a[1 + 0 * lda] = -ONE;
            a[0 + 1 * lda] = ONE;
            a[1 + 1 * lda] = -dpar[0];
            a[3 + 2 * lda] = -ONE;
            a[2 + 3 * lda] = ONE;
            SLC_DLASET("A", n, m, &dbl1, &dbl1, b, &ldb);
            SLC_DLASET("A", p, n, &dbl1, &dbl1, c, &ldc);
            d[0 + 0 * ldd] = ZERO;

        } else if (nr2 == 3) {
            strncpy(note, "Vertical acceleration of a rigid guided missile", 70);
            if (lsame_local(def[0], 'D')) ipar[0] = 1;
            if (ipar[0] < 1 || ipar[0] > 10) { *info = -4; }
            *n = 3;
            *m = 1;
            *p = 1;
            if (lde < *n) { *info = -10; }
            if (lda < *n) { *info = -12; }
            if (ldb < *n) { *info = -14; }
            if (ldc < *p) { *info = -16; }
            if (ldd < *p) { *info = -18; }
            if (*info != 0) return;

            *info = 1;
            return;

        } else if (nr2 == 4) {
            strncpy(note, "Senning 1980: hydraulic positioning system", 70);
            if (lsame_local(def[0], 'D')) {
                dpar[0] = 1.4e4;
                dpar[1] = 0.1287;
                dpar[2] = 0.15;
                dpar[3] = 0.01;
                dpar[4] = 0.002;
                dpar[5] = 0.24;
                dpar[6] = 10.75;
            }
            if ((dpar[0] <= 9e3) || (dpar[0] >= 1.6e4) ||
                (dpar[1] <= 0.05) || (dpar[1] >= 0.3) ||
                (dpar[2] <= 0.05) || (dpar[2] >= 5.0) ||
                (dpar[3] <= ZERO) || (dpar[3] >= 0.05) ||
                (dpar[4] <= 1.03e-4) || (dpar[4] >= 3.5e-3) ||
                (dpar[5] <= 0.01) || (dpar[5] >= 15.0) ||
                (dpar[6] <= 10.5) || (dpar[6] >= 11.1)) {
                *info = -3;
            }
            *n = 3;
            *m = 1;
            *p = 1;
            if (lde < *n) { *info = -10; }
            if (lda < *n) { *info = -12; }
            if (ldb < *n) { *info = -14; }
            if (ldc < *p) { *info = -16; }
            if (ldd < *p) { *info = -18; }
            if (*info != 0) return;

            SLC_DLASET("A", n, n, &dbl0, &dbl1, e, &lde);
            SLC_DLASET("A", n, n, &dbl0, &dbl0, a, &lda);
            a[0 + 1 * lda] = ONE;
            a[1 + 1 * lda] = -(dpar[2] + FOUR * dpar[3] / PI) / dpar[1];
            a[1 + 2 * lda] = dpar[6] / dpar[1];
            a[2 + 1 * lda] = -FOUR * dpar[6] * dpar[0] / 874.0;
            a[2 + 2 * lda] = -FOUR * dpar[0] * (dpar[5] + dpar[4]) / 874.0;
            SLC_DLASET("A", n, m, &dbl0, &dbl0, b, &ldb);
            b[2 + 0 * ldb] = -FOUR * dpar[0] / 874.0;
            SLC_DLASET("A", p, n, &dbl0, &dbl1, c, &ldc);
            d[0 + 0 * ldd] = ZERO;

        } else if (nr2 == 5) {
            strncpy(note, "Kwakernaak/Westdyk 1985: cascade of inverted pendula", 70);
            if (lsame_local(def[0], 'D')) ipar[0] = 1;
            if (ipar[0] < 1 || ipar[0] > 7) { *info = -4; }
            if (*info != 0) return;

            *info = 1;
            return;

        } else if (nr2 == 6) {
            strncpy(note, "Kallstrom/Astrom 1981: regulation of a ship heading", 70);
            if (lsame_local(def[0], 'D')) ipar[0] = 1;
            if (ipar[0] < 1 || ipar[0] > 5) { *info = -4; }
            if (*info != 0) return;

            *info = 1;
            return;

        } else if (nr2 == 7) {
            strncpy(note, "Ackermann 1989: track-guided bus", 70);
            if (lsame_local(def[0], 'D')) {
                dpar[0] = 15.0;
                dpar[1] = 10.0;
            }
            if (dpar[0] < 9.95 || dpar[0] > 16.0) { *info = -3; }
            if (dpar[1] < 1.0 || dpar[1] > 20.0) { *info = -3; }
            *n = 5;
            *m = 1;
            *p = 1;
            if (lde < *n) { *info = -10; }
            if (lda < *n) { *info = -12; }
            if (ldb < *n) { *info = -14; }
            if (ldc < *p) { *info = -16; }
            if (ldd < *p) { *info = -18; }
            if (*info != 0) return;

            SLC_DLASET("A", n, n, &dbl0, &dbl1, e, &lde);
            SLC_DLASET("A", n, n, &dbl0, &dbl0, a, &lda);
            a[0 + 0 * lda] = -668.0 / (dpar[0] * dpar[1]);
            a[0 + 1 * lda] = -ONE + 180.4 / (dpar[0] * dpar[1] * dpar[1]);
            a[1 + 0 * lda] = 180.4 / (10.86 * dpar[0]);
            a[1 + 1 * lda] = -4417.5452 / (10.86 * dpar[0] * dpar[1]);
            a[0 + 4 * lda] = 198.0 / (dpar[0] * dpar[1]);
            a[1 + 4 * lda] = 726.66 / (10.86 * dpar[0]);
            a[2 + 0 * lda] = dpar[1];
            a[2 + 3 * lda] = dpar[1];
            a[3 + 1 * lda] = ONE;
            SLC_DLASET("A", n, m, &dbl0, &dbl0, b, &ldb);
            b[4 + 0 * ldb] = ONE;
            SLC_DLASET("A", p, n, &dbl0, &dbl0, c, &ldc);
            c[0 + 2 * ldc] = ONE;
            c[0 + 3 * ldc] = 6.12;
            d[0 + 0 * ldd] = ZERO;

        } else {
            *info = -2;
        }

    } else if (nr1 == 3) {
        if (!(lsame_local(def[0], 'D') || lsame_local(def[0], 'N'))) {
            *info = -1;
            return;
        }

        if (nr2 == 1) {
            strncpy(note, "Laub 1979, Ex.4: string of high speed vehicles", 70);
            if (lsame_local(def[0], 'D')) ipar[0] = 20;
            if (ipar[0] < 2) { *info = -4; }
            *n = 2 * ipar[0] - 1;
            *m = ipar[0];
            *p = ipar[0] - 1;
            if (lde < *n) { *info = -10; }
            if (lda < *n) { *info = -12; }
            if (ldb < *n) { *info = -14; }
            if (ldc < *p) { *info = -16; }
            if (ldd < *p) { *info = -18; }
            if (*info != 0) return;

            SLC_DLASET("A", n, n, &dbl0, &dbl1, e, &lde);
            SLC_DLASET("A", n, n, &dbl0, &dbl0, a, &lda);
            SLC_DLASET("A", n, m, &dbl0, &dbl0, b, &ldb);
            SLC_DLASET("A", p, n, &dbl0, &dbl0, c, &ldc);
            for (i = 0; i < *n; i++) {
                i32 idx = i + 1;
                if (idx % 2 == 1) {
                    a[i + i * lda] = -ONE;
                    b[i + (idx / 2) * ldb] = ONE;
                } else {
                    a[i + (i - 1) * lda] = ONE;
                    a[i + (i + 1) * lda] = -ONE;
                    c[(idx / 2 - 1) + i * ldc] = ONE;
                }
            }
            SLC_DLASET("A", p, m, &dbl0, &dbl0, d, &ldd);

        } else if (nr2 == 2) {
            strncpy(note, "Hodel et al. 1996: heat flow in a thin rod", 70);
            if (lsame_local(def[0], 'D')) ipar[0] = 100;
            if (ipar[0] < 1) { *info = -4; }
            *n = ipar[0];
            *m = 1;
            *p = *n;
            if (lde < *n) { *info = -10; }
            if (lda < *n) { *info = -12; }
            if (ldb < *n) { *info = -14; }
            if (ldc < *p) { *info = -16; }
            if (ldd < *p) { *info = -18; }
            if (*info != 0) return;

            temp = (f64)(*n + 1);
            f64 neg2temp = -TWO * temp;
            SLC_DLASET("A", n, n, &dbl0, &dbl1, e, &lde);
            SLC_DLASET("A", n, n, &dbl0, &neg2temp, a, &lda);
            a[0 + 0 * lda] = -temp;
            for (i = 0; i < *n - 1; i++) {
                a[i + (i + 1) * lda] = temp;
                a[(i + 1) + i * lda] = temp;
            }
            SLC_DLASET("A", n, m, &dbl0, &dbl0, b, &ldb);
            b[(*n - 1) + 0 * ldb] = temp;
            SLC_DLASET("A", p, n, &dbl0, &dbl1, c, &ldc);
            SLC_DLASET("A", p, m, &dbl0, &dbl0, d, &ldd);

        } else if (nr2 == 3) {
            strncpy(note, "Laub 1979, Ex.6", 70);
            if (lsame_local(def[0], 'D')) ipar[0] = 21;
            if (ipar[0] < 1) { *info = -4; }
            *n = ipar[0];
            *m = 1;
            *p = 1;
            if (lde < *n) { *info = -10; }
            if (lda < *n) { *info = -12; }
            if (ldb < *n) { *info = -14; }
            if (ldc < *p) { *info = -16; }
            if (ldd < *p) { *info = -18; }
            if (*info != 0) return;

            SLC_DLASET("A", n, n, &dbl0, &dbl1, e, &lde);
            SLC_DLASET("A", n, n, &dbl0, &dbl0, a, &lda);
            i32 nm1 = *n - 1;
            SLC_DLASET("A", &nm1, &nm1, &dbl0, &dbl1, &a[0 + 1 * lda], &lda);
            SLC_DLASET("A", n, m, &dbl0, &dbl0, b, &ldb);
            b[(*n - 1) + 0 * ldb] = ONE;
            SLC_DLASET("A", p, n, &dbl0, &dbl0, c, &ldc);
            c[0 + 0 * ldc] = ONE;
            SLC_DLASET("A", p, m, &dbl0, &dbl0, d, &ldd);

        } else if (nr2 == 4) {
            strncpy(note, "Lang/Penzl 1994: rotating axle", 70);
            if (lsame_local(def[0], 'D')) ipar[0] = 211;
            if (ipar[0] < 1 || ipar[0] > 211) { *info = -4; }
            if (*info != 0) return;

            *info = 1;
            return;

        } else {
            *info = -2;
        }

    } else if (nr1 == 4) {
        if (!(lsame_local(def[0], 'D') || lsame_local(def[0], 'N'))) {
            *info = -1;
            return;
        }

        if (nr2 == 1) {
            strncpy(note, "Rosen/Wang 1995: control of 1-dim. heat flow", 70);
            if (lsame_local(def[0], 'D')) {
                ipar[0] = 100;
                dpar[0] = 0.01;
                dpar[1] = ONE;
                dpar[2] = ONE;
                dpar[3] = 0.2;
                dpar[4] = 0.3;
                dpar[5] = 0.2;
                dpar[6] = 0.3;
            }
            if (ipar[0] < 2) { *info = -4; }
            *n = ipar[0];
            *m = 1;
            *p = 1;
            if (lde < *n) { *info = -10; }
            if (lda < *n) { *info = -12; }
            if (ldb < *n) { *info = -14; }
            if (ldc < *p) { *info = -16; }
            if (ldd < *p) { *info = -18; }
            if (*info != 0) return;

            vec[3] = true;
            appind = (f64)(*n + 1);
            ttemp = -dpar[0] * appind;
            temp = ONE / (6.0 * appind);
            f64 four_temp = FOUR * temp;
            f64 two_ttemp = TWO * ttemp;
            SLC_DLASET("A", n, n, &dbl0, &four_temp, e, &lde);
            SLC_DLASET("A", n, n, &dbl0, &two_ttemp, a, &lda);
            f64 neg_ttemp = -ttemp;
            for (i = 0; i < *n - 1; i++) {
                a[(i + 1) + i * lda] = neg_ttemp;
                a[i + (i + 1) * lda] = neg_ttemp;
                e[(i + 1) + i * lde] = temp;
                e[i + (i + 1) * lde] = temp;
            }

            for (i = 0; i < *n; i++) {
                b1 = fmax((f64)(i) / appind, dpar[3]);
                b2 = fmin((f64)(i + 2) / appind, dpar[4]);
                c1 = fmax((f64)(i) / appind, dpar[5]);
                c2 = fmin((f64)(i + 2) / appind, dpar[6]);

                if (b1 >= b2) {
                    b[i + 0 * ldb] = ZERO;
                } else {
                    b[i + 0 * ldb] = b2 - b1;
                    temp = fmin(b2, (f64)(i + 1) / appind);
                    if (b1 < temp) {
                        b[i + 0 * ldb] += appind * (temp * temp - b1 * b1) / TWO;
                        b[i + 0 * ldb] += (f64)(i + 1) * (b1 - temp);
                    }
                    temp = fmax(b1, (f64)(i + 1) / appind);
                    if (temp < b2) {
                        b[i + 0 * ldb] -= appind * (b2 * b2 - temp * temp) / TWO;
                        b[i + 0 * ldb] -= (f64)(i + 1) * (temp - b2);
                    }
                }

                if (c1 >= c2) {
                    c[0 + i * ldc] = ZERO;
                } else {
                    c[0 + i * ldc] = c2 - c1;
                    temp = fmin(c2, (f64)(i + 1) / appind);
                    if (c1 < temp) {
                        c[0 + i * ldc] += appind * (temp * temp - c1 * c1) / TWO;
                        c[0 + i * ldc] += (f64)(i + 1) * (c1 - temp);
                    }
                    temp = fmax(c1, (f64)(i + 1) / appind);
                    if (temp < c2) {
                        c[0 + i * ldc] -= appind * (c2 * c2 - temp * temp) / TWO;
                        c[0 + i * ldc] -= (f64)(i + 1) * (temp - c2);
                    }
                }
            }
            SLC_DSCAL(n, &dpar[1], b, &int1);
            SLC_DSCAL(n, &dpar[2], c, &ldc);
            SLC_DLASET("A", p, m, &dbl0, &dbl0, d, &ldd);

        } else if (nr2 == 2) {
            strncpy(note, "Hench et al. 1995: coupled springs, dashpots, masses", 70);
            if (lsame_local(def[0], 'D')) {
                ipar[0] = 30;
                dpar[0] = FOUR;
                dpar[1] = FOUR;
                dpar[2] = ONE;
            }
            if (ipar[0] < 2) { *info = -4; }
            l = ipar[0];
            *n = 2 * l;
            *m = 2;
            *p = 2 * l;
            if (lde < *n) { *info = -10; }
            if (lda < *n) { *info = -12; }
            if (ldb < *n) { *info = -14; }
            if (ldc < *p) { *info = -16; }
            if (ldd < *p) { *info = -18; }
            if (*info != 0) return;

            vec[3] = true;
            SLC_DLASET("A", n, n, &dbl0, &dpar[0], e, &lde);
            SLC_DLASET("A", n, n, &dbl0, &dbl0, a, &lda);
            f64 neg2dpar2 = -TWO * dpar[2];
            f64 negdpar1 = -dpar[1];
            for (i = 0; i < l; i++) {
                e[i + i * lde] = ONE;
                a[i + (i + l) * lda] = ONE;
                a[(i + l) + (i + l) * lda] = negdpar1;
                if (i < l - 1) {
                    a[(i + l) + (i + 1) * lda] = dpar[2];
                    a[(i + l + 1) + i * lda] = dpar[2];
                    if (i > 0) {
                        a[(i + l) + i * lda] = neg2dpar2;
                    }
                }
            }
            a[l + 0 * lda] = -dpar[2];
            a[(*n - 1) + (l - 1) * lda] = -dpar[2];
            SLC_DLASET("A", n, m, &dbl0, &dbl0, b, &ldb);
            b[l + 0 * ldb] = ONE;
            b[(*n - 1) + 1 * ldb] = -ONE;
            SLC_DLASET("A", p, n, &dbl0, &dbl1, c, &ldc);
            SLC_DLASET("A", p, m, &dbl0, &dbl0, d, &ldd);

        } else {
            *info = -2;
        }

    } else {
        *info = -2;
    }
}
