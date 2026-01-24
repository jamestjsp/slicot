/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

/*
 * BD02AD - Benchmark examples for discrete-time dynamical systems
 *
 * Purpose:
 *   Generate benchmark examples for time-invariant, discrete-time
 *   dynamical systems (E, A, B, C, D):
 *       E x_{k+1} = A x_k + B u_k
 *             y_k = C x_k + D u_k
 *
 *   This implements the DTDSX benchmark library from SLICOT Working Note 1998-10.
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

static bool lsame_local(char a, char b) {
    return toupper((unsigned char)a) == toupper((unsigned char)b);
}

void bd02ad(const char* def, const i32* nr, f64* dpar, i32* ipar,
            bool* vec, i32* n, i32* m, i32* p,
            f64* e, const i32 lde, f64* a, const i32 lda,
            f64* b, const i32 ldb, f64* c, const i32 ldc,
            f64* d, const i32 ldd, char* note,
            f64* dwork, const i32 ldwork, i32* info)
{
    f64 temp;
    f64 dbl0 = ZERO, dbl1 = ONE;
    (void)dwork;
    (void)ldwork;

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
            strncpy(note, "Laub 1979, Ex. 2: uncontrollable-unobservable data", 70);
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

        } else if (nr2 == 2) {
            strncpy(note, "Laub 1979, Ex. 3", 70);
            *n = 2;
            *m = 2;
            *p = 2;
            if (lde < *n) { *info = -10; }
            if (lda < *n) { *info = -12; }
            if (ldb < *n) { *info = -14; }
            if (ldc < *p) { *info = -16; }
            if (ldd < *p) { *info = -18; }
            if (*info != 0) return;

            SLC_DLASET("A", n, n, &dbl0, &dbl1, e, &lde);
            SLC_DLASET("A", n, n, &dbl0, &dbl0, a, &lda);
            a[0 + 0 * lda] = 0.9512;
            a[1 + 1 * lda] = 0.9048;
            b[0 + 0 * ldb] = 4.877;
            b[0 + 1 * ldb] = 4.877;
            b[1 + 0 * ldb] = -1.1895;
            b[1 + 1 * ldb] = 3.569;
            SLC_DLASET("A", p, n, &dbl0, &dbl1, c, &ldc);
            SLC_DLASET("A", p, m, &dbl0, &dbl0, d, &ldd);

        } else if (nr2 == 3) {
            strncpy(note, "Van Dooren 1981, Ex. II", 70);
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
            a[0 + 0 * lda] = TWO;
            a[1 + 0 * lda] = ONE;
            a[0 + 1 * lda] = -ONE;
            a[1 + 1 * lda] = ZERO;
            SLC_DLASET("A", n, m, &dbl0, &dbl1, b, &ldb);
            SLC_DLASET("A", p, n, &dbl1, &dbl0, c, &ldc);
            d[0 + 0 * ldd] = ZERO;

        } else if (nr2 == 4) {
            strncpy(note, "Ionescu/Weiss 1992", 70);
            *n = 2;
            *m = 2;
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
            a[1 + 1 * lda] = -ONE;
            SLC_DLASET("A", n, m, &dbl0, &dbl1, b, &ldb);
            b[1 + 0 * ldb] = TWO;
            SLC_DLASET("A", p, n, &dbl0, &dbl1, c, &ldc);
            SLC_DLASET("A", p, m, &dbl0, &dbl0, d, &ldd);

        } else if (nr2 == 5) {
            strncpy(note, "Jonckheere 1981", 70);
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
            SLC_DLASET("A", n, m, &dbl1, &dbl0, b, &ldb);
            SLC_DLASET("A", p, n, &dbl0, &dbl1, c, &ldc);
            SLC_DLASET("A", p, m, &dbl0, &dbl0, d, &ldd);

        } else if (nr2 == 6) {
            strncpy(note, "Ackerson/Fu 1970: satellite control problem", 70);
            *n = 4;
            *m = 2;
            *p = 4;
            if (lde < *n) { *info = -10; }
            if (lda < *n) { *info = -12; }
            if (ldb < *n) { *info = -14; }
            if (ldc < *p) { *info = -16; }
            if (ldd < *p) { *info = -18; }
            if (*info != 0) return;

            SLC_DLASET("A", n, n, &dbl0, &dbl1, e, &lde);
            SLC_DLASET("A", p, n, &dbl0, &dbl1, c, &ldc);
            SLC_DLASET("A", p, m, &dbl0, &dbl0, d, &ldd);

            *info = 1;
            return;

        } else if (nr2 == 7) {
            strncpy(note, "Litkouhi 1983: system with slow and fast modes", 70);
            *n = 4;
            *m = 2;
            *p = 4;
            if (lde < *n) { *info = -10; }
            if (lda < *n) { *info = -12; }
            if (ldb < *n) { *info = -14; }
            if (ldc < *p) { *info = -16; }
            if (ldd < *p) { *info = -18; }
            if (*info != 0) return;

            SLC_DLASET("A", n, n, &dbl0, &dbl1, e, &lde);
            SLC_DLASET("A", p, n, &dbl0, &dbl1, c, &ldc);
            SLC_DLASET("A", p, m, &dbl0, &dbl0, d, &ldd);

            *info = 1;
            return;

        } else if (nr2 == 8) {
            strncpy(note, "Lu/Lin 1993, Ex. 4.3", 70);
            *n = 4;
            *m = 4;
            *p = 4;
            if (lde < *n) { *info = -10; }
            if (lda < *n) { *info = -12; }
            if (ldb < *n) { *info = -14; }
            if (ldc < *p) { *info = -16; }
            if (ldd < *p) { *info = -18; }
            if (*info != 0) return;

            SLC_DLASET("A", n, n, &dbl0, &dbl1, e, &lde);
            SLC_DLASET("U", p, n, &dbl1, &dbl1, c, &ldc);
            c[0 + 2 * ldc] = TWO;
            c[0 + 3 * ldc] = FOUR;
            c[1 + 3 * ldc] = TWO;
            SLC_DLASET("A", p, m, &dbl0, &dbl0, d, &ldd);

            *info = 1;
            return;

        } else if (nr2 == 9) {
            strncpy(note, "Gajic/Shen 1993, Section 2.7.4: chemical plant", 70);
            *n = 5;
            *m = 2;
            *p = 5;
            if (lde < *n) { *info = -10; }
            if (lda < *n) { *info = -12; }
            if (ldb < *n) { *info = -14; }
            if (ldc < *p) { *info = -16; }
            if (ldd < *p) { *info = -18; }
            if (*info != 0) return;

            SLC_DLASET("A", n, n, &dbl0, &dbl1, e, &lde);
            SLC_DLASET("A", p, n, &dbl0, &dbl1, c, &ldc);
            SLC_DLASET("A", p, m, &dbl0, &dbl0, d, &ldd);

            *info = 1;
            return;

        } else if (nr2 == 10) {
            strncpy(note, "Davison/Wang 1974", 70);
            *n = 6;
            *m = 2;
            *p = 2;
            if (lde < *n) { *info = -10; }
            if (lda < *n) { *info = -12; }
            if (ldb < *n) { *info = -14; }
            if (ldc < *p) { *info = -16; }
            if (ldd < *p) { *info = -18; }
            if (*info != 0) return;
            vec[7] = true;

            SLC_DLASET("A", n, n, &dbl0, &dbl1, e, &lde);
            SLC_DLASET("A", n, n, &dbl0, &dbl0, a, &lda);
            a[0 + 1 * lda] = ONE;
            a[1 + 2 * lda] = ONE;
            a[3 + 4 * lda] = ONE;
            a[4 + 5 * lda] = ONE;
            SLC_DLASET("A", n, m, &dbl0, &dbl0, b, &ldb);
            b[2 + 0 * ldb] = ONE;
            b[5 + 1 * ldb] = ONE;
            SLC_DLASET("A", p, n, &dbl0, &dbl0, c, &ldc);
            c[0 + 0 * ldc] = ONE;
            c[0 + 1 * ldc] = ONE;
            c[1 + 3 * ldc] = ONE;
            c[1 + 4 * ldc] = -ONE;
            SLC_DLASET("A", p, m, &dbl0, &dbl0, d, &ldd);
            d[0 + 0 * ldd] = ONE;
            d[1 + 0 * ldd] = ONE;

        } else if (nr2 == 11) {
            strncpy(note, "Patnaik et al. 1980: tubular ammonia reactor", 70);
            *n = 9;
            *m = 3;
            *p = 2;
            if (lde < *n) { *info = -10; }
            if (lda < *n) { *info = -12; }
            if (ldb < *n) { *info = -14; }
            if (ldc < *p) { *info = -16; }
            if (ldd < *p) { *info = -18; }
            if (*info != 0) return;

            SLC_DLASET("A", n, n, &dbl0, &dbl1, e, &lde);
            SLC_DLASET("A", p, n, &dbl0, &dbl0, c, &ldc);
            c[0 + 0 * ldc] = ONE;
            c[1 + 4 * ldc] = ONE;
            SLC_DLASET("A", p, m, &dbl0, &dbl0, d, &ldd);

            *info = 1;
            return;

        } else if (nr2 == 12) {
            strncpy(note, "Smith 1969: two-stand cold rolling mill", 70);
            *n = 10;
            *m = 3;
            *p = 5;
            if (lde < *n) { *info = -10; }
            if (lda < *n) { *info = -12; }
            if (ldb < *n) { *info = -14; }
            if (ldc < *p) { *info = -16; }
            if (ldd < *p) { *info = -18; }
            if (*info != 0) return;
            vec[7] = true;

            SLC_DLASET("A", n, n, &dbl0, &dbl1, e, &lde);
            SLC_DLASET("A", n, n, &dbl0, &dbl0, a, &lda);
            i32 nm1 = *n - 1;
            SLC_DLASET("A", &nm1, &nm1, &dbl0, &dbl1, &a[1 + 0 * lda], &lda);
            a[0 + (*n - 1) * lda] = 0.112;
            SLC_DLASET("A", n, m, &dbl0, &dbl0, b, &ldb);
            b[0 + 0 * ldb] = 2.76;
            b[0 + 1 * ldb] = -1.35;
            b[0 + 2 * ldb] = -0.46;
            SLC_DLASET("A", p, n, &dbl0, &dbl0, c, &ldc);
            c[0 + 0 * ldc] = ONE;
            c[1 + (*n - 1) * ldc] = 0.894;
            c[2 + (*n - 1) * ldc] = -16.93;
            c[3 + (*n - 1) * ldc] = 0.07;
            c[4 + (*n - 1) * ldc] = 0.398;

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
            strncpy(note, "Pappas et al. 1980: process control of paper machine", 70);
            if (lsame_local(def[0], 'D')) {
                dpar[0] = 1e8;
                dpar[1] = ONE;
                dpar[2] = ONE;
            }
            if (dpar[0] == ZERO) { *info = -3; }
            *n = 4;
            *m = 1;
            *p = 1;
            if (lde < *n) { *info = -10; }
            if (lda < *n) { *info = -12; }
            if (ldb < *n) { *info = -14; }
            if (ldc < *p) { *info = -16; }
            if (ldd < *p) { *info = -18; }
            if (*info != 0) return;

            temp = dpar[1] / dpar[0];
            SLC_DLASET("A", n, n, &dbl0, &dbl1, e, &lde);
            SLC_DLASET("A", n, n, &dbl0, &dbl0, a, &lda);
            i32 nm1 = *n - 1;
            SLC_DLASET("A", &nm1, &nm1, &dbl0, &dbl1, &a[1 + 0 * lda], &lda);
            a[0 + 0 * lda] = ONE - temp;
            SLC_DLASET("A", n, m, &dbl0, &dbl0, b, &ldb);
            b[0 + 0 * ldb] = dpar[2] * temp;
            SLC_DLASET("A", p, n, &dbl0, &dbl0, c, &ldc);
            c[0 + 3 * ldc] = ONE;
            SLC_DLASET("A", p, m, &dbl0, &dbl0, d, &ldd);

        } else {
            *info = -2;
        }

    } else if (nr1 == 3) {
        if (!(lsame_local(def[0], 'D') || lsame_local(def[0], 'N'))) {
            *info = -1;
            return;
        }

        if (nr2 == 1) {
            strncpy(note, "Pappas et al. 1980, Ex. 3", 70);
            if (lsame_local(def[0], 'D')) ipar[0] = 100;
            if (ipar[0] < 2) { *info = -4; }
            *n = ipar[0];
            *m = 1;
            *p = *n;
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
            SLC_DLASET("A", p, n, &dbl0, &dbl1, c, &ldc);
            SLC_DLASET("A", p, m, &dbl0, &dbl0, d, &ldd);

        } else {
            *info = -2;
        }

    } else {
        *info = -2;
    }
}
