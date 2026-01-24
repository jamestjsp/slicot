/*
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * NF01BV - Compute J'*J + c*I for single output Jacobian (Cholesky method).
 *
 * Computes the matrix J'*J + c*I for the Jacobian J as received from NF01BY
 * for one output variable. Used with MD03AD direct (Cholesky) solver.
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <ctype.h>

void nf01bv(const char *stor, const char *uplo, const i32 *n,
            const i32 *ipar, const i32 *lipar,
            const f64 *dpar, const i32 *ldpar,
            const f64 *j, const i32 *ldj,
            f64 *jtj, const i32 *ldjtj,
            f64 *dwork, const i32 *ldwork, i32 *info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    i32 int0 = 0;
    i32 int1 = 1;

    char stor_c = toupper((unsigned char)stor[0]);
    char uplo_c = toupper((unsigned char)uplo[0]);

    bool full = (stor_c == 'F');
    bool upper = (uplo_c == 'U');

    *info = 0;

    if (!full && stor_c != 'P') {
        *info = -1;
    } else if (!upper && uplo_c != 'L') {
        *info = -2;
    } else if (*n < 0) {
        *info = -3;
    } else if (*lipar < 1) {
        *info = -5;
    } else if (*ldpar < 1) {
        *info = -7;
    } else if (*ldjtj < 1 || (full && *ldjtj < *n)) {
        *info = -11;
    } else if (*ldwork < 0) {
        *info = -13;
    } else {
        i32 m = ipar[0];
        if (m < 0) {
            *info = -4;
        } else if (*ldj < 1 || (m > 0 && *ldj < m)) {
            *info = -9;
        }
    }

    if (*info != 0) {
        return;
    }

    f64 c = dpar[0];
    i32 nn = *n;
    i32 m = ipar[0];

    if (nn == 0) {
        return;
    }

    if (m == 0) {
        if (full) {
            SLC_DLASET(uplo, &nn, &nn, &ZERO, &c, jtj, ldjtj);
        } else {
            i32 len = (nn * (nn + 1)) / 2;
            f64 dum = ZERO;
            SLC_DCOPY(&len, &dum, &int0, jtj, &int1);

            if (upper) {
                i32 ii = 0;
                for (i32 i = 1; i <= nn; i++) {
                    ii += i;
                    jtj[ii - 1] = c;
                }
            } else {
                i32 ii = 0;
                for (i32 i = nn; i >= 1; i--) {
                    jtj[ii] = c;
                    ii += i;
                }
            }
        }
        return;
    }

    if (full) {
        SLC_DLASET(uplo, &nn, &nn, &ZERO, &c, jtj, ldjtj);
        SLC_DSYRK(uplo, "T", &nn, &m, &ONE, j, ldj, &ONE, jtj, ldjtj);
    } else if (upper) {
        i32 ii = 0;
        for (i32 i = 1; i <= nn; i++) {
            SLC_DGEMV("T", &m, &i, &ONE, j, ldj, &j[(i - 1) * (*ldj)], &int1,
                      &ZERO, &jtj[ii], &int1);
            ii += i;
            jtj[ii - 1] += c;
        }
    } else {
        i32 ii = 0;
        for (i32 i = nn; i >= 1; i--) {
            i32 col = nn - i;
            SLC_DGEMV("T", &m, &i, &ONE, &j[col * (*ldj)], ldj,
                      &j[col * (*ldj)], &int1, &ZERO, &jtj[ii], &int1);
            jtj[ii] += c;
            ii += i;
        }
    }
}
