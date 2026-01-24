/*
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * NF01BU - Compute J'*J + c*I for full Wiener system Jacobian (Cholesky method).
 *
 * Computes the matrix J'*J + c*I for the Jacobian J as received from NF01BD
 * which has a block structure with diagonal blocks J_k and off-diagonal L_k.
 * Used with MD03AD direct (Cholesky) solver for the full Wiener system.
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <ctype.h>

void nf01bu(const char *stor, const char *uplo, const i32 *n,
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
    } else if (*lipar < 4) {
        *info = -5;
    } else if (*ldpar < 1) {
        *info = -7;
    } else if (*ldjtj < 1 || (full && *ldjtj < *n)) {
        *info = -11;
    } else if (*ldwork < 0) {
        *info = -13;
    } else {
        i32 st = ipar[0];
        i32 bn = ipar[1];
        i32 bsm = ipar[2];
        i32 bsn = ipar[3];
        i32 nths = bn * bsn;
        i32 m = (bn > 1) ? bn * bsm : bsm;

        if (st < 0 || bn < 0 || bsm < 0 || bsn < 0) {
            *info = -4;
        } else if (*n != nths + st) {
            *info = -3;
        } else if (*ldj < 1 || (m > 0 && *ldj < m)) {
            *info = -9;
        }
    }

    if (*info != 0) {
        return;
    }

    i32 nn = *n;
    if (nn == 0) {
        return;
    }

    f64 c = dpar[0];
    i32 st = ipar[0];
    i32 bn = ipar[1];
    i32 bsm = ipar[2];
    i32 bsn = ipar[3];
    i32 nths = bn * bsn;
    i32 m = (bn > 1) ? bn * bsm : bsm;
    i32 ldjj = *ldj;

    if (bn <= 1 || bsn == 0 || bsm == 0) {
        i32 itmp[1] = {m};
        nf01bv(stor, uplo, n, itmp, &int1, dpar, &int1, j, ldj,
               jtj, ldjtj, dwork, ldwork, info);
        return;
    }

    i32 jl = bsn;  /* column index of L block (0-based) */
    i32 nbsn = nn * bsn;

    if (full) {
        if (upper) {
            SLC_DLASET(uplo, &bsn, &bsn, &ZERO, &c, jtj, ldjtj);
            SLC_DSYRK(uplo, "T", &bsn, &bsm, &ONE, j, ldj, &ONE, jtj, ldjtj);

            i32 ibsn = bsn;
            i32 i1 = nbsn;

            for (i32 ibsm = bsm; ibsm < m; ibsm += bsm) {
                i32 ii = i1 + ibsn;
                SLC_DLASET("F", &ibsn, &bsn, &ZERO, &ZERO, &jtj[i1], ldjtj);
                i1 += nbsn;
                SLC_DLASET(uplo, &bsn, &bsn, &ZERO, &c, &jtj[ii], ldjtj);
                SLC_DSYRK(uplo, "T", &bsn, &bsm, &ONE, &j[ibsm], ldj, &ONE, &jtj[ii], ldjtj);
                ibsn += bsn;
            }

            if (st > 0) {
                for (i32 ibsm = 0; ibsm < m; ibsm += bsm) {
                    SLC_DGEMM("T", "N", &bsn, &st, &bsm, &ONE,
                              &j[ibsm], ldj, &j[ibsm + jl * ldjj], ldj,
                              &ZERO, &jtj[i1], ldjtj);
                    i1 += bsn;
                }
                SLC_DLASET(uplo, &st, &st, &ZERO, &c, &jtj[i1], ldjtj);
                SLC_DSYRK(uplo, "T", &st, &m, &ONE, &j[jl * ldjj], ldj, &ONE, &jtj[i1], ldjtj);
            }
        } else {
            i32 ibsn = nths;
            i32 ii = 0;

            for (i32 ibsm = 0; ibsm < m; ibsm += bsm) {
                i32 i1 = ii + bsn;
                SLC_DLASET(uplo, &bsn, &bsn, &ZERO, &c, &jtj[ii], ldjtj);
                SLC_DSYRK(uplo, "T", &bsn, &bsm, &ONE, &j[ibsm], ldj, &ONE, &jtj[ii], ldjtj);
                ibsn -= bsn;
                SLC_DLASET("F", &ibsn, &bsn, &ZERO, &ZERO, &jtj[i1], ldjtj);
                ii = i1 + nbsn;
                if (st > 0) {
                    SLC_DGEMM("T", "N", &st, &bsn, &bsm, &ONE,
                              &j[ibsm + jl * ldjj], ldj, &j[ibsm], ldj,
                              &ZERO, &jtj[i1 + ibsn], ldjtj);
                }
            }

            if (st > 0) {
                SLC_DLASET(uplo, &st, &st, &ZERO, &c, &jtj[ii], ldjtj);
                SLC_DSYRK(uplo, "T", &st, &m, &ONE, &j[jl * ldjj], ldj, &ONE, &jtj[ii], ldjtj);
            }
        }
    } else {
        f64 tmp = ZERO;

        if (upper) {
            i32 ibsn = 0;
            i32 i1 = 0;

            for (i32 ibsm = 0; ibsm < m; ibsm += bsm) {
                for (i32 k = 0; k < bsn; k++) {
                    i32 ii = i1 + ibsn;
                    i32 kp1 = k + 1;
                    SLC_DCOPY(&ibsn, &tmp, &int0, &jtj[i1], &int1);
                    SLC_DGEMV("T", &bsm, &kp1, &ONE, &j[ibsm], ldj,
                              &j[ibsm + k * ldjj], &int1, &ZERO, &jtj[ii], &int1);
                    i1 = ii + kp1;
                    jtj[i1 - 1] += c;
                }
                ibsn += bsn;
            }

            for (i32 k = 0; k < st; k++) {
                for (i32 ibsm = 0; ibsm < m; ibsm += bsm) {
                    SLC_DGEMV("T", &bsm, &bsn, &ONE, &j[ibsm], ldj,
                              &j[ibsm + (bsn + k) * ldjj], &int1, &ZERO, &jtj[i1], &int1);
                    i1 += bsn;
                }
                i32 kp1 = k + 1;
                SLC_DGEMV("T", &m, &kp1, &ONE, &j[jl * ldjj], ldj,
                          &j[(bsn + k) * ldjj], &int1, &ZERO, &jtj[i1], &int1);
                i1 += kp1;
                jtj[i1 - 1] += c;
            }
        } else {
            i32 ibsn = nths;
            i32 ii = 0;

            for (i32 ibsm = 0; ibsm < m; ibsm += bsm) {
                ibsn -= bsn;
                for (i32 k = 0; k < bsn; k++) {
                    i32 i1 = ii + bsn - k;
                    i32 cnt = bsn - k;
                    SLC_DCOPY(&ibsn, &tmp, &int0, &jtj[i1], &int1);
                    SLC_DGEMV("T", &bsm, &cnt, &ONE, &j[ibsm + k * ldjj], ldj,
                              &j[ibsm + k * ldjj], &int1, &ZERO, &jtj[ii], &int1);
                    jtj[ii] += c;
                    i1 += ibsn;
                    ii = i1 + st;
                    if (st > 0) {
                        SLC_DGEMV("T", &bsm, &st, &ONE, &j[ibsm + jl * ldjj], ldj,
                                  &j[ibsm + k * ldjj], &int1, &ZERO, &jtj[i1], &int1);
                    }
                }
            }

            for (i32 k = 0; k < st; k++) {
                i32 cnt = st - k;
                SLC_DGEMV("T", &m, &cnt, &ONE, &j[(bsn + k) * ldjj], ldj,
                          &j[(bsn + k) * ldjj], &int1, &ZERO, &jtj[ii], &int1);
                jtj[ii] += c;
                ii += cnt;
            }
        }
    }
}
