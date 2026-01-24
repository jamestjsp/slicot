/**
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * SB04NX - Solve Hessenberg system with two offdiagonals and two RHS
 *
 * Solves a system of equations in Hessenberg form with two consecutive
 * offdiagonals and two right-hand sides.
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>

void sb04nx(
    const char* rc,
    const char* ul,
    const i32 m,
    const f64* a,
    const i32 lda,
    const f64 lambd1,
    const f64 lambd2,
    const f64 lambd3,
    const f64 lambd4,
    f64* d,
    const f64 tol,
    i32* iwork,
    f64* dwork,
    const i32 lddwor,
    i32* info)
{
    const f64 zero = 0.0;
    i32 int1 = 1, int2 = 2;
    i32 j, j1, j2, m2, mj, ml;
    f64 c_val, s_val, r_val, rcond;
    char trans;

    *info = 0;

    if (m == 0)
        return;

    m2 = m * 2;
    bool lulu = (*ul == 'U' || *ul == 'u');
    bool lrcr = (*rc == 'R' || *rc == 'r');

    if (lulu) {
        for (j = 0; j < m; j++) {
            j2 = (j + 1) * 2;
            ml = (j + 2 <= m) ? (j + 2) : m;

            f64 zero_val = 0.0;
            i32 m2_32 = m2;
            SLC_DLASET("F", &m2_32, &int2, &zero_val, &zero_val,
                       &dwork[(j2 - 2) * lddwor], &lddwor);

            SLC_DCOPY(&ml, &a[j * lda], &int1, &dwork[(j2 - 2) * lddwor], &int2);
            SLC_DCOPY(&ml, &a[j * lda], &int1, &dwork[1 + (j2 - 1) * lddwor], &int2);

            dwork[(j2 - 2) + (j2 - 2) * lddwor] += lambd1;
            dwork[(j2 - 1) + (j2 - 2) * lddwor] = lambd3;
            dwork[(j2 - 2) + (j2 - 1) * lddwor] = lambd2;
            dwork[(j2 - 1) + (j2 - 1) * lddwor] += lambd4;
        }

        if (lrcr) {
            trans = 'N';
            for (j = 0; j < m2 - 1; j++) {
                mj = m2 - j - 1;
                if (j < m2 - 2) {
                    if (dwork[(j + 2) + j * lddwor] != zero) {
                        SLC_DLARTG(&dwork[(j + 1) + j * lddwor], &dwork[(j + 2) + j * lddwor],
                                   &c_val, &s_val, &r_val);
                        dwork[(j + 1) + j * lddwor] = r_val;
                        dwork[(j + 2) + j * lddwor] = zero;
                        SLC_DROT(&mj, &dwork[(j + 1) + (j + 1) * lddwor], &lddwor,
                                 &dwork[(j + 2) + (j + 1) * lddwor], &lddwor, &c_val, &s_val);
                        SLC_DROT(&int1, &d[j + 1], &int1, &d[j + 2], &int1, &c_val, &s_val);
                    }
                }
                if (dwork[(j + 1) + j * lddwor] != zero) {
                    SLC_DLARTG(&dwork[j + j * lddwor], &dwork[(j + 1) + j * lddwor],
                               &c_val, &s_val, &r_val);
                    dwork[j + j * lddwor] = r_val;
                    dwork[(j + 1) + j * lddwor] = zero;
                    SLC_DROT(&mj, &dwork[j + (j + 1) * lddwor], &lddwor,
                             &dwork[(j + 1) + (j + 1) * lddwor], &lddwor, &c_val, &s_val);
                    SLC_DROT(&int1, &d[j], &int1, &d[j + 1], &int1, &c_val, &s_val);
                }
            }
        } else {
            trans = 'T';
            for (j = 0; j < m2 - 1; j++) {
                mj = m2 - j - 1;
                if (j < m2 - 2) {
                    if (dwork[mj + (mj - 2) * lddwor] != zero) {
                        SLC_DLARTG(&dwork[mj + (mj - 1) * lddwor], &dwork[mj + (mj - 2) * lddwor],
                                   &c_val, &s_val, &r_val);
                        dwork[mj + (mj - 1) * lddwor] = r_val;
                        dwork[mj + (mj - 2) * lddwor] = zero;
                        SLC_DROT(&mj, &dwork[(mj - 1) * lddwor], &int1,
                                 &dwork[(mj - 2) * lddwor], &int1, &c_val, &s_val);
                        SLC_DROT(&int1, &d[mj - 1], &int1, &d[mj - 2], &int1, &c_val, &s_val);
                    }
                }
                if (dwork[mj + (mj - 1) * lddwor] != zero) {
                    SLC_DLARTG(&dwork[mj + mj * lddwor], &dwork[mj + (mj - 1) * lddwor],
                               &c_val, &s_val, &r_val);
                    dwork[mj + mj * lddwor] = r_val;
                    dwork[mj + (mj - 1) * lddwor] = zero;
                    SLC_DROT(&mj, &dwork[mj * lddwor], &int1,
                             &dwork[(mj - 1) * lddwor], &int1, &c_val, &s_val);
                    SLC_DROT(&int1, &d[mj], &int1, &d[mj - 1], &int1, &c_val, &s_val);
                }
            }
        }
    } else {
        for (j = 0; j < m; j++) {
            j2 = (j + 1) * 2;
            j1 = (j > 0) ? j : 1;
            j1--;
            ml = ((m - j + 1) <= m) ? (m - j + 1) : m;
            if (j == 0) ml = m;

            f64 zero_val = 0.0;
            i32 m2_32 = m2;
            SLC_DLASET("F", &m2_32, &int2, &zero_val, &zero_val,
                       &dwork[(j2 - 2) * lddwor], &lddwor);

            SLC_DCOPY(&ml, &a[j1 + j * lda], &int1, &dwork[j1 * 2 + (j2 - 2) * lddwor], &int2);
            SLC_DCOPY(&ml, &a[j1 + j * lda], &int1, &dwork[j1 * 2 + 1 + (j2 - 1) * lddwor], &int2);

            dwork[(j2 - 2) + (j2 - 2) * lddwor] += lambd1;
            dwork[(j2 - 1) + (j2 - 2) * lddwor] = lambd3;
            dwork[(j2 - 2) + (j2 - 1) * lddwor] = lambd2;
            dwork[(j2 - 1) + (j2 - 1) * lddwor] += lambd4;
        }

        if (lrcr) {
            trans = 'N';
            for (j = 0; j < m2 - 1; j++) {
                mj = m2 - j - 1;
                if (j < m2 - 2) {
                    if (dwork[(mj - 2) + mj * lddwor] != zero) {
                        SLC_DLARTG(&dwork[(mj - 1) + mj * lddwor], &dwork[(mj - 2) + mj * lddwor],
                                   &c_val, &s_val, &r_val);
                        dwork[(mj - 1) + mj * lddwor] = r_val;
                        dwork[(mj - 2) + mj * lddwor] = zero;
                        SLC_DROT(&mj, &dwork[mj - 1], &lddwor,
                                 &dwork[mj - 2], &lddwor, &c_val, &s_val);
                        SLC_DROT(&int1, &d[mj - 1], &int1, &d[mj - 2], &int1, &c_val, &s_val);
                    }
                }
                if (dwork[(mj - 1) + mj * lddwor] != zero) {
                    SLC_DLARTG(&dwork[mj + mj * lddwor], &dwork[(mj - 1) + mj * lddwor],
                               &c_val, &s_val, &r_val);
                    dwork[mj + mj * lddwor] = r_val;
                    dwork[(mj - 1) + mj * lddwor] = zero;
                    SLC_DROT(&mj, &dwork[mj], &lddwor, &dwork[mj - 1], &lddwor,
                             &c_val, &s_val);
                    SLC_DROT(&int1, &d[mj], &int1, &d[mj - 1], &int1, &c_val, &s_val);
                }
            }
        } else {
            trans = 'T';
            for (j = 0; j < m2 - 1; j++) {
                mj = m2 - j - 1;
                if (j < m2 - 2) {
                    if (dwork[j + (j + 2) * lddwor] != zero) {
                        SLC_DLARTG(&dwork[j + (j + 1) * lddwor], &dwork[j + (j + 2) * lddwor],
                                   &c_val, &s_val, &r_val);
                        dwork[j + (j + 1) * lddwor] = r_val;
                        dwork[j + (j + 2) * lddwor] = zero;
                        SLC_DROT(&mj, &dwork[(j + 1) + (j + 1) * lddwor], &int1,
                                 &dwork[(j + 1) + (j + 2) * lddwor], &int1, &c_val, &s_val);
                        SLC_DROT(&int1, &d[j + 1], &int1, &d[j + 2], &int1, &c_val, &s_val);
                    }
                }
                if (dwork[j + (j + 1) * lddwor] != zero) {
                    SLC_DLARTG(&dwork[j + j * lddwor], &dwork[j + (j + 1) * lddwor],
                               &c_val, &s_val, &r_val);
                    dwork[j + j * lddwor] = r_val;
                    dwork[j + (j + 1) * lddwor] = zero;
                    SLC_DROT(&mj, &dwork[(j + 1) + j * lddwor], &int1,
                             &dwork[(j + 1) + (j + 1) * lddwor], &int1, &c_val, &s_val);
                    SLC_DROT(&int1, &d[j], &int1, &d[j + 1], &int1, &c_val, &s_val);
                }
            }
        }
    }

    i32 info_trcon = 0;
    SLC_DTRCON("1", ul, "N", &m2, dwork, &lddwor, &rcond,
               &dwork[m2 * lddwor], iwork, &info_trcon);

    if (rcond <= tol) {
        *info = 1;
    } else {
        SLC_DTRSV(ul, &trans, "N", &m2, dwork, &lddwor, d, &int1);
    }
}
