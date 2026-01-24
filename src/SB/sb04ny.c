/**
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * SB04NY - Solve Hessenberg system with one offdiagonal and one RHS
 *
 * Solves a system of equations in Hessenberg form with one offdiagonal
 * and one right-hand side.
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>

void sb04ny(
    const char* rc,
    const char* ul,
    const i32 m,
    const f64* a,
    const i32 lda,
    const f64 lambda,
    f64* d,
    const f64 tol,
    i32* iwork,
    f64* dwork,
    const i32 lddwor,
    i32* info)
{
    const f64 zero = 0.0;
    i32 int1 = 1;
    i32 j, j1, mj;
    f64 c_val, s_val, r_val, rcond;
    char trans;

    *info = 0;

    if (m == 0)
        return;

    bool lulu = (*ul == 'U' || *ul == 'u');
    bool lrcr = (*rc == 'R' || *rc == 'r');

    if (lulu) {
        for (j = 0; j < m; j++) {
            i32 len = (j + 2 <= m) ? (j + 2) : m;
            SLC_DCOPY(&len, &a[j * lda], &int1, &dwork[j * lddwor], &int1);
            dwork[j + j * lddwor] += lambda;
        }

        if (lrcr) {
            trans = 'N';
            for (j = 0; j < m - 1; j++) {
                mj = m - j - 1;
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
            for (j = 0; j < m - 1; j++) {
                mj = m - j - 1;
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
            j1 = (j > 0) ? j : 1;
            j1--;
            i32 len = m - j1;
            SLC_DCOPY(&len, &a[j1 + j * lda], &int1, &dwork[j1 + j * lddwor], &int1);
            dwork[j + j * lddwor] += lambda;
        }

        if (lrcr) {
            trans = 'N';
            for (j = 0; j < m - 1; j++) {
                mj = m - j - 1;
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
            for (j = 0; j < m - 1; j++) {
                mj = m - j - 1;
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
    SLC_DTRCON("1", ul, "N", &m, dwork, &lddwor, &rcond,
               &dwork[m * lddwor], iwork, &info_trcon);

    if (rcond <= tol) {
        *info = 1;
    } else {
        SLC_DTRSV(ul, &trans, "N", &m, dwork, &lddwor, d, &int1);
    }
}
