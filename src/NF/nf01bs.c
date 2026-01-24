/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <ctype.h>
#include <string.h>

/**
 * @brief QR factorization of Jacobian in compressed form.
 *
 * Computes QR factorization with column pivoting of Jacobian J in compressed form.
 *
 * @param[in] n Number of columns of J.
 * @param[in] ipar Integer parameters (st, bn, bsm, bsn).
 * @param[in] lipar Length of ipar.
 * @param[in] fnorm Norm of error vector.
 * @param[in,out] j Jacobian matrix (ldj x nc).
 * @param[in] ldj Leading dimension of J.
 * @param[in,out] e Error vector.
 * @param[out] jnorms Euclidean norms of columns of J.
 * @param[out] gnorm 1-norm of scaled gradient.
 * @param[out] ipvt Permutation matrix P.
 * @param[out] dwork Workspace.
 * @param[in] ldwork Length of dwork.
 * @param[out] info Exit code.
 */
static void local_dlacpy_safe(i32 m, i32 n, const f64 *a, i32 lda, f64 *b, i32 ldb) {
    for (i32 j = 0; j < n; j++) {
        memmove(&b[j * ldb], &a[j * lda], m * sizeof(f64));
    }
}

void nf01bs(i32 n, const i32 *ipar, i32 lipar, f64 fnorm, f64 *j, i32 *ldj, 
            f64 *e, f64 *jnorms, f64 *gnorm, i32 *ipvt, f64 *dwork, 
            i32 ldwork, i32 *info)
{
    /* Local variables */
    i32 bn, bsm, bsn, i, ibsm, ibsn, ibsni, itau, jl, jlm, jwork, k, l, m, mmn, nths, st, wrkopt;
    f64 sum, zero = 0.0, one = 1.0;
    i32 inc_1 = 1;

    *info = 0;
    if (n < 0) *info = -1;
    else if (lipar < 4) *info = -3;
    else if (fnorm < zero) *info = -4;
    else if (*ldj < (n > 1 ? n : 1)) *info = -6; /* Check ldj >= max(1,n) */
    else {
        st = ipar[0];
        bn = ipar[1];
        bsm = ipar[2];
        bsn = ipar[3];
        nths = bn * bsn;
        mmn = bsm - bsn;
        m = (bn > 0) ? bn * bsm : n;
        
        if (st < 0 || bn < 0 || bsm < 0 || bsn < 0) *info = -2;
        else if (n != nths + st) *info = -1;
        else if (m < n) *info = -2;
        else if (*ldj < (m > 1 ? m : 1)) *info = -6;
        else {
            if (n == 0) jwork = 1;
            else if (bn <= 1 || bsn == 0) {
                if (bn <= 1 && bsm == 1 && n == 1) jwork = 1;
                else jwork = 4 * n + 1;
            } else {
                jwork = bsn + ((3 * bsn + 1 > st) ? 3 * bsn + 1 : st);
                if (bsm > bsn) {
                    jwork = (jwork > 4 * st + 1) ? jwork : 4 * st + 1;
                    if (bsm < 2 * bsn) {
                        i32 term = mmn * (bn - 1);
                        jwork = (jwork > term) ? jwork : term;
                    }
                }
            }
            if (ldwork < jwork) *info = -12;
        }
    }

    if (*info != 0) {
        i32 err_code = -(*info);
        SLC_XERBLA("NF01BS", &err_code);
        return;
    }

    *gnorm = zero;
    if (n == 0) {
        *ldj = 1;
        dwork[0] = one;
        return;
    }

    if (bn <= 1 || bsn == 0) {
        /* Special case: full matrix */
        /* Call MD03BX( M, N, FNORM, J, LDJ, E, JNORMS, GNORM, IPVT, DWORK, LDWORK, INFO ) */
        /* Note: MD03BX expects LDJ as int*, but uses it as input/output? 
           MD03BX signature in slicot.h:
           void md03bx(i32 m, i32 n, f64 fnorm, f64* j, i32* ldj, f64* e, ...);
           Yes, pointers.
        */
        md03bx(m, n, fnorm, j, ldj, e, jnorms, gnorm, ipvt, dwork, ldwork, info);
        return;
    }

    /* General case: bn > 1 and bsn > 0 */
    for (i = 0; i < n; i++) ipvt[i] = 0;

    wrkopt = 1;
    ibsn = 0; /* 0-based */
    jl = (*ldj) * bsn; /* 0-based start of last block column? 
        Fortran: JL = LDJ*BSN + 1.
        J(JL) is first element of last block column.
        In C: j[ldj * bsn].
    */
    jwork = bsn; /* 0-based index for workspace start */
    
    for (ibsm = 0; ibsm < m; ibsm += bsm) {
        /* DGEQP3( BSM, BSN, J(IBSM), LDJ, IPVT(IBSN), DWORK, DWORK(JWORK), ... ) */
        /* J(IBSM) -> &j[ibsm]. Stride ldj.
           IPVT(IBSN) -> &ipvt[ibsn].
           DWORK(JWORK) -> &dwork[jwork].
        */
        i32 lwork_qp = ldwork - jwork;
        SLC_DGEQP3(&bsm, &bsn, &j[ibsm], ldj, &ipvt[ibsn], dwork, 
                   &dwork[jwork], &lwork_qp, info);
        
        if ((i32)dwork[jwork] + jwork > wrkopt) wrkopt = (i32)dwork[jwork] + jwork;
        
        if (ibsm > 0) {
            /* Adjust pivoting indices */
            for (i = ibsn; i < ibsn + bsn; i++) {
                ipvt[i] += ibsn;
            }
        }
        
        if (st > 0) {
            /* DORMQR('L', 'T', BSM, ST, BSN, J(IBSM), LDJ, DWORK, J(JL), LDJ, ... ) */
            /* Apply Q' to last block column */
            /* J(JL) -> &j[jl + ibsm]. (Rows match current block) */
            i32 lwork_mq = ldwork - jwork;
            SLC_DORMQR("Left", "Transpose", &bsm, &st, &bsn, &j[ibsm], ldj, 
                       dwork, &j[jl + ibsm], ldj, &dwork[jwork], &lwork_mq, info);
            
            if ((i32)dwork[jwork] + jwork > wrkopt) wrkopt = (i32)dwork[jwork] + jwork;
        }
        
        /* Apply Q' to e */
        /* DORMQR(..., E(IBSM), BSM, ... ) */
        i32 one_i = 1;
        i32 lwork_mq = ldwork - jwork;
        SLC_DORMQR("Left", "Transpose", &bsm, &one_i, &bsn, &j[ibsm], ldj, 
                   dwork, &e[ibsm], &bsm, &dwork[jwork], &lwork_mq, info);
                   
        /* Update pointers for next block */
        /* Note: jl stays pointing to start of last block col? 
           Fortran: JL = JL + BSM.
           But in C 0-based: &j[jl + ibsm] handled row offset.
           Wait.
           Fortran: J(JL). JL = LDJ*BSN + 1 initially.
           Loop 1: J(LDJ*BSN + 1).
           Loop 2: J(LDJ*BSN + 1 + BSM).
           
           My C: &j[jl + ibsm]. ibsm increases by bsm.
           So this matches.
        */
        ibsn += bsn;
    }

    if (mmn > 0) {
        l = ipvt[0] - 1;
        jnorms[l] = fabs(j[0]);
        ibsm = bsm;
        ibsn = bsn;

        for (k = 0; k < bn - 1; k++) {
            j[ibsn] = j[ibsm];
            l = ipvt[ibsn] - 1;
            jnorms[l] = fabs(j[ibsn]);
            ibsm += bsm;
            ibsn += bsn;
        }

        ibsn += st;

        for (i = 1; i < bsn; i++) {
            ibsm = i * (*ldj);
            jlm = i;

            for (k = 0; k < bn; k++) {
                for (i32 jj = 0; jj <= i; jj++) {
                    j[ibsn + jj] = j[ibsm + jj];
                }
                l = ipvt[jlm] - 1;
                i32 len_i = i + 1;
                jnorms[l] = SLC_DNRM2(&len_i, &j[ibsn], &inc_1);
                ibsm += bsm;
                ibsn += bsn;
                jlm += bsn;
            }

            ibsn += st;
        }

        jl = (*ldj) * bsn;
        if (bsm >= 2 * bsn) {
            for (i = 0; i < st; i++) {
                ibsn = bsn;
                for (ibsm = bsm; ibsm < m; ibsm += bsm) {
                    SLC_DSWAP(&mmn, &j[jl + ibsm], &inc_1, &j[jl + ibsn], &inc_1);
                    ibsn += bsn;
                }
                jl += (*ldj);
            }

            ibsn = bsn;
            for (ibsm = bsm; ibsm < m; ibsm += bsm) {
                SLC_DSWAP(&mmn, &e[ibsm], &inc_1, &e[ibsn], &inc_1);
                ibsn += bsn;
            }
        } else {
            for (i = 0; i < st; i++) {
                ibsn = bsn;
                jlm = jl + ibsn;
                jwork = 0;

                for (ibsm = bsm; ibsm < m; ibsm += bsm) {
                    SLC_DCOPY(&mmn, &j[jlm], &inc_1, &dwork[jwork], &inc_1);
                    for (k = jl; k < jl + bsn; k++) {
                        j[ibsn + k - jl] = j[ibsm + k - jl];
                    }
                    jlm += bsm;
                    ibsn += bsn;
                    jwork += mmn;
                }

                i32 len = mmn * (bn - 1);
                SLC_DCOPY(&len, dwork, &inc_1, &j[jl + ibsn - bsm], &inc_1);
                jl += (*ldj);
            }

            ibsn = bsn;
            jlm = ibsn;
            jwork = 0;

            for (ibsm = bsm; ibsm < m; ibsm += bsm) {
                SLC_DCOPY(&mmn, &e[jlm], &inc_1, &dwork[jwork], &inc_1);
                for (k = 0; k < bsn; k++) {
                    e[ibsn + k] = e[ibsm + k];
                }
                jlm += bsm;
                ibsn += bsn;
                jwork += mmn;
            }

            i32 len = mmn * (bn - 1);
            SLC_DCOPY(&len, dwork, &inc_1, &e[ibsn], &inc_1);
        }

        if (st > 0) {
            jl = ((*ldj) + bn) * bsn;
            itau = 0;
            jwork = itau + st;
            i32 mmn_bn = mmn * bn;
            i32 lwork_qp = ldwork - jwork;
            for (i = nths; i < n; i++) ipvt[i] = 0;
            SLC_DGEQP3(&mmn_bn, &st, &j[jl], ldj, &ipvt[nths], &dwork[itau], &dwork[jwork], &lwork_qp, info);
            if ((i32)dwork[jwork] + jwork > wrkopt) wrkopt = (i32)dwork[jwork] + jwork;

            i32 flag_true = 1;
            SLC_DLAPMT(&flag_true, &nths, &st, &j[jl - nths], ldj, &ipvt[nths]);

            for (i = nths; i < n; i++) ipvt[i] += nths;

            lwork_qp = ldwork - jwork;
            SLC_DORMQR("Left", "Transpose", &mmn_bn, &inc_1, &st, &j[jl], ldj, &dwork[itau], &e[ibsn], ldj, &dwork[jwork], &lwork_qp, info);
            if ((i32)dwork[jwork] + jwork > wrkopt) wrkopt = (i32)dwork[jwork] + jwork;

            ibsn = n * bsn;
            local_dlacpy_safe(n, st, &j[(*ldj) * bsn], *ldj, &j[ibsn], n);

            ibsni = ibsn;
            for (i = nths; i < n; i++) {
                l = ipvt[i] - 1;
                i32 len_i = i + 1;
                jnorms[l] = SLC_DNRM2(&len_i, &j[ibsni], &inc_1);
                ibsni += n;
            }
        }
    } else {
        ibsn = 0;
        for (i = 0; i < bsn; i++) {
            jlm = i;
            for (k = 0; k < bn; k++) {
                l = ipvt[jlm] - 1;
                i32 len_i = i + 1;
                jnorms[l] = SLC_DNRM2(&len_i, &j[ibsn], &inc_1);
                ibsn += bsn;
                jlm += bsn;
            }
            ibsn += st;
        }

        for (i = nths; i < n; i++) ipvt[i] = i + 1;
    }

    if (fnorm != zero) {
        for (ibsn = 0; ibsn < nths; ibsn += bsn) {
            ibsni = ibsn;
            for (i = 0; i < bsn; i++) {
                l = ipvt[ibsn + i] - 1;
                if (jnorms[l] != zero) {
                    i32 len_i = i + 1;
                    sum = SLC_DDOT(&len_i, &j[ibsni], &inc_1, &e[ibsn], &inc_1) / fnorm;
                    *gnorm = (*gnorm > fabs(sum / jnorms[l])) ? *gnorm : fabs(sum / jnorms[l]);
                }
                ibsni += n;
            }
        }

        ibsni = n * bsn;
        for (i = nths; i < n; i++) {
            l = ipvt[i] - 1;
            if (jnorms[l] != zero) {
                i32 len_i = i + 1;
                sum = SLC_DDOT(&len_i, &j[ibsni], &inc_1, e, &inc_1) / fnorm;
                *gnorm = (*gnorm > fabs(sum / jnorms[l])) ? *gnorm : fabs(sum / jnorms[l]);
            }
            ibsni += n;
        }
    }

    *ldj = n;
    dwork[0] = (f64)wrkopt;
}
