/*
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * IB01ND - SVD system order via block Hankel
 *
 * Computes singular value decomposition of triangular factor R from
 * QR factorization of concatenated block Hankel matrices.
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <float.h>
#include <stdbool.h>
#include <stdlib.h>

i32 SLC_IB01ND(char meth, char jobd, i32 nobr, i32 m, i32 l,
               f64 *r, i32 ldr, f64 *sv, f64 tol,
               i32 *iwork, f64 *dwork, i32 ldwork,
               i32 *iwarn, i32 *info)
{
    bool moesp = (meth == 'M' || meth == 'm');
    bool n4sid = (meth == 'N' || meth == 'n');
    bool jobdm = (jobd == 'M' || jobd == 'm');

    i32 mnobr = m * nobr;
    i32 lnobr = l * nobr;
    i32 llnobr = lnobr + lnobr;
    i32 lmnobr = lnobr + mnobr;
    i32 mmnobr = mnobr + mnobr;
    i32 lmmnob = mmnobr + lnobr;
    i32 nr = lmnobr + lmnobr;

    *iwarn = 0;
    *info = 0;

    /* Parameter validation */
    if (!moesp && !n4sid) {
        *info = -1;
        return *info;
    }
    if (moesp && !(jobdm || jobd == 'N' || jobd == 'n')) {
        *info = -2;
        return *info;
    }
    if (nobr <= 0) {
        *info = -3;
        return *info;
    }
    if (m < 0) {
        *info = -4;
        return *info;
    }
    if (l <= 0) {
        *info = -5;
        return *info;
    }
    if (ldr < nr || (moesp && jobdm && ldr < 3 * mnobr)) {
        *info = -7;
        return *info;
    }

    /* Compute workspace requirements */
    i32 minwrk = 1;
    if (ldwork >= 1) {
        if (moesp) {
            minwrk = 5 * lnobr;
            if (jobdm) {
                i32 t1 = mmnobr - nobr;
                if (t1 < 0) t1 = 0;
                i32 t2 = lmnobr;
                minwrk = (t1 > minwrk) ? t1 : minwrk;
                minwrk = (t2 > minwrk) ? t2 : minwrk;
            }
        } else {
            i32 t = 5 * lmnobr + 1;
            minwrk = (t > minwrk) ? t : minwrk;
        }
    }

    if (ldwork < minwrk) {
        *info = -12;
        dwork[0] = (f64)minwrk;
        return *info;
    }

    /* Compute pointers to the needed blocks of R (0-based) */
    i32 nr2 = mnobr;
    i32 nr3 = mmnobr;
    i32 nr4 = lmmnob;
    i32 itau = 0;
    i32 jwork = itau + mnobr;

    i32 ierr;
    i32 maxwrk = minwrk;

    f64 toll = tol;
    f64 eps = DBL_EPSILON;
    f64 thresh = pow(eps, 2.0 / 3.0);
    f64 rcond1 = 1.0;
    f64 rcond2 = 1.0;

    if (moesp) {
        /* MOESP approach */
        if (m > 0 && jobdm) {
            /* Rearrange blocks: copy (1,1) to (3,2) and (1,4) to (3,3) */
            for (i32 j = 0; j < mnobr; j++) {
                for (i32 i = 0; i <= j; i++) {
                    r[(nr3 + i) + (nr2 + j) * ldr] = r[i + j * ldr];
                }
            }
            for (i32 j = 0; j < lnobr; j++) {
                for (i32 i = 0; i < mnobr; i++) {
                    r[(nr3 + i) + (nr3 + j) * ldr] = r[i + (nr4 + j) * ldr];
                }
            }

            /* Triangularize R_1c using MB04OD, then apply to R_2c using MB04ID */
            mb04od("U", nobr, lnobr, mnobr,
                   &r[nr2 + nr2 * ldr], ldr,
                   &r[nr3 + nr2 * ldr], ldr,
                   &r[nr2 + nr3 * ldr], ldr,
                   &r[nr3 + nr3 * ldr], ldr,
                   &dwork[itau], &dwork[jwork]);

            mb04id(mmnobr, mnobr, mnobr - 1, lnobr,
                   &r[0 + nr2 * ldr], ldr,
                   &r[0 + nr3 * ldr], ldr,
                   &dwork[itau], &dwork[jwork], ldwork - jwork, &ierr);

            if (dwork[jwork] + jwork > maxwrk)
                maxwrk = (i32)dwork[jwork] + jwork;

            /* Copy leading submatrices to final positions */
            for (i32 j = 0; j < mnobr; j++) {
                for (i32 i = 0; i <= j; i++) {
                    r[(lmnobr + i) + j * ldr] = r[i + (nr2 + j) * ldr];
                }
            }
            for (i32 j = 0; j < lnobr; j++) {
                for (i32 i = 0; i < mnobr; i++) {
                    r[i + (nr2 + j) * ldr] = r[i + (nr3 + j) * ldr];
                }
            }
        }

        /* Copy [R_24' R_34']' to [R_22' R_32']' */
        for (i32 j = 0; j < lnobr; j++) {
            for (i32 i = 0; i < lmnobr; i++) {
                r[(nr2 + i) + (nr2 + j) * ldr] = r[(nr2 + i) + (nr4 + j) * ldr];
            }
        }

        /* Triangularize via DGEQRF */
        jwork = itau + lnobr;
        i32 ldw = ldwork - jwork;
        SLC_DGEQRF(&lmnobr, &lnobr, &r[nr2 + nr2 * ldr], &ldr,
                   &dwork[itau], &dwork[jwork], &ldw, &ierr);

    } else {
        /* N4SID approach */
        i32 llmnob = llnobr + mnobr;

        if (m > 0) {
            /* Interchange first two block-columns */
            for (i32 i = 0; i < mnobr; i++) {
                /* Swap columns i and mnobr+i up to row i */
                for (i32 k = 0; k <= i; k++) {
                    f64 temp = r[k + i * ldr];
                    r[k + i * ldr] = r[k + (mnobr + i) * ldr];
                    r[k + (mnobr + i) * ldr] = temp;
                }
                /* Copy R(i+1:mnobr+i, mnobr+i) to R(i+1:mnobr+i, i) */
                for (i32 k = i + 1; k <= mnobr + i && k < mmnobr; k++) {
                    r[k + i * ldr] = r[k + (mnobr + i) * ldr];
                }
                /* Zero out R(i+1:mmnobr, mnobr+i) */
                for (i32 k = i + 1; k < mmnobr; k++) {
                    r[k + (mnobr + i) * ldr] = 0.0;
                }
            }

            /* Triangularize U_f using MB04ID */
            mb04id(mmnobr, mnobr, mnobr - 1, llmnob,
                   r, ldr, &r[0 + nr2 * ldr], ldr,
                   &dwork[itau], &dwork[jwork], ldwork - jwork, &ierr);

            if (dwork[jwork] + jwork > maxwrk)
                maxwrk = (i32)dwork[jwork] + jwork;

            /* Save Y_f (transposed) to last block-row */
            ma02ad("F", lmmnob, lnobr,
                   &r[0 + nr4 * ldr], ldr,
                   &r[nr4 + 0 * ldr], ldr);

            /* Check condition of triangular factor */
            SLC_DTRCON("1", "U", "N", &mnobr, r, &ldr,
                       &rcond1, &dwork[jwork], iwork, &ierr);

            if (toll <= 0.0)
                toll = (f64)(mnobr * mnobr) * eps;

            f64 thresh_test = (toll > thresh) ? toll : thresh;

            if (rcond1 > thresh_test) {
                /* Full rank - set residual area to zero */
                for (i32 j = 0; j < llmnob; j++) {
                    for (i32 i = 0; i < mnobr; i++) {
                        r[i + (nr2 + j) * ldr] = 0.0;
                    }
                }
            } else {
                /* Rank-deficient: save q info, use QR with pivoting */
                for (i32 i = 0; i < mnobr - 1; i++) {
                    for (i32 j = mmnobr - 1; j >= nr2; j--) {
                        r[j + i * ldr] = r[(j - mnobr + i) + i * ldr];
                    }
                    for (i32 k = i + 1; k < mnobr; k++) {
                        r[k + i * ldr] = 0.0;
                    }
                    iwork[i] = 0;
                }
                iwork[mnobr - 1] = 0;

                i32 itau2 = jwork;
                jwork = itau2 + mnobr;
                f64 svlmax = 0.0;
                i32 rank;
                f64 sval[3];

                mb03od("Q", mnobr, mnobr, r, ldr, iwork, toll, svlmax,
                       &dwork[itau2], &rank, sval, &dwork[jwork],
                       ldwork - jwork, &ierr);

                if (dwork[jwork] + jwork > maxwrk)
                    maxwrk = (i32)dwork[jwork] + jwork;

                /* Apply Q^T to right-hand side */
                i32 ldw = ldwork - jwork;
                SLC_DORMQR("L", "T", &mnobr, &llmnob, &mnobr,
                           r, &ldr, &dwork[itau2], &r[0 + nr2 * ldr], &ldr,
                           &dwork[jwork], &ldw, &ierr);

                if (rank < mnobr)
                    *iwarn = 4;

                /* Zero out rank rows and apply Q */
                for (i32 j = 0; j < llmnob; j++) {
                    for (i32 i = 0; i < rank; i++) {
                        r[i + (nr2 + j) * ldr] = 0.0;
                    }
                }

                SLC_DORMQR("L", "N", &mnobr, &llmnob, &mnobr,
                           r, &ldr, &dwork[itau2], &r[0 + nr2 * ldr], &ldr,
                           &dwork[jwork], &ldw, &ierr);

                jwork = itau2;

                /* Restore q transformation */
                for (i32 i = 0; i < mnobr - 1; i++) {
                    for (i32 j = nr2; j < mmnobr; j++) {
                        r[(j - mnobr + i) + i * ldr] = r[j + i * ldr];
                    }
                }
            }

            /* Apply q transformations backward */
            mb04iy("L", "N", mmnobr, llmnob, mnobr, mnobr - 1,
                   r, ldr, &dwork[itau], &r[0 + nr2 * ldr], ldr,
                   &dwork[jwork], ldwork - jwork, &ierr);

            if (dwork[jwork] + jwork > maxwrk)
                maxwrk = (i32)dwork[jwork] + jwork;

        } else {
            /* M = 0: just save Y_f transposed */
            ma02ad("F", lmmnob, lnobr,
                   &r[0 + nr4 * ldr], ldr,
                   &r[nr4 + 0 * ldr], ldr);
            rcond1 = 1.0;
        }

        /* Triangularize r_1 using DGEQRF */
        i32 ldw = ldwork - jwork;
        SLC_DGEQRF(&mmnobr, &mnobr, &r[0 + nr2 * ldr], &ldr,
                   &dwork[itau], &dwork[jwork], &ldw, &ierr);

        /* Apply Q^T to r_2 */
        SLC_DORMQR("L", "T", &mmnobr, &llnobr, &mnobr,
                   &r[0 + nr2 * ldr], &ldr, &dwork[itau],
                   &r[0 + nr3 * ldr], &ldr,
                   &dwork[jwork], &ldw, &ierr);

        i32 nrsave = nr2;
        i32 itau2 = jwork;
        jwork = itau2 + lnobr;

        mb04id(lmnobr, lnobr, lnobr - 1, lnobr,
               &r[nr2 + nr3 * ldr], ldr,
               &r[nr2 + nr4 * ldr], ldr,
               &dwork[itau2], &dwork[jwork], ldwork - jwork, &ierr);

        if (dwork[jwork] + jwork > maxwrk)
            maxwrk = (i32)dwork[jwork] + jwork;

        /* Check condition of triangular matrix */
        SLC_DTRCON("1", "U", "N", &lmnobr, &r[0 + nr2 * ldr], &ldr,
                   &rcond2, &dwork[jwork], iwork, &ierr);

        if (tol <= 0.0)
            toll = (f64)(lmnobr * lmnobr) * eps;

        f64 thresh_test2 = (toll > thresh) ? toll : thresh;

        if (rcond2 <= thresh_test2) {
            if (m > 0) {
                /* Save Q info */
                for (i32 j = 0; j < mnobr; j++) {
                    for (i32 i = 1; i < mmnobr; i++) {
                        r[i + j * ldr] = r[i + (nr2 + j) * ldr];
                    }
                }
                nrsave = 0;

                for (i32 i = nr2; i < lmnobr; i++) {
                    for (i32 k = 0; k < mnobr; k++) {
                        r[(mnobr + i) + k * ldr] = r[(i + 1) + (mnobr + i) * ldr];
                    }
                }
            }

            /* Zero lower triangle */
            for (i32 j = 0; j < lmnobr - 1; j++) {
                for (i32 i = j + 1; i < lmnobr; i++) {
                    r[(1 + i) + (nr2 + j) * ldr] = 0.0;
                }
            }

            /* QR with pivoting */
            for (i32 i = 0; i < lmnobr; i++)
                iwork[i] = 0;

            i32 itau3 = jwork;
            jwork = itau3 + lmnobr;
            f64 svlmax = 0.0;
            i32 rank1;
            f64 sval[3];

            mb03od("Q", lmnobr, lmnobr, &r[0 + nr2 * ldr], ldr,
                   iwork, toll, svlmax, &dwork[itau3], &rank1, sval,
                   &dwork[jwork], ldwork - jwork, &ierr);

            if (dwork[jwork] + jwork > maxwrk)
                maxwrk = (i32)dwork[jwork] + jwork;

            /* Apply Q^T */
            ldw = ldwork - jwork;
            SLC_DORMQR("L", "T", &lmnobr, &lnobr, &lmnobr,
                       &r[0 + nr2 * ldr], &ldr, &dwork[itau3],
                       &r[0 + nr4 * ldr], &ldr,
                       &dwork[jwork], &ldw, &ierr);

            if (rank1 < lmnobr)
                *iwarn = 5;

            /* Zero out and apply Q */
            for (i32 j = 0; j < lnobr; j++) {
                for (i32 i = rank1; i < lmnobr; i++) {
                    r[i + (nr4 + j) * ldr] = 0.0;
                }
            }

            SLC_DORMQR("L", "N", &lmnobr, &lnobr, &lmnobr,
                       &r[0 + nr2 * ldr], &ldr, &dwork[itau3],
                       &r[0 + nr4 * ldr], &ldr,
                       &dwork[jwork], &ldw, &ierr);

            jwork = itau3;

            if (m > 0) {
                /* Restore transpose from R_31 */
                for (i32 i = nr2; i < lmnobr; i++) {
                    for (i32 k = 0; k < mnobr; k++) {
                        r[(i + 1) + (mnobr + i) * ldr] = r[(mnobr + i) + k * ldr];
                    }
                }
            }
        }

        /* Apply MB04IY backward */
        mb04iy("L", "N", lmnobr, lnobr, lnobr, lnobr - 1,
               &r[nr2 + nr3 * ldr], ldr, &dwork[itau2],
               &r[nr2 + nr4 * ldr], ldr,
               &dwork[jwork], ldwork - jwork, &ierr);

        if (dwork[jwork] + jwork > maxwrk)
            maxwrk = (i32)dwork[jwork] + jwork;

        /* Apply Q using DORMQR */
        jwork = itau2;
        ldw = ldwork - jwork;
        SLC_DORMQR("L", "N", &mmnobr, &lnobr, &mnobr,
                   &r[0 + nrsave * ldr], &ldr, &dwork[itau],
                   &r[0 + nr4 * ldr], &ldr,
                   &dwork[jwork], &ldw, &ierr);

        /* Triangularize P' using DGEQRF */
        jwork = itau + lnobr;
        ldw = ldwork - jwork;
        SLC_DGEQRF(&lmmnob, &lnobr, &r[0 + nr4 * ldr], &ldr,
                   &dwork[itau], &dwork[jwork], &ldw, &ierr);

        /* Copy triangular factor to R_22 */
        for (i32 j = 0; j < lnobr; j++) {
            for (i32 i = 0; i <= j; i++) {
                r[(nr2 + i) + (nr2 + j) * ldr] = r[i + (nr4 + j) * ldr];
            }
        }

        /* Restore Y_f */
        ma02ad("F", lnobr, lmmnob,
               &r[nr4 + 0 * ldr], ldr,
               &r[0 + nr4 * ldr], ldr);
    }

    /* SVD of R_22 */
    f64 dum_arr = 0.0;
    mb03ud('N', 'V', lnobr, &r[nr2 + nr2 * ldr], ldr, &dum_arr, 1, sv,
           dwork, ldwork, &ierr);

    if (ierr != 0) {
        *info = 2;
        return *info;
    }

    if (dwork[0] > maxwrk)
        maxwrk = (i32)dwork[0];

    /* Transpose R_22 in-place (columns become singular vectors) */
    for (i32 i = nr2 + 1; i < lmnobr; i++) {
        i32 len = lmnobr - i;
        for (i32 k = 0; k < len; k++) {
            f64 temp = r[(i + k) + (i - 1) * ldr];
            r[(i + k) + (i - 1) * ldr] = r[(i - 1) + (i + k) * ldr];
            r[(i - 1) + (i + k) * ldr] = temp;
        }
    }

    /* Return optimal workspace and condition numbers */
    dwork[0] = (f64)maxwrk;
    if (n4sid) {
        dwork[1] = rcond1;
        dwork[2] = rcond2;
    }

    return 0;
}
