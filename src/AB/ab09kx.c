/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

/*
 * AB09KX - Stable projection of V*G*W or conj(V)*G*conj(W)
 *
 * Purpose:
 *   To construct a state-space representation (A,BS,CS,DS) of the
 *   stable projection of V*G*W or conj(V)*G*conj(W) from the
 *   state-space representations (A,B,C,D), (AV,BV,CV,DV), and
 *   (AW,BW,CW,DW) of the transfer-function matrices G, V and W,
 *   respectively. G is assumed to be a stable transfer-function
 *   matrix and the state matrix A must be in a real Schur form.
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>

#define A(i,j)  a[(i) + (j)*lda]
#define B(i,j)  b[(i) + (j)*ldb]
#define C(i,j)  c[(i) + (j)*ldc]
#define D(i,j)  d[(i) + (j)*ldd]
#define AV(i,j) av[(i) + (j)*ldav]
#define BV(i,j) bv[(i) + (j)*ldbv]
#define CV(i,j) cv[(i) + (j)*ldcv]
#define DV(i,j) dv[(i) + (j)*lddv]
#define AW(i,j) aw[(i) + (j)*ldaw]
#define BW(i,j) bw[(i) + (j)*ldbw]
#define CW(i,j) cw[(i) + (j)*ldcw]
#define DW(i,j) dw[(i) + (j)*lddw]
#define DWORK(i) dwork[i]

void ab09kx(
    const char* job,
    const char* dico,
    const char* weight,
    const i32 n,
    const i32 nv,
    const i32 nw,
    const i32 m,
    const i32 p,
    const f64* a,
    const i32 lda,
    f64* b,
    const i32 ldb,
    f64* c,
    const i32 ldc,
    f64* d,
    const i32 ldd,
    f64* av,
    const i32 ldav,
    f64* bv,
    const i32 ldbv,
    f64* cv,
    const i32 ldcv,
    const f64* dv,
    const i32 lddv,
    f64* aw,
    const i32 ldaw,
    f64* bw,
    const i32 ldbw,
    f64* cw,
    const i32 ldcw,
    const f64* dw,
    const i32 lddw,
    f64* dwork,
    const i32 ldwork,
    i32* iwarn,
    i32* info)
{
    const f64 zero = 0.0;
    const f64 one = 1.0;

    bool conjs, discr, frwght, leftw, rightw;
    f64 scale, work;
    i32 i, ia, ib, ierr, kw, ldw, ldwn, lw;

    char job_c = job[0];
    char dico_c = dico[0];
    char weight_c = weight[0];

    conjs = (job_c == 'C' || job_c == 'c');
    discr = (dico_c == 'D' || dico_c == 'd');
    leftw = (weight_c == 'L' || weight_c == 'l' || weight_c == 'B' || weight_c == 'b');
    rightw = (weight_c == 'R' || weight_c == 'r' || weight_c == 'B' || weight_c == 'b');
    frwght = leftw || rightw;

    *iwarn = 0;
    *info = 0;

    if (discr && conjs) {
        ia = 2 * nv;
        ib = 2 * nw;
    } else {
        ia = 0;
        ib = 0;
    }

    lw = 1;
    if (leftw) {
        i32 t1 = nv * (nv + 5);
        i32 t2a = (ia > p * n) ? ia : p * n;
        i32 t2 = (t2a > p * m) ? t2a : p * m;
        i32 t3 = nv * n + t2;
        lw = (lw > t1) ? lw : t1;
        lw = (lw > t3) ? lw : t3;
    }
    if (rightw) {
        i32 t1 = nw * (nw + 5);
        i32 t2a = (ib > m * n) ? ib : m * n;
        i32 t2 = (t2a > p * m) ? t2a : p * m;
        i32 t3 = nw * n + t2;
        lw = (lw > t1) ? lw : t1;
        lw = (lw > t3) ? lw : t3;
    }

    if (!(job_c == 'N' || job_c == 'n' || conjs)) {
        *info = -1;
    } else if (!(dico_c == 'C' || dico_c == 'c' || discr)) {
        *info = -2;
    } else if (!(frwght || weight_c == 'N' || weight_c == 'n')) {
        *info = -3;
    } else if (n < 0) {
        *info = -4;
    } else if (nv < 0) {
        *info = -5;
    } else if (nw < 0) {
        *info = -6;
    } else if (m < 0) {
        *info = -7;
    } else if (p < 0) {
        *info = -8;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -10;
    } else if (ldb < (n > 1 ? n : 1)) {
        *info = -12;
    } else if (ldc < (p > 1 ? p : 1)) {
        *info = -14;
    } else if (ldd < (p > 1 ? p : 1)) {
        *info = -16;
    } else if (ldav < 1 || (leftw && ldav < nv)) {
        *info = -18;
    } else if (ldbv < 1 || (leftw && ldbv < nv)) {
        *info = -20;
    } else if (ldcv < 1 || (leftw && ldcv < p)) {
        *info = -22;
    } else if (lddv < 1 || (leftw && lddv < p)) {
        *info = -24;
    } else if (ldaw < 1 || (rightw && ldaw < nw)) {
        *info = -26;
    } else if (ldbw < 1 || (rightw && ldbw < nw)) {
        *info = -28;
    } else if (ldcw < 1 || (rightw && ldcw < m)) {
        *info = -30;
    } else if (lddw < 1 || (rightw && lddw < m)) {
        *info = -32;
    } else if (ldwork < lw) {
        *info = -34;
    }

    if (*info != 0) {
        return;
    }

    if (!frwght || (m < 1) || (p < 1)) {
        dwork[0] = one;
        return;
    }

    work = one;

    if (leftw && nv > 0) {
        kw = nv * (nv + 2);
        tb01wd(nv, p, p, av, ldav, bv, ldbv, cv, ldcv,
               &dwork[2 * nv], nv, dwork, &dwork[nv],
               &dwork[kw], ldwork - kw, &ierr);
        if (ierr != 0) {
            *info = 1;
            return;
        }
        f64 workopt = dwork[kw] + (f64)(kw);
        work = (work > workopt) ? work : workopt;

        if (conjs) {
            if (discr) {
                for (i = 0; i < nv; i++) {
                    if (SLC_DLAPY2(&dwork[i], &dwork[nv + i]) >= one) {
                        *iwarn = 1;
                        break;
                    }
                }
            } else {
                for (i = 0; i < nv; i++) {
                    if (dwork[i] >= zero) {
                        *iwarn = 1;
                        break;
                    }
                }
            }
        } else {
            if (discr) {
                for (i = 0; i < nv; i++) {
                    if (SLC_DLAPY2(&dwork[i], &dwork[nv + i]) <= one) {
                        *iwarn = 1;
                        break;
                    }
                }
            } else {
                for (i = 0; i < nv; i++) {
                    if (dwork[i] <= zero) {
                        *iwarn = 1;
                        break;
                    }
                }
            }
        }
    }

    if (rightw && nw > 0) {
        kw = nw * (nw + 2);
        tb01wd(nw, m, m, aw, ldaw, bw, ldbw, cw, ldcw,
               &dwork[2 * nw], nw, dwork, &dwork[nw],
               &dwork[kw], ldwork - kw, &ierr);
        if (ierr != 0) {
            *info = 2;
            return;
        }
        f64 workopt = dwork[kw] + (f64)(kw);
        work = (work > workopt) ? work : workopt;

        if (conjs) {
            if (discr) {
                for (i = 0; i < nw; i++) {
                    if (SLC_DLAPY2(&dwork[i], &dwork[nw + i]) >= one) {
                        *iwarn = *iwarn + 2;
                        break;
                    }
                }
            } else {
                for (i = 0; i < nw; i++) {
                    if (dwork[i] >= zero) {
                        *iwarn = *iwarn + 2;
                        break;
                    }
                }
            }
        } else {
            if (discr) {
                for (i = 0; i < nw; i++) {
                    if (SLC_DLAPY2(&dwork[i], &dwork[nw + i]) <= one) {
                        *iwarn = *iwarn + 2;
                        break;
                    }
                }
            } else {
                for (i = 0; i < nw; i++) {
                    if (dwork[i] <= zero) {
                        *iwarn = *iwarn + 2;
                        break;
                    }
                }
            }
        }
    }

    if (leftw) {
        ldw = (nv > 1) ? nv : 1;
        kw = nv * n;

        if (conjs) {
            i32 nn = n;
            i32 pp = p;
            i32 nvv = nv;
            i32 mm = m;
            f64 neg_one = -one;

            SLC_DGEMM("T", "N", &nvv, &nn, &pp, &neg_one, cv, &ldcv, c, &ldc,
                      &zero, dwork, &ldw);

            if (discr) {
                sb04py('T', 'N', -1, nv, n, av, ldav, a, lda,
                       dwork, ldw, &scale, &dwork[kw], &ierr);
                if (ierr != 0) {
                    *info = 3;
                    return;
                }

                SLC_DGEMM("T", "N", &pp, &nn, &pp, &one, dv, &lddv, c, &ldc,
                          &zero, &dwork[kw], &pp);
                SLC_DLACPY("F", &pp, &nn, &dwork[kw], &pp, c, &ldc);

                SLC_DGEMM("T", "N", &pp, &mm, &pp, &one, dv, &lddv, d, &ldd,
                          &zero, &dwork[kw], &pp);
                SLC_DLACPY("F", &pp, &mm, &dwork[kw], &pp, d, &ldd);

                f64 scale_inv = one / scale;
                SLC_DGEMM("T", "N", &pp, &nn, &nvv, &scale_inv, bv, &ldbv,
                          dwork, &ldw, &zero, &dwork[kw], &pp);
                SLC_DGEMM("N", "N", &pp, &nn, &nn, &one, &dwork[kw], &pp, a, &lda,
                          &one, c, &ldc);

                SLC_DGEMM("N", "N", &pp, &mm, &nn, &one, &dwork[kw], &pp, b, &ldb,
                          &one, d, &ldd);
            } else {
                SLC_DTRSYL("T", "N", &(i32){1}, &nvv, &nn, av, &ldav, a, &lda,
                           dwork, &ldw, &scale, &ierr);
                if (ierr != 0) {
                    *info = 3;
                    return;
                }

                SLC_DGEMM("T", "N", &pp, &nn, &pp, &one, dv, &lddv, c, &ldc,
                          &zero, &dwork[kw], &pp);
                SLC_DLACPY("F", &pp, &nn, &dwork[kw], &pp, c, &ldc);
                f64 scale_inv = one / scale;
                SLC_DGEMM("T", "N", &pp, &nn, &nvv, &scale_inv, bv, &ldbv,
                          dwork, &ldw, &one, c, &ldc);

                SLC_DGEMM("T", "N", &pp, &mm, &pp, &one, dv, &lddv, d, &ldd,
                          &zero, &dwork[kw], &pp);
                SLC_DLACPY("F", &pp, &mm, &dwork[kw], &pp, d, &ldd);
            }
        } else {
            i32 nn = n;
            i32 pp = p;
            i32 nvv = nv;
            i32 mm = m;
            f64 neg_one = -one;

            SLC_DGEMM("N", "N", &nvv, &nn, &pp, &neg_one, bv, &ldbv, c, &ldc,
                      &zero, dwork, &ldw);

            SLC_DTRSYL("N", "N", &(i32){-1}, &nvv, &nn, av, &ldav, a, &lda,
                       dwork, &ldw, &scale, &ierr);
            if (ierr != 0) {
                *info = 3;
                return;
            }

            SLC_DGEMM("N", "N", &pp, &nn, &pp, &one, dv, &lddv, c, &ldc,
                      &zero, &dwork[kw], &pp);
            SLC_DLACPY("F", &pp, &nn, &dwork[kw], &pp, c, &ldc);
            f64 scale_inv = one / scale;
            SLC_DGEMM("N", "N", &pp, &nn, &nvv, &scale_inv, cv, &ldcv,
                      dwork, &ldw, &one, c, &ldc);

            SLC_DGEMM("N", "N", &pp, &mm, &pp, &one, dv, &lddv, d, &ldd,
                      &zero, &dwork[kw], &pp);
            SLC_DLACPY("F", &pp, &mm, &dwork[kw], &pp, d, &ldd);
        }
    }

    if (rightw) {
        ldwn = (n > 1) ? n : 1;
        kw = n * nw;

        if (conjs) {
            i32 nn = n;
            i32 pp = p;
            i32 nww = nw;
            i32 mm = m;
            f64 neg_one = -one;

            ldw = (nw > 1) ? nw : 1;
            SLC_DGEMM("N", "T", &nww, &nn, &mm, &neg_one, bw, &ldbw, b, &ldb,
                      &zero, dwork, &ldw);

            if (discr) {
                sb04py('N', 'T', -1, nw, n, aw, ldaw, a, lda,
                       dwork, ldw, &scale, &dwork[kw], &ierr);
                if (ierr != 0) {
                    *info = 4;
                    return;
                }

                SLC_DGEMM("N", "T", &nn, &mm, &mm, &one, b, &ldb, dw, &lddw,
                          &zero, &dwork[kw], &ldwn);
                SLC_DLACPY("F", &nn, &mm, &dwork[kw], &ldwn, b, &ldb);

                SLC_DGEMM("N", "T", &pp, &mm, &mm, &one, d, &ldd, dw, &lddw,
                          &zero, &dwork[kw], &pp);
                SLC_DLACPY("F", &pp, &mm, &dwork[kw], &pp, d, &ldd);

                f64 scale_inv = one / scale;
                SLC_DGEMM("T", "T", &nn, &mm, &nww, &scale_inv, dwork, &ldw,
                          cw, &ldcw, &zero, &dwork[kw], &ldwn);
                SLC_DGEMM("N", "N", &nn, &mm, &nn, &one, a, &lda,
                          &dwork[kw], &ldwn, &one, b, &ldb);

                SLC_DGEMM("N", "N", &pp, &mm, &nn, &one, c, &ldc,
                          &dwork[kw], &ldwn, &one, d, &ldd);
            } else {
                SLC_DTRSYL("N", "T", &(i32){1}, &nww, &nn, aw, &ldaw, a, &lda,
                           dwork, &ldw, &scale, &ierr);
                if (ierr != 0) {
                    *info = 4;
                    return;
                }

                SLC_DGEMM("N", "T", &nn, &mm, &mm, &one, b, &ldb, dw, &lddw,
                          &zero, &dwork[kw], &ldwn);
                SLC_DLACPY("F", &nn, &mm, &dwork[kw], &ldwn, b, &ldb);
                f64 scale_inv = one / scale;
                SLC_DGEMM("T", "T", &nn, &mm, &nww, &scale_inv, dwork, &ldw,
                          cw, &ldcw, &one, b, &ldb);

                SLC_DGEMM("N", "T", &pp, &mm, &mm, &one, d, &ldd, dw, &lddw,
                          &zero, &dwork[kw], &pp);
                SLC_DLACPY("F", &pp, &mm, &dwork[kw], &pp, d, &ldd);
            }
        } else {
            i32 nn = n;
            i32 pp = p;
            i32 nww = nw;
            i32 mm = m;

            SLC_DGEMM("N", "N", &nn, &nww, &mm, &one, b, &ldb, cw, &ldcw,
                      &zero, dwork, &ldwn);

            SLC_DTRSYL("N", "N", &(i32){-1}, &nn, &nww, a, &lda, aw, &ldaw,
                       dwork, &ldwn, &scale, &ierr);
            if (ierr != 0) {
                *info = 4;
                return;
            }

            SLC_DGEMM("N", "N", &nn, &mm, &mm, &one, b, &ldb, dw, &lddw,
                      &zero, &dwork[kw], &ldwn);
            SLC_DLACPY("F", &nn, &mm, &dwork[kw], &ldwn, b, &ldb);
            f64 scale_inv = one / scale;
            SLC_DGEMM("N", "N", &nn, &mm, &nww, &scale_inv, dwork, &ldwn,
                      bw, &ldbw, &one, b, &ldb);

            SLC_DGEMM("N", "N", &pp, &mm, &mm, &one, d, &ldd, dw, &lddw,
                      &zero, &dwork[kw], &pp);
            SLC_DLACPY("F", &pp, &mm, &dwork[kw], &pp, d, &ldd);
        }
    }

    dwork[0] = (work > (f64)lw) ? work : (f64)lw;
}
