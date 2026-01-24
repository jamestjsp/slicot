/*
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * MB04RZ - Block-diagonalization of generalized complex Schur form
 *
 * Reduces a complex matrix pair (A,B) in generalized complex Schur form to
 * block-diagonal form using well-conditioned non-unitary equivalence
 * transformations.
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <complex.h>
#include <math.h>
#include <stdbool.h>

void mb04rz(const char *jobx, const char *joby, const char *sort, i32 n,
            f64 pmax, c128 *a, i32 lda, c128 *b, i32 ldb, c128 *x, i32 ldx,
            c128 *y, i32 ldy, i32 *nblcks, i32 *blsize, c128 *alpha,
            c128 *beta, f64 tol, i32 *iwork, i32 *info) {
    const f64 ZERO = 0.0, ONE = 1.0, TEN = 10.0;
    const c128 CZERO = 0.0 + 0.0 * I;
    const c128 CONE = 1.0 + 0.0 * I;
    const c128 NEG_CONE = -1.0 + 0.0 * I;

    bool wantx = (jobx[0] == 'U' || jobx[0] == 'u');
    bool wanty = (joby[0] == 'U' || joby[0] == 'u');
    bool lsorn = (sort[0] == 'N' || sort[0] == 'n');
    bool lsors = (sort[0] == 'S' || sort[0] == 's');
    bool lsort = (sort[0] == 'B' || sort[0] == 'b') || lsors;

    i32 iwantx = wantx ? 1 : 0;
    i32 iwanty = wanty ? 1 : 0;

    *info = 0;

    if (!wantx && !(jobx[0] == 'N' || jobx[0] == 'n')) {
        *info = -1;
    } else if (!wanty && !(joby[0] == 'N' || joby[0] == 'n')) {
        *info = -2;
    } else if (!lsorn && !lsort && !(sort[0] == 'C' || sort[0] == 'c')) {
        *info = -3;
    } else if (n < 0) {
        *info = -4;
    } else if (pmax < ONE) {
        *info = -5;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -7;
    } else if (ldb < (n > 1 ? n : 1)) {
        *info = -9;
    } else if (ldx < 1 || (wantx && ldx < n)) {
        *info = -11;
    } else if (ldy < 1 || (wanty && ldy < n)) {
        *info = -13;
    }

    if (*info != 0) {
        i32 neg_info = -(*info);
        SLC_XERBLA("MB04RZ", &neg_info);
        return;
    }

    *nblcks = 0;
    if (n == 0) {
        return;
    } else if (n == 1) {
        *nblcks = 1;
        blsize[0] = 1;
        alpha[0] = a[0];
        beta[0] = b[0];
        return;
    }

    f64 eps = SLC_DLAMCH("Precision");
    f64 safemn = SLC_DLAMCH("Safe minimum");
    f64 bignum = ONE / safemn;

    f64 dum[1];
    f64 nrmb = SLC_ZLANTR("Froben", "Upper", "NoDiag", &n, &n, b, &ldb, dum);
    f64 scale = SLC_ZLANTR("Froben", "Upper", "NoDiag", &n, &n, a, &lda, dum) / nrmb;
    f64 tolb = TEN * eps * nrmb;

    i32 int1 = 1;
    i32 lda_p1 = lda + 1;
    i32 ldb_p1 = ldb + 1;
    SLC_ZCOPY(&n, a, &lda_p1, alpha, &int1);
    SLC_ZCOPY(&n, b, &ldb_p1, beta, &int1);
    SLC_ZDSCAL(&n, &scale, beta, &int1);

    f64 thresh = 0.0;
    f64 mxev = 0.0;

    if (lsort) {
        thresh = fabs(tol);
        if (thresh == ZERO) {
            thresh = sqrt(sqrt(eps));
        }

        if (tol <= ZERO) {
            mxev = ZERO;
            for (i32 i = 0; i < n; i++) {
                f64 absa = cabs(alpha[i]);
                f64 blr = creal(beta[i]);
                if (blr >= ONE) {
                    mxev = fmax(mxev, absa / blr);
                } else if (absa < bignum * blr) {
                    mxev = fmax(mxev, absa / blr);
                }
            }

            if (thresh <= ONE) {
                if (mxev >= ONE) {
                    thresh = thresh * mxev;
                } else {
                    thresh = fmax(thresh * mxev, eps);
                }
            } else if (mxev < bignum / thresh) {
                thresh = thresh * mxev;
            } else {
                thresh = bignum;
            }
        } else {
            thresh = thresh / scale;
        }
    }

    i32 l11 = 0;
    i32 da11, da22, l22, l22m1;
    i32 ierr;

    while (l11 < n) {
        (*nblcks)++;
        da11 = 1;
        l22 = l11 + da11;

        if (lsort && l11 < n - 1) {
            f64 elr = creal(alpha[l11]);
            f64 eli = cimag(alpha[l11]);
            f64 blr_ref = creal(beta[l11]);

            for (i32 k = l22; k < n; k++) {
                f64 ekr = creal(alpha[k]);
                f64 eki = cimag(alpha[k]);
                f64 bkr = creal(beta[k]);

                f64 d_val, dc_val;
                i32 ma01dz_info = 0;
                ma01dz(elr, eli, blr_ref, ekr, eki, bkr, eps, safemn, &d_val,
                       &dc_val, &ma01dz_info);
                if (ma01dz_info == 1) {
                    *info = 1;
                    return;
                }

                if (dc_val != ZERO && d_val <= thresh) {
                    if (k > l22) {
                        i32 k1 = k + 1;
                        i32 l22_1 = l22 + 1;
                        SLC_ZTGEXC(&iwantx, &iwanty, &n, a, &lda, b, &ldb, x,
                                   &ldx, y, &ldy, &k1, &l22_1, &ierr);

                        if (bkr == ZERO && cabs(b[l22 + l22 * ldb]) < tolb) {
                            b[l22 + l22 * ldb] = CZERO;
                        }

                        for (i32 i = l22; i <= k; i++) {
                            if (cimag(b[i + i * ldb]) != ZERO ||
                                creal(b[i + i * ldb]) < ZERO) {
                                f64 absb = cabs(b[i + i * ldb]);
                                if (absb > safemn) {
                                    c128 sc2 = b[i + i * ldb] / absb;
                                    c128 sc1 = conj(sc2);
                                    b[i + i * ldb] = absb;
                                    i32 nmi = n - i;
                                    if (nmi > 0) {
                                        SLC_ZSCAL(&nmi, &sc1, &b[i + (i + 1) * ldb], &ldb);
                                    }
                                    i32 nmip1 = n - i + 1;
                                    SLC_ZSCAL(&nmip1, &sc1, &a[i + i * lda], &lda);
                                    if (wantx) {
                                        SLC_ZSCAL(&n, &sc2, &x[i * ldx], &int1);
                                    }
                                } else {
                                    b[i + i * ldb] = CZERO;
                                }
                            }

                            alpha[i] = a[i + i * lda];
                            beta[i] = b[i + i * ldb] * scale;
                        }
                    }

                    da11 = da11 + 1;
                    l22 = l11 + da11;
                }
            }
        }

        while (l22 < n) {
            l22m1 = l22 - 1;
            da22 = n - l22m1 - 1;

            ma02az("Transpose", "Full", da11, da22, &a[l11 + l22 * lda], lda,
                   &a[l22 + l11 * lda], lda);
            ma02az("Transpose", "Full", da11, da22, &b[l11 + l22 * ldb], ldb,
                   &b[l22 + l11 * ldb], ldb);

            f64 sc;
            mb04rw(da11, da22, pmax, &a[l11 + l11 * lda], lda,
                   &a[l22 + l22 * lda], lda, &a[l11 + l22 * lda], lda,
                   &b[l11 + l11 * ldb], ldb, &b[l22 + l22 * ldb], ldb,
                   &b[l11 + l22 * ldb], ldb, &sc, iwork, &ierr);

            if (ierr >= 1) {
                ma02az("Transpose", "Full", da22, da11, &a[l22 + l11 * lda], lda,
                       &a[l11 + l22 * lda], lda);
                SLC_ZLASET("Full", &da22, &da11, &CZERO, &CZERO,
                           &a[l22 + l11 * lda], &lda);
                ma02az("Transpose", "Full", da22, da11, &b[l22 + l11 * ldb], ldb,
                       &b[l11 + l22 * ldb], ldb);
                SLC_ZLASET("Full", &da22, &da11, &CZERO, &CZERO,
                           &b[l22 + l11 * ldb], &ldb);

                i32 k_val, l_val;
                f64 d_val;

                if (lsorn || lsors) {
                    c128 av = CZERO;
                    f64 avr, avi, dav;
                    bool done_avg = false;

                    for (i32 i = l11; i <= l22m1; i++) {
                        c128 ei = alpha[i];
                        f64 bir = creal(beta[i]);
                        if (bir >= ONE) {
                            ei = ei / bir;
                            av = av + ei;
                        } else if (fmax(fabs(creal(ei)), fabs(cimag(ei))) <
                                   bignum * bir) {
                            ei = ei / bir;
                            av = av + ei;
                        } else {
                            avr = (creal(ei) > 0) ? ONE : -ONE;
                            avi = ZERO;
                            dav = ZERO;
                            done_avg = true;
                            break;
                        }
                    }

                    if (!done_avg) {
                        avr = creal(av) / da11;
                        avi = cimag(av) / da11;
                        dav = ONE;
                    }

                    d_val = bignum;
                    k_val = l22;

                    for (l_val = l22; l_val < n; l_val++) {
                        f64 elr = creal(alpha[l_val]);
                        f64 eli = cimag(alpha[l_val]);
                        f64 blr = creal(beta[l_val]);

                        if (fmax(blr, dav) == ZERO) {
                            d_val = ZERO;
                            k_val = l_val;
                            break;
                        } else {
                            f64 c_v, dc_v;
                            i32 ma01_info = 0;
                            ma01dz(elr, eli, blr, avr, avi, dav, eps, safemn,
                                   &c_v, &dc_v, &ma01_info);
                            if (ma01_info == 1) {
                                *info = 1;
                                return;
                            }
                            if (dc_v != ZERO && c_v < d_val) {
                                d_val = c_v;
                                k_val = l_val;
                            }
                        }
                    }
                } else {
                    d_val = bignum;
                    k_val = l22;
                    i32 i_ref = l22m1;

                    f64 eir = creal(alpha[i_ref]);
                    f64 eii = cimag(alpha[i_ref]);
                    f64 bir = creal(beta[i_ref]);

                    for (l_val = l22; l_val < n; l_val++) {
                        f64 elr = creal(alpha[l_val]);
                        f64 eli = cimag(alpha[l_val]);
                        f64 blr = creal(beta[l_val]);

                        if (fmax(bir, blr) == ZERO) {
                            d_val = ZERO;
                            k_val = l_val;
                            break;
                        } else {
                            f64 c_v, dc_v;
                            i32 ma01_info = 0;
                            ma01dz(eir, eii, bir, elr, eli, blr, eps, safemn,
                                   &c_v, &dc_v, &ma01_info);
                            if (ma01_info == 1) {
                                *info = 1;
                                return;
                            }
                            if (dc_v != ZERO && c_v < d_val) {
                                d_val = c_v;
                                k_val = l_val;
                            }
                        }
                    }
                }

                if (k_val > l22) {
                    i32 k1 = k_val + 1;
                    i32 l22_1 = l22 + 1;
                    SLC_ZTGEXC(&iwantx, &iwanty, &n, a, &lda, b, &ldb, x, &ldx,
                               y, &ldy, &k1, &l22_1, &ierr);

                    f64 bkr_val = creal(beta[k_val]);
                    if (bkr_val == ZERO && cabs(b[l22 + l22 * ldb]) < tolb) {
                        b[l22 + l22 * ldb] = CZERO;
                    }

                    for (i32 i = l22; i <= k_val; i++) {
                        if (cimag(b[i + i * ldb]) != ZERO ||
                            creal(b[i + i * ldb]) < ZERO) {
                            f64 absb = cabs(b[i + i * ldb]);
                            if (absb > safemn) {
                                c128 sc2 = b[i + i * ldb] / absb;
                                c128 sc1 = conj(sc2);
                                b[i + i * ldb] = absb;
                                i32 nmi = n - i;
                                if (nmi > 0) {
                                    SLC_ZSCAL(&nmi, &sc1, &b[i + (i + 1) * ldb], &ldb);
                                }
                                i32 nmip1 = n - i + 1;
                                SLC_ZSCAL(&nmip1, &sc1, &a[i + i * lda], &lda);
                                if (wantx) {
                                    SLC_ZSCAL(&n, &sc2, &x[i * ldx], &int1);
                                }
                            } else {
                                b[i + i * ldb] = CZERO;
                            }
                        }

                        alpha[i] = a[i + i * lda];
                        beta[i] = b[i + i * ldb] * scale;
                    }
                }

                da11 = da11 + 1;
                l22 = l11 + da11;
                continue;
            }

            break;
        }

        if (l22 < n) {
            if (wantx) {
                SLC_ZGEMM("NoTran", "CTrans", &n, &da11, &da22, &CONE,
                          &x[l22 * ldx], &ldx, &b[l11 + l22 * ldb], &ldb,
                          &CONE, &x[l11 * ldx], &ldx);
            }

            if (wanty) {
                SLC_ZGEMM("NoTran", "NoTran", &n, &da22, &da11, &NEG_CONE,
                          &y[l11 * ldy], &ldy, &a[l11 + l22 * lda], &lda, &CONE,
                          &y[l22 * ldy], &ldy);

                for (i32 i = l11; i <= l22m1; i++) {
                    f64 sc_val = SLC_DZNRM2(&n, &y[i * ldy], &int1);
                    if (fabs(sc_val - ONE) > eps && sc_val > safemn) {
                        sc_val = ONE / sc_val;
                        SLC_ZDSCAL(&da11, &sc_val, &a[l11 + i * lda], &int1);
                        SLC_ZDSCAL(&da11, &sc_val, &b[l11 + i * ldb], &int1);
                        SLC_ZDSCAL(&n, &sc_val, &y[i * ldy], &int1);
                    }
                }
            }

            SLC_ZLASET("Full", &da11, &da22, &CZERO, &CZERO, &a[l11 + l22 * lda],
                       &lda);
            SLC_ZLASET("Full", &da22, &da11, &CZERO, &CZERO, &a[l22 + l11 * lda],
                       &lda);
            SLC_ZLASET("Full", &da11, &da22, &CZERO, &CZERO, &b[l11 + l22 * ldb],
                       &ldb);
            SLC_ZLASET("Full", &da22, &da11, &CZERO, &CZERO, &b[l22 + l11 * ldb],
                       &ldb);
        }

        blsize[*nblcks - 1] = da11;
        l11 = l22;
    }

    if (wantx) {
        i32 l11_x = 0;
        for (i32 l = 0; l < *nblcks; l++) {
            da11 = blsize[l];
            l22 = l11_x + da11;

            for (i32 i = l11_x; i < l22; i++) {
                f64 sc_val = SLC_DZNRM2(&n, &x[i * ldx], &int1);
                if (fabs(sc_val - ONE) > eps && sc_val > safemn) {
                    sc_val = ONE / sc_val;
                    SLC_ZDSCAL(&da11, &sc_val, &a[i + l11_x * lda], &lda);
                    SLC_ZDSCAL(&da11, &sc_val, &b[i + l11_x * ldb], &ldb);
                    SLC_ZDSCAL(&n, &sc_val, &x[i * ldx], &int1);
                }
            }

            l11_x = l22;
        }
    }

    if (wanty) {
        i32 l11_y = n - da11;

        for (i32 i = l11_y; i < n; i++) {
            f64 sc_val = SLC_DZNRM2(&n, &y[i * ldy], &int1);
            if (fabs(sc_val - ONE) > eps && sc_val > safemn) {
                sc_val = ONE / sc_val;
                SLC_ZDSCAL(&da11, &sc_val, &a[l11_y + i * lda], &int1);
                SLC_ZDSCAL(&da11, &sc_val, &b[l11_y + i * ldb], &int1);
                SLC_ZDSCAL(&n, &sc_val, &y[i * ldy], &int1);
            }
        }
    }

    f64 inv_scale = ONE / scale;
    SLC_ZDSCAL(&n, &inv_scale, beta, &int1);
}
