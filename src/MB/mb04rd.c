/*
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * MB04RD - Block-diagonalization of generalized real Schur form
 *
 * Reduces a matrix pair (A,B) in generalized real Schur form to block-diagonal
 * form using well-conditioned non-orthogonal equivalence transformations.
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <stdbool.h>
#include <string.h>

void mb04rd(const char *jobx, const char *joby, const char *sort, i32 n,
            f64 pmax, f64 *a, i32 lda, f64 *b, i32 ldb, f64 *x, i32 ldx,
            f64 *y, i32 ldy, i32 *nblcks, i32 *blsize, f64 *alphar,
            f64 *alphai, f64 *beta, f64 tol, i32 *iwork, f64 *dwork,
            i32 ldwork, i32 *info) {
    const f64 ZERO = 0.0, ONE = 1.0, TEN = 10.0;

    bool wantx = (jobx[0] == 'U' || jobx[0] == 'u');
    bool wanty = (joby[0] == 'U' || joby[0] == 'u');
    bool lsorn = (sort[0] == 'N' || sort[0] == 'n');
    bool lsors = (sort[0] == 'S' || sort[0] == 's');
    bool lsort = (sort[0] == 'B' || sort[0] == 'b') || lsors;
    bool lquery = (ldwork == -1);

    i32 iwantx = wantx ? 1 : 0;
    i32 iwanty = wanty ? 1 : 0;

    i32 minwrk, maxwrk;

    *info = 0;

    if (n <= 1) {
        minwrk = 1;
    } else {
        minwrk = 4 * n + 16;
    }

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
    } else {
        if (lquery) {
            i32 ifst = 1, ilst = n, ierr;
            SLC_DTGEXC(&iwantx, &iwanty, &n, a, &lda, b, &ldb, x, &ldx, y, &ldy,
                       &ifst, &ilst, dwork, &ldwork, &ierr);
            maxwrk = (i32)dwork[0];
            dwork[0] = (f64)maxwrk;
            return;
        } else {
            if (ldwork < minwrk) {
                *info = -22;
                dwork[0] = (f64)minwrk;
                if (ldwork == 0)
                    return;
            } else {
                maxwrk = minwrk;
            }
        }
    }

    if (*info != 0) {
        i32 neg_info = -(*info);
        SLC_XERBLA("MB04RD", &neg_info);
        return;
    }

    *nblcks = 0;
    if (n == 0) {
        dwork[0] = ONE;
        return;
    } else if (n == 1) {
        *nblcks = 1;
        blsize[0] = 1;
        alphar[0] = a[0];
        alphai[0] = ZERO;
        beta[0] = b[0];
        dwork[0] = ONE;
        return;
    }

    f64 eps = SLC_DLAMCH("Precision");
    f64 safemn = SLC_DLAMCH("Safe minimum");
    f64 bignum = ONE / safemn;

    f64 dum[1];
    f64 nrmb = SLC_DLANTR("Froben", "Upper", "NoDiag", &n, &n, b, &ldb, dum);
    f64 scale = SLC_DLANHS("Froben", &n, a, &lda, dum) / nrmb;
    f64 tolb = TEN * eps * nrmb;

    i32 ierr;
    mb03qv(n, a, lda, b, ldb, alphar, alphai, beta, &ierr);

    i32 int1 = 1;
    SLC_DSCAL(&n, &scale, beta, &int1);

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
                f64 absa = SLC_DLAPY2(&alphar[i], &alphai[i]);
                f64 blr = beta[i];
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

    while (l11 < n) {
        (*nblcks)++;
        if (alphai[l11] > ZERO) {
            da11 = 2;
        } else {
            da11 = 1;
        }
        l22 = l11 + da11;

        if (lsort && l11 < n - 1) {
            f64 elr = alphar[l11];
            f64 eli = alphai[l11];
            f64 blr_ref = beta[l11];

            i32 k = l22;
            i32 kf = l22;
            if (k < n && alphai[k] > ZERO)
                kf = kf + 1;

            while (k < n) {
                bool pinf = false;
                f64 ekr = alphar[k];
                f64 eki = alphai[k];
                f64 bkr = beta[k];
                if (bkr == ZERO) {
                    pinf = a[k + k * lda] > ZERO;
                }

                f64 c_val, dc_val;
                i32 ma01dz_info = 0;
                ma01dz(elr, eli, blr_ref, ekr, eki, bkr, eps, safemn, &c_val,
                       &dc_val, &ma01dz_info);
                if (ma01dz_info == 1) {
                    *info = 1;
                    return;
                }

                if (dc_val != ZERO && c_val <= thresh) {
                    if (k > l22) {
                        i32 k1 = k + 1;
                        i32 l22_1 = l22 + 1;
                        SLC_DTGEXC(&iwantx, &iwanty, &n, a, &lda, b, &ldb, x,
                                   &ldx, y, &ldy, &k1, &l22_1, dwork, &ldwork,
                                   &ierr);

                        if (k < n - 1) {
                            if (a[(k + 1) + k * lda] != ZERO)
                                kf = k + 1;
                        }

                        if (bkr == ZERO && fabs(b[l22 + l22 * lda]) < tolb) {
                            b[l22 + l22 * lda] = ZERO;
                            if (pinf && a[l22 + l22 * lda] < ZERO) {
                                f64 neg1 = -ONE;
                                i32 l22_len = l22;
                                SLC_DSCAL(&l22_len, &neg1, &b[l22 * ldb], &int1);
                                i32 l22p1 = l22 + 1;
                                SLC_DSCAL(&l22p1, &neg1, &a[l22 * lda], &int1);
                                if (wanty) {
                                    SLC_DSCAL(&n, &neg1, &y[l22 * ldy], &int1);
                                }
                            }
                        }

                        for (i32 i = l22; i <= kf; i++) {
                            if (b[i + i * ldb] < ZERO) {
                                f64 neg1 = -ONE;
                                i32 ip1 = i + 1;
                                SLC_DSCAL(&ip1, &neg1, &b[i * ldb], &int1);
                                SLC_DSCAL(&ip1, &neg1, &a[i * lda], &int1);
                                if (i < n - 1) {
                                    if (a[(i + 1) + i * lda] != ZERO)
                                        a[(i + 1) + i * lda] =
                                            -a[(i + 1) + i * lda];
                                }
                                if (wanty) {
                                    SLC_DSCAL(&n, &neg1, &y[i * ldy], &int1);
                                }
                            }
                        }

                        i32 nqv = kf - l22 + 1;
                        mb03qv(nqv, &a[l22 + l22 * lda], lda,
                               &b[l22 + l22 * ldb], ldb, &alphar[l22],
                               &alphai[l22], &beta[l22], &ierr);

                        SLC_DSCAL(&nqv, &scale, &beta[l22], &int1);
                    }

                    if (alphai[l22] > ZERO) {
                        da11 = da11 + 2;
                    } else {
                        da11 = da11 + 1;
                    }
                    l22 = l11 + da11;
                }
                if (alphai[k] > ZERO) {
                    k = k + 2;
                } else {
                    k = k + 1;
                }
                kf = k;
            }
        }

        while (l22 < n) {
            l22m1 = l22 - 1;
            da22 = n - l22m1 - 1;

            ma02ad("Full", da11, da22, &a[l11 + l22 * lda], lda,
                   &a[l22 + l11 * lda], lda);
            ma02ad("Full", da11, da22, &b[l11 + l22 * ldb], ldb,
                   &b[l22 + l11 * ldb], ldb);

            f64 sc;
            mb04rt(da11, da22, pmax, &a[l11 + l11 * lda], lda,
                   &a[l22 + l22 * lda], lda, &a[l11 + l22 * lda], lda,
                   &b[l11 + l11 * ldb], ldb, &b[l22 + l22 * ldb], ldb,
                   &b[l11 + l22 * ldb], ldb, &sc, iwork, &ierr);

            if (ierr >= 1) {
                ma02ad("Full", da22, da11, &a[l22 + l11 * lda], lda,
                       &a[l11 + l22 * lda], lda);
                SLC_DLASET("Full", &da22, &da11, &ZERO, &ZERO,
                           &a[l22 + l11 * lda], &lda);
                ma02ad("Full", da22, da11, &b[l22 + l11 * ldb], ldb,
                       &b[l11 + l22 * ldb], ldb);
                SLC_DLASET("Full", &da22, &da11, &ZERO, &ZERO,
                           &b[l22 + l11 * ldb], &ldb);

                bool goon =
                    (l22 == n - 1 && da11 == 1) || l22 < n - 2;
                i32 k_val, kf_val, l_val;
                bool pinf_val;
                f64 d_val;

                if ((lsorn || lsors) && goon) {
                    f64 avr = ZERO, avi = ZERO, dav;
                    bool done_avg = false;

                    for (i32 i = l11; i <= l22m1; i++) {
                        f64 eir = alphar[i];
                        f64 eii = alphai[i];
                        f64 bir = beta[i];
                        if (bir >= ONE) {
                            eir = eir / bir;
                            eii = eii / bir;
                            avr = avr + eir;
                            avi = avi + eii;
                        } else if (fmax(fabs(eir), fabs(eii)) <
                                   bignum * bir) {
                            eir = eir / bir;
                            eii = eii / bir;
                            avr = avr + eir;
                            avi = avi + eii;
                        } else {
                            avr = (eir > 0) ? ONE : -ONE;
                            avi = ZERO;
                            dav = ZERO;
                            done_avg = true;
                            break;
                        }
                    }

                    if (!done_avg) {
                        avr = avr / da11;
                        avi = avi / da11;
                        dav = ONE;
                    }

                    d_val = bignum;
                    k_val = l22;
                    l_val = l22;
                    pinf_val = false;

                    while (l_val < n) {
                        f64 elr = alphar[l_val];
                        f64 eli = alphai[l_val];
                        f64 blr = beta[l_val];
                        if (fmax(blr, dav) == ZERO) {
                            d_val = ZERO;
                            k_val = l_val;
                            pinf_val = a[l_val + l_val * lda] > ZERO;
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
                            if (alphai[l_val] > ZERO) {
                                l_val = l_val + 2;
                            } else {
                                l_val = l_val + 1;
                            }
                        }
                    }

                    if (alphai[k_val] > ZERO) {
                        kf_val = k_val + 1;
                    } else {
                        kf_val = k_val;
                    }
                } else {
                    d_val = bignum;
                    k_val = l22;
                    l_val = l22;
                    i32 i_ref = l22m1;

                    f64 eir = alphar[i_ref];
                    f64 eii = alphai[i_ref];
                    f64 bir = beta[i_ref];

                    pinf_val = false;

                    while (l_val < n) {
                        f64 elr = alphar[l_val];
                        f64 eli = alphai[l_val];
                        f64 blr = beta[l_val];

                        if (fmax(bir, blr) == ZERO) {
                            d_val = ZERO;
                            k_val = l_val;
                            pinf_val = a[l_val + l_val * lda] > ZERO;
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
                            if (alphai[l_val] > ZERO) {
                                l_val = l_val + 2;
                            } else {
                                l_val = l_val + 1;
                            }
                        }
                    }

                    if (alphai[k_val] > ZERO) {
                        kf_val = k_val + 1;
                    } else {
                        kf_val = k_val;
                    }
                }

                if (k_val > l22) {
                    i32 k1 = k_val + 1;
                    i32 l22_1 = l22 + 1;
                    SLC_DTGEXC(&iwantx, &iwanty, &n, a, &lda, b, &ldb, x, &ldx,
                               y, &ldy, &k1, &l22_1, dwork, &ldwork, &ierr);

                    if (k_val < n - 1) {
                        if (a[(k_val + 1) + k_val * lda] != ZERO)
                            kf_val = k_val + 1;
                    }

                    f64 blr_kval = beta[k_val];
                    if (blr_kval == ZERO &&
                        fabs(b[l22 + l22 * ldb]) < tolb) {
                        b[l22 + l22 * ldb] = ZERO;
                        if (pinf_val && a[l22 + l22 * lda] < ZERO) {
                            f64 neg1 = -ONE;
                            SLC_DSCAL(&l22m1, &neg1, &b[l22 * ldb], &int1);
                            i32 l22_len = l22 + 1;
                            SLC_DSCAL(&l22_len, &neg1, &a[l22 * lda], &int1);
                            if (wanty) {
                                SLC_DSCAL(&n, &neg1, &y[l22 * ldy], &int1);
                            }
                        }
                    }

                    for (i32 i = l22; i <= kf_val; i++) {
                        if (b[i + i * ldb] < ZERO) {
                            f64 neg1 = -ONE;
                            i32 ip1 = i + 1;
                            SLC_DSCAL(&ip1, &neg1, &b[i * ldb], &int1);
                            SLC_DSCAL(&ip1, &neg1, &a[i * lda], &int1);
                            if (i < n - 1) {
                                if (a[(i + 1) + i * lda] != ZERO)
                                    a[(i + 1) + i * lda] =
                                        -a[(i + 1) + i * lda];
                            }
                            if (wanty) {
                                SLC_DSCAL(&n, &neg1, &y[i * ldy], &int1);
                            }
                        }
                    }

                    i32 nqv = kf_val - l22 + 1;
                    mb03qv(nqv, &a[l22 + l22 * lda], lda, &b[l22 + l22 * ldb],
                           ldb, &alphar[l22], &alphai[l22], &beta[l22], &ierr);

                    SLC_DSCAL(&nqv, &scale, &beta[l22], &int1);
                }

                if (alphai[l22] > ZERO) {
                    da11 = da11 + 2;
                } else {
                    da11 = da11 + 1;
                }
                l22 = l11 + da11;
                continue;
            }

            break;
        }

        if (l22 < n) {
            if (wantx) {
                f64 alpha = ONE, beta_val = ONE;
                SLC_DGEMM("NoTran", "Trans", &n, &da11, &da22, &alpha,
                          &x[l22 * ldx], &ldx, &b[l11 + l22 * ldb], &ldb,
                          &beta_val, &x[l11 * ldx], &ldx);
            }

            if (wanty) {
                f64 neg1 = -ONE, one = ONE;
                SLC_DGEMM("NoTran", "NoTran", &n, &da22, &da11, &neg1,
                          &y[l11 * ldy], &ldy, &a[l11 + l22 * lda], &lda, &one,
                          &y[l22 * ldy], &ldy);

                for (i32 i = l11; i <= l22m1; i++) {
                    f64 sc_val = SLC_DNRM2(&n, &y[i * ldy], &int1);
                    if (fabs(sc_val - ONE) > eps && sc_val > safemn) {
                        sc_val = ONE / sc_val;
                        SLC_DSCAL(&da11, &sc_val, &a[l11 + i * lda], &int1);
                        SLC_DSCAL(&da11, &sc_val, &b[l11 + i * ldb], &int1);
                        SLC_DSCAL(&n, &sc_val, &y[i * ldy], &int1);
                    }
                }
            }

            SLC_DLASET("Full", &da11, &da22, &ZERO, &ZERO, &a[l11 + l22 * lda],
                       &lda);
            SLC_DLASET("Full", &da22, &da11, &ZERO, &ZERO, &a[l22 + l11 * lda],
                       &lda);
            SLC_DLASET("Full", &da11, &da22, &ZERO, &ZERO, &b[l11 + l22 * ldb],
                       &ldb);
            SLC_DLASET("Full", &da22, &da11, &ZERO, &ZERO, &b[l22 + l11 * ldb],
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
                f64 sc_val = SLC_DNRM2(&n, &x[i * ldx], &int1);
                if (fabs(sc_val - ONE) > eps && sc_val > safemn) {
                    sc_val = ONE / sc_val;
                    SLC_DSCAL(&da11, &sc_val, &a[i + l11_x * lda], &lda);
                    SLC_DSCAL(&da11, &sc_val, &b[i + l11_x * ldb], &ldb);
                    SLC_DSCAL(&n, &sc_val, &x[i * ldx], &int1);
                }
            }

            l11_x = l22;
        }
    }

    if (wanty) {
        i32 l11_y = n - da11;

        for (i32 i = l11_y; i < n; i++) {
            f64 sc_val = SLC_DNRM2(&n, &y[i * ldy], &int1);
            if (fabs(sc_val - ONE) > eps && sc_val > safemn) {
                sc_val = ONE / sc_val;
                SLC_DSCAL(&da11, &sc_val, &a[l11_y + i * lda], &int1);
                SLC_DSCAL(&da11, &sc_val, &b[l11_y + i * ldb], &int1);
                SLC_DSCAL(&n, &sc_val, &y[i * ldy], &int1);
            }
        }
    }

    f64 inv_scale = ONE / scale;
    SLC_DSCAL(&n, &inv_scale, beta, &int1);

    dwork[0] = (f64)maxwrk;
}
